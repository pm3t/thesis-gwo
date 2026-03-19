import customtkinter as ctk
from tkinter import messagebox, filedialog
from models.ensemble import WeightedEnsembleModel
from utils.metrics import get_metrics
from utils.visualizer import Visualizer
from ui.widgets import MetricTable, ComparisonTable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class EnsembleTab(ctk.CTkFrame):
    def __init__(self, master, data_processor, model_results, gwo_results):
        super().__init__(master)
        
        self.data_processor = data_processor
        self.model_results = model_results
        self.gwo_results = gwo_results
        self.canvas_comp = None
        self.canvas_error = None
        
        # Configure grid
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top Frame: Controls & Weights
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        
        self.lbl_weights = ctk.CTkLabel(self.top_frame, text="Current Weights (from GWO):", font=ctk.CTkFont(weight="bold"))
        self.lbl_weights.pack(side="left", padx=10)
        
        self.btn_run_ensemble = ctk.CTkButton(self.top_frame, text="Run Ensemble", command=self.run_ensemble)
        self.btn_run_ensemble.pack(side="left", padx=10)
        
        self.btn_export = ctk.CTkButton(self.top_frame, text="Export Results (CSV)", command=self.export_csv)
        self.btn_export.pack(side="left", padx=10)

        # Middle: Split View
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        # Left: Error Chart (Scrollable)
        self.left_results = ctk.CTkScrollableFrame(self, label_text="Comparison Visualization", orientation="horizontal")
        self.left_results.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.error_canvas_frame = ctk.CTkFrame(self.left_results)
        self.error_canvas_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Right: Comparison Table (Scrollable)
        self.right_results = ctk.CTkScrollableFrame(self, label_text="Comparison Metrics", orientation="horizontal")
        self.right_results.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        
        self.comparison_table = ComparisonTable(self.right_results)
        self.comparison_table.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.lbl_improvement = ctk.CTkLabel(self, text="Accuracy Improvement: -", font=ctk.CTkFont(size=14, weight="bold"))
        self.lbl_improvement.grid(row=2, column=0, columnspan=2, pady=10)

    def run_ensemble(self):
        if 'best_weights' not in self.gwo_results:
            messagebox.showwarning("Warning", "Please run GWO optimization first.")
            return

        try:
            weights = self.gwo_results['best_weights']
            y_pred_ma = self.model_results['MA']['pred']
            y_pred_es = self.model_results['ES']['pred']
            y_pred_lr = self.model_results['LR']['pred']
            _, _, _, y_test = self.data_processor.get_train_test_data()
            dates_test, _ = self.data_processor.test_df[self.data_processor.date_col], self.data_processor.test_df[self.data_processor.target_col]

            model = WeightedEnsembleModel()
            model.set_weights(weights)
            y_ens_pred = model.predict(y_pred_ma, y_pred_es, y_pred_lr)
            
            # 1. Metrics for GWO Ensemble
            metrics = get_metrics(y_test, y_ens_pred)
            # self.ens_metrics.update_metrics(metrics) # Removed
            self.model_results['GWO Ensemble'] = {
                'pred': y_ens_pred,
                'metrics': metrics
            }
            
            # 2. Equal Average Ensemble
            y_avg_pred = (y_pred_ma + y_pred_es + y_pred_lr) / 3.0
            avg_metrics = get_metrics(y_test, y_avg_pred)
            self.model_results['Equal Average'] = {
                'pred': y_avg_pred,
                'metrics': avg_metrics
            }
            
            # 3. Update Table
            self.comparison_table.update_data(self.model_results)
            
            # 3. Accuracy Improvement Calc
            best_baseline_mape = min(
                self.model_results['MA']['metrics']['MAPE'],
                self.model_results['ES']['metrics']['MAPE'],
                self.model_results['LR']['metrics']['MAPE']
            )
            improvement = ((best_baseline_mape - metrics['MAPE']) / best_baseline_mape) * 100
            self.lbl_improvement.configure(text=f"GWO Optimization vs Best Baseline: {improvement:.2f}% Improvement")
            
            messagebox.showinfo("Success", "Ensemble model evaluated.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ensemble failed: {e}")

        # 4. Error Chart Comparison
        if self.canvas_error is not None: self.canvas_error.get_tk_widget().destroy()
        
        metrics_all = {k: v['metrics'] for k, v in self.model_results.items() if 'metrics' in v}
        fig_error, _ = Visualizer.plot_error_comparison(metrics_all)
        self.canvas_error = FigureCanvasTkAgg(fig_error, master=self.error_canvas_frame)
        self.canvas_error.draw()
        self.canvas_error.get_tk_widget().pack(fill="both", expand=True)

    def export_csv(self):
        if 'GWO Ensemble' not in self.model_results:
            messagebox.showwarning("Warning", "Run ensemble first.")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                dates_test = self.data_processor.test_df[self.data_processor.date_col]
                y_test = self.data_processor.test_df[self.data_processor.target_col]
                y_pred = self.model_results['GWO Ensemble']['pred']
                
                export_df = pd.DataFrame({
                    'Date': dates_test,
                    'Actual': y_test,
                    'MA': self.model_results['MA']['pred'],
                    'ES': self.model_results['ES']['pred'],
                    'LR': self.model_results['LR']['pred'],
                    'Equal_Average': self.model_results['Equal Average']['pred'],
                    'GWO_Ensemble': self.model_results['GWO Ensemble']['pred']
                })
                export_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
                # Also save weights
                weight_path = file_path.replace(".csv", "_weights.txt")
                weights = self.gwo_results['best_weights']
                with open(weight_path, "w") as f:
                    f.write(f"Optimal Weights Found by GWO:\n")
                    f.write(f"w1 (Moving Average): {weights[0]:.6f}\n")
                    f.write(f"w2 (Exponential Smoothing): {weights[1]:.6f}\n")
                    f.write(f"w3 (Linear Regression): {weights[2]:.6f}\n")
                    f.write(f"\nModel Performance (MAPE):\n")
                    for k, v in self.model_results.items():
                        f.write(f"{k}: {v['metrics']['MAPE']:.4f}%\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
