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
        
        self.btn_export_report = ctk.CTkButton(self.top_frame, text="Export Thesis Report (TXT)", command=self.export_thesis_report)
        self.btn_export_report.pack(side="left", padx=10)

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
        self.lbl_improvement.grid(row=2, column=0, columnspan=2, pady=(10, 10))

    def run_ensemble(self):
        if 'best_weights' not in self.gwo_results:
            messagebox.showwarning("Warning", "Please run GWO optimization first.")
            return

        try:
            weights = self.gwo_results['best_weights']
            y_pred_ma = self.model_results['MA']['pred']
            y_pred_es = self.model_results['ES']['pred']
            y_pred_rnn = self.model_results['RNN']['pred']
            _, _, _, y_test = self.data_processor.get_train_test_data()
            dates_test, _ = self.data_processor.test_df[self.data_processor.date_col], self.data_processor.test_df[self.data_processor.target_col]

            model = WeightedEnsembleModel()
            model.set_weights(weights)
            y_ens_pred = model.predict(y_pred_ma, y_pred_es, y_pred_rnn)
            
            # 1. Metrics for GWO Ensemble
            metrics = get_metrics(y_test, y_ens_pred)
            self.model_results['GWO Ensemble'] = {
                'pred': y_ens_pred,
                'metrics': metrics
            }
            
            # 2. Equal Average Ensemble
            y_avg_pred = (y_pred_ma + y_pred_es + y_pred_rnn) / 3.0
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
                self.model_results['RNN']['metrics']['MAPE']
            )
            improvement = ((best_baseline_mape - metrics['MAPE']) / best_baseline_mape) * 100
            self.lbl_improvement.configure(text=f"Optimization vs Best Baseline: {improvement:.2f}% Improvement")
            
            messagebox.showinfo("Success", "Ensemble model evaluated successfully.")
            
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
                    'RNN': self.model_results['RNN']['pred'],
                    'Equal_Average': self.model_results['Equal Average']['pred'],
                    'GWO_Ensemble': self.model_results['GWO Ensemble']['pred']
                })
                export_df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {file_path}")
                
                # Also save weights
                weight_path = file_path.replace(".csv", "_weights.txt")
                weights = self.gwo_results['best_weights']
                opt_name = self.gwo_results.get('optimizer_name', 'GWO')
                with open(weight_path, "w") as f:
                    f.write(f"Optimal Weights Found by {opt_name}:\n")
                    f.write(f"w1 (Moving Average): {weights[0]:.6f}\n")
                    f.write(f"w2 (Exponential Smoothing): {weights[1]:.6f}\n")
                    f.write(f"w3 (Simple RNN): {weights[2]:.6f}\n")
                    f.write(f"\nModel Performance (MAPE):\n")
                    for k, v in self.model_results.items():
                        if 'metrics' in v:
                            f.write(f"{k}: {v['metrics']['MAPE']:.4f}%\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")

    def export_thesis_report(self):
        if 'GWO Ensemble' not in self.model_results:
            messagebox.showwarning("Warning", "Please run the Ensemble evaluation first before exporting the full report.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if not file_path:
            return

        try:
            with open(file_path, "w") as f:
                f.write("=========================================================================\n")
                f.write("             LAPORAN EKSPERIMEN TESIS - FORECASTING ENSEMBLE             \n")
                f.write("=========================================================================\n\n")

                # 1. Dataset Statistics
                f.write("1. STATISTIK DESKRIPTIF DATASET\n")
                f.write("-----------------------------------------\n")
                
                # Awal
                stats = self.data_processor.get_stats()
                if stats is not None:
                    f.write(">>> DATASET AWAL <<<\n")
                    if self.data_processor.raw_data is not None:
                        f.write(f"Jumlah Baris & Kolom: {self.data_processor.raw_data.shape}\n")
                        f.write("--- 10 Data Teratas (Head) ---\n")
                        f.write(self.data_processor.raw_data.head(10).to_string())
                        f.write("\n\n")
                    f.write("--- Statistik Deskriptif ---\n")
                    f.write(stats.to_string())
                    f.write("\n\n")
                
                # Preprocessed
                if self.data_processor.df is not None:
                    f.write(">>> DATASET HASIL PREPROCESS <<<\n")
                    f.write(f"Jumlah Baris & Kolom: {self.data_processor.df.shape}\n")
                    f.write("--- 10 Data Teratas (Head) ---\n")
                    f.write(self.data_processor.df.head(10).to_string())
                    f.write("\n\n")
                    f.write("--- Statistik Deskriptif ---\n")
                    f.write(self.data_processor.df.describe().to_string())
                    f.write("\n\n")
                
                # Split
                if self.data_processor.train_df is not None and self.data_processor.test_df is not None:
                    f.write(">>> DATASET HASIL SPLIT <<<\n")
                    f.write(f"Data Latih (Train) shape: {self.data_processor.train_df.shape}\n")
                    f.write(f"Data Uji (Test) shape: {self.data_processor.test_df.shape}\n\n")
                    f.write("--- Statistik Data Latih ---\n")
                    f.write(self.data_processor.train_df.describe().to_string())
                    f.write("\n\n--- Statistik Data Uji ---\n")
                    f.write(self.data_processor.test_df.describe().to_string())
                    f.write("\n")
                
                if stats is None and self.data_processor.df is None:
                    f.write("Data statistik tidak tersedia.\n")
                f.write("\n")

                # 2. Model Baseline Performance
                f.write("2. PERFORMA MODEL INDIVIDU (BASELINE)\n")
                f.write("-----------------------------------------\n")
                baselines = ['MA', 'ES', 'RNN']
                for model_name in baselines:
                    if model_name in self.model_results:
                        m_res = self.model_results[model_name]['metrics']
                        f.write(f"Model: {model_name}\n")
                        f.write(f"  MAPE: {m_res['MAPE']:.4f}%\n")
                        f.write(f"  MAE:  {m_res['MAE']:.4f}\n")
                        f.write(f"  MSE:  {m_res['MSE']:.4f}\n")
                        f.write(f"  RMSE: {m_res['RMSE']:.4f}\n")
                        f.write(f"  R2:   {m_res['R2']:.4f}\n\n")
                
                # 3. Metaheuristic Optimization Results & Stability
                f.write("3. HASIL OPTIMASI & UJI KESTABILAN (GWO)\n")
                f.write("-----------------------------------------\n")
                import os
                found_stability = False
                for short_name in ["GWO"]:
                    stats_path = f"Analysis/Stability_Stats_{short_name}.txt"
                    if os.path.exists(stats_path):
                        found_stability = True
                        f.write(f"\n>>> STATISTIK KESTABILAN {short_name} (30 RUNS) <<<\n")
                        with open(stats_path, "r") as sf:
                            f.write(sf.read())
                        f.write("\n")
                
                if not found_stability:
                    f.write("Hasil pengujian stabilitas multi-run tidak ditemukan di folder Analysis/.\n")
                f.write("\n")

                # 4. Ensemble Model Comparison
                f.write("4. PERBANDINGAN MODEL ENSEMBLE\n")
                f.write("-----------------------------------------\n")
                ensembles = ['Equal Average', 'GWO Ensemble']
                for model_name in ensembles:
                    if model_name in self.model_results:
                        m_res = self.model_results[model_name]['metrics']
                        f.write(f"Model: {model_name}\n")
                        f.write(f"  MAPE: {m_res['MAPE']:.4f}%\n")
                        f.write(f"  MAE:  {m_res['MAE']:.4f}\n")
                        f.write(f"  MSE:  {m_res['MSE']:.4f}\n")
                        f.write(f"  RMSE: {m_res['RMSE']:.4f}\n")
                        f.write(f"  R2:   {m_res['R2']:.4f}\n\n")
                

                f.write("\n=========================================================================\n")
                f.write("                        AKHIR DARI LAPORAN EKSPERIMEN                    \n")
                f.write("=========================================================================\n")

            messagebox.showinfo("Success", f"Full thesis report exported successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {e}")
