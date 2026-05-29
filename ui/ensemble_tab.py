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
        self.lbl_improvement.grid(row=2, column=0, columnspan=2, pady=(10, 2))
        
        self.lbl_wilcoxon = ctk.CTkLabel(self, text="Wilcoxon Signed-Rank Test: -", font=ctk.CTkFont(size=13, weight="bold"), text_color="#2ba37a")
        self.lbl_wilcoxon.grid(row=3, column=0, columnspan=2, pady=2)

        self.diag_frame = ctk.CTkFrame(self)
        self.diag_frame.grid(row=4, column=0, columnspan=2, pady=10)

        self.btn_residuals = ctk.CTkButton(self.diag_frame, text="Show Residual Analysis", command=self.show_residuals, state="disabled")
        self.btn_residuals.pack(side="left", padx=10)

        self.btn_correlation = ctk.CTkButton(self.diag_frame, text="Show Correlation Plot", command=self.show_correlation, state="disabled")
        self.btn_correlation.pack(side="left", padx=10)

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
            
            # 3.5 Calculate Wilcoxon signed-rank test against the best baseline
            baselines = ['MA', 'ES', 'RNN']
            mapes = [self.model_results[b]['metrics']['MAPE'] for b in baselines]
            best_baseline_name = baselines[np.argmin(mapes)]
            y_best_baseline_pred = self.model_results[best_baseline_name]['pred']
            
            from utils.metrics import calculate_wilcoxon_test
            stat, p_val = calculate_wilcoxon_test(y_test, y_ens_pred, y_best_baseline_pred)
            
            conclusion = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "NOT SIGNIFICANT (p >= 0.05)"
            self.lbl_wilcoxon.configure(text=f"Wilcoxon Test vs Best Baseline ({best_baseline_name}): p-value = {p_val:.6f} ({conclusion})")
            
            # Enable diagnostic buttons
            self.btn_residuals.configure(state="normal")
            self.btn_correlation.configure(state="normal")
            
            messagebox.showinfo("Success", "Ensemble model evaluated and statistical tests completed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Ensemble failed: {e}")

        # 4. Error Chart Comparison
        if self.canvas_error is not None: self.canvas_error.get_tk_widget().destroy()
        
        metrics_all = {k: v['metrics'] for k, v in self.model_results.items() if 'metrics' in v}
        fig_error, _ = Visualizer.plot_error_comparison(metrics_all)
        self.canvas_error = FigureCanvasTkAgg(fig_error, master=self.error_canvas_frame)
        self.canvas_error.draw()
        self.canvas_error.get_tk_widget().pack(fill="both", expand=True)

    def show_residuals(self):
        if 'GWO Ensemble' not in self.model_results:
            return
        
        _, _, _, y_test = self.data_processor.get_train_test_data()
        y_ens_pred = self.model_results['GWO Ensemble']['pred']
        
        fig = Visualizer.plot_residuals(y_test, y_ens_pred)
        
        top = ctk.CTkToplevel(self)
        top.title("Residual Analysis Diagnostics")
        top.geometry("900x500")
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_correlation(self):
        if 'GWO Ensemble' not in self.model_results:
            return
            
        _, _, _, y_test = self.data_processor.get_train_test_data()
        y_ens_pred = self.model_results['GWO Ensemble']['pred']
        
        fig, _ = Visualizer.plot_actual_vs_predicted(y_test, y_ens_pred)
        
        top = ctk.CTkToplevel(self)
        top.title("Correlation Scatter Plot")
        top.geometry("600x600")
        
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

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
                stats = self.data_processor.get_stats()
                if stats is not None:
                    f.write(stats.to_string())
                    f.write("\n")
                else:
                    f.write("Data statistik tidak tersedia.\n")
                f.write("\n")

                # 2. ADF Test Results
                f.write("2. UJI STASIONERITAS (AUGMENTED DICKEY-FULLER TEST)\n")
                f.write("-----------------------------------------\n")
                adf_res = getattr(self.data_processor, 'adf_results', None)
                if adf_res is None and self.data_processor.df is not None:
                    adf_res = self.data_processor.run_adf_test()
                
                if adf_res is not None:
                    conclusion = "STATIONARY (Stasioner)" if adf_res['is_stationary'] else "NON-STATIONARY (Tidak Stasioner)"
                    f.write(f"ADF Statistic: {adf_res['adf_stat']:.6f}\n")
                    f.write(f"p-value: {adf_res['p_value']:.6e}\n")
                    f.write(f"Conclusion: {conclusion}\n\n")
                    f.write("Critical Values:\n")
                    for k, v in adf_res['critical_values'].items():
                        f.write(f"  {k}: {v:.6f}\n")
                else:
                    f.write("Uji ADF belum dijalankan atau data tidak tersedia.\n")
                f.write("\n")

                # 3. Model Baseline Performance
                f.write("3. PERFORMA MODEL INDIVIDU (BASELINE)\n")
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
                
                # 4. Metaheuristic Optimization Results & Stability
                f.write("4. HASIL OPTIMASI & UJI KESTABILAN (GWO / PSO)\n")
                f.write("-----------------------------------------\n")
                import os
                found_stability = False
                for short_name in ["GWO", "PSO"]:
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

                # 5. Ensemble Model Comparison
                f.write("5. PERBANDINGAN MODEL ENSEMBLE\n")
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
                
                # 6. Wilcoxon Signed-Rank Test & Improvement
                f.write("6. UJI PERBANDINGAN SIGNIFIKANSI STATISTIK\n")
                f.write("-----------------------------------------\n")
                if 'GWO Ensemble' in self.model_results:
                    baselines_list = ['MA', 'ES', 'RNN']
                    mapes = [self.model_results[b]['metrics']['MAPE'] for b in baselines_list if b in self.model_results]
                    if mapes:
                        best_baseline_name = baselines_list[np.argmin(mapes)]
                        y_test = self.data_processor.test_df[self.data_processor.target_col]
                        y_ens_pred = self.model_results['GWO Ensemble']['pred']
                        y_best_baseline_pred = self.model_results[best_baseline_name]['pred']
                        
                        from utils.metrics import calculate_wilcoxon_test
                        stat, p_val = calculate_wilcoxon_test(y_test, y_ens_pred, y_best_baseline_pred)
                        conclusion = "SIGNIFICANT (p < 0.05)" if p_val < 0.05 else "NOT SIGNIFICANT (p >= 0.05)"
                        
                        best_mape = self.model_results[best_baseline_name]['metrics']['MAPE']
                        ens_mape = self.model_results['GWO Ensemble']['metrics']['MAPE']
                        improvement = ((best_mape - ens_mape) / best_mape) * 100
                        
                        f.write(f"Perbandingan: Ensemble vs Best Baseline ({best_baseline_name})\n")
                        f.write(f"  Akurasi Improvement: {improvement:.2f}%\n")
                        f.write(f"  Wilcoxon Test Statistic: {stat:.4f}\n")
                        f.write(f"  Wilcoxon Test p-value:   {p_val:.6f}\n")
                        f.write(f"  Kesimpulan Uji Statistik:  {conclusion}\n")
                
                f.write("\n=========================================================================\n")
                f.write("                        AKHIR DARI LAPORAN EKSPERIMEN                    \n")
                f.write("=========================================================================\n")

            messagebox.showinfo("Success", f"Full thesis report exported successfully to {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {e}")
