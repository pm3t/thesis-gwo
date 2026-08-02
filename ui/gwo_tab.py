import customtkinter as ctk
from tkinter import messagebox, filedialog
from optimizers.gwo import GreyWolfOptimizer
from utils.visualizer import Visualizer
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

class GWOTab(ctk.CTkFrame):
    def __init__(self, master, data_processor, model_results, gwo_results):
        super().__init__(master)
        
        self.data_processor = data_processor
        self.model_results = model_results
        self.gwo_results = gwo_results  # { 'best_weights': [], 'convergence': [] }
        self.canvas = None

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ── Left Frame: Optimization Controls (Scrollable) ───────────────────
        self.left_frame = ctk.CTkScrollableFrame(self, label_text="Parameters & Controls")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        ctk.CTkLabel(self.left_frame, text="Optimizer: Grey Wolf Optimizer (GWO)", font=ctk.CTkFont(weight="bold")).pack(pady=(5, 10))

        ctk.CTkLabel(self.left_frame, text="Population Size (n_wolves):").pack(pady=(10, 0))
        self.entry_pop = ctk.CTkEntry(self.left_frame, placeholder_text="20")
        self.entry_pop.pack(pady=5)
        self.entry_pop.insert(0, "20")

        ctk.CTkLabel(self.left_frame, text="Max Iterations:").pack(pady=(10, 0))
        self.entry_iter = ctk.CTkEntry(self.left_frame, placeholder_text="100")
        self.entry_iter.pack(pady=5)
        self.entry_iter.insert(0, "100")

        self.var_multirun = ctk.BooleanVar(value=False)
        self.check_multirun = ctk.CTkCheckBox(self.left_frame, text="Run 30x (Stability Analysis)", variable=self.var_multirun)
        self.check_multirun.pack(pady=10)

        self.btn_run = ctk.CTkButton(
            self.left_frame, text="Run Optimization",
            command=self.run_optimization
        )
        self.btn_run.pack(pady=5)

        # Export Buttons Section
        ctk.CTkLabel(self.left_frame, text="Opsi Ekspor Grafik:", font=ctk.CTkFont(weight="bold")).pack(pady=(12, 2))

        self.btn_export_conv = ctk.CTkButton(
            self.left_frame, text="Export Konvergensi GWO (PNG)",
            command=self.export_convergence_plot,
            fg_color="#2b5b84", hover_color="#1f3f5e"
        )
        self.btn_export_conv.pack(pady=3)

        self.btn_export_pred = ctk.CTkButton(
            self.left_frame, text="Export Prediksi Ensemble (PNG)",
            command=self.export_prediction_plot,
            fg_color="#2b5b84", hover_color="#1f3f5e"
        )
        self.btn_export_pred.pack(pady=3)

        self.btn_export_combined = ctk.CTkButton(
            self.left_frame, text="Export Both Plots (Combined)",
            command=self.export_plot,
            fg_color="#3d609c", hover_color="#2b4673"
        )
        self.btn_export_combined.pack(pady=3)

        # Results section
        ctk.CTkLabel(
            self.left_frame, text="Optimal Weights (Best Run):",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=(15, 5))

        self.lbl_w1 = ctk.CTkLabel(self.left_frame, text="w1 (MA): -")
        self.lbl_w1.pack(pady=2)
        self.lbl_w2 = ctk.CTkLabel(self.left_frame, text="w2 (ES): -")
        self.lbl_w2.pack(pady=2)
        self.lbl_w3 = ctk.CTkLabel(self.left_frame, text="w3 (LR): -")
        self.lbl_w3.pack(pady=2)

        self.lbl_best_fitness = ctk.CTkLabel(
            self.left_frame, text="Best Fitness (MAPE): -",
            font=ctk.CTkFont(weight="bold")
        )
        self.lbl_best_fitness.pack(pady=5)

        # Stability results section
        self.lbl_stability_title = ctk.CTkLabel(
            self.left_frame, text="Stability Stats (30 Runs):",
            font=ctk.CTkFont(weight="bold")
        )
        self.lbl_stability_title.pack(pady=(15, 2))

        self.lbl_worst_fitness = ctk.CTkLabel(self.left_frame, text="Worst Fitness (MAPE): -")
        self.lbl_worst_fitness.pack(pady=2)

        self.lbl_mean_fitness = ctk.CTkLabel(self.left_frame, text="Mean Fitness (MAPE): -")
        self.lbl_mean_fitness.pack(pady=2)

        self.lbl_std_fitness = ctk.CTkLabel(self.left_frame, text="Std Dev (MAPE): -")
        self.lbl_std_fitness.pack(pady=2)

        # ── Right Frame: Visualization ────────────────────────────────────────
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(0, weight=3)
        self.right_frame.grid_rowconfigure(1, weight=2)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.canvas_frame = ctk.CTkFrame(self.right_frame)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Table for multi-run results
        self.table_frame = ctk.CTkScrollableFrame(self.right_frame, label_text="Stability Analysis - 30 Runs Details")
        self.table_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.placeholder_lbl = ctk.CTkLabel(self.table_frame, text="Run 30x (Stability Analysis) to see run-by-run metrics.")
        self.placeholder_lbl.pack(pady=20)

        # State storage for export
        self._current_convergence = None
        self._current_y_test = None
        self._current_y_ens_pred = None
        self._current_test_dates = None

        # Build initial figure or render existing results
        self.update_visualization()

    # ─────────────────────────────────────────────────────────────────────────
    # Figure helpers
    # ─────────────────────────────────────────────────────────────────────────
    def update_visualization(self):
        """Render plots if gwo_results contain data, else build placeholder."""
        if 'convergence' in self.gwo_results and 'best_weights' in self.gwo_results:
            conv = self.gwo_results['convergence']
            best_w = self.gwo_results['best_weights']
            self._current_convergence = conv

            if all(k in self.model_results for k in ['MA', 'ES', 'LR']) and self.data_processor is not None:
                train_dates, y_train, test_dates, y_test = self.data_processor.get_train_test_data()
                y_pred_ma = self.model_results['MA']['pred']
                y_pred_es = self.model_results['ES']['pred']
                y_pred_lr = self.model_results['LR']['pred']

                w_arr = np.array(best_w)
                if np.sum(w_arr) > 0:
                    w_arr = w_arr / np.sum(w_arr)
                y_ens_pred = w_arr[0] * y_pred_ma + w_arr[1] * y_pred_es + w_arr[2] * y_pred_lr

                self._current_y_test = y_test
                self._current_y_ens_pred = y_ens_pred
                self._current_test_dates = test_dates

                self._plot_static_convergence(conv, y_test=y_test, y_ens_pred=y_ens_pred, test_dates=test_dates)
            else:
                self._plot_static_convergence(conv)
        else:
            self._build_placeholder_figure()

    def _build_placeholder_figure(self):
        """Show empty axis with axis labels before optimization runs."""
        fig, (ax_conv, ax_pred) = plt.subplots(2, 1, figsize=(9, 6.5), facecolor="white")

        ax_conv.set_facecolor("white")
        ax_conv.tick_params(colors="black", labelsize=8)
        for spine in ax_conv.spines.values():
            spine.set_edgecolor("#cccccc")

        ax_conv.set_title("Kurva Konvergensi GWO", color="black", fontsize=10, fontweight="bold")
        ax_conv.set_xlabel("Iterasi", color="#333333", fontsize=8)
        ax_conv.set_ylabel("Best MAPE (%)", color="#333333", fontsize=8)
        ax_conv.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
        ax_conv.text(0.5, 0.5, "Belum ada data (Silakan klik 'Run Optimization')",
                     ha="center", va="center", color="#777777",
                     fontsize=10, transform=ax_conv.transAxes)

        ax_pred.set_facecolor("white")
        ax_pred.tick_params(colors="black", labelsize=8)
        for spine in ax_pred.spines.values():
            spine.set_edgecolor("#cccccc")

        ax_pred.set_title("Perbandingan Prediksi GWO Ensemble dan Data Uji", color="black", fontsize=10, fontweight="bold")
        ax_pred.set_xlabel("Tanggal / Sampel", color="#333333", fontsize=8)
        ax_pred.set_ylabel("Penjualan", color="#333333", fontsize=8)
        ax_pred.grid(True, linestyle=":", alpha=0.6, color="#cccccc")
        ax_pred.text(0.5, 0.5, "Belum ada data (Silakan klik 'Run Optimization')",
                     ha="center", va="center", color="#777777",
                     fontsize=10, transform=ax_pred.transAxes)

        fig.tight_layout()
        self._embed_figure(fig)

    def _embed_figure(self, fig):
        """Destroy previous canvas and embed a new one."""
        self.current_fig = fig
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def export_convergence_plot(self):
        """Export ONLY the GWO Convergence plot to image file."""
        if not hasattr(self, '_current_convergence') or self._current_convergence is None or len(self._current_convergence) == 0:
            if 'convergence' in self.gwo_results:
                self._current_convergence = self.gwo_results['convergence']
            else:
                messagebox.showwarning("Warning", "Belum ada data konvergensi. Silakan jalankan 'Run Optimization' terlebih dahulu.")
                return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("PDF files", "*.pdf")],
            title="Simpan Grafik Konvergensi GWO"
        )
        if file_path:
            try:
                fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
                ax.set_facecolor("white")
                ax.tick_params(colors="black", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#cccccc")

                ax.set_title("Kurva Konvergensi Grey Wolf Optimizer (GWO)", color="black", fontsize=11, fontweight="bold", pad=12)
                ax.set_xlabel("Iterasi", color="#333333", fontsize=9)
                ax.set_ylabel("Best MAPE (%)", color="#333333", fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")

                n_iter = len(self._current_convergence)
                ax.set_xlim(0, n_iter)
                y_min = min(self._current_convergence)
                y_max = max(self._current_convergence)
                y_pad = (y_max - y_min) * 0.05
                if y_pad == 0: y_pad = y_min * 0.05
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

                ax.plot(range(1, n_iter + 1), self._current_convergence, color="#1f77b4", linewidth=2, zorder=2)
                ax.plot([n_iter], [self._current_convergence[-1]], "o", color="#d62728", markersize=6, zorder=3)
                ax.text(
                    0.98, 0.95, f"Best MAPE: {self._current_convergence[-1]:.4f}%", transform=ax.transAxes,
                    color="#333333", fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9)
                )

                fig.tight_layout()
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                messagebox.showinfo("Success", f"Grafik Konvergensi GWO berhasil disimpan ke:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan grafik konvergensi: {e}")

    def export_prediction_plot(self):
        """Export ONLY the GWO Ensemble Prediction vs Test Data plot to image file."""
        if not hasattr(self, '_current_y_test') or self._current_y_test is None or self._current_y_ens_pred is None:
            if 'best_weights' in self.gwo_results and all(k in self.model_results for k in ['MA', 'ES', 'LR']):
                train_dates, y_train, test_dates, y_test = self.data_processor.get_train_test_data()
                y_pred_ma = self.model_results['MA']['pred']
                y_pred_es = self.model_results['ES']['pred']
                y_pred_lr = self.model_results['LR']['pred']
                best_w = self.gwo_results['best_weights']
                w_arr = np.array(best_w)
                if np.sum(w_arr) > 0: w_arr = w_arr / np.sum(w_arr)
                self._current_y_ens_pred = w_arr[0] * y_pred_ma + w_arr[1] * y_pred_es + w_arr[2] * y_pred_lr
                self._current_y_test = y_test
                self._current_test_dates = test_dates
            else:
                messagebox.showwarning("Warning", "Belum ada data prediksi. Silakan jalankan 'Run Optimization' terlebih dahulu.")
                return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("PDF files", "*.pdf")],
            title="Simpan Grafik Perbandingan Prediksi GWO Ensemble"
        )
        if file_path:
            try:
                fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
                ax.set_facecolor("white")
                ax.tick_params(colors="black", labelsize=9)
                for spine in ax.spines.values():
                    spine.set_edgecolor("#cccccc")

                ax.set_title("Perbandingan Prediksi GWO Ensemble dan Data Uji", color="black", fontsize=11, fontweight="bold", pad=12)
                ax.set_xlabel("Tanggal", color="#333333", fontsize=9)
                ax.set_ylabel("Penjualan", color="#333333", fontsize=9)
                ax.grid(True, linestyle=":", alpha=0.6, color="#cccccc")

                if hasattr(self.data_processor, 'inverse_transform') and self.data_processor.scaler is not None:
                    y_test_plot = self.data_processor.inverse_transform(self._current_y_test)
                    y_pred_plot = self.data_processor.inverse_transform(self._current_y_ens_pred)
                else:
                    y_test_plot = np.array(self._current_y_test)
                    y_pred_plot = np.array(self._current_y_ens_pred)

                x_axis = self._current_test_dates if self._current_test_dates is not None else range(1, len(y_test_plot) + 1)

                ax.plot(x_axis, y_test_plot, label="Data Uji (Aktual)", color="#1f77b4", linewidth=1.8, alpha=0.85)
                ax.plot(x_axis, y_pred_plot, label="Prediksi GWO Ensemble", color="#ff7f0e", linestyle="--", linewidth=1.8, alpha=0.9)
                ax.legend(loc="best", fontsize=9)

                fig.tight_layout()
                fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close(fig)
                messagebox.showinfo("Success", f"Grafik Perbandingan Prediksi berhasil disimpan ke:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan grafik prediksi: {e}")

    def export_plot(self):
        """Export current combined plots to image file."""
        if not hasattr(self, 'current_fig') or self.current_fig is None:
            messagebox.showwarning("Warning", "Belum ada grafik untuk diekspor. Silakan jalankan 'Run Optimization' terlebih dahulu.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg"), ("PDF files", "*.pdf")],
            title="Simpan Kedua Grafik (Combined)"
        )
        if file_path:
            try:
                self.current_fig.savefig(file_path, dpi=300, bbox_inches='tight', facecolor='white')
                messagebox.showinfo("Success", f"Grafik berhasil disimpan ke:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Gagal menyimpan grafik: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # Optimization
    # ─────────────────────────────────────────────────────────────────────────
    # Optimization
    # ─────────────────────────────────────────────────────────────────────────
    def run_optimization(self):
        if not all(k in self.model_results for k in ['MA', 'ES', 'LR']):
            messagebox.showwarning("Warning", "Please run all three baseline models first.")
            return

        _, _, _, y_test = self.data_processor.get_train_test_data()
        y_pred_ma = self.model_results['MA']['pred']
        y_pred_es = self.model_results['ES']['pred']
        y_pred_lr = self.model_results['LR']['pred']

        optimizer_name = "Grey Wolf Optimizer (GWO)"
        short_name = "GWO"

        try:
            n_wolves = int(self.entry_pop.get())
            max_iter = int(self.entry_iter.get())
            
            import os
            import pandas as pd
            os.makedirs("Analysis", exist_ok=True)

            if self.var_multirun.get():
                    n_runs = 30
                    all_fitness = []
                    all_weights = []
                    best_run_fitness = float("inf")
                    
                    best_weights = None
                    best_convergence = None
                    best_positions_history = None

                    self.btn_run.configure(text="Running Multi-run...", state="disabled")
                    self.update()

                    run_results = []

                    for run in range(n_runs):
                        self.btn_run.configure(text=f"Running {run+1}/{n_runs}...")
                        self.update()

                        from optimizers.gwo import GreyWolfOptimizer
                        opt = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)

                        w, conv, pos_hist = opt.optimize(y_test, y_pred_ma, y_pred_es, y_pred_lr)
                        run_best_fitness = conv[-1]

                        all_fitness.append(run_best_fitness)
                        all_weights.append(w)

                        if run_best_fitness < best_run_fitness:
                            best_run_fitness = run_best_fitness
                            best_weights = w
                            best_convergence = conv
                            best_positions_history = pos_hist

                        # Calculate full metrics for this run (normalized scale)
                        from utils.metrics import get_metrics
                        w_arr = np.array(w)
                        if np.sum(w_arr) > 0:
                            w_arr = w_arr / np.sum(w_arr)
                        y_ens_pred = w_arr[0] * y_pred_ma + w_arr[1] * y_pred_es + w_arr[2] * y_pred_lr
                        metrics = get_metrics(y_test, y_ens_pred)

                        run_results.append({
                            'Run': run + 1,
                            'n (wolf)': n_wolves,
                            'iteration': max_iter,
                            'MAPE': metrics['MAPE'],
                            'w1': w_arr[0],
                            'w2': w_arr[1],
                            'w3': w_arr[2]
                        })

                    # Calculate stability stats
                    all_fitness = np.array(all_fitness)
                    all_weights = np.array(all_weights)

                    best_fit = np.min(all_fitness)
                    worst_idx = np.argmax(all_fitness)
                    worst_fit = all_fitness[worst_idx]
                    worst_weights = all_weights[worst_idx]
                    
                    mean_fit = np.mean(all_fitness)
                    std_fit = np.std(all_fitness)

                    avg_w = np.mean(all_weights, axis=0)
                    std_w = np.std(all_weights, axis=0)

                    # Format Ringkasan Parameter Statistik Optimasi
                    summary_table_lines = [
                        "| Parameter Statistik | Nilai Fitness (MAPE) | Bobot w1 (MA) | Bobot w2 (ES) | Bobot w3 (LR) |",
                        "|---------------------|----------------------|---------------|---------------|----------------|",
                        f"| Terbaik             | {best_fit:19.6f}% | {best_weights[0]:13.6f} | {best_weights[1]:13.6f} | {best_weights[2]:14.6f} |",
                        f"| Terburuk            | {worst_fit:19.6f}% | {worst_weights[0]:13.6f} | {worst_weights[1]:13.6f} | {worst_weights[2]:14.6f} |",
                        f"| Rata-rata           | {mean_fit:19.6f}% | {avg_w[0]:13.6f} | {avg_w[1]:13.6f} | {avg_w[2]:14.6f} |",
                        f"| Std. Deviation      | {std_fit:19.6f}% | {std_w[0]:13.6f} | {std_w[1]:13.6f} | {std_w[2]:14.6f} |"
                    ]
                    summary_table_str = "\n".join(summary_table_lines)

                    # Format stability runs as an ASCII table
                    n_header = "n (wolf)"
                    table_lines = []
                    table_lines.append(f"| Run | {n_header:<8} | iteration |   MAPE (%)   |      w1      |      w2      |      w3      |")
                    table_lines.append(f"|-----|{'-'*10}|-----------|--------------|--------------|--------------|--------------|")
                    for res in run_results:
                        table_lines.append(
                            f"| {res['Run']:3d} | {res['n (wolf)']:8d} | {res['iteration']:9d} | {res['MAPE']:11.4f}% | {res['w1']:12.4f} | {res['w2']:12.4f} | {res['w3']:12.4f} |"
                        )
                    table_str = "\n".join(table_lines)

                    # Save stats to file
                    stats_path = f"Analysis/Stability_Stats_{short_name}.txt"
                    with open(stats_path, "w") as f:
                        f.write(f"STABILITY ANALYSIS ({n_runs} RUNS) FOR {optimizer_name.upper()}\n")
                        f.write(f"=========================================\n")
                        f.write(f"Best Fitness (MAPE): {best_fit:.6f}%\n")
                        f.write(f"Worst Fitness (MAPE): {worst_fit:.6f}%\n")
                        f.write(f"Mean Fitness (MAPE): {mean_fit:.6f}%\n")
                        f.write(f"Standard Deviation (MAPE): {std_fit:.6f}%\n\n")
                        
                        f.write(f"RINGKASAN PARAMETER STATISTIK OPTIMASI:\n")
                        f.write(f"{summary_table_str}\n\n")
                        
                        f.write(f"Best Run Weights:\n")
                        f.write(f"  w1 (MA): {best_weights[0]:.6f}\n")
                        f.write(f"  w2 (ES): {best_weights[1]:.6f}\n")
                        f.write(f"  w3 (LR): {best_weights[2]:.6f}\n\n")
                        f.write(f"Average Weights Across All Runs:\n")
                        f.write(f"  w1 (MA): {avg_w[0]:.6f} (std: {std_w[0]:.6f})\n")
                        f.write(f"  w2 (ES): {avg_w[1]:.6f} (std: {std_w[1]:.6f})\n")
                        f.write(f"  w3 (LR): {avg_w[2]:.6f} (std: {std_w[2]:.6f})\n\n")
                        f.write(f"Runs Performance Details Table:\n")
                        f.write(f"{table_str}\n")

                    # Save to CSV
                    runs_df = pd.DataFrame(run_results)
                    runs_df.to_csv(f"Analysis/Stability_Runs_{short_name}.csv", index=False)

                    # Update labels
                    self.lbl_w1.configure(text=f"w1 (MA): {best_weights[0]:.4f}")
                    self.lbl_w2.configure(text=f"w2 (ES): {best_weights[1]:.4f}")
                    self.lbl_w3.configure(text=f"w3 (LR): {best_weights[2]:.4f}")
                    self.lbl_best_fitness.configure(text=f"Best Fitness (MAPE): {best_fit:.4f}%")

                    self.lbl_worst_fitness.configure(text=f"Worst Fitness (MAPE): {worst_fit:.4f}%")
                    self.lbl_mean_fitness.configure(text=f"Mean Fitness (MAPE): {mean_fit:.4f}%")
                    self.lbl_std_fitness.configure(text=f"Std Dev (MAPE): {std_fit:.4f}%")

                    self.gwo_results['best_weights'] = best_weights
                    self.gwo_results['convergence'] = best_convergence
                    self.gwo_results['optimizer_name'] = short_name
                    self._positions_history = best_positions_history

                    # Save convergence history of best run
                    conv_path = f"Analysis/Convergence_Data_{short_name}.csv"
                    pd.DataFrame({'Iteration': range(1, len(best_convergence) + 1), 'Best_Fitness_MAPE': best_convergence}).to_csv(conv_path, index=False)

                    # Update Thesis Data.txt if it exists
                    thesis_report_path = "Analysis/Thesis Data.txt"
                    if os.path.exists(thesis_report_path):
                        try:
                            with open(stats_path, "r") as sf:
                                new_stats_text = sf.read()
                            with open(thesis_report_path, "r") as tf:
                                report_content = tf.read()
                            
                            start_marker = f">>> STATISTIK KESTABILAN {short_name} (30 RUNS) <<<"
                            start_idx = report_content.find(start_marker)
                            if start_idx != -1:
                                end_idx = report_content.find("4. PERBANDINGAN MODEL ENSEMBLE", start_idx)
                                if end_idx != -1:
                                    updated_report = report_content[:start_idx] + start_marker + "\n" + new_stats_text + "\n\n" + report_content[end_idx:]
                                else:
                                    end_line_pos = report_content.find("=========================================================================", start_idx)
                                    if end_line_pos != -1:
                                        updated_report = report_content[:start_idx] + start_marker + "\n" + new_stats_text + "\n\n" + report_content[end_line_pos:]
                                    else:
                                        updated_report = report_content[:start_idx] + start_marker + "\n" + new_stats_text
                                        
                                with open(thesis_report_path, "w") as tf:
                                    tf.write(updated_report)
                        except Exception as ex:
                            print(f"Error updating Thesis Data: {ex}")

                    # Update table in UI
                    self.update_stability_table(run_results, short_name)
                    messagebox.showinfo("Success", f"Multi-run {short_name} completed. Stats saved to Analysis/Stability_Stats_{short_name}.txt")

            else:
                # Single run
                from optimizers.gwo import GreyWolfOptimizer
                opt = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)

                best_weights, convergence, positions_history = opt.optimize(
                    y_test, y_pred_ma, y_pred_es, y_pred_lr
                )

                # Store results
                self.gwo_results['best_weights'] = best_weights
                self.gwo_results['convergence'] = convergence
                self.gwo_results['optimizer_name'] = short_name
                self._positions_history = positions_history

                # Update labels
                self.lbl_w1.configure(text=f"w1 (MA): {best_weights[0]:.4f}")
                self.lbl_w2.configure(text=f"w2 (ES): {best_weights[1]:.4f}")
                self.lbl_w3.configure(text=f"w3 (LR): {best_weights[2]:.4f}")
                self.lbl_best_fitness.configure(text=f"Best Fitness (MAPE): {convergence[-1]:.4f}%")

                # Clear stability labels
                self.lbl_worst_fitness.configure(text="Worst Fitness (MAPE): -")
                self.lbl_mean_fitness.configure(text="Mean Fitness (MAPE): -")
                self.lbl_std_fitness.configure(text="Std Dev (MAPE): -")

                # Save convergence history
                conv_path = f"Analysis/Convergence_Data_{short_name}.csv"
                pd.DataFrame({'Iteration': range(1, len(convergence) + 1), 'Best_Fitness_MAPE': convergence}).to_csv(conv_path, index=False)

                # Clear/Reset table in UI for single run
                for widget in self.table_frame.winfo_children():
                    widget.destroy()
                self.placeholder_lbl = ctk.CTkLabel(self.table_frame, text="Run 30x (Stability Analysis) to see run-by-run metrics.")
                self.placeholder_lbl.pack(pady=20)

                messagebox.showinfo("Success", f"Single run {short_name} completed. Convergence saved to Analysis/Convergence_Data_{short_name}.csv")

            # Calculate GWO Ensemble prediction for visualization
            best_w = self.gwo_results['best_weights']
            w_arr = np.array(best_w)
            if np.sum(w_arr) > 0:
                w_arr = w_arr / np.sum(w_arr)
            y_ens_pred = w_arr[0] * y_pred_ma + w_arr[1] * y_pred_es + w_arr[2] * y_pred_lr

            # Plot static convergence & prediction comparison
            self._plot_static_convergence(
                self.gwo_results['convergence'],
                y_test=y_test,
                y_ens_pred=y_ens_pred,
                test_dates=test_dates
            )

        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {e}")
        finally:
            self.btn_run.configure(text="Run Optimization", state="normal")
            self.update()

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Static Visualization
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_static_convergence(self, convergence, y_test=None, y_ens_pred=None, test_dates=None):
        """Build and show static convergence curve and prediction vs test data plot with white background."""
        n_iter = len(convergence)
        fig, (ax_conv, ax_pred) = plt.subplots(2, 1, figsize=(9, 6.5), facecolor="white")

        # ── 1. Convergence Plot (Background Putih) ───────────────────────────
        ax_conv.set_facecolor("white")
        ax_conv.tick_params(colors="black", labelsize=8)
        for spine in ax_conv.spines.values():
            spine.set_edgecolor("#cccccc")

        ax_conv.set_title("Kurva Konvergensi GWO", color="black", fontsize=10, fontweight="bold")
        ax_conv.set_xlabel("Iterasi", color="#333333", fontsize=8)
        ax_conv.set_ylabel("Best MAPE (%)", color="#333333", fontsize=8)
        ax_conv.grid(True, linestyle=":", alpha=0.6, color="#cccccc")

        if n_iter > 0:
            ax_conv.set_xlim(0, n_iter)
            y_min = min(convergence)
            y_max = max(convergence)
            y_pad = (y_max - y_min) * 0.05
            if y_pad == 0: y_pad = y_min * 0.05
            
            ax_conv.set_ylim(y_min - y_pad, y_max + y_pad)

            ax_conv.plot(range(1, n_iter + 1), convergence,
                         color="#1f77b4", linewidth=2, zorder=2, label="Best Fitness")
                         
            ax_conv.plot([n_iter], [convergence[-1]], "o", color="#d62728", markersize=6, zorder=3)
            ax_conv.text(
                0.98, 0.95, f"Best MAPE: {convergence[-1]:.4f}%", transform=ax_conv.transAxes,
                color="#333333", fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.9)
            )

        # ── 2. Plot Perbandingan Prediksi GWO Ensemble dan Data Uji ─────────
        ax_pred.set_facecolor("white")
        ax_pred.tick_params(colors="black", labelsize=8)
        for spine in ax_pred.spines.values():
            spine.set_edgecolor("#cccccc")

        ax_pred.set_title("Perbandingan Prediksi GWO Ensemble dan Data Uji", color="black", fontsize=10, fontweight="bold")
        ax_pred.set_xlabel("Tanggal", color="#333333", fontsize=8)
        ax_pred.set_ylabel("Penjualan", color="#333333", fontsize=8)
        ax_pred.grid(True, linestyle=":", alpha=0.6, color="#cccccc")

        if y_test is not None and y_ens_pred is not None:
            # Denormalize data if scaler is available
            if hasattr(self.data_processor, 'inverse_transform') and self.data_processor.scaler is not None:
                y_test_plot = self.data_processor.inverse_transform(y_test)
                y_pred_plot = self.data_processor.inverse_transform(y_ens_pred)
            else:
                y_test_plot = np.array(y_test)
                y_pred_plot = np.array(y_ens_pred)

            x_axis = test_dates if test_dates is not None else range(1, len(y_test_plot) + 1)

            ax_pred.plot(x_axis, y_test_plot, label="Data Uji (Aktual)", color="#1f77b4", linewidth=1.8, alpha=0.85)
            ax_pred.plot(x_axis, y_pred_plot, label="Prediksi GWO Ensemble", color="#ff7f0e", linestyle="--", linewidth=1.8, alpha=0.9)
            ax_pred.legend(loc="best", fontsize=8)

        fig.tight_layout()
        self._embed_figure(fig)

    def update_stability_table(self, run_results, short_name):
        # Clear existing widgets in self.table_frame
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        n_header = "n (wolf)"
        headers = ["Run", n_header, "iteration", "MAPE", "w1", "w2", "w3"]
        
        # Configure columns of the table frame to expand/space evenly
        for col_idx in range(len(headers)):
            self.table_frame.grid_columnconfigure(col_idx, weight=1)

        # Headers
        for col_idx, h in enumerate(headers):
            lbl = ctk.CTkLabel(
                self.table_frame, text=h,
                font=ctk.CTkFont(weight="bold", size=11)
            )
            lbl.grid(row=0, column=col_idx, padx=10, pady=5, sticky="nsew")

        # Rows
        for row_idx, res in enumerate(run_results):
            r = row_idx + 1
            # Run
            ctk.CTkLabel(self.table_frame, text=str(res['Run'])).grid(row=r, column=0, padx=10, pady=2)
            # n (wolf)
            ctk.CTkLabel(self.table_frame, text=str(res['n (wolf)'])).grid(row=r, column=1, padx=10, pady=2)
            # iteration
            ctk.CTkLabel(self.table_frame, text=str(res['iteration'])).grid(row=r, column=2, padx=10, pady=2)
            # MAPE
            ctk.CTkLabel(self.table_frame, text=f"{res['MAPE']:.4f}%").grid(row=r, column=3, padx=10, pady=2)
            # w1
            ctk.CTkLabel(self.table_frame, text=f"{res['w1']:.4f}").grid(row=r, column=4, padx=10, pady=2)
            # w2
            ctk.CTkLabel(self.table_frame, text=f"{res['w2']:.4f}").grid(row=r, column=5, padx=10, pady=2)
            # w3
            ctk.CTkLabel(self.table_frame, text=f"{res['w3']:.4f}").grid(row=r, column=6, padx=10, pady=2)
