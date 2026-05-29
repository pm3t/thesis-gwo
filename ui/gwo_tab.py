import customtkinter as ctk
from tkinter import messagebox
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

        ctk.CTkLabel(self.left_frame, text="Select Optimizer:").pack(pady=(5, 0))
        self.combo_optimizer = ctk.CTkComboBox(self.left_frame, values=["Grey Wolf Optimizer (GWO)", "Particle Swarm Optimization (PSO)"])
        self.combo_optimizer.pack(pady=5)
        self.combo_optimizer.set("Grey Wolf Optimizer (GWO)")

        ctk.CTkLabel(self.left_frame, text="Population Size (n_wolves/particles):").pack(pady=(10, 0))
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
        self.btn_run.pack(pady=10)

        # Results section
        ctk.CTkLabel(
            self.left_frame, text="Optimal Weights (Best Run):",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=(15, 5))

        self.lbl_w1 = ctk.CTkLabel(self.left_frame, text="w1 (MA): -")
        self.lbl_w1.pack(pady=2)
        self.lbl_w2 = ctk.CTkLabel(self.left_frame, text="w2 (ES): -")
        self.lbl_w2.pack(pady=2)
        self.lbl_w3 = ctk.CTkLabel(self.left_frame, text="w3 (RNN): -")
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

        # Build initial placeholder figure
        self._build_placeholder_figure()

    # ─────────────────────────────────────────────────────────────────────────
    # Figure helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _build_placeholder_figure(self):
        """Show empty axis with axis labels before optimization runs."""
        fig = plt.figure(figsize=(9, 5), facecolor="#2b2b2b")
        ax_conv = fig.add_subplot(111)

        ax_conv.set_facecolor("#1e1e1e")
        ax_conv.tick_params(colors="#cccccc", labelsize=8)
        for spine in ax_conv.spines.values():
            spine.set_edgecolor("#555555")

        ax_conv.set_title("Convergence Curve", color="#e0e0e0", fontsize=10)
        ax_conv.set_xlabel("Iteration", color="#aaaaaa", fontsize=8)
        ax_conv.set_ylabel("Best MAPE (%)", color="#aaaaaa", fontsize=8)
        ax_conv.text(0.5, 0.5, "No data yet",
                     ha="center", va="center", color="#555555",
                     fontsize=10, transform=ax_conv.transAxes)

        self._embed_figure(fig)

    def _embed_figure(self, fig):
        """Destroy previous canvas and embed a new one."""
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Optimization
    # ─────────────────────────────────────────────────────────────────────────
    # Optimization
    # ─────────────────────────────────────────────────────────────────────────
    def run_optimization(self):
        if not all(k in self.model_results for k in ['MA', 'ES', 'RNN']):
            messagebox.showwarning("Warning", "Please run all three baseline models first.")
            return

        _, _, _, y_test = self.data_processor.get_train_test_data()
        y_pred_ma = self.model_results['MA']['pred']
        y_pred_es = self.model_results['ES']['pred']
        y_pred_rnn = self.model_results['RNN']['pred']

        optimizer_name = self.combo_optimizer.get()
        short_name = "GWO" if "Grey Wolf" in optimizer_name else "PSO"

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

                    if "Grey Wolf" in optimizer_name:
                        from optimizers.gwo import GreyWolfOptimizer
                        opt = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
                    else:
                        from optimizers.pso import ParticleSwarmOptimizer
                        opt = ParticleSwarmOptimizer(n_particles=n_wolves, max_iter=max_iter)

                    w, conv, pos_hist = opt.optimize(y_test, y_pred_ma, y_pred_es, y_pred_rnn)
                    run_best_fitness = conv[-1]

                    all_fitness.append(run_best_fitness)
                    all_weights.append(w)

                    if run_best_fitness < best_run_fitness:
                        best_run_fitness = run_best_fitness
                        best_weights = w
                        best_convergence = conv
                        best_positions_history = pos_hist

                    # Calculate full metrics for this run
                    from utils.metrics import get_metrics
                    w_arr = np.array(w)
                    if np.sum(w_arr) > 0:
                        w_arr = w_arr / np.sum(w_arr)
                    y_ens_pred = w_arr[0] * y_pred_ma + w_arr[1] * y_pred_es + w_arr[2] * y_pred_rnn
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
                worst_fit = np.max(all_fitness)
                mean_fit = np.mean(all_fitness)
                std_fit = np.std(all_fitness)

                avg_w = np.mean(all_weights, axis=0)
                std_w = np.std(all_weights, axis=0)

                # Format stability runs as an ASCII table
                n_header = "n (wolf)" if "GWO" in short_name else "n (particle)"
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
                    f.write(f"Best Run Weights:\n")
                    f.write(f"  w1 (MA): {best_weights[0]:.6f}\n")
                    f.write(f"  w2 (ES): {best_weights[1]:.6f}\n")
                    f.write(f"  w3 (RNN): {best_weights[2]:.6f}\n\n")
                    f.write(f"Average Weights Across All Runs:\n")
                    f.write(f"  w1 (MA): {avg_w[0]:.6f} (std: {std_w[0]:.6f})\n")
                    f.write(f"  w2 (ES): {avg_w[1]:.6f} (std: {std_w[1]:.6f})\n")
                    f.write(f"  w3 (RNN): {avg_w[2]:.6f} (std: {std_w[2]:.6f})\n\n")
                    f.write(f"Runs Performance Details Table:\n")
                    f.write(f"{table_str}\n")

                # Save to CSV
                runs_df = pd.DataFrame(run_results)
                runs_df.to_csv(f"Analysis/Stability_Runs_{short_name}.csv", index=False)

                # Update labels
                self.lbl_w1.configure(text=f"w1 (MA): {best_weights[0]:.4f}")
                self.lbl_w2.configure(text=f"w2 (ES): {best_weights[1]:.4f}")
                self.lbl_w3.configure(text=f"w3 (RNN): {best_weights[2]:.4f}")
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
                            end_idx = -1
                            markers = [
                                ">>> STATISTIK KESTABILAN PSO (30 RUNS) <<<",
                                ">>> STATISTIK KESTABILAN GWO (30 RUNS) <<<",
                                "5. PERBANDINGAN MODEL ENSEMBLE"
                            ]
                            for marker in markers:
                                if marker != start_marker:
                                    pos = report_content.find(marker, start_idx + len(start_marker))
                                    if pos != -1:
                                        if end_idx == -1 or pos < end_idx:
                                            end_idx = pos
                            
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
                if "Grey Wolf" in optimizer_name:
                    from optimizers.gwo import GreyWolfOptimizer
                    opt = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
                else:
                    from optimizers.pso import ParticleSwarmOptimizer
                    opt = ParticleSwarmOptimizer(n_particles=n_wolves, max_iter=max_iter)

                best_weights, convergence, positions_history = opt.optimize(
                    y_test, y_pred_ma, y_pred_es, y_pred_rnn
                )

                # Store results
                self.gwo_results['best_weights'] = best_weights
                self.gwo_results['convergence'] = convergence
                self.gwo_results['optimizer_name'] = short_name
                self._positions_history = positions_history

                # Update labels
                self.lbl_w1.configure(text=f"w1 (MA): {best_weights[0]:.4f}")
                self.lbl_w2.configure(text=f"w2 (ES): {best_weights[1]:.4f}")
                self.lbl_w3.configure(text=f"w3 (RNN): {best_weights[2]:.4f}")
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

            # Plot static convergence
            self._plot_static_convergence(self.gwo_results['convergence'])

        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {e}")
        finally:
            self.btn_run.configure(text="Run Optimization", state="normal")
            self.update()

    # ─────────────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────────────
    # Static Visualization
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_static_convergence(self, convergence):
        """Build and show the static convergence curve."""
        n_iter = len(convergence)
        fig = plt.figure(figsize=(9, 5), facecolor="#2b2b2b")
        ax_conv = fig.add_subplot(111)

        ax_conv.set_facecolor("#1e1e1e")
        ax_conv.tick_params(colors="#cccccc", labelsize=8)
        for spine in ax_conv.spines.values():
            spine.set_edgecolor("#555555")

        ax_conv.set_title("Convergence Curve", color="#e0e0e0", fontsize=10)
        ax_conv.set_xlabel("Iteration", color="#aaaaaa", fontsize=8)
        ax_conv.set_ylabel("Best MAPE (%)", color="#aaaaaa", fontsize=8)
        
        if n_iter > 0:
            ax_conv.set_xlim(0, n_iter)
            # Add small padding to y-limits
            y_min = min(convergence)
            y_max = max(convergence)
            y_pad = (y_max - y_min) * 0.05
            if y_pad == 0: y_pad = y_min * 0.05
            
            ax_conv.set_ylim(y_min - y_pad, y_max + y_pad)

            ax_conv.plot(range(1, n_iter + 1), convergence,
                         color="#5599ff", linewidth=2, zorder=2)
                         
            ax_conv.plot([n_iter], [convergence[-1]], "o", color="#ff4444", markersize=5, zorder=3)
            ax_conv.text(
                0.98, 0.97, f"MAPE: {convergence[-1]:.4f}%", transform=ax_conv.transAxes,
                color="#aaaaaa", fontsize=8, va="top", ha="right"
            )

        self._embed_figure(fig)

    def update_stability_table(self, run_results, short_name):
        # Clear existing widgets in self.table_frame
        for widget in self.table_frame.winfo_children():
            widget.destroy()

        n_header = "n (wolf)" if "GWO" in short_name else "n (particle)"
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
