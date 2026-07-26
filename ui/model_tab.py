import customtkinter as ctk
from tkinter import messagebox, filedialog
from models.moving_average import MovingAverageModel
from models.exponential_smoothing import ExponentialSmoothingModel
from models.linear_regression import LinearRegressionModel
from utils.metrics import get_metrics
from ui.widgets import MetricTable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PredictionPlotWindow(ctk.CTkToplevel):
    def __init__(self, master, title, dates, y_true, y_pred, color):
        super().__init__(master)
        self.title(title)
        self.geometry("700x600+50+150")

        self.lift()
        self.focus_force()

        self.fig, ax = plt.subplots(figsize=(7, 4.8), dpi=100)
        ax.plot(dates, y_true, label="Data Uji", color="#2d2d2d", linewidth=2)
        ax.plot(dates, y_pred, label="Prediksi", color=color, linewidth=2)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("date", fontsize=11)
        ax.set_ylabel("sales", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="lower left", fontsize=12, framealpha=0.8, edgecolor="#cccccc")
        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        self.btn_save = ctk.CTkButton(self, text="Save Plot", command=self.save_plot)
        self.btn_save.pack(pady=(0, 10))

    def save_plot(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")]
        )
        if file_path:
            try:
                self.fig.savefig(file_path)
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")


class ModelTab(ctk.CTkFrame):
    def __init__(self, master, data_processor, model_results):
        super().__init__(master)

        self.data_processor = data_processor
        self.model_results  = model_results

        # Configure grid
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Moving Average Section
        self.ma_frame, self.btn_ma = self.create_model_frame("Moving Average (Seasonal MA)", 0, self.train_ma)
        self.lbl_ma_win = ctk.CTkLabel(self.ma_frame, text="Window Size (minggu):")
        self.lbl_ma_win.pack(pady=5)
        self.entry_ma_win = ctk.CTkEntry(self.ma_frame, placeholder_text="4")
        self.entry_ma_win.pack(pady=5)
        self.entry_ma_win.insert(0, "4")
        self.ma_metrics = MetricTable(self.ma_frame)
        self.ma_metrics.pack(fill="x", padx=10, pady=10)

        # 2. Exponential Smoothing Section
        self.es_frame, self.btn_es = self.create_model_frame("Exponential Smoothing (Holt-Winters)", 1, self.train_es)
        self.lbl_es_alpha = ctk.CTkLabel(self.es_frame, text="Alpha level (0-1):")
        self.lbl_es_alpha.pack(pady=(5, 0))
        self.entry_es_alpha = ctk.CTkEntry(self.es_frame, placeholder_text="0.3")
        self.entry_es_alpha.pack(pady=2)
        self.entry_es_alpha.insert(0, "0.3")

        self.lbl_es_beta = ctk.CTkLabel(self.es_frame, text="Beta trend (0-1):")
        self.lbl_es_beta.pack(pady=(5, 0))
        self.entry_es_beta = ctk.CTkEntry(self.es_frame, placeholder_text="0.02")
        self.entry_es_beta.pack(pady=2)
        self.entry_es_beta.insert(0, "0.02")

        self.lbl_es_gamma = ctk.CTkLabel(self.es_frame, text="Gamma seasonal (0-1):")
        self.lbl_es_gamma.pack(pady=(5, 0))
        self.entry_es_gamma = ctk.CTkEntry(self.es_frame, placeholder_text="0.3")
        self.entry_es_gamma.pack(pady=2)
        self.entry_es_gamma.insert(0, "0.3")

        self.es_metrics = MetricTable(self.es_frame)
        self.es_metrics.pack(fill="x", padx=10, pady=10)

        # 3. Linear Regression Section
        self.lr_frame, self.btn_lr = self.create_model_frame("Linear Regression (LR)", 2, self.train_lr)
        ctk.CTkLabel(self.lr_frame, text="(Tren time_idx + Day-of-Week + Month)").pack(pady=(8, 2))
        self.lr_metrics = MetricTable(self.lr_frame)
        self.lr_metrics.pack(fill="x", padx=10, pady=10)

    # ─────────────────────────────────────────────────────────────────────
    def create_model_frame(self, title, col, command):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")
        ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=15, weight="bold")).pack(pady=10)
        btn = ctk.CTkButton(frame, text="Train & Predict", command=command)
        btn.pack(pady=10)
        return frame, btn

    def get_data(self):
        x_train, y_train, x_test, y_test = self.data_processor.get_train_test_data()
        if y_train is None:
            messagebox.showwarning("Warning", "Please load and split data first.")
            return None, None, None, None
        return x_train, y_train, x_test, y_test

    # ─────────────────────────────────────────────────────────────────────
    def train_ma(self):
        x_train, y_train, x_test, y_test = self.get_data()
        if y_train is None:
            return

        self.btn_ma.configure(text="Training...", state="disabled")
        self.update()

        try:
            win = int(self.entry_ma_win.get())
            model = MovingAverageModel(window_size=win)
            model.fit(y_train, dates=x_train)
            test_preds = model.walk_forward(y_test, dates_test=x_test)

            metrics = get_metrics(y_test, test_preds)
            self.ma_metrics.update_metrics(metrics)
            self.model_results['MA'] = {'pred': np.array(test_preds), 'metrics': metrics}

            messagebox.showinfo("Success", f"Seasonal MA completed (R²: {metrics['R2']:.4f}).")
            PredictionPlotWindow(self, "Prediksi Seasonal MA vs Data Uji", x_test, y_test, test_preds, "#4e94ff")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_ma.configure(text="Train & Predict", state="normal")
            self.update()

    def train_es(self):
        _, y_train, x_test, y_test = self.get_data()
        if y_train is None:
            return

        self.btn_es.configure(text="Training...", state="disabled")
        self.update()

        try:
            alpha = float(self.entry_es_alpha.get())
            beta  = float(self.entry_es_beta.get())
            gamma = float(self.entry_es_gamma.get())

            model = ExponentialSmoothingModel(alpha=alpha, beta=beta, gamma=gamma, period=7)
            model.fit(y_train)
            test_preds = model.walk_forward(y_test)

            metrics = get_metrics(y_test, test_preds)
            self.es_metrics.update_metrics(metrics)
            self.model_results['ES'] = {'pred': np.array(test_preds), 'metrics': metrics}

            messagebox.showinfo("Success", f"Holt-Winters ES completed (R²: {metrics['R2']:.4f}).")
            PredictionPlotWindow(self, "Prediksi Holt-Winters ES vs Data Uji", x_test, y_test, test_preds, "#ffa500")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_es.configure(text="Train & Predict", state="normal")
            self.update()

    def train_lr(self):
        x_train, y_train, x_test, y_test = self.get_data()
        if y_train is None:
            return

        self.btn_lr.configure(text="Training...", state="disabled")
        self.update()

        try:
            model = LinearRegressionModel()
            model.fit(x_train, y_train)
            test_preds = model.walk_forward(x_test, y_test)

            metrics = get_metrics(y_test, test_preds)
            self.lr_metrics.update_metrics(metrics)
            self.model_results['LR'] = {'pred': np.array(test_preds), 'metrics': metrics}

            messagebox.showinfo("Success", f"Linear Regression completed (R²: {metrics['R2']:.4f}).")
            PredictionPlotWindow(self, "Prediksi LR vs Data Uji", x_test, y_test, test_preds, "#2ecc71")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_lr.configure(text="Train & Predict", state="normal")
            self.update()
