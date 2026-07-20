import customtkinter as ctk
from tkinter import messagebox, filedialog
from models.moving_average import MovingAverageModel
from models.exponential_smoothing import ExponentialSmoothingModel
from models.simple_rnn import SimpleRNNModel
from utils.metrics import get_metrics
from ui.widgets import MetricTable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class PredictionPlotWindow(ctk.CTkToplevel):
    def __init__(self, master, title, dates, y_true, y_pred, color):
        super().__init__(master)
        self.title(title)
        self.geometry("700x600+50+150")  # Positioned on the left, slightly taller for the button
        
        self.lift()
        self.focus_force()
        
        # Build Matplotlib figure
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

        # Save Button
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


class RNNLossPlotWindow(ctk.CTkToplevel):
    def __init__(self, master, loss_history):
        super().__init__(master)
        self.title("Kurva Training Loss RNN")
        self.geometry("700x550+800+150")  # Positioned on the right, slightly taller for the button
        
        self.lift()
        self.focus_force()
        
        # Build Matplotlib figure
        self.fig, ax = plt.subplots(figsize=(7, 4.3), dpi=100)
        epochs = range(1, len(loss_history) + 1)
        ax.plot(epochs, loss_history, label="Training Loss", color="#ff5b60", linewidth=2.5)
        
        ax.set_title("Kurva Training Loss RNN", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Loss (MSE)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper right", fontsize=12, framealpha=0.8, edgecolor="#cccccc")
        plt.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

        # Save Button
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
        self.model_results = model_results  # { 'MA': { 'pred': [], 'metrics': {} }, ... }

        # Configure grid
        self.grid_columnconfigure((0, 1, 2), weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. Moving Average Section
        self.ma_frame, self.btn_ma = self.create_model_frame("Moving Average (MA)", 0, self.train_ma)
        self.lbl_ma_win = ctk.CTkLabel(self.ma_frame, text="Window Size (n):")
        self.lbl_ma_win.pack(pady=5)
        self.entry_ma_win = ctk.CTkEntry(self.ma_frame, placeholder_text="3")
        self.entry_ma_win.pack(pady=5)
        self.entry_ma_win.insert(0, "3")
        self.ma_metrics = MetricTable(self.ma_frame)
        self.ma_metrics.pack(fill="x", padx=10, pady=10)

        # 2. Exponential Smoothing Section
        self.es_frame, self.btn_es = self.create_model_frame("Exponential Smoothing (ES)", 1, self.train_es)
        self.lbl_es_alpha = ctk.CTkLabel(self.es_frame, text="Alpha (0-1):")
        self.lbl_es_alpha.pack(pady=5)
        self.entry_es_alpha = ctk.CTkEntry(self.es_frame, placeholder_text="0.5")
        self.entry_es_alpha.pack(pady=5)
        self.entry_es_alpha.insert(0, "0.5")
        self.es_metrics = MetricTable(self.es_frame)
        self.es_metrics.pack(fill="x", padx=10, pady=10)

        # 3. Simple RNN Section
        self.rnn_frame, self.btn_rnn = self.create_model_frame("Simple RNN (RNN)", 2, self.train_rnn)

        self.lbl_rnn_lookback = ctk.CTkLabel(self.rnn_frame, text="Lookback (timesteps):")
        self.lbl_rnn_lookback.pack(pady=(10, 0))
        self.entry_rnn_lookback = ctk.CTkEntry(self.rnn_frame, placeholder_text="40")
        self.entry_rnn_lookback.pack(pady=5)
        self.entry_rnn_lookback.insert(0, "40")

        self.lbl_rnn_epochs = ctk.CTkLabel(self.rnn_frame, text="Epochs:")
        self.lbl_rnn_epochs.pack(pady=(5, 0))
        self.entry_rnn_epochs = ctk.CTkEntry(self.rnn_frame, placeholder_text="100")
        self.entry_rnn_epochs.pack(pady=5)
        self.entry_rnn_epochs.insert(0, "100")

        self.rnn_metrics = MetricTable(self.rnn_frame)
        self.rnn_metrics.pack(fill="x", padx=10, pady=10)

    # ─────────────────────────────────────────────────────────────────────
    def create_model_frame(self, title, col, command):
        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=col, padx=10, pady=10, sticky="nsew")

        lbl = ctk.CTkLabel(frame, text=title, font=ctk.CTkFont(size=16, weight="bold"))
        lbl.pack(pady=10)

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
        _, y_train, x_test, y_test = self.get_data()
        if y_train is None:
            return
            
        self.btn_ma.configure(text="Training...", state="disabled")
        self.update()
        
        try:
            win = int(self.entry_ma_win.get())
            model = MovingAverageModel(window_size=win)
            model.fit(y_train)
            test_preds = model.forecast(len(y_test))

            metrics = get_metrics(y_test, test_preds)
            self.ma_metrics.update_metrics(metrics)

            self.model_results['MA'] = {
                'pred': np.array(test_preds),
                'metrics': metrics,
            }
            messagebox.showinfo("Success", "MA Model completed.")
            # Show prediction plot window
            PredictionPlotWindow(
                self, 
                title="Prediksi MA vs Data Uji", 
                dates=x_test, 
                y_true=y_test, 
                y_pred=test_preds, 
                color="#4e94ff"
            )
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
            model = ExponentialSmoothingModel(alpha=alpha)
            model.fit(y_train)
            test_preds = model.forecast(len(y_test))

            metrics = get_metrics(y_test, test_preds)
            self.es_metrics.update_metrics(metrics)

            self.model_results['ES'] = {
                'pred': np.array(test_preds),
                'metrics': metrics,
            }
            messagebox.showinfo("Success", "ES Model completed.")
            # Show prediction plot window
            PredictionPlotWindow(
                self, 
                title="Prediksi ES vs Data Uji", 
                dates=x_test, 
                y_true=y_test, 
                y_pred=test_preds, 
                color="#ffa500"
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_es.configure(text="Train & Predict", state="normal")
            self.update()

    def train_rnn(self):
        _, y_train, x_test, y_test = self.get_data()
        if y_train is None:
            return
            
        self.btn_rnn.configure(text="Training... Please wait", state="disabled")
        self.update()
        
        try:
            lookback = int(self.entry_rnn_lookback.get())
            epochs = int(self.entry_rnn_epochs.get())

            model = SimpleRNNModel(lookback=lookback, epochs=epochs)
            model.fit(y_train)
            test_preds = model.forecast(len(y_test))

            metrics = get_metrics(y_test, test_preds)
            self.rnn_metrics.update_metrics(metrics)

            self.model_results['RNN'] = {
                'pred': np.array(test_preds),
                'metrics': metrics,
            }
            messagebox.showinfo("Success", "RNN Model completed.")
            # Show prediction plot window
            PredictionPlotWindow(
                self, 
                title="Prediksi RNN vs Data Uji", 
                dates=x_test, 
                y_true=y_test, 
                y_pred=test_preds, 
                color="#2ecc71"
            )
            # Show RNN training loss window
            if hasattr(model, 'loss_history') and model.loss_history is not None:
                RNNLossPlotWindow(self, loss_history=model.loss_history)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.btn_rnn.configure(text="Train & Predict", state="normal")
            self.update()
