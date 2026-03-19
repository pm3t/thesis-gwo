import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils.visualizer import Visualizer
from tkinter import messagebox, filedialog
import matplotlib.pyplot as plt

class VisualizeTab(ctk.CTkFrame):
    def __init__(self, master, data_processor):
        super().__init__(master)
        
        self.data_processor = data_processor
        self.canvas = None
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Top Frame: Controls
        self.ctrl_frame = ctk.CTkFrame(self)
        self.ctrl_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        
        self.lbl_plot = ctk.CTkLabel(self.ctrl_frame, text="Select Plot Type:")
        self.lbl_plot.pack(side="left", padx=10)
        
        self.combo_plot = ctk.CTkComboBox(self.ctrl_frame, values=["Time Series Plot", "Decomposition Plot", "Distribution Plot"])
        self.combo_plot.pack(side="left", padx=10)
        
        self.btn_plot = ctk.CTkButton(self.ctrl_frame, text="Generate Plot", command=self.generate_plot)
        self.btn_plot.pack(side="left", padx=10)
        
        self.btn_save = ctk.CTkButton(self.ctrl_frame, text="Save Plot", command=self.save_plot)
        self.btn_save.pack(side="left", padx=10)

        # Bottom Frame: Canvas
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        self.fig = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

    def generate_plot(self):
        plot_type = self.combo_plot.get()
        dates, values = self.data_processor.get_full_data()
        
        if dates is None or values is None:
            messagebox.showwarning("Warning", "Please preprocess data in the DATA tab first.")
            return

        # Clear existing canvas
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        try:
            if plot_type == "Time Series Plot":
                self.fig, self.ax = Visualizer.plot_time_series(dates, values)
            elif plot_type == "Decomposition Plot":
                df = self.data_processor.df
                date_col = self.data_processor.date_col
                target_col = self.data_processor.target_col
                self.fig = Visualizer.plot_decomposition(df, date_col, target_col)
            elif plot_type == "Distribution Plot":
                self.fig, self.ax = Visualizer.plot_distribution(values)
            
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plot: {e}")

    def save_plot(self):
        if self.canvas is None:
            messagebox.showwarning("Warning", "No plot to save. Generate a plot first.")
            return
            
        file_path = filedialog.asksaveasfilename(defaultextension=".png", 
                                                filetypes=[("PNG files", "*.png"), ("JPG files", "*.jpg")])
        if file_path:
            try:
                self.fig.savefig(file_path)
                messagebox.showinfo("Success", f"Plot saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save plot: {e}")
