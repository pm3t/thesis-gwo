import customtkinter as ctk

class MetricTable(ctk.CTkFrame):
    def __init__(self, master, metrics_dict=None, title="Evaluation Metrics", **kwargs):
        super().__init__(master, **kwargs)
        
        self.title_label = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=14, weight="bold"))
        self.title_label.pack(pady=5)
        
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="x", padx=10, pady=5)
        
        self.labels = {}
        self.values = {}
        
        metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        
        for i, m in enumerate(metrics):
            lbl = ctk.CTkLabel(self.table_frame, text=m, font=ctk.CTkFont(weight="bold"))
            lbl.grid(row=0, column=i, padx=10, pady=2)
            
            val = ctk.CTkLabel(self.table_frame, text="-")
            val.grid(row=1, column=i, padx=10, pady=2)
            
            self.labels[m] = lbl
            self.values[m] = val

        if metrics_dict:
            self.update_metrics(metrics_dict)

    def update_metrics(self, metrics_dict):
        for m, v in metrics_dict.items():
            if m in self.values:
                # Format to 4 decimal places, MAPE with %
                if m == 'MAPE':
                    self.values[m].configure(text=f"{v:.4f}%")
                else:
                    self.values[m].configure(text=f"{v:.4f}")

class ComparisonTable(ctk.CTkFrame):
    def __init__(self, master, title="Model Comparison Overview", **kwargs):
        super().__init__(master, **kwargs)
        
        self.title_label = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=16, weight="bold"))
        self.title_label.pack(pady=10)
        
        self.table_frame = ctk.CTkFrame(self)
        self.table_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        self.rows = {} # { 'Model Name': { 'MAE': Label, ... } }
        
        # Headers
        ctk.CTkLabel(self.table_frame, text="Model/Method", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=15, pady=5)
        for i, m in enumerate(self.metrics):
            ctk.CTkLabel(self.table_frame, text=m, font=ctk.CTkFont(weight="bold")).grid(row=0, column=i+1, padx=15, pady=5)

    def update_data(self, all_results_dict):
        """
        all_results_dict: { 'MA': { 'metrics': {...} }, ... }
        """
        # Clear existing rows (if needed, but simple overwrite is fine)
        for i, (model_name, data) in enumerate(all_results_dict.items()):
            # Model Name Label
            ctk.CTkLabel(self.table_frame, text=model_name, font=ctk.CTkFont(weight="bold")).grid(row=i+1, column=0, padx=15, pady=5, sticky="w")
            
            metrics = data['metrics']
            for j, m in enumerate(self.metrics):
                val = metrics.get(m, 0.0)
                text = f"{val:.4f}%" if m == 'MAPE' else f"{val:.4f}"
                ctk.CTkLabel(self.table_frame, text=text).grid(row=i+1, column=j+1, padx=15, pady=5)

class IterationTable(ctk.CTkFrame):
    def __init__(self, master, title="GWO Convergence History", **kwargs):
        super().__init__(master, **kwargs)
        
        self.title_label = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(size=16, weight="bold"))
        self.title_label.pack(pady=5)
        
        # Scrollable frame for the table
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Headers
        ctk.CTkLabel(self.scroll_frame, text="Iteration", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=20, pady=5)
        ctk.CTkLabel(self.scroll_frame, text="Best Fitness (MAPE)", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1, padx=20, pady=5)

    def update_history(self, convergence_curve):
        # Clear previous rows (except headers)
        for widget in self.scroll_frame.winfo_children():
            if int(widget.grid_info()["row"]) > 0:
                widget.destroy()
                
        for i, fitness in enumerate(convergence_curve):
            ctk.CTkLabel(self.scroll_frame, text=f"Iterasi {i+1}").grid(row=i+1, column=0, padx=20, pady=2)
            ctk.CTkLabel(self.scroll_frame, text=f"{fitness:.6f}%").grid(row=i+1, column=1, padx=20, pady=2)

class ScrollableText(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.text = ctk.CTkTextbox(self, wrap="none")
        self.text.pack(side="left", fill="both", expand=True)
        
    def write(self, message):
        self.text.insert("end", message + "\n")
        self.text.see("end")
        
    def clear(self):
        self.text.delete("1.0", "end")
