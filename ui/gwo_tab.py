import customtkinter as ctk
from tkinter import messagebox
from optimizers.gwo import GreyWolfOptimizer
from ui.widgets import IterationTable
import numpy as np

class GWOTab(ctk.CTkFrame):
    def __init__(self, master, data_processor, model_results, gwo_results):
        super().__init__(master)
        
        self.data_processor = data_processor
        self.model_results = model_results
        self.gwo_results = gwo_results # { 'best_weights': [], 'convergence': [] }
        self.canvas = None
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        # Left Frame: GWO Controls
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.lbl_title = ctk.CTkLabel(self.left_frame, text="GWO Parameters", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_title.pack(pady=10)
        
        self.lbl_pop = ctk.CTkLabel(self.left_frame, text="Population Size (n_wolves):")
        self.lbl_pop.pack(pady=5)
        self.entry_pop = ctk.CTkEntry(self.left_frame, placeholder_text="20")
        self.entry_pop.pack(pady=5)
        self.entry_pop.insert(0, "20")
        
        self.lbl_iter = ctk.CTkLabel(self.left_frame, text="Max Iterations:")
        self.lbl_iter.pack(pady=5)
        self.entry_iter = ctk.CTkEntry(self.left_frame, placeholder_text="100")
        self.entry_iter.pack(pady=5)
        self.entry_iter.insert(0, "100")
        
        self.btn_run = ctk.CTkButton(self.left_frame, text="Run GWO Optimization", command=self.run_optimization)
        self.btn_run.pack(pady=20)
        
        # Results section
        self.res_frame = ctk.CTkLabel(self.left_frame, text="Optimal Weights:", font=ctk.CTkFont(weight="bold"))
        self.res_frame.pack(pady=5)
        
        self.lbl_w1 = ctk.CTkLabel(self.left_frame, text="w1 (MA): -")
        self.lbl_w1.pack(pady=2)
        self.lbl_w2 = ctk.CTkLabel(self.left_frame, text="w2 (ES): -")
        self.lbl_w2.pack(pady=2)
        self.lbl_w3 = ctk.CTkLabel(self.left_frame, text="w3 (LR): -")
        self.lbl_w3.pack(pady=2)
        
        self.lbl_best_fitness = ctk.CTkLabel(self.left_frame, text="Best Fitness (MAPE): -", font=ctk.CTkFont(weight="bold"))
        self.lbl_best_fitness.pack(pady=10)

        # Right Frame: Convergence Plot
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_plot_title = ctk.CTkLabel(self.right_frame, text="Optimization History", font=ctk.CTkFont(weight="bold"))
        self.lbl_plot_title.pack(pady=5)
        
        self.iteration_table = IterationTable(self.right_frame)
        self.iteration_table.pack(fill="both", expand=True, padx=10, pady=10)

    def run_optimization(self):
        # Check if baseline models are ready
        if not all(k in self.model_results for k in ['MA', 'ES', 'LR']):
            messagebox.showwarning("Warning", "Please run all three baseline models first.")
            return
            
        _, _, _, y_test = self.data_processor.get_train_test_data()
        y_pred_ma = self.model_results['MA']['pred']
        y_pred_es = self.model_results['ES']['pred']
        y_pred_lr = self.model_results['LR']['pred']
        
        try:
            n_wolves = int(self.entry_pop.get())
            max_iter = int(self.entry_iter.get())
            
            gwo = GreyWolfOptimizer(n_wolves=n_wolves, max_iter=max_iter)
            best_weights, convergence = gwo.optimize(y_test, y_pred_ma, y_pred_es, y_pred_lr)
            
            # Store results
            self.gwo_results['best_weights'] = best_weights
            self.gwo_results['convergence'] = convergence
            
            # Update UI
            self.lbl_w1.configure(text=f"w1 (MA): {best_weights[0]:.4f}")
            self.lbl_w2.configure(text=f"w2 (ES): {best_weights[1]:.4f}")
            self.lbl_w3.configure(text=f"w3 (LR): {best_weights[2]:.4f}")
            self.lbl_best_fitness.configure(text=f"Best Fitness (MAPE): {convergence[-1]:.4f}%")
            
            # Update Iteration Table
            self.iteration_table.update_history(convergence)
            
            messagebox.showinfo("Success", "GWO Optimization completed.")
            
        except Exception as e:
            messagebox.showerror("Error", f"GWO failed: {e}")
