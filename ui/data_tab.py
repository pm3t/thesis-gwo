import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd

class DataTab(ctk.CTkFrame):
    def __init__(self, master, data_processor):
        super().__init__(master)
        
        self.data_processor = data_processor
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Left Frame: Controls
        self.left_frame = ctk.CTkFrame(self)
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        self.btn_load = ctk.CTkButton(self.left_frame, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(pady=10)
        
        self.lbl_file = ctk.CTkLabel(self.left_frame, text="No file loaded", wraplength=200)
        self.lbl_file.pack(pady=5)
        
        # Column selection
        self.lbl_date = ctk.CTkLabel(self.left_frame, text="Select Date Column:")
        self.lbl_date.pack(pady=5)
        self.combo_date = ctk.CTkComboBox(self.left_frame, values=[])
        self.combo_date.pack(pady=5)
        
        self.lbl_target = ctk.CTkLabel(self.left_frame, text="Select Target Column:")
        self.lbl_target.pack(pady=5)
        self.combo_target = ctk.CTkComboBox(self.left_frame, values=[])
        self.combo_target.pack(pady=5)
        
        # Preprocessing options
        self.check_outliers = ctk.CTkCheckBox(self.left_frame, text="Handle Outliers (IQR)")
        self.check_outliers.pack(pady=5)
        
        self.btn_process = ctk.CTkButton(self.left_frame, text="Preprocess Data", command=self.preprocess_data)
        self.btn_process.pack(pady=10)
        
        # Split options
        self.lbl_ratio = ctk.CTkLabel(self.left_frame, text="Training Ratio (default 0.8):")
        self.lbl_ratio.pack(pady=5)
        self.entry_ratio = ctk.CTkEntry(self.left_frame, placeholder_text="0.8")
        self.entry_ratio.pack(pady=5)
        self.entry_ratio.insert(0, "0.8")
        
        self.btn_split = ctk.CTkButton(self.left_frame, text="Split Data (80/20)", command=self.split_data)
        self.btn_split.pack(pady=10)

        # Right Frame: Preview & Stats
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        
        self.lbl_preview = ctk.CTkLabel(self.right_frame, text="Data Preview (First 10 rows):", font=ctk.CTkFont(weight="bold"))
        self.lbl_preview.pack(pady=5)
        
        self.text_preview = ctk.CTkTextbox(self.right_frame, height=200)
        self.text_preview.pack(fill="x", padx=10, pady=5)
        
        self.lbl_stats = ctk.CTkLabel(self.right_frame, text="Basic Statistics:", font=ctk.CTkFont(weight="bold"))
        self.lbl_stats.pack(pady=5)
        
        self.text_stats = ctk.CTkTextbox(self.right_frame, height=200)
        self.text_stats.pack(fill="x", padx=10, pady=5)

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.lbl_file.configure(text=file_path.split("/")[-1])
            df = self.data_processor.load_csv(file_path)
            
            # Update combo boxes
            cols = list(df.columns)
            self.combo_date.configure(values=cols)
            self.combo_target.configure(values=cols)
            
            # Update Preview & Stats
            self.update_preview()
            self.update_stats()

    def update_preview(self, df=None):
        if df is None:
            df = self.data_processor.raw_data
            
        self.text_preview.delete("1.0", "end")
        if df is not None:
            self.text_preview.insert("1.0", df.head(10).to_string())

    def update_stats(self):
        stats = self.data_processor.get_stats()
        self.text_stats.delete("1.0", "end")
        if stats is not None:
            self.text_stats.insert("1.0", stats.to_string())

    def preprocess_data(self):
        date_col = self.combo_date.get()
        target_col = self.combo_target.get()
        handle_outliers = self.check_outliers.get()
        
        if not date_col or not target_col:
            messagebox.showwarning("Warning", "Please select date and target columns.")
            return
            
        try:
            df_processed = self.data_processor.preprocess(date_col, target_col, handle_outliers)
            self.update_preview(df_processed)
            messagebox.showinfo("Success", "Data preprocessed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess: {e}")

    def split_data(self):
        try:
            ratio = float(self.entry_ratio.get())
            train_df, test_df = self.data_processor.split_data(ratio)
            
            if train_df is not None:
                msg = f"Data Split Successful!\nTraining set: {len(train_df)} rows\nTesting set: {len(test_df)} rows"
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showwarning("Warning", "Please preprocess data first.")
        except ValueError:
            messagebox.showerror("Error", "Training ratio must be a number between 0 and 1.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to split data: {e}")
