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

        # Left Frame: Controls (Scrollable)
        self.left_frame = ctk.CTkScrollableFrame(self, label_text="Data Parameters")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Mode Selection
        self.lbl_mode = ctk.CTkLabel(self.left_frame, text="Data Mode:")
        self.lbl_mode.pack(pady=(5, 0))
        self.segmented_mode = ctk.CTkSegmentedButton(
            self.left_frame, 
            values=["Single CSV", "Pre-split CSVs"],
            command=self.toggle_mode
        )
        self.segmented_mode.pack(pady=5)
        self.segmented_mode.set("Single CSV")

        # Load buttons
        self.btn_load = ctk.CTkButton(self.left_frame, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(pady=10)
        
        self.btn_load_train = ctk.CTkButton(self.left_frame, text="Load Train CSV", command=self.load_train_csv)
        self.btn_load_test = ctk.CTkButton(self.left_frame, text="Load Test CSV", command=self.load_test_csv)
        
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
        self.btn_process = ctk.CTkButton(self.left_frame, text="Preprocess Data", command=self.preprocess_data)
        self.btn_process.pack(pady=10)
        
        self.btn_adf = ctk.CTkButton(self.left_frame, text="Run ADF Stationarity Test", command=self.run_adf_test)
        self.btn_adf.pack(pady=5)
        
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
        
        if not date_col or not target_col:
            messagebox.showwarning("Warning", "Please select date and target columns.")
            return
            
        try:
            df_processed = self.data_processor.preprocess(date_col, target_col)
            self.update_preview(df_processed)
            
            if self.segmented_mode.get() == "Pre-split CSVs":
                msg = f"Data preprocessed successfully!\nTraining set: {len(self.data_processor.train_df)} rows\nTesting set: {len(self.data_processor.test_df)} rows"
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showinfo("Success", "Data preprocessed successfully. Now please split the data.")
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

    def run_adf_test(self):
        if self.data_processor.df is None:
            messagebox.showwarning("Warning", "Please preprocess data first before running ADF test.")
            return
            
        try:
            res = self.data_processor.run_adf_test()
            if res is None:
                messagebox.showerror("Error", "Failed to run ADF test. Check if data is loaded correctly.")
                return
                
            adf_stat = res['adf_stat']
            p_val = res['p_value']
            crit = res['critical_values']
            conclusion = "STATIONARY (Stasioner)" if res['is_stationary'] else "NON-STATIONARY (Tidak Stasioner)"
            
            # Format output text
            adf_text = (
                f"\n====================================\n"
                f"AUGMENTED DICKEY-FULLER (ADF) TEST\n"
                f"====================================\n"
                f"ADF Statistic: {adf_stat:.6f}\n"
                f"p-value: {p_val:.6e}\n"
                f"Conclusion: {conclusion}\n\n"
                f"Critical Values:\n"
                f"  1%:  {crit['1%']:.6f}\n"
                f"  5%:  {crit['5%']:.6f}\n"
                f"  10%: {crit['10%']:.6f}\n"
                f"====================================\n"
            )
            
            # Append to text_stats
            self.text_stats.insert("end", adf_text)
            self.text_stats.see("end")
            messagebox.showinfo("Success", f"ADF Test completed. Conclusion: {conclusion}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to run ADF test: {e}")

    def toggle_mode(self, mode):
        self.lbl_file.configure(text="No file loaded")
        
        self.data_processor.raw_data = None
        self.data_processor.raw_train_data = None
        self.data_processor.raw_test_data = None
        self.data_processor.df = None
        self.data_processor.train_df = None
        self.data_processor.test_df = None
        
        if hasattr(self, 'train_path'): del self.train_path
        if hasattr(self, 'test_path'): del self.test_path

        if mode == "Single CSV":
            self.btn_load_train.pack_forget()
            self.btn_load_test.pack_forget()
            self.btn_load.pack(after=self.segmented_mode, pady=10)
            
            self.lbl_ratio.pack(after=self.btn_adf, pady=5)
            self.entry_ratio.pack(after=self.lbl_ratio, pady=5)
            self.btn_split.pack(after=self.entry_ratio, pady=10)
        else:
            self.btn_load.pack_forget()
            self.btn_load_train.pack(after=self.segmented_mode, pady=5)
            self.btn_load_test.pack(after=self.btn_load_train, pady=5)
            
            self.lbl_ratio.pack_forget()
            self.entry_ratio.pack_forget()
            self.btn_split.pack_forget()

    def load_train_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.train_path = file_path
            test_info = f"\nTest: {self.test_path.split('/')[-1]}" if hasattr(self, 'test_path') else ""
            self.lbl_file.configure(text=f"Train: {file_path.split('/')[-1]}{test_info}")
            
            df_train = pd.read_csv(file_path)
            self.data_processor.raw_train_data = df_train
            
            cols = list(df_train.columns)
            self.combo_date.configure(values=cols)
            self.combo_target.configure(values=cols)
            
            if hasattr(self, 'test_path') and self.data_processor.raw_test_data is not None:
                self.data_processor.raw_data = pd.concat([df_train, self.data_processor.raw_test_data], ignore_index=True)
                self.update_preview(df_train)
                self.update_stats()

    def load_test_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.test_path = file_path
            train_info = f"Train: {self.train_path.split('/')[-1]}\n" if hasattr(self, 'train_path') else ""
            self.lbl_file.configure(text=f"{train_info}Test: {file_path.split('/')[-1]}")
            
            df_test = pd.read_csv(file_path)
            self.data_processor.raw_test_data = df_test
            
            if hasattr(self, 'train_path') and self.data_processor.raw_train_data is not None:
                self.data_processor.raw_data = pd.concat([self.data_processor.raw_train_data, df_test], ignore_index=True)
                self.update_preview(self.data_processor.raw_train_data)
                self.update_stats()
