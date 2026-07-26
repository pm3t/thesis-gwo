import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class DataTab(ctk.CTkFrame):
    def __init__(self, master, data_processor):
        super().__init__(master)

        self.data_processor = data_processor

        # Configure grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # ── Left Frame: Controls (Scrollable) ────────────────────────────────
        self.left_frame = ctk.CTkScrollableFrame(self, label_text="Data Parameters")
        self.left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Load button
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

        # ── Outlier Handling ─────────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_frame, text="── Penanganan Outlier ──",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=(15, 5))

        # Method selector
        ctk.CTkLabel(self.left_frame, text="Metode Deteksi:").pack(pady=(5, 0))
        self.combo_outlier_method = ctk.CTkComboBox(
            self.left_frame,
            values=["IQR (Interquartile Range)", "Z-Score", "None (Tidak ada)"],
            state="readonly",
            width=210
        )
        self.combo_outlier_method.pack(pady=5)
        self.combo_outlier_method.set("IQR (Interquartile Range)")

        # IQR multiplier (visible only for IQR)
        self.lbl_iqr_k = ctk.CTkLabel(self.left_frame, text="IQR multiplier (k):")
        self.lbl_iqr_k.pack(pady=(5, 0))
        self.entry_iqr_k = ctk.CTkEntry(self.left_frame, placeholder_text="1.5", width=100)
        self.entry_iqr_k.pack(pady=5)
        self.entry_iqr_k.insert(0, "1.5")

        # Z-Score threshold (visible only for Z-Score)
        self.lbl_z_thresh = ctk.CTkLabel(self.left_frame, text="Z-Score threshold:")
        self.lbl_z_thresh.pack(pady=(5, 0))
        self.entry_z_thresh = ctk.CTkEntry(self.left_frame, placeholder_text="3.0", width=100)
        self.entry_z_thresh.pack(pady=5)
        self.entry_z_thresh.insert(0, "3.0")

        # Action: clip or remove
        ctk.CTkLabel(self.left_frame, text="Tindakan:").pack(pady=(5, 0))
        self.combo_outlier_action = ctk.CTkComboBox(
            self.left_frame,
            values=["Clip (Winsorize)", "Remove (Hapus baris)"],
            state="readonly",
            width=210
        )
        self.combo_outlier_action.pack(pady=5)
        self.combo_outlier_action.set("Clip (Winsorize)")

        # Detect & preview outliers
        self.btn_detect = ctk.CTkButton(
            self.left_frame, text="🔍 Deteksi Outlier",
            command=self.detect_outliers,
            fg_color="#e67e22", hover_color="#ca6f1e"
        )
        self.btn_detect.pack(pady=8)

        # Outlier info label
        self.lbl_outlier_info = ctk.CTkLabel(
            self.left_frame, text="Outlier: belum dideteksi",
            wraplength=200, justify="left"
        )
        self.lbl_outlier_info.pack(pady=5)

        # ── Preprocessing / Split ────────────────────────────────────────────
        ctk.CTkLabel(
            self.left_frame, text="── Preprocessing & Split ──",
            font=ctk.CTkFont(weight="bold")
        ).pack(pady=(15, 5))

        # Remove low-sales rows (threshold)
        self.var_remove_zeros = ctk.BooleanVar(value=True)
        self.check_remove_zeros = ctk.CTkCheckBox(
            self.left_frame,
            text="Hapus baris Sales ≤ threshold",
            variable=self.var_remove_zeros
        )
        self.check_remove_zeros.pack(pady=(5, 0))

        ctk.CTkLabel(self.left_frame, text="Threshold minimum sales:").pack(pady=(6, 0))
        self.entry_zero_threshold = ctk.CTkEntry(self.left_frame, placeholder_text="20000", width=120)
        self.entry_zero_threshold.pack(pady=(2, 5))
        self.entry_zero_threshold.insert(0, "20000")

        self.btn_process = ctk.CTkButton(
            self.left_frame, text="Preprocess Data", command=self.preprocess_data
        )
        self.btn_process.pack(pady=10)

        ctk.CTkLabel(self.left_frame, text="Training Ratio (default 0.8):").pack(pady=5)
        self.entry_ratio = ctk.CTkEntry(self.left_frame, placeholder_text="0.8")
        self.entry_ratio.pack(pady=5)
        self.entry_ratio.insert(0, "0.8")

        self.btn_split = ctk.CTkButton(
            self.left_frame, text="Split Data (80/20)", command=self.split_data
        )
        self.btn_split.pack(pady=10)

        # ── Right Frame: Preview & Stats ─────────────────────────────────────
        self.right_frame = ctk.CTkTabview(self)
        self.right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        self.tab_preview = self.right_frame.add("Preview & Statistik")
        self.tab_outlier = self.right_frame.add("Visualisasi Outlier")

        # Preview tab
        self.tab_preview.grid_columnconfigure(0, weight=1)
        self.tab_preview.grid_rowconfigure(0, weight=1)
        self.tab_preview.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(self.tab_preview, text="Data Preview (First 10 rows):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        self.text_preview = ctk.CTkTextbox(self.tab_preview, height=180)
        self.text_preview.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        ctk.CTkLabel(self.tab_preview, text="Basic Statistics:", font=ctk.CTkFont(weight="bold")).grid(row=2, column=0, sticky="w", padx=10, pady=(10, 0))
        self.text_stats = ctk.CTkTextbox(self.tab_preview, height=200)
        self.text_stats.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)
        self.tab_preview.grid_rowconfigure(3, weight=1)

        # Outlier visualisation tab
        self.tab_outlier.grid_columnconfigure(0, weight=1)
        self.tab_outlier.grid_rowconfigure(0, weight=1)
        self.outlier_canvas_frame = ctk.CTkFrame(self.tab_outlier)
        self.outlier_canvas_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.outlier_canvas = None
        self._outlier_placeholder()

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _outlier_placeholder(self):
        lbl = ctk.CTkLabel(
            self.outlier_canvas_frame,
            text="Klik 🔍 Deteksi Outlier untuk melihat visualisasi.",
            text_color="gray"
        )
        lbl.pack(expand=True)

    def _get_raw_series(self):
        """Return the aggregated raw series before normalization."""
        date_col   = self.combo_date.get()
        target_col = self.combo_target.get()
        if not date_col or not target_col:
            return None, None, None, None
        if self.data_processor.raw_data is None:
            return None, None, None, None

        df = self.data_processor.raw_data.copy()
        df = df.dropna(subset=[date_col, target_col])
        df = df.drop_duplicates()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        df = df.groupby(date_col)[target_col].sum().reset_index()
        return df, df[date_col], df[target_col], target_col

    def _compute_bounds(self, series):
        """Return (lower, upper, mask_outliers) based on current UI settings."""
        method = self.combo_outlier_method.get()
        if method.startswith("IQR"):
            try:
                k = float(self.entry_iqr_k.get())
            except ValueError:
                k = 1.5
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
        elif method.startswith("Z-Score"):
            try:
                thresh = float(self.entry_z_thresh.get())
            except ValueError:
                thresh = 3.0
            mean = series.mean()
            std  = series.std()
            lower = mean - thresh * std
            upper = mean + thresh * std
        else:
            lower = series.min()
            upper = series.max()
        mask = (series < lower) | (series > upper)
        return lower, upper, mask

    # ─────────────────────────────────────────────────────────────────────────
    # Actions
    # ─────────────────────────────────────────────────────────────────────────
    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.lbl_file.configure(text=file_path.split("/")[-1])
            df = self.data_processor.load_csv(file_path)

            cols = list(df.columns)
            self.combo_date.configure(values=cols)
            self.combo_target.configure(values=cols)

            self.update_preview()
            self.update_stats()
            self.lbl_outlier_info.configure(text="Outlier: belum dideteksi")

    def detect_outliers(self):
        df, dates, series, target_col = self._get_raw_series()
        if df is None:
            messagebox.showwarning("Warning", "Silakan load CSV dan pilih kolom terlebih dahulu.")
            return

        method = self.combo_outlier_method.get()
        if method.startswith("None"):
            self.lbl_outlier_info.configure(text="Penanganan outlier: Tidak aktif.")
            return

        lower, upper, mask = self._compute_bounds(series)
        n_outliers = int(mask.sum())
        pct = n_outliers / len(series) * 100

        # Update info label
        action = self.combo_outlier_action.get()
        self.lbl_outlier_info.configure(
            text=(
                f"Terdeteksi: {n_outliers} outlier ({pct:.1f}%)\n"
                f"Batas bawah: {lower:,.2f}\n"
                f"Batas atas:  {upper:,.2f}\n"
                f"Tindakan: {action}"
            )
        )

        # ── Draw visualisation ──────────────────────────────────────────────
        # Destroy previous canvas
        for w in self.outlier_canvas_frame.winfo_children():
            w.destroy()
        plt.close("all")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=95)
        fig.patch.set_facecolor("#f5f5f5")

        # Panel 1: Time-series with outliers highlighted
        ax1 = axes[0]
        ax1.set_facecolor("#ffffff")
        ax1.plot(dates, series, color="#3498db", linewidth=1.2, label="Data asli", zorder=2)
        ax1.axhline(upper, color="#e74c3c", linestyle="--", linewidth=1, label=f"Batas atas ({upper:,.0f})")
        ax1.axhline(lower, color="#e67e22", linestyle="--", linewidth=1, label=f"Batas bawah ({lower:,.0f})")
        if n_outliers > 0:
            ax1.scatter(dates[mask], series[mask], color="#e74c3c", s=40, zorder=5, label=f"Outlier ({n_outliers})")
        ax1.set_title("Time-Series + Batas Outlier", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Tanggal", fontsize=9)
        ax1.set_ylabel("Sales", fontsize=9)
        ax1.legend(fontsize=7, loc="upper left")
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.tick_params(axis="x", rotation=30, labelsize=7)

        # Panel 2: Box plot
        ax2 = axes[1]
        ax2.set_facecolor("#ffffff")
        bp = ax2.boxplot(
            series.dropna(), vert=True, patch_artist=True,
            boxprops=dict(facecolor="#aed6f1", color="#2980b9"),
            medianprops=dict(color="#e74c3c", linewidth=2),
            flierprops=dict(marker="o", color="#e74c3c", markersize=5, alpha=0.7),
            whiskerprops=dict(color="#2980b9"),
            capprops=dict(color="#2980b9")
        )
        ax2.axhline(upper, color="#e74c3c", linestyle="--", linewidth=1, label=f"Batas atas ({upper:,.0f})")
        ax2.axhline(lower, color="#e67e22", linestyle="--", linewidth=1, label=f"Batas bawah ({lower:,.0f})")
        ax2.set_title("Box Plot", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Sales", fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

        plt.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=self.outlier_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self.outlier_canvas = canvas

        # Switch to visualisation tab
        self.right_frame.set("Visualisasi Outlier")

        if n_outliers == 0:
            messagebox.showinfo("Outlier", f"Tidak ditemukan outlier dengan metode {method}.")
        else:
            messagebox.showinfo(
                "Outlier Terdeteksi",
                f"Ditemukan {n_outliers} outlier ({pct:.1f}%).\n"
                f"Saat Preprocess, outlier akan di-{action.split()[0].lower()}."
            )

    def update_preview(self, df=None):
        if df is None:
            df = self.data_processor.raw_data
        self.text_preview.delete("1.0", "end")
        if df is not None:
            self.text_preview.insert("1.0", df.head(10).to_string())

    def update_stats(self):
        self.text_stats.delete("1.0", "end")

        raw_stats = self.data_processor.get_stats()
        if raw_stats is not None:
            self.text_stats.insert("end", "=== 1. DATASET AWAL ===\n")
            if self.data_processor.raw_data is not None:
                self.text_stats.insert("end", f"Jumlah Baris & Kolom: {self.data_processor.raw_data.shape}\n")
            self.text_stats.insert("end", raw_stats.to_string())
            self.text_stats.insert("end", "\n\n")

        if self.data_processor.df is not None:
            self.text_stats.insert("end", "=== 2. DATASET HASIL PREPROCESS ===\n")
            self.text_stats.insert("end", f"Jumlah Baris & Kolom: {self.data_processor.df.shape}\n")
            self.text_stats.insert("end", self.data_processor.df.describe().to_string())
            self.text_stats.insert("end", "\n\n")

        if self.data_processor.train_df is not None and self.data_processor.test_df is not None:
            self.text_stats.insert("end", "=== 3. DATASET HASIL SPLIT ===\n")
            self.text_stats.insert("end", f"Data Latih (Train) shape: {self.data_processor.train_df.shape}\n")
            self.text_stats.insert("end", f"Data Uji (Test) shape: {self.data_processor.test_df.shape}\n\n")
            self.text_stats.insert("end", "--- Statistik Data Latih ---\n")
            self.text_stats.insert("end", self.data_processor.train_df.describe().to_string())
            self.text_stats.insert("end", "\n\n--- Statistik Data Uji ---\n")
            self.text_stats.insert("end", self.data_processor.test_df.describe().to_string())
            self.text_stats.insert("end", "\n")

    def preprocess_data(self):
        date_col   = self.combo_date.get()
        target_col = self.combo_target.get()

        if not date_col or not target_col:
            messagebox.showwarning("Warning", "Please select date and target columns.")
            return

        method  = self.combo_outlier_method.get()
        action  = self.combo_outlier_action.get()

        # Build kwargs for data_processor
        handle = not method.startswith("None")
        try:
            iqr_k   = float(self.entry_iqr_k.get())
        except ValueError:
            iqr_k = 1.5
        try:
            z_thresh = float(self.entry_z_thresh.get())
        except ValueError:
            z_thresh = 3.0
        try:
            zero_threshold = float(self.entry_zero_threshold.get())
        except ValueError:
            zero_threshold = 20000.0

        try:
            df_processed = self.data_processor.preprocess(
                date_col, target_col,
                handle_outliers=handle,
                outlier_method=method,
                iqr_k=iqr_k,
                z_thresh=z_thresh,
                outlier_action=action,
                remove_zeros=self.var_remove_zeros.get(),
                zero_threshold=zero_threshold
            )
            self.update_preview(df_processed)
            self.update_stats()
            messagebox.showinfo("Success", "Data preprocessed successfully. Now please split the data.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to preprocess: {e}")

    def split_data(self):
        try:
            ratio = float(self.entry_ratio.get())
            train_df, test_df = self.data_processor.split_data(ratio)

            if train_df is not None:
                self.update_stats()
                msg = (
                    f"Data Split Successful!\n"
                    f"Training set: {len(train_df)} rows\n"
                    f"Testing set:  {len(test_df)} rows"
                )
                messagebox.showinfo("Success", msg)
            else:
                messagebox.showwarning("Warning", "Please preprocess data first.")
        except ValueError:
            messagebox.showerror("Error", "Training ratio must be a number between 0 and 1.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to split data: {e}")
