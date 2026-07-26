import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self):
        self.raw_data = None
        self.df = None
        self.train_df = None
        self.test_df = None
        self.date_col = None
        self.target_col = None
        self.scaler = None

    def load_csv(self, file_path):
        """
        Load CSV data and store it.
        """
        self.raw_data = pd.read_csv(file_path)
        return self.raw_data

    def get_stats(self):
        """
        Return basic stats for the dataframe.
        """
        if self.raw_data is None:
            return None
        return self.raw_data.describe()

    def preprocess(self, date_col, target_col,
                   handle_outliers=True,
                   outlier_method="IQR (Interquartile Range)",
                   iqr_k=1.5,
                   z_thresh=3.0,
                   outlier_action="Clip (Winsorize)",
                   remove_zeros=True,
                   zero_threshold=20000.0):
        """
        Cleaning, sorting, aggregating, outlier handling, and normalisation.

        Parameters
        ----------
        handle_outliers : bool   – whether to apply outlier handling at all
        outlier_method  : str    – "IQR (Interquartile Range)" | "Z-Score" | "None ..."
        iqr_k           : float  – multiplier for IQR bounds  (default 1.5)
        z_thresh        : float  – standard-deviation threshold for Z-Score (default 3.0)
        outlier_action  : str    – "Clip (Winsorize)" clips to bounds;
                                   "Remove (Hapus baris)" drops rows outside bounds
        remove_zeros    : bool   – if True, drop rows where target_col <= zero_threshold
        zero_threshold  : float  – minimum sales value to keep (default 20000)
        """
        self.date_col   = date_col
        self.target_col = target_col

        def _compute_bounds(series, fit_series=None):
            """Return (lower, upper) from the chosen method, fitted on fit_series."""
            src = fit_series if fit_series is not None else series
            if outlier_method.startswith("IQR"):
                q1 = src.quantile(0.25)
                q3 = src.quantile(0.75)
                iqr = q3 - q1
                return q1 - iqr_k * iqr, q3 + iqr_k * iqr
            elif outlier_method.startswith("Z-Score"):
                mean, std = src.mean(), src.std()
                return mean - z_thresh * std, mean + z_thresh * std
            else:
                return series.min(), series.max()

        def _apply_outlier(df, col, lower, upper):
            if outlier_action.startswith("Clip"):
                df[col] = df[col].clip(lower, upper)
            else:  # Remove
                df = df[(df[col] >= lower) & (df[col] <= upper)]
            return df

        # Validate columns exist
        if hasattr(self, 'raw_train_data') and self.raw_train_data is not None:
            for src_name, src_df in [("Train", self.raw_train_data), ("Test", self.raw_test_data)]:
                if date_col not in src_df.columns:
                    raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan di {src_name} dataset.")
                if target_col not in src_df.columns:
                    raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di {src_name} dataset.")
        if self.raw_data is not None:
            if date_col not in self.raw_data.columns:
                raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan di dataset.")
            if target_col not in self.raw_data.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.")

        # ── Pre-split path (train.csv + test.csv loaded separately) ──────────
        if hasattr(self, 'raw_train_data') and self.raw_train_data is not None \
                and hasattr(self, 'raw_test_data') and self.raw_test_data is not None:

            def _clean(src):
                d = src.copy()
                d = d.dropna(subset=[date_col, target_col])
                d = d.drop_duplicates()
                d[date_col] = pd.to_datetime(d[date_col])
                d = d.sort_values(by=date_col)
                d = d.groupby(date_col)[target_col].sum().reset_index()
                return d

            df_train = _clean(self.raw_train_data)
            df_test  = _clean(self.raw_test_data)

            if remove_zeros:
                df_train = df_train[df_train[target_col] > zero_threshold].reset_index(drop=True)
                df_test  = df_test[df_test[target_col]  > zero_threshold].reset_index(drop=True)

            if handle_outliers:
                lower, upper = _compute_bounds(df_train[target_col])
                df_train = _apply_outlier(df_train, target_col, lower, upper)
                df_test  = _apply_outlier(df_test,  target_col, lower, upper)

            # Normalise (fit on train only to prevent leakage)
            self.scaler = MinMaxScaler()
            df_train[target_col] = self.scaler.fit_transform(df_train[[target_col]]).flatten()
            df_test[target_col]  = self.scaler.transform(df_test[[target_col]]).flatten()

            self.train_df = df_train
            self.test_df  = df_test
            self.df = pd.concat([df_train, df_test], ignore_index=True)
            return self.df

        # ── Single-CSV path ──────────────────────────────────────────────────
        if self.raw_data is None:
            return None

        df = self.raw_data.copy()
        df = df.dropna(subset=[date_col, target_col])
        df = df.drop_duplicates()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        df = df.groupby(date_col)[target_col].sum().reset_index()

        if remove_zeros:
            df = df[df[target_col] > zero_threshold].reset_index(drop=True)

        if handle_outliers:
            lower, upper = _compute_bounds(df[target_col])
            df = _apply_outlier(df, target_col, lower, upper)

        # Normalise
        self.scaler = MinMaxScaler()
        df[target_col] = self.scaler.fit_transform(df[[target_col]]).flatten()

        self.df = df
        return df

    def split_data(self, train_ratio=0.8):
        """
        Chronological split or return pre-split data.
        """
        if self.train_df is not None and self.test_df is not None:
            return self.train_df, self.test_df
            
        if self.df is None:
            return None, None
            
        train_size = int(len(self.df) * train_ratio)
        self.train_df = self.df.iloc[:train_size]
        self.test_df = self.df.iloc[train_size:]
        
        return self.train_df, self.test_df

    def get_train_test_data(self):
        """
        Return training and testing data for models.
        """
        if self.train_df is None or self.test_df is None:
            return None, None, None, None
            
        y_train = self.train_df[self.target_col]
        y_test = self.test_df[self.target_col]
        
        return self.train_df[self.date_col], y_train, self.test_df[self.date_col], y_test

    def get_full_data(self):
        """
        Return full cleaned data.
        """
        if self.df is None:
            return None, None
        return self.df[self.date_col], self.df[self.target_col]

    def run_adf_test(self):
        """
        Run Augmented Dickey-Fuller (ADF) test on the target column of preprocessed data.
        """
        if self.df is None or self.target_col is None:
            return None
            
        from statsmodels.tsa.stattools import adfuller
        series = self.df[self.target_col].dropna()
        result = adfuller(series)
        
        adf_stat = result[0]
        p_val = result[1]
        crit_values = result[4]
        
        is_stationary = p_val < 0.05
        
        return {
            'adf_stat': adf_stat,
            'p_value': p_val,
            'critical_values': crit_values,
            'is_stationary': is_stationary
        }

    def inverse_transform(self, y):
        """
        Inverse transform a 1D array/series using the global MinMaxScaler.
        """
        if self.scaler is None or y is None:
            return y
        y_arr = np.array(y).reshape(-1, 1)
        return self.scaler.inverse_transform(y_arr).flatten()
