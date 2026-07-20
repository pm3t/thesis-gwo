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

    def preprocess(self, date_col, target_col, handle_outliers=True):
        """
        Cleaning, sorting, and aggregating data.
        """
        self.date_col = date_col
        self.target_col = target_col
        
        # Validate columns exist
        if hasattr(self, 'raw_train_data') and self.raw_train_data is not None:
            if date_col not in self.raw_train_data.columns:
                raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan di Train dataset.\nKolom yang tersedia: {list(self.raw_train_data.columns)}")
            if target_col not in self.raw_train_data.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di Train dataset.\nKolom yang tersedia: {list(self.raw_train_data.columns)}")
        if hasattr(self, 'raw_test_data') and self.raw_test_data is not None:
            if date_col not in self.raw_test_data.columns:
                raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan di Test dataset.\nKolom yang tersedia: {list(self.raw_test_data.columns)}")
            if target_col not in self.raw_test_data.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di Test dataset.\nCatatan: Dataset test harus memiliki nilai aktual target untuk mengukur akurasi/MAPE.\nKolom yang tersedia: {list(self.raw_test_data.columns)}")
        if self.raw_data is not None:
            if date_col not in self.raw_data.columns:
                raise ValueError(f"Kolom tanggal '{date_col}' tidak ditemukan di dataset.\nKolom yang tersedia: {list(self.raw_data.columns)}")
            if target_col not in self.raw_data.columns:
                raise ValueError(f"Kolom target '{target_col}' tidak ditemukan di dataset.\nKolom yang tersedia: {list(self.raw_data.columns)}")

        # Check if we have pre-split train/test data loaded
        if hasattr(self, 'raw_train_data') and self.raw_train_data is not None and hasattr(self, 'raw_test_data') and self.raw_test_data is not None:
            # Preprocess train
            df_train = self.raw_train_data.copy()
            df_train = df_train.dropna(subset=[date_col, target_col])
            df_train = df_train.drop_duplicates()
            df_train[date_col] = pd.to_datetime(df_train[date_col])
            df_train = df_train.sort_values(by=date_col)
            df_train = df_train.groupby(date_col)[target_col].sum().reset_index()
            
            # Preprocess test
            df_test = self.raw_test_data.copy()
            df_test = df_test.dropna(subset=[date_col, target_col])
            df_test = df_test.drop_duplicates()
            df_test[date_col] = pd.to_datetime(df_test[date_col])
            df_test = df_test.sort_values(by=date_col)
            df_test = df_test.groupby(date_col)[target_col].sum().reset_index()
            
            if handle_outliers:
                # Calculate bounds on train to avoid data leakage
                q1 = df_train[target_col].quantile(0.25)
                q3 = df_train[target_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                df_train[target_col] = df_train[target_col].clip(lower_bound, upper_bound)
                df_test[target_col] = df_test[target_col].clip(lower_bound, upper_bound)
                
            # Global Normalization
            self.scaler = MinMaxScaler()
            df_train[target_col] = self.scaler.fit_transform(df_train[[target_col]]).flatten()
            df_test[target_col] = self.scaler.transform(df_test[[target_col]]).flatten()
            
            self.train_df = df_train
            self.test_df = df_test
            self.df = pd.concat([df_train, df_test], ignore_index=True)
            return self.df
            
        else:
            if self.raw_data is None:
                return None
            
            df = self.raw_data.copy()
            df = df.dropna(subset=[date_col, target_col])
            df = df.drop_duplicates()
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(by=date_col)
            df = df.groupby(date_col)[target_col].sum().reset_index()
            
            if handle_outliers:
                q1 = df[target_col].quantile(0.25)
                q3 = df[target_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                df[target_col] = df[target_col].clip(lower_bound, upper_bound)
                
            # Global Normalization
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
