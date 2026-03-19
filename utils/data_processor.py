import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self):
        self.raw_data = None
        self.df = None
        self.train_df = None
        self.test_df = None
        self.date_col = None
        self.target_col = None

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

    def preprocess(self, date_col, target_col, handle_outliers=False):
        """
        Cleaning, sorting, and aggregating data.
        """
        if self.raw_data is None:
            return None
        
        self.date_col = date_col
        self.target_col = target_col
        
        df = self.raw_data.copy()
        
        # 1. Clean missing values and duplicates
        df = df.dropna(subset=[date_col, target_col])
        df = df.drop_duplicates()
        
        # 2. Convert date column
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 3. Sort by date
        df = df.sort_values(by=date_col)
        
        # 4. Agregasi by date (total sales per date) - khusus dataset Kaggle Store Sales
        # Seringkali ditanya total penjualan per hari untuk forecasting
        df = df.groupby(date_col)[target_col].sum().reset_index()
        
        # 5. Handle outliers (Simple clipping using IQR)
        if handle_outliers:
            q1 = df[target_col].quantile(0.25)
            q3 = df[target_col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[target_col] = df[target_col].clip(lower_bound, upper_bound)
            
        self.df = df
        return df

    def split_data(self, train_ratio=0.8):
        """
        Chronological split (training first, testing last).
        """
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
