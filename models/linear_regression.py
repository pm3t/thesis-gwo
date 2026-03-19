from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.last_index = 0

    def _prepare_features(self, dates, start_idx=0):
        """
        Create time index and day-of-week dummy variables.
        """
        df = pd.DataFrame({'date': pd.to_datetime(dates)})
        df['time_idx'] = np.arange(start_idx, start_idx + len(df))
        
        # Day of week features (0-6)
        df['day_of_week'] = df['date'].dt.dayofweek
        
        # One-hot encoding for day of week (to avoid ordinal bias)
        dummies = pd.get_dummies(df['day_of_week'], prefix='day').reindex(
            columns=[f'day_{i}' for i in range(7)], fill_value=0
        )
        
        X = pd.concat([df[['time_idx']], dummies], axis=1)
        return X

    def fit(self, dates, y):
        """
        Fit Linear Regression model based on time and seasonality.
        """
        X = self._prepare_features(dates)
        self.model.fit(X, y)
        self.last_index = len(y)
        self.last_dates = pd.to_datetime(dates)
        return self

    def forecast(self, steps=1):
        """
        Forecast future values by extending time index and dates.
        """
        # Create future dates (assuming daily frequency)
        last_date = self.last_dates.iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)
        
        X_future = self._prepare_features(future_dates, start_idx=self.last_index)
        return self.model.predict(X_future)
