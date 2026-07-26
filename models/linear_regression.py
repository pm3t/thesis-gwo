from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()
        self.last_index = 0
        self.last_dates = None
        self._feature_cols = None

    def _prepare_features(self, dates, start_idx=0):
        """
        Create time index, day-of-week, and month dummy variables.
        """
        df = pd.DataFrame({'date': pd.to_datetime(dates)})
        df['time_idx'] = np.arange(start_idx, start_idx + len(df))

        # Extract calendar features
        df['dow']   = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month

        # One-hot encoding for day of week (day_0 .. day_6)
        dow_dummies = pd.get_dummies(df['dow'], prefix='day').reindex(
            columns=[f'day_{i}' for i in range(7)], fill_value=0
        )

        # One-hot encoding for month (month_1 .. month_12)
        month_dummies = pd.get_dummies(df['month'], prefix='m').reindex(
            columns=[f'm_{i}' for i in range(1, 13)], fill_value=0
        )

        X = pd.concat([df[['time_idx']], dow_dummies, month_dummies], axis=1)
        return X

    def fit(self, dates, y):
        """
        Fit Linear Regression model on training dates and values.
        """
        X = self._prepare_features(dates, start_idx=0)
        self._feature_cols = list(X.columns)
        self.model.fit(X, y)
        self.last_index = len(y)
        self.last_dates = pd.to_datetime(dates)
        return self

    def forecast(self, steps=1, dates_test=None):
        """
        Predict test values for the given test dates or future step count.
        """
        if self.last_dates is None:
            raise ValueError("Model must be fitted first.")

        if dates_test is not None:
            future_dates = pd.to_datetime(dates_test)
        else:
            last_date = self.last_dates.iloc[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)

        X_future = self._prepare_features(future_dates, start_idx=self.last_index)
        return self.model.predict(X_future)

    def walk_forward(self, dates_test, y_test=None):
        """
        Predict values for test dates.
        """
        return self.forecast(len(dates_test), dates_test=dates_test)
