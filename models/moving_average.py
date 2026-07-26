import pandas as pd
import numpy as np


class MovingAverageModel:
    def __init__(self, window_size=4):
        """
        Seasonal Moving Average with Walk-Forward Forecasting.
        
        Parameters
        ----------
        window_size : int
            Number of weeks (same day of week) to average (default 4).
        """
        self.window_size = window_size
        self._df_history = None

    def fit(self, y, dates=None):
        """
        Store training data for forecasting.
        If dates are provided, extract day of week.
        """
        if dates is not None:
            dates = pd.to_datetime(dates)
            self._df_history = pd.DataFrame({
                'date': dates,
                'dow': dates.dt.dayofweek,
                'y': np.array(y, dtype=float)
            })
        else:
            y_arr = np.array(y, dtype=float)
            # Dummy daily dates starting from day 0 if dates not given
            dates_dummy = pd.date_range("2020-01-01", periods=len(y_arr))
            self._df_history = pd.DataFrame({
                'date': dates_dummy,
                'dow': dates_dummy.dayofweek,
                'y': y_arr
            })
        return self

    def forecast(self, steps=1):
        """
        Static multi-step forecast using last window_size weeks for each day of week.
        """
        if self._df_history is None:
            raise ValueError("Model must be fitted first.")

        df = self._df_history.copy()
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps)

        preds = []
        for fdate in future_dates:
            dow = fdate.dayofweek
            same_dow_vals = df[df['dow'] == dow]['y'].iloc[-self.window_size:].values
            if len(same_dow_vals) == 0:
                pred = df['y'].mean()
            else:
                pred = np.mean(same_dow_vals)
            preds.append(pred)

            # Append prediction to df to simulate multi-step extension
            df = pd.concat([df, pd.DataFrame({'date': [fdate], 'dow': [dow], 'y': [pred]})], ignore_index=True)

        return np.array(preds)

    def walk_forward(self, y_test, dates_test=None):
        """
        Walk-Forward 1-step forecast for test period using actual observations.

        Parameters
        ----------
        y_test : array-like
            True values for the test period.
        dates_test : array-like, optional
            Test dates. If omitted, generated daily sequentially.

        Returns
        -------
        np.ndarray of 1-step-ahead walk-forward predictions.
        """
        if self._df_history is None:
            raise ValueError("Model must be fitted first.")

        df = self._df_history.copy()
        y_test_arr = np.array(y_test, dtype=float)

        if dates_test is not None:
            dates_test = pd.to_datetime(dates_test)
        else:
            last_date = df['date'].iloc[-1]
            dates_test = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(y_test_arr))

        preds = []
        for fdate, true_val in zip(dates_test, y_test_arr):
            dow = fdate.dayofweek
            same_dow_vals = df[df['dow'] == dow]['y'].iloc[-self.window_size:].values
            if len(same_dow_vals) == 0:
                pred = df['y'].mean()
            else:
                pred = np.mean(same_dow_vals)
            preds.append(pred)

            # Update history with ACTUAL test observation (walk-forward)
            df = pd.concat([df, pd.DataFrame({'date': [fdate], 'dow': [dow], 'y': [true_val]})], ignore_index=True)

        return np.array(preds)
