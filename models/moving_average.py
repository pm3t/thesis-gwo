import pandas as pd
import numpy as np

class MovingAverageModel:
    def __init__(self, window_size=3):
        self.window_size = window_size
        self.last_values = None

    def fit(self, y):
        """
        Moving average doesn't have a 'training' phase in the traditional sense,
        but we store the last values for future prediction if needed.
        """
        self.last_values = y.iloc[-self.window_size:].values
        return self

    def predict(self, data):
        """
        Calculate simple moving average.
        For historical data, it uses a rolling window.
        """
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.rolling(window=self.window_size).mean().bfill()
        else:
            # Fallback for single value or array
            return np.mean(data[-self.window_size:])

    def forecast(self, steps=1):
        """
        Forecast future values based on the last window.
        Simple MA forecast is usually the average of the last 'n' observations.
        """
        if self.last_values is None:
            raise ValueError("Model must be fitted first.")
        
        forecast_val = np.mean(self.last_values)
        return np.full(steps, forecast_val)
