from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pandas as pd
import numpy as np

class ExponentialSmoothingModel:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.model = None
        self.fitted_model = None

    def fit(self, y):
        """
        Fit Holt-Winters Exponential Smoothing model.
        Supports trend and seasonality (default weekly: 7 days).
        """
        # Triple Exponential Smoothing (Holt-Winters)
        # Assuming daily data with weekly seasonality (period=7)
        self.model = ExponentialSmoothing(
            y, 
            trend='add', 
            seasonal='add', 
            seasonal_periods=7,
            initialization_method="estimated"
        )
        
        # If alpha is provided, we use it for smoothing_level (alpha)
        # Others (beta, gamma) are optimized automatically for better fit
        self.fitted_model = self.model.fit(
            smoothing_level=self.alpha,
            optimized=True # Allow optimization of other parameters
        )
        return self

    def predict(self, y=None):
        """
        Predict (fitted values) for the training period.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")
        
        return self.fitted_model.fittedvalues

    def forecast(self, steps=1):
        """
        Forecast future values.
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first.")
        
        return self.fitted_model.forecast(steps)
