import numpy as np
import pandas as pd


class ExponentialSmoothingModel:
    def __init__(self, alpha=0.3, beta=0.02, gamma=0.3, period=7):
        """
        Triple Exponential Smoothing (Holt-Winters) with Walk-Forward Update.

        Parameters
        ----------
        alpha : float (0-1)
            Smoothing factor for level (default 0.3).
        beta : float (0-1)
            Smoothing factor for trend (default 0.02).
        gamma : float (0-1)
            Smoothing factor for seasonality (default 0.3).
        period : int
            Seasonal period length in timesteps (default 7 for weekly).
        """
        self.alpha  = alpha
        self.beta   = beta
        self.gamma  = gamma
        self.period = period

        self._l = None
        self._b = None
        self._s = None
        self._y_train = None

    def fit(self, y):
        """
        Fit Holt-Winters additive level, trend, and seasonal components.
        """
        y_arr = np.array(y, dtype=float)
        self._y_train = y_arr
        p = self.period

        if len(y_arr) < 2 * p:
            # Fallback initialization for very short series
            self._l = np.mean(y_arr)
            self._b = 0.0
            self._s = [0.0] * p
            return self

        # Initialise level, trend, and seasonal components
        l = np.mean(y_arr[:p])
        b = (np.mean(y_arr[p:2*p]) - np.mean(y_arr[:p])) / p
        s = [y_arr[i] - l for i in range(p)]

        # Warm-up / fit across training series
        for t in range(len(y_arr)):
            val = y_arr[t]
            s_idx = t % p
            l_prev, b_prev = l, b

            # Level update
            l = self.alpha * (val - s[s_idx]) + (1 - self.alpha) * (l_prev + b_prev)
            # Trend update
            b = self.beta * (l - l_prev) + (1 - self.beta) * b_prev
            # Seasonal update
            s[s_idx] = self.gamma * (val - l_prev - b_prev) + (1 - self.gamma) * s[s_idx]

        self._l = l
        self._b = b
        self._s = s
        return self

    def forecast(self, steps=1):
        """
        Multi-step static forecast starting from the end of training data.
        """
        if self._l is None:
            raise ValueError("Model must be fitted first.")

        p = self.period
        n_train = len(self._y_train)
        preds = []

        for m in range(1, steps + 1):
            s_idx = (n_train + m - 1) % p
            pred = self._l + m * self._b + self._s[s_idx]
            preds.append(pred)

        return np.array(preds)

    def walk_forward(self, y_test):
        """
        Walk-Forward 1-step forecast for test period using actual observations.

        Parameters
        ----------
        y_test : array-like
            True target values for the test period.

        Returns
        -------
        np.ndarray of 1-step-ahead walk-forward predictions.
        """
        if self._l is None:
            raise ValueError("Model must be fitted first.")

        y_test_arr = np.array(y_test, dtype=float)
        p = self.period
        n_train = len(self._y_train)

        l = self._l
        b = self._b
        s = list(self._s)
        preds = []

        for t, true_val in enumerate(y_test_arr):
            s_idx = (n_train + t) % p
            # 1-step ahead prediction
            pred = l + b + s[s_idx]
            preds.append(pred)

            # Walk-forward update with TRUE observation
            l_prev, b_prev = l, b
            l = self.alpha * (true_val - s[s_idx]) + (1 - self.alpha) * (l_prev + b_prev)
            b = self.beta * (l - l_prev) + (1 - self.beta) * b_prev
            s[s_idx] = self.gamma * (true_val - l_prev - b_prev) + (1 - self.gamma) * s[s_idx]

        return np.array(preds)
