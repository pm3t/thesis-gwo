import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # Force CPU only

import numpy as np
from sklearn.preprocessing import MinMaxScaler


class SimpleRNNModel:
    """
    Simple RNN untuk time series forecasting.
    Arsitektur: Input → SimpleRNN(32) → Dropout(0.2) → Dense(1)
    Menggunakan MinMaxScaler untuk normalisasi.
    """

    def __init__(self, lookback: int = 7, units: int = 32,
                 dropout: float = 0.2, epochs: int = 50,
                 batch_size: int = 16, random_state: int = 42):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self._last_sequence = None  # window terakhir untuk forecasting

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────
    def _build_model(self):
        """Lazy import Keras supaya tidak wajib install TF saat import modul."""
        import tensorflow as tf
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)

        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.lookback, 1)),
            tf.keras.layers.SimpleRNN(self.units, activation='tanh'),
            tf.keras.layers.Dropout(self.dropout),
            tf.keras.layers.Dense(1),
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def _make_sequences(self, data: np.ndarray):
        """Buat pasangan (X, y) dari data 1D menggunakan sliding window."""
        X, y = [], []
        for i in range(len(data) - self.lookback):
            X.append(data[i: i + self.lookback])
            y.append(data[i + self.lookback])
        return np.array(X), np.array(y)

    # ─────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────
    def fit(self, y_train):
        """
        Normalisasi, buat sequences, dan latih model.

        Parameters
        ----------
        y_train : array-like
            Data latih (1D).
        """
        y = np.array(y_train, dtype=float).reshape(-1, 1)
        y_scaled = self.scaler.fit_transform(y).flatten()

        X, Y = self._make_sequences(y_scaled)
        # Reshape ke (samples, timesteps, features)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        self.model = self._build_model()
        self.model.fit(
            X, Y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
            shuffle=False,
        )

        # Simpan window terakhir untuk forecasting iteratif
        self._last_sequence = y_scaled[-self.lookback:].copy()
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """
        Prediksi `steps` langkah ke depan secara iteratif.

        Parameters
        ----------
        steps : int
            Jumlah titik yang ingin diprediksi.

        Returns
        -------
        np.ndarray
            Prediksi dalam skala asli.
        """
        if self.model is None:
            raise RuntimeError("Model belum dilatih. Panggil fit() terlebih dahulu.")

        current_seq = self._last_sequence.copy()
        predictions_scaled = []

        for _ in range(steps):
            x_input = current_seq.reshape(1, self.lookback, 1)
            pred = self.model.predict(x_input, verbose=0)[0, 0]
            predictions_scaled.append(pred)
            # Geser window
            current_seq = np.append(current_seq[1:], pred)

        preds = np.array(predictions_scaled).reshape(-1, 1)
        return self.scaler.inverse_transform(preds).flatten()
