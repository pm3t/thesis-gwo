import numpy as np

class WeightedEnsembleModel:
    def __init__(self):
        self.weights = None

    def set_weights(self, weights):
        """
        Set weights manually or from GWO.
        weights: array-like [w_ma, w_es, w_lr]
        """
        self.weights = np.array(weights)
        # Ensure weights sum to 1
        if np.sum(self.weights) != 0:
            self.weights = self.weights / np.sum(self.weights)

    def predict(self, predictions_ma, predictions_es, predictions_lr):
        """
        Calculate weighted ensemble prediction.
        """
        if self.weights is None:
            raise ValueError("Weights must be set before prediction.")
        
        return (self.weights[0] * np.array(predictions_ma) + 
                self.weights[1] * np.array(predictions_es) + 
                self.weights[2] * np.array(predictions_lr))
