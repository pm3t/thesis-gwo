import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def get_metrics(y_true, y_pred):
    """
    Return a dictionary of common evaluation metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def calculate_wilcoxon_test(y_true, y_pred_ensemble, y_pred_best_baseline):
    """
    Perform Wilcoxon Signed-Rank Test on absolute errors of Ensemble vs Best Baseline.
    """
    from scipy.stats import wilcoxon
    y_true = np.array(y_true)
    y_pred_ensemble = np.array(y_pred_ensemble)
    y_pred_best_baseline = np.array(y_pred_best_baseline)
    
    # Calculate absolute errors
    errors_ens = np.abs(y_true - y_pred_ensemble)
    errors_base = np.abs(y_true - y_pred_best_baseline)
    
    try:
        statistic, p_value = wilcoxon(errors_ens, errors_base)
        return statistic, p_value
    except Exception as e:
        return 0.0, 1.0
