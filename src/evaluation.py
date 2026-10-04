import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    # Math
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE" : mse,
        "RMSE" : rmse,
        "MAPE" : mape,
        "R2" : r2
    }