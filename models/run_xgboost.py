import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from xgboost_model import build_xgboost_model
from src.evaluation import evaluate_predictions

def inverse_scale_target(scaler, y_scaled, n_features=20):
    """
    Helper function to inverse scale the normalized predictions and true values 
    back to their original monetary scale for accurate metric evaluation.
    """
    # Create a dummy array matching the width of the scaler (features + target)
    dummy_arr = np.zeros((len(y_scaled), n_features + 1))
    # Place the target values in the last column (index -1)
    dummy_arr[:, -1] = y_scaled.flatten()
    # Inverse transform and extract the target column
    inversed = scaler.inverse_transform(dummy_arr)
    return inversed[:, -1]

def prepare_xgboost_data(df: pd.DataFrame, feature_cols: list, target_col: str, train_ratio: float = 0.8):
    # Prepare 2D tabular datra
    data = df[feature_cols + [target_col]].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    
    split_idx = int(len(scaled_data) * train_ratio)
    
    x_train, x_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return x_train, x_test, y_train, y_test, scaler

def run_xgboost_model():
    # Check if it can read the dataset
    print("Loading processed dataset for XGBoost...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")

    # Set target column as target variable
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"
    
    print("Preparing windowless 80/20 train/test split...")
    x_train, x_test, y_train, y_test, scaler = prepare_xgboost_data(
        df=final_df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=config.TRAIN_SPLIT
    )

    # Use hardcoded parameters
    print("\n--- Training Final XGBoost Model With Parameters ---")
    xgb_model = build_xgboost_model()
    
    # Train directly on the features without feedback loop
    xgb_model.fit(x_train, y_train, verbose=False)
    
    print("Predicting across test set...")
    y_pred_scaled = xgb_model.predict(x_test)

    # Inverse transform predictions and actual values to real prices
    y_pred = inverse_scale_target(scaler, y_pred_scaled, n_features=len(feature_cols))
    y_true = inverse_scale_target(scaler, y_test, n_features=len(feature_cols))

    # Specialized refression metrics
    metrics = evaluate_predictions(y_true, y_pred)

    # Math
    xgb_results = [{
        "Model": "XGBoost",
        "MAE": round(metrics["MAE"], 5),
        "MSE": round(metrics["MSE"], 5),
        "RMSE": round(metrics["RMSE"], 5),
        "MAPE": f"{metrics['MAPE']:.2f}%",
        "R2": round(metrics['R2'], 5)
    }]
    
    summary_df = pd.DataFrame(xgb_results)
    
    print("\n==================================================")
    print("                 XGBOOST RESULTS                   ")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================")

if __name__ == "__main__":
    run_xgboost_model()