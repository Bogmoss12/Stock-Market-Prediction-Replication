import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config

def prepare_xgboost_tuning_data(df: pd.DataFrame, feature_cols: list, target_col: str, train_ratio: float = 0.8):
    # Prepare 80-20 training and testing data
    data = df[feature_cols + [target_col]].values
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    
    split_idx = int(len(scaled_data) * train_ratio)
    
    x_train, x_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return x_train, x_test, y_train, y_test, scaler

def run_xgboost_grid_search():
    print("Loading processed dataset for XGBoost tuning...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")
    
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"
    
    x_train, x_test, y_train, y_test, scaler = prepare_xgboost_tuning_data(
        df=final_df,
        feature_cols=feature_cols,
        target_col=target_col,
        train_ratio=config.TRAIN_SPLIT
    )
    
    print("Initializing XGBoost base regressor...")
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1
    )
    
    # Define the parameters
    param_grid = {
        'n_estimators': [100, 500, 1000, 1500, 2000],
        'colsample_bytree': [0.3, 0.5, 0.7, 0.8, 1.0],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5],
        'gamma': [0, 0.1]
    }

    # Apply GridSearchCV
    print("Setting up GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=5,
        verbose=1,
        n_jobs=-1
    )
    
    print("\nRunning GridSearchCV search...")
    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_

    ordered_params = [
        'n_estimators', 
        'colsample_bytree', 
        'learning_rate', 
        'max_depth', 
        'gamma'
    ]
    
    print("\n==================================================")
    print("     OPTIMIZED HYPERPARAMETERS FOUND  (XGBoost)    ")
    print("==================================================")
    for param in ordered_params:
        value = best_params.get(param, "N/A")
        print(f"Parameter: {param:<18} | Value: {value}")
    print("==================================================")
    
    # Save the best model weights
    best_model = grid_search.best_estimator_
    os.makedirs("models/saved_models", exist_ok=True)
    best_model.save_model("models/saved_models/optimized_xgboost_table20.json")
    print("\nSaved optimized XGBoost model to 'models/saved_models/optimized_xgboost_table20.json'")

if __name__ == "__main__":
    run_xgboost_grid_search()