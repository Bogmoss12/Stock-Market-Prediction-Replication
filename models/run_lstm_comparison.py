import pandas as pd
import numpy as np

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.data_preparation import prepare_model_data
from lstm_model import build_lstm_model
from utils import get_callbacks
from src.evaluation import evaluate_predictions

def inverse_scale_target(scaler, y_scaled, n_features=20):
    """
    Helper function to inverse scale the normalized predictions and true values 
    back to their original monetary scale for accurate metric evaluation.
    """
    # Create a dummy array matching the width of the scaler (features + target)
    dummy = np.zeros((len(y_scaled), n_features + 1))
    # Place the target values in the last column (index -1)
    dummy[:, -1] = y_scaled.flatten()
    # Inverse transform and extract the target column
    inversed = scaler.inverse_transform(dummy)
    return inversed[:, -1]

def run_lstm_layer_comparison():
    print("Loading processed dataset...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")
    
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"
    
    print("Preparing training and testing sequences...")
    x_train, x_test, y_train, y_test, scaler = prepare_model_data(
        df=final_df,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config
    )
    
    layer_configs = [2, 3, 4, 5]
    table_7_results = []
    
    for n in layer_configs:
        print(f"\n--- Training {n} LSTM Layers Model ---")
        
        # Build model with initial hyperparameters (256 units, 0.1 dropout)
        model = build_lstm_model(num_layers=n, units=256, dropout_rate=0.1, input_shape=(x_train.shape[1], x_train.shape[2]))
        callbacks = get_callbacks(patience_early_stop=10, patience_lr=4)
        
        # Train model using settings specified in the paper
        model.fit(
            x_train, y_train,
            epochs=50,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Predict on test set
        y_pred_scaled = model.predict(x_test)
        
        # Inverse transform predictions and actual values to real prices for evaluation
        y_pred = inverse_scale_target(scaler, y_pred_scaled, n_features=len(feature_cols))
        y_true = inverse_scale_target(scaler, y_test, n_features=len(feature_cols))
        
        # Calculate specialized regression metrics
        metrics = evaluate_predictions(y_true, y_pred)
        
        # Format MAPE nicely with a percentage sign matching Table 7 style
        model_name = f"{n} LSTM"
        table_7_results.append({
            "Model": model_name,
            "MAE": round(metrics["MAE"], 5),
            "MSE": round(metrics["MSE"], 5),
            "RMSE": round(metrics["RMSE"], 5),
            "MAPE": f"{metrics['MAPE']:.2f}%",
            "R2": round(metrics['R2'], 5)
        })
        
    # Convert results into a summary DataFrame matching Table 7 format
    summary_df = pd.DataFrame(table_7_results)
    
    print("\n==================================================")
    print("          REPLICATED TABLE 7 RESULTS             ")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================")
    
    # CSV results
    summary_df.to_csv("data/processed/table_7_replication_results.csv", index=False)

if __name__ == "__main__":
    run_lstm_layer_comparison()