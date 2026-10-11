import pandas as pd
import numpy as np
import random
import tensorflow as tf

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.data_preparation import prepare_model_data
from rnn_lstm_model import build_rnn_lstm_model
from rnn_gru_model import build_rnn_gru_model
from utils import get_callbacks
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

def run_rnn_hybrid_comparison():
    # Check if it can read the dataset
    print("Loading processed dataset...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")

    # Set target column as target variable
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"
    
    print("Preparing training and testing sequences...")
    x_train, x_test, y_train, y_test, scaler = prepare_model_data(
        df=final_df,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config
    )
    
    # Combinations configs
    hybrid_configs = [
        ('rnn_gru',  2, 1, 'rnn_first', "2 RNN + 1 GRU"),
        ('rnn_lstm', 2, 1, 'rnn_first', "2 RNN + 1 LSTM"),
        ('rnn_gru',  3, 1, 'rnn_first', "3 RNN + 1 GRU"),
        ('rnn_gru',  3, 2, 'rnn_first', "3 RNN + 2 GRU"),
        ('rnn_lstm', 3, 1, 'rnn_first', "3 RNN + 1 LSTM"),
        ('rnn_lstm', 3, 2, 'rnn_first', "3 RNN + 2 LSTM"),
        ('rnn_gru',  2, 1, 'gru_first', "1 GRU + 2 RNN"),
        ('rnn_gru',  2, 2, 'gru_first', "2 GRU + 2 RNN"),
        ('rnn_lstm', 2, 1, 'lstm_first', "1 LSTM + 2 RNN"),
        ('rnn_lstm', 2, 2, 'lstm_first', "2 LSTM + 2 RNN")
    ]
    table_16_results = []
    
    for mod_type, r_layers, o_layers, order, model_name in hybrid_configs:
        print(f"\n--- Training {model_name} Model ---")
        
        # Route to correct hybrid combination
        if mod_type == 'rnn_gru':
            model = build_rnn_gru_model(
                rnn_layers=r_layers, 
                gru_layers=o_layers, 
                stack_order=order,
                units=config.GRU_UNITS,
                dropout_rate=config.DROPOUT_RATE,
                input_shape=(x_train.shape[1], x_train.shape[2])
            )
        elif mod_type == 'rnn_lstm':
            model = build_rnn_lstm_model(
                rnn_layers=r_layers, 
                lstm_layers=o_layers, 
                stack_order=order,
                units=256,
                dropout_rate=config.DROPOUT_RATE,
                input_shape=(x_train.shape[1], x_train.shape[2])
            )
        callbacks = get_callbacks(patience_early_stop=10, patience_lr=4)

        # Train using specifications in paper
        model.fit(
            x_train, y_train,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Predictions on test set
        y_pred_scaled = model.predict(x_test)

        # Inverse transform predictions and actual values to real prices
        y_pred = inverse_scale_target(scaler, y_pred_scaled, n_features=len(feature_cols))
        y_true = inverse_scale_target(scaler, y_test, n_features=len(feature_cols))

        # Specialized regression metrics
        metrics = evaluate_predictions(y_true, y_pred)

        # Math
        table_16_results.append({
            "Model": model_name,
            "MAE": round(metrics["MAE"], 5),
            "MSE": round(metrics["MSE"], 5),
            "RMSE": round(metrics["RMSE"], 5),
            "MAPE": f"{metrics['MAPE']:.2f}%",
            "R2": round(metrics['R2'], 5)
        })
        
    summary_df = pd.DataFrame(table_16_results)

    # Convert results into a summary DataFrame
    print("\n==================================================")
    print("      REPLICATED TABLE 16 (RNN HYBRIDS) RESULTS   ")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================")

    # CSV results
    summary_df.to_csv("data/processed/table_16_replication_results.csv", index=False)

if __name__ == "__main__":
    run_rnn_hybrid_comparison()