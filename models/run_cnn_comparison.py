import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, LSTM, GRU, Flatten, Dropout, Dense

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.data_preparation import prepare_model_data
from utils import get_callbacks, compile_model
from src.evaluation import evaluate_predictions

def build_table18_model(model_config: str, input_shape=(100, 20)):
    """
    Unified builder function to construct any pure CNN or hybrid architecture 
    specified in Table 18 of the reference paper.
    """
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    # Parse the configuration name to build the correct architecture layout
    if model_config == "2 CNN":
        for _ in range(2):
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        
    elif model_config == "3 CNN":
        for i in range(3):
            f = 64 * (2 ** i)
            model.add(Conv1D(filters=f, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        
    elif model_config == "4 CNN":
        for i in range(4):
            f = 64 * (2 ** i if i < 2 else 2)
            model.add(Conv1D(filters=f, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))

    # --- CNN + Sequence Hybrids ---
    elif "CNN +" in model_config:
        parts = model_config.split(" + ")
        cnn_count = int(parts[0].split()[0])
        seq_count = int(parts[1].split()[0])
        seq_type = parts[1].split()[1].lower() # 'gru' or 'lstm'
        
        # CNN blocks first
        for i in range(cnn_count):
            model.add(Conv1D(filters=64 * (2 ** i), kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.1))
            
        # Sequence blocks after
        for j in range(seq_count):
            return_seq = (j < seq_count - 1)
            if seq_type == 'lstm':
                model.add(LSTM(units=128, return_sequences=return_seq))
            else:
                model.add(GRU(units=128, return_sequences=return_seq))
            model.add(Dropout(0.1))

    # --- Sequence + CNN Hybrids (e.g., 2 LSTM + 2 CNN) ---
    elif "+ CNN" in model_config:
        parts = model_config.split(" + ")
        seq_count = int(parts[0].split()[0])
        seq_type = parts[0].split()[1].lower() # 'lstm' or 'gru'
        cnn_count = int(parts[1].split()[0])
        
        # Sequence blocks first (must return sequences to feed Conv1D)
        for j in range(seq_count):
            if seq_type == 'lstm':
                model.add(LSTM(units=128, return_sequences=True))
            else:
                model.add(GRU(units=128, return_sequences=True))
            model.add(Dropout(0.1))
            
        # CNN blocks after
        for i in range(cnn_count):
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(0.1))
            
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))

    # Final output regression layer
    model.add(Dense(units=1, activation='linear'))
    return compile_model(model, learning_rate=0.001)

def inverse_scale_target(scaler, y_scaled, y_target_shape, n_features=20):
    """
    Safely inverse scale normalized predictions back to original monetary scale,
    ensuring shape alignment with y_test.
    """
    # Flatten just in case, but trim or reshape to match y_target_shape if necessary
    flat_preds = y_scaled.flatten()
    
    # If prediction length somehow expanded to match timesteps, take the last step of each window
    if len(flat_preds) != y_target_shape:
        # Reshape assuming windows of length 100 or match y_target_shape directly
        flat_preds = flat_preds[-y_target_shape:]
        
    dummy = np.zeros((y_target_shape, n_features + 1))
    dummy[:, -1] = flat_preds
    return scaler.inverse_transform(dummy)[:, -1]

def run_full_table18_benchmark():
    print("Loading processed dataset...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    
    x_train, x_test, y_train, y_test, scaler = prepare_model_data(
        df=final_df, feature_cols=feature_cols, target_col="Target", config=config
    )
    
    # Complete list of all models matching Table 18 precisely
    table_18_models = [
        "2 CNN", "3 CNN", "4 CNN",
        "2 CNN + 1 GRU", "2 CNN + 2 GRU",
        "2 CNN + 1 LSTM", "2 CNN + 2 LSTM",
        "3 CNN + 1 GRU", "3 CNN + 2 GRU",
        "3 CNN + 1 LSTM", "3 CNN + 2 LSTM", "3 CNN + 3 LSTM",
        "2 LSTM + 2 CNN", "2 LSTM + 3 CNN",
        "1 LSTM + 2 CNN", "1 LSTM + 3 CNN",
        "2 GRU + 2 CNN", "2 GRU + 3 CNN",
        "1 GRU + 2 CNN", "1 GRU + 3 CNN"
    ]
    
    results = []
    
    for name in table_18_models:
        print(f"\n==================================================")
        print(f"Training Table 18 Architecture: {name}")
        print(f"==================================================")
        
        model = build_table18_model(model_config=name, input_shape=(x_train.shape[1], x_train.shape[2]))
        callbacks = get_callbacks(patience_early_stop=8, patience_lr=3)
        
        model.fit(
            x_train, y_train,
            epochs=35,
            batch_size=32,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )
        
        y_pred_scaled = model.predict(x_test)
        
        # Pass y_test length to ensure exact row alignment
        y_pred = inverse_scale_target(scaler, y_pred_scaled, y_test.shape[0], len(feature_cols))
        y_true = inverse_scale_target(scaler, y_test, y_test.shape[0], len(feature_cols))
        
        metrics = evaluate_predictions(y_true, y_pred)
        
        results.append({
            "Model": name,
            "MAE": round(metrics["MAE"], 5),
            "MSE": round(metrics["MSE"], 5),
            "RMSE": round(metrics["RMSE"], 5),
            "MAPE": f"{metrics['MAPE']:.2f}%",
            "R2": round(metrics["R2"], 5)
        })
        
    summary_df = pd.DataFrame(results)
    print("\n\n==================================================")
    print("      COMPLETE TABLE 18 BENCHMARK COMPARISON      ")
    print("==================================================")
    print(summary_df.to_string(index=False))
    print("==================================================")
    
    summary_df.to_csv("data/processed/table_18_full_comparison.csv", index=False)

if __name__ == "__main__":
    run_full_table18_benchmark()