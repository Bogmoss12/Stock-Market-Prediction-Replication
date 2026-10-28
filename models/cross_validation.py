import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import GRU, LSTM, Conv1D, Dense, Dropout, Input, MaxPooling1D, SimpleRNN
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.evaluation import evaluate_predictions

WINDOW_SIZE = getattr(config, "LOOKBACK_WINDOW", 100)

# Final optimized builders for all models using hardcoded best features
def build_final_lstm(n_features=20):
    # LSTM Model with 2 LSTM layers
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        LSTM(256, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_final_gru(n_features=20):
    # GRU Model with 2 GRU layers
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        GRU(192, return_sequences=True),
        Dropout(0.1),
        GRU(256, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_final_lstm_gru(n_features=20):
    # LSTM + GRU Model with 2 LSTM + 1 GRU
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        GRU(256, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_final_cnn_gru(n_features=20):
    # CNN + GRU Model with 3 CNN + 1 GRU
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=224, kernel_size=3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        GRU(256, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_final_cnn_lstm(n_features=20):
    # CNN + LSTM model with 2 CNN + 2 LSTM
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        Conv1D(filters=256, kernel_size=3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        LSTM(224, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def build_final_rnn(n_features=20):
    # RNN Model with 2 layers
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        SimpleRNN(256, return_sequences=True),
        Dropout(0.1),
        SimpleRNN(256, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_final_gru_rnn(n_features=20):
    # GRU + RNN Model with 2 GRU + 2 RNN
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        GRU(256, return_sequences=True),
        Dropout(0.1),
        GRU(224, return_sequences=True),
        Dropout(0.1),
        SimpleRNN(128, return_sequences=True),
        Dropout(0.1),
        SimpleRNN(128, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_final_lstm_rnn(n_features=20):
    # LSTM + RNN Model with 1 LSTM + 2 RNN
    model = Sequential([
        Input(shape=(WINDOW_SIZE, n_features)),
        LSTM(256, return_sequences=True),
        Dropout(0.1),
        SimpleRNN(128, return_sequences=True),
        Dropout(0.1),
        SimpleRNN(128, return_sequences=False),
        Dropout(0.1),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model


def build_final_xgboost():
    # XGBoost with parameters
    return xgb.XGBRegressor(
        n_estimators=2000,
        colsample_bytree=1,
        learning_rate=0.2,
        max_depth=3,
        gamma=0,
        objective='reg:squarederror',
        n_jobs=-1
    )

# Data scaling and sequence helpers
def create_sequences(scaled_array, window_size=100):
    # Create 3D sliding window sequences
    X, y = [], []
    for i in range(window_size, len(scaled_array)):
        X.append(scaled_array[i - window_size:i, :-1])
        y.append(scaled_array[i, -1])
    return np.array(X), np.array(y)


def inverse_scale_target(scaler, y_scaled, n_features=20):
    # Create a dummy array matching the width of the scaler (features + target)
    dummy_arr = np.zeros((len(y_scaled), n_features + 1))
    # Place the target values in the last column (index -1)
    dummy_arr[:, -1] = y_scaled.flatten()
    # Inverse transform and extract the target column
    inversed = scaler.inverse_transform(dummy_arr)
    return inversed[:, -1]


def prepare_fold_subsets(raw_data, train_idx, test_idx, window_size=100):
    # Minmax scaler to provide constant 0,1 input
    train_raw = raw_data[train_idx]
    test_raw = raw_data[test_idx]

    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))

    train_scaled = scaler_train.fit_transform(train_raw)
    test_scaled = scaler_test.fit_transform(test_raw)

    # 2D data for XGBoost
    x_train_2d, y_train_2d = train_scaled[:, :-1], train_scaled[:, -1]
    x_test_2d, y_test_2d = test_scaled[:, :-1], test_scaled[:, -1]

    # 3D data for recurrent models
    x_train_3d, y_train_3d = create_sequences(train_scaled, window_size)
    test_with_lookback = np.vstack([train_scaled[-window_size:], test_scaled])
    x_test_3d, y_test_3d = create_sequences(test_with_lookback, window_size)

    return (
        (x_train_3d, y_train_3d, x_test_3d, y_test_3d),
        (x_train_2d, y_train_2d, x_test_2d, y_test_2d),
        scaler_test
    )

# 10 fold time series cross validation engine
def run_10_fold_cross_validation():
    print("Loading processed dataset for 10-Fold Time Series Cross-Validation...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")

    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"
    n_features = len(feature_cols)

    raw_data = final_df[feature_cols + [target_col]].values

    # Initialize 10-Fold time series split
    tscv = TimeSeriesSplit(n_splits=10)

    # Models to evaluate
    models_to_evaluate = {
        "LSTM": ("dl", lambda: build_final_lstm(n_features)),
        "GRU": ("dl", lambda: build_final_gru(n_features)),
        "LSTM + GRU": ("dl", lambda: build_final_lstm_gru(n_features)),
        "CNN + GRU": ("dl", lambda: build_final_cnn_gru(n_features)),
        "CNN + LSTM": ("dl", lambda: build_final_cnn_lstm(n_features)),
        "RNN": ("dl", lambda: build_final_rnn(n_features)),
        "GRU + RNN": ("dl", lambda: build_final_gru_rnn(n_features)),
        "LSTM + RNN": ("dl", lambda: build_final_lstm_rnn(n_features)),
        "XGBoost": ("xgb", build_final_xgboost)
    }

    all_model_fold_Metrics = {name: [] for name in models_to_evaluate}

    os.makedirs("results/cross_validation", exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(raw_data), start=1):
        print(f"\n==================================================")
        print(f"       RUNNING FOLD {fold_idx}/10 (Train: {len(train_idx)} | Test: {len(test_idx)})")
        print(f"==================================================")

        dl_data, xgb_data, scaler_test = prepare_fold_subsets(
            raw_data, train_idx, test_idx, window_size=WINDOW_SIZE
        )
        x_train_3d, y_train_3d, x_test_3d, y_test_3d = dl_data
        x_train_2d, y_train_2d, x_test_2d, y_test_2d = xgb_data

        for model_name, (model_type, builder_fn) in models_to_evaluate.items():
            print(f" -> Training {model_name} on Fold {fold_idx}...")

            if model_type == "dl":
                tf.keras.backend.clear_session()
                model = builder_fn()

                callbacks = [
                    EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=0),
                    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
                ]

                model.fit(
                    x_train_3d, y_train_3d,
                    epochs=getattr(config, "EPOCHS", 50),
                    batch_size=getattr(config, "BATCH_SIZE", 32),
                    callbacks=callbacks,
                    verbose=0
                )

                y_pred_scaled = model.predict(x_test_3d, verbose=0)
                y_pred = inverse_scale_target(scaler_test, y_pred_scaled, n_features)
                y_true = inverse_scale_target(scaler_test, y_test_3d, n_features)

            else:
                model = builder_fn()
                model.fit(x_train_2d, y_train_2d, verbose=False)

                y_pred_scaled = model.predict(x_test_2d)
                y_pred = inverse_scale_target(scaler_test, y_pred_scaled, n_features)
                y_true = inverse_scale_target(scaler_test, y_test_2d, n_features)

            metrics = evaluate_predictions(y_true, y_pred)

            all_model_fold_Metrics[model_name].append({
                "Fold": f"Fold {fold_idx}",
                "MAE": metrics["MAE"],
                "MSE": metrics["MSE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
                "R2": metrics["R2"]
            })

    # Display individual table results
    avg_summary_rows = []
    std_summary_rows = []

    for model_name, fold_list in all_model_fold_Metrics.items():
        fold_df = pd.DataFrame(fold_list)

        # Save raw numeric fold results
        safe_name = model_name.lower().replace(" + ", "_").replace(" ", "_")
        fold_df.to_csv(f"results/cross_validation/{safe_name}_10_folds.csv", index=False)

        # Calculate mean and standard deviation across the 10 folds
        avg_summary_rows.append({
            "Model": model_name,
            "MAE": round(fold_df["MAE"].mean(), 5),
            "MSE": round(fold_df["MSE"].mean(), 5),
            "RMSE": round(fold_df["RMSE"].mean(), 5),
            "MAPE": f"{fold_df['MAPE'].mean():.2f}%",
            "R2": round(fold_df["R2"].mean(), 5)
        })

        std_summary_rows.append({
            "Model": model_name,
            "MAE": round(fold_df["MAE"].std(), 5),
            "MSE": round(fold_df["MSE"].std(), 5),
            "RMSE": round(fold_df["RMSE"].std(), 5),
            "MAPE": f"{fold_df['MAPE'].std():.2f}%",
            "R2": round(fold_df["R2"].std(), 5)
        })

        # Format individual fold table
        display_df = fold_df.copy()
        for col in ["MAE", "MSE", "RMSE", "R2"]:
            display_df[col] = display_df[col].round(5)
        display_df["MAPE"] = display_df["MAPE"].apply(lambda x: f"{x:.2f}%")

        print(f"\n==================================================")
        print(f"       10-FOLD RESULTS: {model_name.upper()} MODEL")
        print(f"==================================================")
        print(display_df.to_string(index=False))
        print(f"==================================================")

    # Display summary table results
    avg_df = pd.DataFrame(avg_summary_rows)
    std_df = pd.DataFrame(std_summary_rows)

    avg_df.to_csv("results/cross_validation/table30_average_metrics.csv", index=False)
    std_df.to_csv("results/cross_validation/table31_std_metrics.csv", index=False)

    print("\n==================================================================")
    print("            AVERAGE OF CALCULATED METRICS (10 FOLDS)               ")
    print("==================================================================")
    print(avg_df.to_string(index=False))
    print("==================================================================")

    print("\n==================================================================")
    print("        STANDARD DEVIATION OF CALCULATED METRICS (10 FOLDS)        ")
    print("==================================================================")
    print(std_df.to_string(index=False))
    print("==================================================================")


if __name__ == "__main__":
    run_10_fold_cross_validation()