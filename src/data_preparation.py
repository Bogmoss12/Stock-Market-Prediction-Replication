import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def scale_dataset(df: pd.DataFrame, feature_cols: list, target_col: str):
    # Scale data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit scaler on features and target
    data_to_scale = df[feature_cols + [target_col]].values
    scaled_data = scaler.fit_transform(data_to_scale)

    scaled_df = pd.DataFrame(scaled_data, columns=feature_cols + [target_col])
    
    # Exclude date in scaling
    if "Date" in df.columns:
        scaled_df.insert(0, "Date", df["Date"].values)
    
    scaled_df.to_csv("data/processed/final_dataset_scaled.csv", index=False)

    return scaled_data, scaler

def create_sequences(scaled_data: np.ndarray, window_size: int):
    # For the sliding window
    x, y = [], []

    # To target the last column in scaled data with 20 features (21 columns incl target var)
    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size:i, :-1])
        y.append(scaled_data[i, -1])

    return np.array(x), np.array(y)

def split_train_test(x: np.ndarray, y: np.ndarray, train_ratio: float):
    # Splits into training and testing datasets
    split_index = int(len(x) * train_ratio)

    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    return x_train, x_test, y_train, y_test

def prepare_model_data(df: pd.DataFrame, feature_cols: list, target_col: str, config):
    # Data to be inputted in models
    scaled_data, scaler = scale_dataset(df, feature_cols, target_col)
    x, y = create_sequences(scaled_data, config.WINDOW_SIZE)
    x_train, x_test, y_train, y_test = split_train_test(x, y, config.TRAIN_SPLIT)

    return x_train, x_test, y_train, y_test, scaler