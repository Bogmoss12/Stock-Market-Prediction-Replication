import os
import pandas as pd
import numpy as np
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, GRU, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.data_preparation import prepare_model_data

import random
import tensorflow as tf

def build_tunable_gru_rnn_layer(hp):
    """
    Model builder function for Keras Tuner's BayesianOptimization.
    Defines the hyperparameter search space for the RNN GRU layers:
    - Units per layer: 64 to 256 (step 64)
    - Dropout rate: 0.1 to 0.5 (step 0.1)
    """
    model = Sequential()

    # Input layer matching the 100-timestep window with 20 features
    model.add(Input(shape=(100, 20)))
    
    # First GRU Layer
    units_1 = hp.Int('units_1', min_value=64, max_value=256, step=32)
    dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    model.add(GRU(units=units_1, return_sequences=True))
    model.add(Dropout(rate=dropout_1))
    
    # Second GRU Layer
    units_2 = hp.Int('units_2', min_value=64, max_value=256, step=32)
    dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    model.add(GRU(units=units_2, return_sequences=True))
    model.add(Dropout(rate=dropout_2))
    
    # First SimpleRNN Layer
    units_3 = hp.Int('units_3', min_value=64, max_value=256, step=32)
    dropout_3 = hp.Float('dropout_3', min_value=0.1, max_value=0.5, step=0.1)
    model.add(SimpleRNN(units=units_3, return_sequences=True))
    model.add(Dropout(rate=dropout_3))
    
    # Second SimpleRNN Layer
    units_4 = hp.Int('units_4', min_value=64, max_value=256, step=32)
    dropout_4 = hp.Float('dropout_4', min_value=0.1, max_value=0.5, step=0.1)
    model.add(SimpleRNN(units=units_4, return_sequences=False))
    model.add(Dropout(rate=dropout_4))

    # Dense output layer for continuous stock price prediction
    model.add(Dense(units=1, activation='linear'))

    # Compile using paper's standard settings (Adam, lr=0.001, MSE loss)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

def run_gru_rnn_bayesian_optimization():
    print("Loading processed dataset for hyperparameter tuning...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")

    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"

    # Prepare training subsets
    x_train, x_test, y_train, y_test, scaler = prepare_model_data(
        df=final_df, 
        feature_cols=feature_cols, 
        target_col=target_col, 
        config=config
    )

    print("Initializing Bayesian Optimization tuner...")
    tuner = kt.BayesianOptimization(
        hypermodel=build_tunable_gru_rnn_layer,
        objective='loss',
        max_trials=30,
        executions_per_trial=1,
        directory='keras_tuner_logs',
        project_name='gru_rnn_4_layer_bayesian'
    )

    # Setup callbacks for the tuning search process
    tuner_callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
    ]

    print("\nStarting Bayesian Optimization search...")
    tuner.search(
        x_train, y_train, 
        epochs=30, 
        batch_size=32, 
        callbacks=tuner_callbacks, 
        verbose=1)
    
    # Retrieve the optimal hyperparameters found during the search
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n==================================================")
    print("      OPTIMIZED HYPERPARAMETERS FOUND (TABLE 18)  ")
    print("==================================================")
    print(f"Layer 1 (GRU) - Units: {best_hps.get('units_1')} | Dropout: {best_hps.get('dropout_1'):.2f}")
    print(f"Layer 2 (GRU) - Units: {best_hps.get('units_2')} | Dropout: {best_hps.get('dropout_2'):.2f}")
    print(f"Layer 3 (RNN) - Units: {best_hps.get('units_3')} | Dropout: {best_hps.get('dropout_3'):.2f}")
    print(f"Layer 4 (RNN) - Units: {best_hps.get('units_4')} | Dropout: {best_hps.get('dropout_4'):.2f}")
    print("==================================================")
    
    # Build and summarize the final optimized model
    print("\nFinal Optimized Model Architecture (Table 18 Recreation):")
    final_model = tuner.hypermodel.build(best_hps)
    final_model.summary()
    
    # Save the best model weights
    os.makedirs("models/saved_models", exist_ok=True)
    final_model.save("models/saved_models/optimized_gru_rnn_table18.keras")
    print("\nSaved optimized 2 GRU + 2 RNN model to 'models/saved_models/optimized_gru_rnn_table18.keras'")

if __name__ == "__main__":
    run_gru_rnn_bayesian_optimization()