import os
import pandas as pd
import numpy as np
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Input, GRU, Dropout, Dense
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

def build_tunable_gru(hp):
    """
    Model builder function for Keras Tuner's BayesianOptimization.
    Defines the hyperparameter search space for the 2-layer GRU:
    - Units per layer: 64 to 256 (step 64)
    - Dropout rate: 0.1 to 0.5 (step 0.1)
    """
    model = Sequential()
    
    # Input layer matching the 100-timestep window with 20 features
    model.add(Input(shape=(100, 20)))
    
    # First GRU Layer & Hyperparameters
    units_1 = hp.Choice('units_1', values=[64, 128, 192, 256])
    dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)

    # units_1 = hp.Choice('units_1', values=[192])
    # dropout_1 = hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)
    
    model.add(GRU(units=units_1, return_sequences=True))
    model.add(Dropout(rate=dropout_1))
    
    # Second GRU Layer & Hyperparameters
    units_2 = hp.Choice('units_2', values=[64, 128, 192, 256])
    dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)

    # units_2 = hp.Choice('units_2', values=[256])
    # dropout_2 = hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)
    
    model.add(GRU(units=units_2, return_sequences=False))
    model.add(Dropout(rate=dropout_2))
    
    # Dense output layer for continuous stock price prediction
    model.add(Dense(units=1, activation='linear'))
    
    # Compile using paper's standard settings (Adam, lr=0.001, MSE loss)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def run_gru_bayesian_optimization():
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
        hypermodel=build_tunable_gru,
        objective='loss',           # Minimize validation Mean Squared Error
        max_trials=30,              # Total number of trial iterations to test
        executions_per_trial=1,     # Number of models built/trained per trial
        directory='keras_tuner_logs',
        project_name='gru_bayesian_optimization'
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
        verbose=1
    )
    
    # Retrieve the optimal hyperparameters found during the search
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n==================================================")
    print("      OPTIMIZED HYPERPARAMETERS FOUND (TABLE 10)  ")
    print("==================================================")
    print(f"Layer 1 - Units: {best_hps.get('units_1')} | Dropout: {best_hps.get('dropout_1'):.2f}")
    print(f"Layer 2 - Units: {best_hps.get('units_2')} | Dropout: {best_hps.get('dropout_2'):.2f}")
    print("==================================================")
    
    # Build and summarize the final optimized model matching Table 10 specification
    print("\nFinal Optimized Model Architecture (Table 10 Recreation):")
    final_model = tuner.hypermodel.build(best_hps)
    print(final_model.summary())
    
    # Save the best model weights
    os.makedirs("models/saved_models", exist_ok=True)
    final_model.save("models/saved_models/optimized_gru_model.keras")
    print("\nSaved optimized GRU model to 'models/saved_models/optimized_gru_model.keras'")

if __name__ == "__main__":
    run_gru_bayesian_optimization()