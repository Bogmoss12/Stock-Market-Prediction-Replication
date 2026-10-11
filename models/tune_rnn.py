import os
import pandas as pd
import numpy as np
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Input, SimpleRNN, Dropout, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.data_preparation import prepare_model_data

def build_tunable_rnn(hp):
    """
    Model builder function for Keras Tuner's BayesianOptimization.
    Defines the hyperparameter search space for the 3-layer SimpleRNN:
    - Units per layer: 64 to 256 (stepping by 64)
    - Dropout rate: 0.1 to 0.5 (stepping by 0.1)
    """
    model = Sequential()
    
    # Input layer matching the 100-timestep window with 20 features
    model.add(Input(shape=(100, 20)))
    
    # Define hyperparameters and add 3 SimpleRNN layers iteratively
    num_layers = 3
    for i in range(num_layers):
        units = hp.Choice(f'units_{i+1}', values=[64, 160, 256])
        dropout_rate = hp.Float(f'dropout_{i+1}', min_value=0.1, max_value=0.5, step=0.1)
        
        # All layers except the final one must return sequences
        return_seq = (i < num_layers - 1)
        
        model.add(SimpleRNN(units=units, return_sequences=return_seq))
        model.add(Dropout(rate=dropout_rate))
    
    # Dense output layer for continuous stock price prediction
    model.add(Dense(units=1, activation='linear'))
    
    # Compile using standard settings (Adam optimizer, lr=0.001, MSE loss)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def run_rnn_bayesian_optimization():
    print("Loading processed dataset for RNN hyperparameter tuning...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")
    
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"
    
    # Prepare training and validation subsets using your corrected pipeline
    x_train, x_test, y_train, y_test, scaler = prepare_model_data(
        df=final_df,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config
    )
    
    print("Initializing Bayesian Optimization tuner for 3-Layer SimpleRNN...")
    tuner = kt.BayesianOptimization(
        hypermodel=build_tunable_rnn,
        objective='loss',           # Minimize validation Mean Squared Error
        max_trials=30,                  # Total trial budget for searching 3 layers
        executions_per_trial=1,         
        directory='keras_tuner_logs',
        project_name='rnn_bayesian_optimization'
    )
    
    # Setup callbacks for tuning
    tuner_callbacks = [
        EarlyStopping(monitor='loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6, verbose=0)
    ]
    
    print("\nStarting Bayesian Optimization search on 3-Layer SimpleRNN...")
    tuner.search(
        x_train, y_train,
        epochs=30,
        batch_size=32,
        callbacks=tuner_callbacks,
        verbose=1
    )
    
    # Retrieve optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n==================================================")
    print("    OPTIMIZED RNN HYPERPARAMETERS (TABLE CONFIG)   ")
    print("==================================================")
    for i in range(3):
        print(f"Layer {i+1} - Units: {best_hps.get(f'units_{i+1}')} | Dropout: {best_hps.get(f'dropout_{i+1}'):.2f}")
    print("==================================================")
    
    # Build and display final optimized model summary
    print("\nFinal Optimized 3-Layer SimpleRNN Architecture Summary:")
    final_model = tuner.hypermodel.build(best_hps)
    print(final_model.summary())
    
    # Save the best model weights
    os.makedirs("models/saved_models", exist_ok=True)
    final_model.save("models/saved_models/optimized_rnn_model.keras")
    print("\nSaved optimized 3-layer RNN model to 'models/saved_models/optimized_rnn_model.keras'")

if __name__ == "__main__":
    run_rnn_bayesian_optimization()