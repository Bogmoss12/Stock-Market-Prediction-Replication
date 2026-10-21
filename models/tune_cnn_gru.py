import argparse
import json
import os
import sys
from pathlib import Path

import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Conv1D, Dense, Dropout, GRU, Input, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import config
from src.data_preparation import prepare_model_data

# ---------------------------------------------------------------------------
# Architecture: 3 CNN blocks -> 1 GRU -> Dense(1, linear)
# Each CNN block: Conv1D(kernel 3, same padding, ReLU) -> MaxPool(2)
# No dropout after the CNN blocks (paper's Table 14 has none).
# Paper's CNN design: kernel 3, padding 'same', pool size 2 (see Section 3 / Table 17-18).
#
# Search space (ASSUMPTIONS where the paper is silent):
#   - filters per CNN block: 64-256, step 32 (contains 128, 224, 256 seen in the paper's CNN+GRU)
#   - GRU units: 64-256, step 32 (same step as the LSTM tuner)
#   - dropout after GRU: 0.1-0.5, step 0.1 (paper's search range)
# Objective: training loss (MSE), as in the paper; no validation split.
# ---------------------------------------------------------------------------
CNN_BLOCKS = 3
KERNEL_SIZE = 3
POOL_SIZE = 2

FILTER_MIN, FILTER_MAX, FILTER_STEP = 64, 256, 32
UNIT_MIN, UNIT_MAX, UNIT_STEP = 64, 256, 32
DROPOUT_MIN, DROPOUT_MAX, DROPOUT_STEP = 0.1, 0.5, 0.1


def make_hypermodel(input_shape):
    def build_tunable_cnn_gru(hp):
        model = Sequential()
        model.add(Input(shape=input_shape))

        # 1. CNN feature-extraction blocks
        for i in range(CNN_BLOCKS):
            filters = hp.Int(f"filters_{i + 1}", min_value=FILTER_MIN, max_value=FILTER_MAX, step=FILTER_STEP)
            model.add(Conv1D(filters=filters, kernel_size=KERNEL_SIZE, activation="relu", padding="same"))
            model.add(MaxPooling1D(pool_size=POOL_SIZE))

        # 2. Single GRU layer. It is the last recurrent layer, so it returns only its final state.
        gru_units = hp.Int("units_gru", min_value=UNIT_MIN, max_value=UNIT_MAX, step=UNIT_STEP)
        gru_dropout = round(hp.Float("gru_dropout", min_value=DROPOUT_MIN,
                                     max_value=DROPOUT_MAX, step=DROPOUT_STEP), 2)
        model.add(GRU(units=gru_units, return_sequences=False))
        model.add(Dropout(rate=gru_dropout))

        # 3. Output layer
        model.add(Dense(units=1, activation="linear"))

        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    return build_tunable_cnn_gru


def best_hyperparameters(best_hp) -> dict:
    filters = [best_hp.get(f"filters_{i + 1}") for i in range(CNN_BLOCKS)]
    return {
        "cnn_filters": filters,
        "gru_units": best_hp.get("units_gru"),
        "gru_dropout": round(best_hp.get("gru_dropout"), 2),
    }


def print_architecture(model) -> pd.DataFrame:
    """Print the built model in the same layout as the paper's Table 17 (layer, units/filters, dropout, activation, shapes)."""
    rows = []
    for layer in model.layers:
        kind = layer.__class__.__name__
        if kind == "Conv1D":
            rows.append({"Layer Type": kind, "Units/Filters": layer.filters, "Dropout Rate": "-",
                         "Activation": layer.activation.__name__, "Input Shape": str(tuple(layer.input.shape)),
                         "Output Shape": str(tuple(layer.output.shape))})
        elif kind == "MaxPooling1D":
            rows.append({"Layer Type": kind, "Units/Filters": "-", "Dropout Rate": "-", "Activation": "-",
                         "Input Shape": str(tuple(layer.input.shape)),
                         "Output Shape": str(tuple(layer.output.shape))})
        elif kind == "Dropout":
            rows.append({"Layer Type": kind, "Units/Filters": "-", "Dropout Rate": round(layer.rate, 2),
                         "Activation": "-", "Input Shape": str(tuple(layer.input.shape)),
                         "Output Shape": str(tuple(layer.output.shape))})
        elif kind in ("GRU", "Dense"):
            rows.append({"Layer Type": kind, "Units/Filters": layer.units, "Dropout Rate": "-",
                         "Activation": layer.activation.__name__, "Input Shape": str(tuple(layer.input.shape)),
                         "Output Shape": str(tuple(layer.output.shape))})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    return table


def run_cnn_gru_tuning(max_trials: int, epochs: int, batch_size: int, seed: int, evaluate_final: bool):
    print("Loading processed dataset for hyperparameter tuning...")
    final_df = pd.read_csv("data/processed/final_dataset.csv")
    feature_cols = [col for col in final_df.columns if col not in ["Date", "Target"]]
    target_col = "Target"

    x_train, x_test, y_train, y_test, scaler = prepare_model_data(
        df=final_df,
        feature_cols=feature_cols,
        target_col=target_col,
        config=config
    )
    input_shape = (x_train.shape[1], x_train.shape[2])

    keras.utils.set_random_seed(seed)

    print("Initializing Bayesian Optimization tuner for 3 CNN + 1 GRU...")
    tuner = kt.BayesianOptimization(
        hypermodel=make_hypermodel(input_shape),
        objective="loss",              # training MSE, as in the paper
        max_trials=max_trials,
        seed=seed,
        directory="keras_tuner_logs",
        project_name="cnn3_gru1_bayesian_optimization",
        overwrite=True,
    )

    tuner_callbacks = [
        EarlyStopping(monitor="loss", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
    ]

    print(f"\nStarting Bayesian Optimization search ({max_trials} trials)...")
    tuner.search(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=tuner_callbacks,
        verbose=1,
    )

    # Record every trial
    trial_rows = []
    for trial in tuner.oracle.trials.values():
        row = dict(trial.hyperparameters.values)
        row["train_loss_mse"] = trial.score
        trial_rows.append(row)
    trials_df = pd.DataFrame(trial_rows).sort_values("train_loss_mse").reset_index(drop=True)
    os.makedirs("data/processed", exist_ok=True)
    trials_df.to_csv("data/processed/cnn3_gru1_tuning_trials.csv", index=False)

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    best = best_hyperparameters(best_hp)

    print("\n==================================================")
    print("      OPTIMIZED HYPERPARAMETERS (3 CNN + 1 GRU)   ")
    print("==================================================")
    for i in range(CNN_BLOCKS):
        print(f"CNN block {i + 1} - Filters: {best['cnn_filters'][i]}")
    print(f"GRU             - Units:   {best['gru_units']} | Dropout: {best['gru_dropout']:.2f}")
    print("==================================================")

    with open("data/processed/cnn3_gru1_best_hyperparameters.json", "w") as f:
        json.dump(best, f, indent=2)

    # Rebuild the best model and print its architecture
    print("\nFinal optimized 3 CNN + 1 GRU architecture:")
    final_model = tuner.hypermodel.build(best_hp)
    final_model.summary()
    table = print_architecture(final_model)
    table.to_csv("data/processed/cnn3_gru1_architecture.csv", index=False)

    os.makedirs("models/saved_models", exist_ok=True)
    final_model.save("models/saved_models/optimized_cnn3_gru1.keras")
    print("\nSaved optimized model to 'models/saved_models/optimized_cnn3_gru1.keras'")

    if evaluate_final:
        from src.evaluation import evaluate_predictions

        def inverse_scale_target(y_scaled):
            dummy = np.zeros((len(y_scaled), len(feature_cols) + 1))
            dummy[:, -1] = y_scaled.flatten()
            return scaler.inverse_transform(dummy)[:, -1]

        print("\n--- Training final tuned model ---")
        final_model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=tuner_callbacks,
            verbose=1,
        )
        y_pred = inverse_scale_target(final_model.predict(x_test, verbose=0))
        y_true = inverse_scale_target(y_test)
        metrics = evaluate_predictions(y_true, y_pred)
        print("\nTest metrics (tuned 3 CNN + 1 GRU, single run):")
        for name, value in metrics.items():
            print(f"  {name}: {value:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bayesian optimization of the 3 CNN + 1 GRU hybrid")
    parser.add_argument("--trials", type=int, default=30, help="number of Bayesian optimization trials")
    parser.add_argument("--epochs", type=int, default=30, help="max epochs per trial (same as the sample tuner)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true", help="skip the final retrain and test-set evaluation")
    args = parser.parse_args()

    run_cnn_gru_tuning(
        max_trials=args.trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        evaluate_final=not args.skip_eval,
    )