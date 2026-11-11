import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Para hindi lahat ng folds ipplot
PLOT_ALL_FOLDS = False

# Which fold to plot
PAPER_FIGURES = [
    ("lstm", 9, "LSTM_Fold_9"),
    ("gru", 9, "GRU_Fold_9"),
    ("lstm_gru", 9, "LSTM_GRU_Fold_9"),
    ("cnn_gru", 6, "CNN_GRU_Fold_6"),
    ("cnn_lstm", 6, "CNN_LSTM_Fold_6"),
    ("gru_rnn", 9, "GRU_RNN_Fold_9"),
    ("lstm_rnn", 9, "LSTM_RNN_Fold_9"),
    ("rnn", 9, "RNN_Fold_9"),
    ("xgboost", 9, "XGBoost_Fold_9"),
]


def plot_from_csv(csv_path, save_path):
    # Read csv fold file
    df = pd.read_csv(csv_path)
    dates = df["Date"].values
    y_true = df["Real_Values"].values
    y_pred = df["Predicted_Values"].values

    plt.figure(figsize=(14, 6))
    plt.plot(range(len(y_true)), y_true, color="black", linestyle="-", linewidth=1.5, label="Real Values")
    plt.plot(range(len(y_pred)), y_pred, color="forestgreen", linestyle="--", linewidth=1.5, label="Predicted Values")

    # Place ticks evenly
    tick_indices = np.linspace(0, len(dates) - 1, min(9, len(dates)), dtype=int)
    formatted_dates = pd.to_datetime(dates[tick_indices]).strftime("%b %d %Y")
    plt.xticks(tick_indices, formatted_dates, rotation=45, fontsize=9)

    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Price ($)", fontsize=11)
    plt.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="gray")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def generate_discussion_plots():
    pred_dir = "results/cross_validation/fold_predictions"
    out_dir = "results/discussion_figures"
    os.makedirs(out_dir, exist_ok=True)

    # Generate figures
    for file_name, target_fold, fig_name in PAPER_FIGURES:
        csv_file = f"{pred_dir}/{file_name}_fold_{target_fold}.csv"
        if os.path.exists(csv_file):
            save_path = f"{out_dir}/{fig_name}.png"
            plot_from_csv(csv_file, save_path)
            print(f"Saved: {save_path}")
        else:
            print(f"Skipped (CSV not found yet): {csv_file}")

    # Generate figures for all (if needed lang naman)
    if PLOT_ALL_FOLDS:
        for safe_name, _, _ in PAPER_FIGURES:
            model_out_dir = f"{out_dir}/all_folds/{safe_name}"
            os.makedirs(model_out_dir, exist_ok=True)
            for fold_idx in range(1, 11):
                csv_file = f"{pred_dir}/{safe_name}_fold_{fold_idx}.csv"
                if os.path.exists(csv_file):
                    plot_from_csv(csv_file, f"{model_out_dir}/fold_{fold_idx}.png")
        print(f"All 10-fold plots saved under '{out_dir}/all_folds/'.")


if __name__ == "__main__":
    generate_discussion_plots()