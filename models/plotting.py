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

FOLD_6_TICKS = [
    "Aug 27, 2009", "Dec 29, 2009", "May 4, 2010",
    "Sep 3, 2010", "Jan 5, 2011", "May 10, 2011",
    "Sep 9, 2011", "Jan 12, 2012", "May 15, 2012"
]

FOLD_9_TICKS = [
    "Dec 28, 2017", "May 21, 2018", "Oct 9, 2018",
    "Mar 4, 2019", "Jul 23, 2019", "Dec 10, 2019",
    "May 1, 2020", "Sep 21, 2020", "Feb 10, 2021"
]


def plot_from_csv(csv_path, save_path, target_fold=None):
    # Read csv fold file
    df = pd.read_csv(csv_path)

    if "xgboost" not in os.path.basename(csv_path).lower():
        df = df.iloc[15:].reset_index(drop=True)

    dates = df["Date"].values
    y_true = df["Real_Values"].values
    y_pred = df["Predicted_Values"].values

    plt.figure(figsize=(14, 6))
    plt.plot(range(len(y_true)), y_true, color="black", linestyle="-", linewidth=1.5, label="Real Values")
    plt.plot(range(len(y_pred)), y_pred, color="forestgreen", linestyle="--", linewidth=1.5, label="Predicted Values")

    # Place ticks evenly
    tick_indices = np.linspace(0, len(dates) - 1, 9, dtype=int)
    if target_fold == 6:
        tick_labels = FOLD_6_TICKS
    elif target_fold == 9:
        tick_labels = FOLD_9_TICKS
    else:
        tick_labels = pd.to_datetime(dates[tick_indices]).strftime("%b %d %Y")

    plt.xticks(tick_indices, tick_labels, rotation=45, fontsize=9)
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
            plot_from_csv(csv_file, save_path, target_fold=target_fold)
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
                    plot_from_csv(csv_file, f"{model_out_dir}/fold_{fold_idx}.png", target_fold=target_fold)
        print(f"All 10-fold plots saved under '{out_dir}/all_folds/'.")


if __name__ == "__main__":
    generate_discussion_plots()