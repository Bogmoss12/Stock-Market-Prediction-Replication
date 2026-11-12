import config
from src.data_acquisition import get_stock_data, get_index_data, merge_all_sources
from src.feature_engineering import build_feature_set
from src.feature_selection import select_top_k_features
from src.data_preparation import prepare_model_data

# acquire stock and index data
stock_df = get_stock_data(config.TICKER)
index_dfs = [get_index_data(idx) for idx in config.INDICES]

raw_df = merge_all_sources(stock_df, index_dfs)
raw_df.to_csv("data/raw/merged_raw.csv", index=False)

# engineer features
feature_df = build_feature_set(raw_df, price_col="Adj Close", config=config)

# select top features
corr_top, skb_top = select_top_k_features(
    feature_df, target_col="Target", exclude_cols=["Date"], k=config.TOP_K_FEATURES
)
print("Correlation top features: ", corr_top)
print("SelectKBest top features: ", skb_top)

final_features = sorted(set(corr_top) & set(skb_top))
final_df = feature_df[["Date"] + final_features + ["Target"]]
final_df.to_csv("data/processed/final_dataset.csv", index=False)

# Data prep and normalization
feature_cols = final_features
target_col = "Target"

x_train, x_test, y_train, y_test, scaler = prepare_model_data(
    df=final_df, 
    feature_cols=feature_cols, 
    target_col=target_col, 
    config=config
)

# Note: I just used these to check samples, time steps, features
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape}")
print(f"y_test shape: {y_test.shape}")