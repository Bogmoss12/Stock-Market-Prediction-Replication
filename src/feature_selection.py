import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression

def correlation_ranking(df: pd.DataFrame, target_col: str, exclude_cols: list) -> pd.Series:
    # rank features by correlation with target 
    features = df.drop(columns=exclude_cols)
    corr = features.corr()[target_col].drop(target_col)
    return corr.reindex(corr.abs().sort_values(ascending=False).index)

def selectkbest_ranking(df: pd.DataFrame, target_col: str, exclude_cols: list) -> pd.Series:
    # rank features by selectkbest f_regression score
    X = df.drop(columns=exclude_cols + [target_col])
    y = df[target_col]
    selector = SelectKBest(score_func = f_regression, k = "all")
    selector.fit(X, y)
    scores = pd.Series(selector.scores_, index = X.columns)
    return scores.sort_values(ascending=False)

def select_top_k_features(df: pd.DataFrame, target_col: str, exclude_cols: list, k: int) -> list:
    # combine both rankings, return agreed top-k feature names
    corr_top = correlation_ranking(df, target_col, exclude_cols).head(k).index.tolist()
    skb_top = selectkbest_ranking(df, target_col, exclude_cols).head(k).index.tolist()
    return corr_top, skb_top
