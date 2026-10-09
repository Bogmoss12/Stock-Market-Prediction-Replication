import pandas as pd
import talib

# indicator functions separated into moving averages, momentum indicators, and spreads

def add_moving_averages(df: pd.DataFrame, price_col: str, sma_windows: list, ema_windows: list) -> pd.DataFrame:
    for w in sma_windows:
        df[f"SMA{w}"] = talib.SMA(df[price_col], timeperiod=w)
    for w in ema_windows:
        df[f"EMA{w}"] = talib.EMA(df[price_col], timeperiod = w)
    return df

def add_momentum_indicators(df: pd.DataFrame, price_col: str, mom_windows: list, roc_windows: list) -> pd.DataFrame:
    for w in mom_windows:
        df[f"MOM{w}"] = talib.MOM(df[price_col], timeperiod=w)
    for w in roc_windows:
        df[f"ROC{w}"] = talib.ROC(df[price_col], timeperiod=w)
    df["MACD"], _, _ = talib.MACD(df[price_col])
    df["RSI"] = talib.RSI(df[price_col])
    return df

def add_interest_rate_spreads(df: pd.DataFrame) -> pd.DataFrame:
    # computations as described in paper under Table 2. Features tested
    df["TE1"] = df["DGS10"] - df["DTB4WK"]
    df["TE2"] = df["DGS10"] - df["DTB3"]
    df["TE3"] = df["DGS10"] - df["DTB6"]
    df["TE5"] = df["DTB3"] - df["DTB4WK"]
    df["TE6"] = df["DTB6"] - df["DTB4WK"]
    df["DE1"] = df["DBAA"] - df["DAAA"]
    df["DE2"] = df["DBAA"] - df["DGS10"]
    df["DE4"] = df["DBAA"] - df["DTB6"]
    df["DE5"] = df["DBAA"] - df["DTB3"]
    df["DE6"] = df["DBAA"] - df["DTB4WK"]
    return df

def build_target(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    # target = next day adj close
    df["Target"] = df[price_col].shift(-1)
    return df

def build_feature_set(df: pd.DataFrame, price_col: str, config) -> pd.DataFrame:
    df = add_moving_averages(df, price_col, config.SMA_WINDOWS, config.EMA_WINDOWS)
    df = add_momentum_indicators(df, price_col, config.MOM_WINDOWS, config.ROC_WINDOWS)
    df = add_interest_rate_spreads(df)
    df = build_target(df, price_col)
    df = df.dropna().reset_index(drop=True) # drop rows before longest window SMA200/EMA200 can compute
    return df