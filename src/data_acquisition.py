import yfinance as yf
import pandas as pd
from fredapi import Fred

def get_stock_data(ticker: str) -> pd.DataFrame:
    # download full historical OHLCV data for a single stock
    df = yf.download(ticker, start="1980-12-12", end="2024-09-28", auto_adjust=False, multi_level_index=False)
    # df = yf.download(ticker, period="40y", auto_adjust=False, multi_level_index=False)
    df = df.drop(columns="Close")
    df = df.reset_index()
    return df

def get_index_data(index_ticker: str) -> pd.DataFrame:
    # download an index's data, keep only Date + Adj Close, rename column
    df = yf.download(index_ticker, start="1980-12-12", end="2024-09-28", auto_adjust=False, multi_level_index=False)
    df = df.drop(columns="Close")
    df = df.reset_index()[["Date", "Adj Close"]]
    col_name = index_ticker.replace("^", "")
    df = df.rename(columns={"Adj Close": col_name})
    return df

def merge_all_sources(stock_df, index_dfs: list) -> pd.DataFrame:
    # merge stock + indices on Date
    merged = stock_df.copy()
    for df in index_dfs:
        merged = merged.merge(df, on="Date", how="left")
    merged = merged.sort_values("Date").ffill()
    return merged
