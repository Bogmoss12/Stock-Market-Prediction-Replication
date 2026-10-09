import yfinance as yf
import pandas as pd
from fredapi import Fred

def get_stock_data(ticker: str) -> pd.DataFrame:
    # download full historical OHLCV data for a single stock
    df = yf.download(ticker, start="1980-12-12", end="2024-09-24", auto_adjust=False, multi_level_index=False)
    # df = yf.download(ticker, period="40y", auto_adjust=False, multi_level_index=False)
    df = df.drop(columns="Close")
    df = df.reset_index()
    return df

def get_index_data(index_ticker: str) -> pd.DataFrame:
    # download an index's data, keep only Date + Adj Close, rename column
    df = yf.download(index_ticker, start="1980-12-12", end="2020-09-24", auto_adjust=False, multi_level_index=False)
    # df = yf.download(index_ticker, period="40y", auto_adjust=False, multi_level_index=False)
    df = df.drop(columns="Close")
    df = df.reset_index()[["Date", "Adj Close"]]
    col_name = index_ticker.replace("^", "")
    df = df.rename(columns={"Adj Close": col_name})
    return df

def get_fred_series(series_id: str, api_key: str) -> pd.DataFrame:
    # download a single FRED series, return as date/value dataframe
    fred = Fred(api_key = api_key)
    series = fred.get_series(series_id)
    df = series.reset_index()
    df.columns = ["Date", series_id]
    return df

def merge_all_sources(stock_df, index_dfs: list, fred_dfs: list) -> pd.DataFrame:
    # merge stock + indices + FRED series on Date, forward-fill FRED gaps
    merged = stock_df.copy()
    for df in index_dfs + fred_dfs:
        merged = merged.merge(df, on = "Date", how = "left")
    merged = merged.sort_values("Date").ffill()
    return merged
