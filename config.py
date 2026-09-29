# necessary variables for feature selection

# tickers, indices, series
TICKER = "AAPL"
INDICES = ["^IXIC", "^GSPC", "^DJI", "^NYA"]
FRED_SERIES = ["DTB4WK", "DTB3", "DTB6", "DGS5", "DGS10", "DAAA", "DBAA", "DCOILWTICO"]

# time windows for technical indicators
SMA_WINDOWS = [5, 25, 50, 100, 200]
EMA_WINDOWS = [10, 12, 20, 26, 50, 100, 200]
MOM_WINDOWS = [2, 3, 4, 5]
ROC_WINDOWS = [5, 10, 15, 20]

TOP_K_FEATURES = 20