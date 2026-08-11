import yfinance as yf
import numpy as np
import pandas as pd


# Nifty 50 index ticker on Yahoo Finance
ticker = "^NSEI"

# Pull 5 years of daily data
data = yf.download(ticker, start = "2019-01-01", end = "2026-08-01")


print(data.head())

print(f"\nTotal rows: {len(data)}")

# Save raw data
data.to_csv("../data/nifty50_raw.csv")

print("\nSaved to data/nifty50_raw.csv")

# --- Compute daily returns ---

data['Return'] = data['Close'].pct_change() # pct_change() computes (today's price - yesterday's price) / yesterday's price —> the daily return

# --- Compute realized volatility (21-day rolling std of returns) ---
# 21 trading days ≈ 1 calendar month — standard window in quant finance

data['Volatility'] = data['Return'].rolling(window = 21).std()  # .rolling(window=21).std() looks at the last 21 days of returns and computes their standard deviation


# --- Additional feature: longer-window volatility average (captures regime, not just yesterday's value) ---
data['Volatility_ma10'] = data['Volatility'].rolling(window=10).mean()

# Drop rows with NaN (from pct_change and rolling window warm-up)
data_clean = data.dropna()

print("\nWith returns and volatility:")

print(data_clean[['Close', 'Return', 'Volatility', 'Volatility_ma10']].head(10))

print(f"\nRows after cleaning: {len(data_clean)}")


# Save processed version
data_clean.to_csv("../data/nifty50_processed.csv")
print("\nSaved to data/nifty50_processed.csv")


# --- Build features (X) and target (y) ---

# Features: lagged returns and lagged volatility (what the network can "see" today)
data_clean['Return_lag1'] = data_clean['Return'].shift(1) # yesterday's return
data_clean['Return_lag2'] = data_clean['Return'].shift(2) # day before yesterday's return
data_clean['Volatility_lag1'] = data_clean['Volatility'].shift(1) # yesterday's volatility 
data_clean['Volatility_ma10_lag1'] = data_clean['Volatility_ma10'].shift(1)


# Target: TODAY's volatility (what I am predicting), using YESTERDAY's info as features
# so I shift features forward by 1, meaning "yesterday's known data predicts today's volatility"

features = ['Return_lag1', 'Return_lag2', 'Volatility_lag1','Volatility_ma10_lag1']
target = 'Volatility'

model_data = data_clean[features + [target]].dropna() # drop rows with NaN from lagging



data_clean['Volatility_ma10_lag1'] = data_clean['Volatility_ma10'].shift(1)       # s 


features = ['Return_lag1', 'Return_lag2', 'Volatility_lag1', 'Volatility_ma10_lag1']
target = 'Volatility'

model_data = data_clean[features + [target]].dropna()


print("\nFinal feature/terget dataset:")

print(model_data.head(10))  # first 10 rows of the final dataset

print(f'\nrows ready for training: {len(model_data)}')


# Save this final modeling dataset

model_data.to_csv("../data/nifty50_model_data.csv")

print("\nSaved to data/nifty50_model_data.csv ")

"""
Why lag the features this way: on any given day, you only actually know yesterday's return 
and yesterday's volatility when the market opens — you don't know today's return yet (that
requires the market to close first). So Return_lag1 (yesterday's return) and Volatility_lag1
(yesterday's volatility) are legitimate, "knowable" inputs for predicting today's volatility.
"""

