import sys
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "market_data.db"

sys.path.insert(0, str(BASE_DIR))
from pair_config import TICKER_A, TICKER_B

# ENFORCE_COINTEGRATION gates signal generation on the ADF test actually
# passing (p < 0.05). With it off, no test is run at all and the pipeline
# treats every spread as tradeable regardless of whether it's actually
# mean-reverting.
ENFORCE_COINTEGRATION = True

def calculate_zscores(db_name=DEFAULT_DB_PATH, window=60):
    print("Connecting to database...")
    conn = sqlite3.connect(db_name)

    # 1. Read the raw data from SQLite into Pandas DataFrames
    print(f"Loading {TICKER_A} and {TICKER_B} data...")
    a_df = pd.read_sql(f"SELECT Date, {TICKER_A}_Close FROM {TICKER_A}", conn)
    b_df = pd.read_sql(f"SELECT Date, {TICKER_B}_Close FROM {TICKER_B}", conn)

    # 2. Merge the data securely (Inner join prevents holiday/weekend mismatch)
    df = pd.merge(a_df, b_df, on="Date", how="inner")
    
    # Force chronological order to prevent data leakage (look-ahead bias)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Merged successfully. Total overlapping trading days: {len(df)}")

    # 3. Rolling OLS beta and spread
    print(f"Calculating {window}-day rolling hedge ratios...")
    
    # Pre-allocate arrays for performance
    spreads = np.full(len(df), np.nan)
    betas = np.full(len(df), np.nan)
    p_values = np.full(len(df), np.nan)

    # PHASE 1: Populate ALL Spread and Beta metrics first
    for i in range(window - 1, len(df)):
        y = df[f'{TICKER_A}_Close'].iloc[i - window + 1 : i + 1].values
        x = df[f'{TICKER_B}_Close'].iloc[i - window + 1 : i + 1].values

        # Run OLS Regression via statsmodels (same library used for the ADF test below)
        X = sm.add_constant(x)
        alpha, beta = sm.OLS(y, X).fit().params

        betas[i] = beta
        spreads[i] = df[f'{TICKER_A}_Close'].iloc[i] - (beta * df[f'{TICKER_B}_Close'].iloc[i])

    # Map our calculated vectors back into the main DataFrame
    df['Beta'] = betas
    df['Spread'] = spreads

    # PHASE 2: Calculate the rolling ADF P-Value using fully populated data
    # We need to wait until we are far enough into the dataframe to have a valid history of spreads
    if ENFORCE_COINTEGRATION:
        print("Running historical ADF Cointegration tests across the series...")
        # We start at (window * 2) - 2 because we need a full 'window' of valid, non-NaN spreads
        for i in range((window * 2) - 2, len(df)):
            spread_window = df['Spread'].iloc[i - window + 1 : i + 1].values
            
            # Run the Augmented Dickey-Fuller test on the historical slice
            adf_result = adfuller(spread_window, maxlag=1)
            p_values[i] = adf_result[1]
            
        df['ADF_P_Value'] = p_values
    else:
        # Placeholder only -- NOT a real p-value. Must stay non-NaN so it
        # survives the dropna() below; ADF_Computed (not this number) is
        # what tells the dashboard whether cointegration was actually tested.
        df['ADF_P_Value'] = 0.0

    # Lets the dashboard tell a genuinely computed p-value apart from a run
    # where cointegration testing was skipped entirely.
    df['ADF_Computed'] = ENFORCE_COINTEGRATION

    # 4. Standardizing the Movement (Z-Score)
    # Now that the spread is beta-neutral, we calculate how far it is stretching
    df['Rolling_Mean'] = df['Spread'].rolling(window=window).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=window).std()
    df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']

    # Drop the initial lookback rows that contain NaN values
    df = df.dropna().reset_index(drop=True)

    # --- Halt if cointegration is broken ---
    if ENFORCE_COINTEGRATION:
        latest_p = df['ADF_P_Value'].iloc[-1]
        print(f"Latest Augmented Dickey-Fuller P-Value: {latest_p:.4f}")
        if latest_p >= 0.05:
            print("CRITICAL HALT: Cointegration leash is broken (p-value >= 0.05).")
            print("Market state is a Random Walk. System halting execution pipeline.")
            conn.close()
            return

    # 5. Save the computed table back to SQLite for the C++ engine to read
    print("Saving computed table to database (table: 'pairs_data')...")
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df['Ticker_A'] = TICKER_A
    df['Ticker_B'] = TICKER_B
    df.to_sql("pairs_data", conn, if_exists="replace", index=False)

    conn.close()

    print("\n--- Latest computed row ---")
    print(df.tail(1)[['Date', 'Beta', 'Spread', 'ADF_P_Value', 'Z_Score']])
    print("---------------------------\n")

if __name__ == "__main__":
    calculate_zscores()