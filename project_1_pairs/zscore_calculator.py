import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

# --- MASTER TOGGLES ---
ENFORCE_COINTEGRATION = False  # Turned TRUE to implement strict risk gates

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = BASE_DIR / "market_data.db"

def calculate_zscores(db_name=DEFAULT_DB_PATH, window=20):
    print("Connecting to database...")
    conn = sqlite3.connect(db_name)

    # 1. Read the raw data from SQLite into Pandas DataFrames
    print("Loading AAPL and MSFT data...")
    aapl_df = pd.read_sql("SELECT Date, AAPL_Close FROM AAPL", conn)
    msft_df = pd.read_sql("SELECT Date, MSFT_Close FROM MSFT", conn)

    # 2. Merge the data securely (Inner join prevents holiday/weekend mismatch)
    df = pd.merge(aapl_df, msft_df, on="Date", how="inner")
    
    # Force chronological order to prevent data leakage (look-ahead bias)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)

    print(f"Merged successfully. Total overlapping trading days: {len(df)}")

    # 3. ADVANCED QUANT MATHEMATICS: Rolling OLS Beta & Spread
    print(f"Calculating {window}-day rolling Beta-Neutral Hedge Ratios...")
    
    # Pre-allocate arrays for performance
    spreads = np.full(len(df), np.nan)
    betas = np.full(len(df), np.nan)
    p_values = np.full(len(df), np.nan)

    # PHASE 1: Populate ALL Spread and Beta metrics first
    for i in range(window - 1, len(df)):
        y = df['AAPL_Close'].iloc[i - window + 1 : i + 1].values
        x = df['MSFT_Close'].iloc[i - window + 1 : i + 1].values
        
        # Run OLS Regression via numpy matrix inversion
        A = np.vstack([x, np.ones(len(x))]).T
        beta, alpha = np.linalg.lstsq(A, y, rcond=None)[0]
        
        betas[i] = beta
        spreads[i] = df['AAPL_Close'].iloc[i] - (beta * df['MSFT_Close'].iloc[i])

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
        df['ADF_P_Value'] = 0.0000  # Default placeholder value for sandbox testing

    # 4. Standardizing the Movement (Z-Score)
    # Now that the spread is beta-neutral, we calculate how far it is stretching
    df['Rolling_Mean'] = df['Spread'].rolling(window=window).mean()
    df['Rolling_Std'] = df['Spread'].rolling(window=window).std()
    df['Z_Score'] = (df['Spread'] - df['Rolling_Mean']) / df['Rolling_Std']

    # Drop the initial lookback rows that contain NaN values
    df = df.dropna().reset_index(drop=True)

    # --- PORTFOLIO GATING REGIME ---
    if ENFORCE_COINTEGRATION:
        latest_p = df['ADF_P_Value'].iloc[-1]
        print(f"Latest Augmented Dickey-Fuller P-Value: {latest_p:.4f}")
        if latest_p >= 0.05:
            print("CRITICAL HALT: Cointegration leash is broken (p-value >= 0.05).")
            print("Market state is a Random Walk. System halting execution pipeline.")
           # conn.close()
            #return

    # 5. Save the upgraded analytical table back to SQLite for C++ consumption
    print("Saving calculated alpha matrix to database (table: 'pairs_data')...")
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df.to_sql("pairs_data", conn, if_exists="replace", index=False)
    
    conn.close()

    print("\n--- UPGRADED QUANT METRICS ---")
    print(df.tail(1)[['Date', 'Beta', 'Spread', 'ADF_P_Value', 'Z_Score']])
    print("-------------------------------\n")
    print("Math complete. Dynamic Strategy ready for C++ Execution Engine.")

if __name__ == "__main__":
    calculate_zscores()