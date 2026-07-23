import os
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "market_data.db"

def run_pipeline():
    # 1. Load hidden keys securely from the .env file
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("CRITICAL ERROR: API Keys missing from .env file!")
        return

    # 2. Authenticate the Client
    client = StockHistoricalDataClient(api_key, secret_key)
    
    # 3. Setup parameters (Past 2 years of daily data)
    tickers = ["AAPL", "MSFT"]
    start_date = datetime.now() - timedelta(days=2 * 365)
    
    request_params = StockBarsRequest(
        symbol_or_symbols=tickers,
        timeframe=TimeFrame.Day,
        start=start_date
    )
    
    print("Connecting to Alpaca Market Data Servers...")
    try:
        # 4. Fetch data and convert instantly to a尊 DataFrame
        bars = client.get_stock_bars(request_params)
        df = bars.df
        
        if df.empty:
            print("No data returned from the exchange.")
            return
            
        print("Data fetched successfully! Structuring database...")
        
        # 5. Process and split data into individual SQL tables
        conn = sqlite3.connect(str(DB_PATH))
        
        for ticker in tickers:
            # Filter rows specifically belonging to this ticker symbol
            ticker_df = df.xs(ticker, level="symbol").reset_index()
            
            # Keep only Date and Close columns for our core strategy
            clean_df = ticker_df[['timestamp', 'close']].copy()
            clean_df.rename(columns={'timestamp': 'Date', 'close': f'{ticker}_Close'}, inplace=True)
            
            # Format timestamp string cleanly (YYYY-MM-DD)
            clean_df['Date'] = clean_df['Date'].dt.strftime('%Y-%m-%d')
            
            # Save to its own table inside SQLite
            clean_df.to_sql(ticker, conn, if_exists="replace", index=False)
            print(f"--> Stored {len(clean_df)} rows for {ticker}")
            
        conn.close()
        print("Database connection closed. Pipeline successfully executed.")
        
    except Exception as e:
        print(f"Pipeline Execution Failed: {e}")

if __name__ == "__main__":
    run_pipeline()