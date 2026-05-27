import sqlite3
import pandas as pd
import yfinance as yf

# --- SIDE NOTE: WHERE IS THE DATA STORED? ---
# We keep our corporate database isolated inside the project_2 folder 
# so it doesn't clutter or accidentally override your Project 1 pairs trading data.
DATABASE_PATH = "project_2_dcf/corporate_data.db"

def fetch_from_yahoo_finance(ticker):
    """
    Connects to Yahoo Finance using the open-source 'yfinance' library.
    Pulls the raw Income Statement, Balance Sheet, and Cash Flow Statement,
    and reshapes them into a clean format for our DCF model.
    """
    print(f"📡 Connecting to Yahoo Finance to grab files for: {ticker}...")
    
    try:
        # Create a tracker object for our target stock
        stock = yf.Ticker(ticker)
        
        # --- LEARNING MOMENT: THE THREE FINANCIAL STATEMENTS ---
        # We need specific line items from all 3 statements to calculate Free Cash Flow.
        # .income_stmt   -> Gives us Revenues and Operating Income (EBIT)
        # .balance_sheet -> Gives us Total Debt (to calculate WACC later)
        # .cashflow      -> Gives us Depreciation (D&A) and Capital Expenditures (CapEx)
        raw_income = stock.income_stmt
        raw_balance = stock.balance_sheet
        raw_cashflow = stock.cashflow
        
        # Quick safety check: If any statement is empty, the company ticker might be mistyped
        if raw_income.empty or raw_balance.empty or raw_cashflow.empty:
            print("⚠️ Ingestion Warning: One or more financial statements came back completely empty.")
            return pd.DataFrame()
            
        # --- LEARNING MOMENT: WHY ARE WE TRANSPOSING (.T)? ---
        # By default, yfinance returns data where accounting lines are rows, and dates are columns.
        # This is backward for data science. We use '.T' to flip (transpose) the matrix.
        # Now, each row represents a 'Fiscal Year', and each column represents an 'Accounting Metric'.
        df_inc = raw_income.T.reset_index()
        df_bal = raw_balance.T.reset_index()
        df_cf = raw_cashflow.T.reset_index()
        
        # Standardize the index column name to 'date' across all dataframes
        df_inc.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        df_bal.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        df_cf.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        
        # We will loop through the flipped tables and extract exactly what our DCF needs
        cleaned_income_rows = []
        
        # Step A: Parse the Income Statement parameters
        for idx, row in df_inc.iterrows():
            cleaned_income_rows.append({
                'date': str(row['date']),
                'revenue': float(row.get('Total Revenue', 0)),
                'operatingIncome': float(row.get('Operating Income', 0))
            })
            
        # Step B: Parse the Balance Sheet parameters (Calculating Total Debt)
        # SIDE NOTE: In accounting, Total Debt = Short-Term Debt + Long-Term Debt.
        # Yahoo Finance uses long, explicit names for these rows. We use a fallback .get()
        # just in case a company uses an alternative naming standard.
        for idx, row in df_bal.iterrows():
            st_debt = float(row.get('Current Debt And Capital Lease Obligation', 0))
            lt_debt = float(row.get('Long Term Debt And Capital Lease Obligation', 0))
            
            if st_debt == 0 and lt_debt == 0:
                st_debt = float(row.get('Current Debt', 0))
                lt_debt = float(row.get('Long Term Debt', 0))
                
            # Inject total debt directly into our structured list
            cleaned_income_rows[idx]['totalDebt'] = st_debt + lt_debt
            
        # Step C: Parse the Cash Flow Statement parameters (D&A and CapEx)
        # SIDE NOTE: Yahoo logs CapEx as a negative number because it represents cash LEAVING the bank.
        # We preserve that negative sign here so our math formulas handle it properly later.
        for idx, row in df_cf.iterrows():
            cleaned_income_rows[idx]['depreciationAndAmortization'] = float(row.get('Depreciation And Amortization', 0))
            cleaned_income_rows[idx]['capitalExpenditure'] = float(row.get('Capital Expenditure', 0))

        # Convert our processed python lists back into beautiful, clean Pandas DataFrames
        final_income = pd.DataFrame(cleaned_income_rows)
        final_balance = pd.DataFrame([{'date': r['date'], 'totalDebt': r['totalDebt']} for r in cleaned_income_rows])
        final_cashflow = pd.DataFrame([{'date': r['date'], 'depreciationAndAmortization': r['depreciationAndAmortization'], 'capitalExpenditure': r['capitalExpenditure']} for r in cleaned_income_rows])
        
        return final_income, final_balance, final_cashflow
        
    except Exception as e:
        print(f"❌ Something went wrong while parsing the web data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def load_or_ingest_asset_data(ticker):
    """
    Manages our local database caching. 
    Checks the local database first to see if we already have the stock saved.
    If we don't, it triggers the Yahoo Finance web scraper loop.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if the database catalog already has tables for this specific stock
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (f"{ticker}_income",))
    saved_table_exists = cursor.fetchone()
    
    if saved_table_exists:
        print(f"💾 Database Hit: Found saved records for {ticker} locally. Skipping internet download!")
        income_df = pd.read_sql(f"SELECT * FROM {ticker}_income", conn)
        balance_df = pd.read_sql(f"SELECT * FROM {ticker}_balance", conn)
        cashflow_df = pd.read_sql(f"SELECT * FROM {ticker}_cashflow", conn)
        conn.close()
        return income_df, balance_df, cashflow_df
    
    # If we don't have it locally, close the connection and fetch from the web
    print(f"🔍 Database Miss: No local files found for {ticker}. Fetching fresh files from the web...")
    conn.close()
    
    income_df, balance_df, cashflow_df = fetch_from_yahoo_finance(ticker)
    
    if income_df.empty:
        print("❌ Error: Could not compile financial matrices. Pipeline stopped.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
    # --- LEARNING MOMENT: SORTING CHRONOLOGICALLY ---
    # APIs return data newest year first. We use sort_values() to flip it oldest year first.
    # Why? Because when we project growth rates later, our loop needs to move forward in time,
    # otherwise we accidentally introduce look-ahead estimation bias!
    for df in [income_df, balance_df, cashflow_df]:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    print(f"💾 Saving clean corporate files for {ticker} into local SQL storage...")
    conn = sqlite3.connect(DATABASE_PATH)
    
    for df, name in zip([income_df, balance_df, cashflow_df], ['income', 'balance', 'cashflow']):
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_sql(f"{ticker}_{name}", conn, if_exists="replace", index=False)
    conn.close()
    
    return income_df, balance_df, cashflow_df

def execute_valuation_audit_pipeline():
    print("==================================================================")
    user_ticker = input("📥 ENTER ANY TICKER YOU WANT TO VALUATE (e.g. AAPL, TSLA, AMD, INTC): ").strip().upper()
    print("==================================================================")
    
    income, balance, cashflow = load_or_ingest_asset_data(user_ticker)
    
    if income.empty:
        return

    print("\n✅ DATA PIPELINE BRIDGE CONNECTED SUCCESSFULLY.")
    print(f"Analyzing a {len(income)}-Year Historical Accounting Horizon.")
    print("------------------------------------------------------------------")
    
    # --- LEARNING MOMENT: SCALING VALUES FOR HUMAN EYES ---
    # Raw corporate data prints out in massive absolute integers (e.g., 383000000000).
    # To make it readable like a real investment banking report, we divide by 1e9 ($10^9$)
    # to display everything cleanly in Billions ($B).
    print(f"📊 HISTORICAL AUDIT SUMMARY LEDGER FOR {user_ticker}:")
    
    fiscal_years = pd.to_datetime(income['date']).dt.year
    
    audit_dataframe = pd.DataFrame({
        "Fiscal Year": fiscal_years,
        "Total Revenue ($B)": pd.to_numeric(income['revenue']) / 1e9,
        "Operating Income / EBIT ($B)": pd.to_numeric(income['operatingIncome']) / 1e9,
        "Total Debt ($B)": pd.to_numeric(balance['totalDebt']) / 1e9,
        "Depreciation & Amort. ($B)": pd.to_numeric(cashflow['depreciationAndAmortization']) / 1e9,
        "Capital Expenditures / CapEx ($B)": (pd.to_numeric(cashflow['capitalExpenditure']) * -1) / 1e9
    })
    
    print(audit_dataframe.to_string(index=False))
    print("==================================================================")
    print("Historical accounting fields verified. Ready to build the valuation forecasting models!")

if __name__ == "__main__":
    execute_valuation_audit_pipeline()