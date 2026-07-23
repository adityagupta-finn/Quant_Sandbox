"""
dcf_ingestion.py — Stage 1 of the DCF Valuation Pipeline
=========================================================

This module handles the complete data acquisition and storage lifecycle for the
Discounted Cash Flow (DCF) valuation engine. It is responsible for:

  1. Connecting to Yahoo Finance via the `yfinance` library and pulling the
     three core financial statements (Income Statement, Balance Sheet, Cash Flow).
  2. Cleaning and reshaping the raw data into a standardised format suitable
     for quantitative analysis (each row = one fiscal year).
  3. Caching the cleaned data in a local SQLite database to avoid redundant
     network calls on subsequent runs (cache-aside pattern).
  4. Providing a standalone audit entry point that prints a human-readable
     summary of the historical financials.

Data Flow
---------
    Yahoo Finance API
        ↓ (yfinance)
    Raw DataFrames (columns = dates, rows = line items)
        ↓ (.T transpose)
    Flipped DataFrames (rows = fiscal years, columns = metrics)
        ↓ (extract & clean)
    Three clean DataFrames: income, balance, cashflow
        ↓ (sort chronologically)
    SQLite tables: {TICKER}_income, {TICKER}_balance, {TICKER}_cashflow

Dependencies
------------
    - sqlite3   : Standard library; local database engine.
    - pandas    : DataFrame manipulation and SQL I/O.
    - yfinance  : Open-source Yahoo Finance scraper.

Usage
-----
    # As a library (called by dcf_forecasting.py):
    income_df, balance_df, cashflow_df = load_or_ingest_asset_data("AAPL")

    # As a standalone script (prints the audit table):
    $ python dcf_ingestion.py
"""

import sqlite3
import pandas as pd
import yfinance as yf

# ─── DATABASE CONFIGURATION ──────────────────────────────────────────────────
# The SQLite file lives inside the project_2_dcf/ folder to keep corporate
# financial data isolated from other projects (e.g., Project 1 pairs trading).
DATABASE_PATH = "project_2_dcf/corporate_data.db"


def fetch_from_yahoo_finance(ticker):
    """
    Fetch and clean the three core financial statements from Yahoo Finance.

    Connects to the Yahoo Finance API via the `yfinance` library, pulls the
    Income Statement, Balance Sheet, and Cash Flow Statement for the given
    ticker, and reshapes them into analysis-ready DataFrames.

    Parameters
    ----------
    ticker : str
        A valid stock ticker symbol (e.g., "AAPL", "TSLA", "MSFT").

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        A 3-tuple of (income_df, balance_df, cashflow_df) on success.
        Each DataFrame has one row per fiscal year and the following columns:
          - income_df   : date, revenue, operatingIncome, totalDebt,
                          depreciationAndAmortization, capitalExpenditure
          - balance_df  : date, totalDebt
          - cashflow_df : date, depreciationAndAmortization, capitalExpenditure

    pd.DataFrame (empty)
        Returns a single empty DataFrame if any statement is missing or if
        an exception occurs during parsing.

    Side Effects
    ------------
    Prints status messages to stdout for pipeline observability.

    Notes
    -----
    - yfinance returns financial statements with dates as COLUMNS and
      accounting line items as ROWS. We transpose (.T) to flip this so that
      each row represents a fiscal year — the standard orientation for
      time-series financial analysis.
    - Yahoo Finance reports CapEx as a NEGATIVE number (cash outflow). We
      preserve the negative sign here; downstream modules handle the sign
      convention in their formulas.
    - Total Debt is computed as Short-Term Debt + Long-Term Debt. Yahoo
      Finance uses varying column names across companies, so we try the
      more verbose names first and fall back to shorter alternatives.
    """
    print(f"📡 Connecting to Yahoo Finance to grab files for: {ticker}...")

    try:
        # ── Step 0: Create the yfinance Ticker object ────────────────────
        # This object lazily fetches data from Yahoo Finance on first access.
        stock = yf.Ticker(ticker)

        # ── Step 1: Pull the three raw financial statement DataFrames ────
        # These are the same three statements found in a company's 10-K/10-Q:
        #   .income_stmt   → Revenue, EBIT (Operating Income)
        #   .balance_sheet → Total Debt (needed for WACC calculation later)
        #   .cashflow      → D&A and CapEx (needed for FCFF calculation)
        raw_income = stock.income_stmt
        raw_balance = stock.balance_sheet
        raw_cashflow = stock.cashflow

        # ── Step 2: Validate that we received non-empty data ─────────────
        # If any statement is empty, the ticker is likely invalid or delisted.
        if raw_income.empty or raw_balance.empty or raw_cashflow.empty:
            print("⚠️ Ingestion Warning: One or more financial statements came back completely empty.")
            return pd.DataFrame()

        # ── Step 3: Transpose — swap rows and columns ────────────────────
        # BEFORE: Rows = line items (Revenue, COGS, ...), Columns = dates
        # AFTER:  Rows = fiscal years, Columns = line items
        # This is the standard data science orientation for time-series data.
        df_inc = raw_income.T.reset_index()
        df_bal = raw_balance.T.reset_index()
        df_cf = raw_cashflow.T.reset_index()

        # Normalise the index column name to 'date' across all DataFrames.
        # The column could be named 'index' or 'Date' depending on yfinance version.
        df_inc.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        df_bal.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)
        df_cf.rename(columns={'index': 'date', 'Date': 'date'}, inplace=True)

        # ── Step 4: Extract the specific line items we need ──────────────
        # We build a single list of dicts (`cleaned_income_rows`) and enrich
        # it across three loops — one per statement. This works because
        # yfinance returns aligned fiscal-year data across all three statements.
        cleaned_income_rows = []

        # ---- Loop A: Income Statement → Revenue and EBIT ----------------
        for idx, row in df_inc.iterrows():
            cleaned_income_rows.append({
                'date': str(row['date']),
                'revenue': float(row.get('Total Revenue', 0)),
                'operatingIncome': float(row.get('Operating Income', 0))
            })

        # ---- Loop B: Balance Sheet → Total Debt --------------------------
        # Total Debt = Short-Term Debt + Long-Term Debt.
        # Yahoo Finance uses verbose column names that can vary by company.
        # Strategy: try the long names first; if both are 0, fall back to
        # shorter alternatives.
        for idx, row in df_bal.iterrows():
            st_debt = float(row.get('Current Debt And Capital Lease Obligation', 0))
            lt_debt = float(row.get('Long Term Debt And Capital Lease Obligation', 0))

            # Fallback: some companies report under shorter column names.
            if st_debt == 0 and lt_debt == 0:
                st_debt = float(row.get('Current Debt', 0))
                lt_debt = float(row.get('Long Term Debt', 0))

            # Inject the computed total debt into the shared row dict.
            cleaned_income_rows[idx]['totalDebt'] = st_debt + lt_debt

        # ---- Loop C: Cash Flow Statement → D&A and CapEx -----------------
        # Note: Yahoo Finance reports CapEx as NEGATIVE (cash leaving the firm).
        # We preserve the sign here; the forecasting module uses abs() when needed.
        for idx, row in df_cf.iterrows():
            cleaned_income_rows[idx]['depreciationAndAmortization'] = float(row.get('Depreciation And Amortization', 0))
            cleaned_income_rows[idx]['capitalExpenditure'] = float(row.get('Capital Expenditure', 0))

        # ── Step 5: Convert the enriched list of dicts into DataFrames ───
        final_income = pd.DataFrame(cleaned_income_rows)
        final_balance = pd.DataFrame([{'date': r['date'], 'totalDebt': r['totalDebt']} for r in cleaned_income_rows])
        final_cashflow = pd.DataFrame([{'date': r['date'], 'depreciationAndAmortization': r['depreciationAndAmortization'], 'capitalExpenditure': r['capitalExpenditure']} for r in cleaned_income_rows])

        return final_income, final_balance, final_cashflow

    except Exception as e:
        print(f"❌ Something went wrong while parsing the web data: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def load_or_ingest_asset_data(ticker):
    """
    Cache-aside data loader: check local DB first, fetch from web on miss.

    Implements a simple cache-aside pattern:
      1. Open the SQLite database and check if tables for this ticker exist.
      2. If YES (cache hit)  → read the three tables and return DataFrames.
      3. If NO  (cache miss) → call fetch_from_yahoo_finance(), sort the
         results chronologically, persist them to SQLite, and return.

    Parameters
    ----------
    ticker : str
        A valid stock ticker symbol (e.g., "AAPL").

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (income_df, balance_df, cashflow_df) — see fetch_from_yahoo_finance()
        for column details.

    Side Effects
    ------------
    - Creates or updates the SQLite database at DATABASE_PATH.
    - Prints cache hit/miss status to stdout.

    Notes
    -----
    The chronological sort (oldest → newest) is critical for the forecasting
    module: CAGR computation reads iloc[0] as the starting revenue and
    iloc[-1] as the ending revenue. If data were still in Yahoo's default
    newest-first order, the growth rate would be inverted.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # ── Check if we already have cached data for this ticker ─────────────
    # We only check for the income table; if it exists, all three should exist
    # because they are always saved together in fetch_from_yahoo_finance().
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (f"{ticker}_income",))
    saved_table_exists = cursor.fetchone()

    if saved_table_exists:
        # ── CACHE HIT: Load from local database ─────────────────────────
        print(f"💾 Database Hit: Found saved records for {ticker} locally. Skipping internet download!")
        income_df = pd.read_sql(f"SELECT * FROM {ticker}_income", conn)
        balance_df = pd.read_sql(f"SELECT * FROM {ticker}_balance", conn)
        cashflow_df = pd.read_sql(f"SELECT * FROM {ticker}_cashflow", conn)
        conn.close()
        return income_df, balance_df, cashflow_df

    # ── CACHE MISS: Fetch from Yahoo Finance ─────────────────────────────
    print(f"🔍 Database Miss: No local files found for {ticker}. Fetching fresh files from the web...")
    conn.close()

    income_df, balance_df, cashflow_df = fetch_from_yahoo_finance(ticker)

    if income_df.empty:
        print("❌ Error: Could not compile financial matrices. Pipeline stopped.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ── Sort chronologically: oldest fiscal year first ────────────────────
    # yfinance returns data newest-first. We need oldest-first so the
    # forecasting loop can compound growth FORWARD in time. Without this
    # sort, the CAGR formula would compute a NEGATIVE growth rate for a
    # company that is actually growing (because it would see revenue
    # declining from new → old).
    for df in [income_df, balance_df, cashflow_df]:
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.reset_index(drop=True, inplace=True)

    # ── Persist to SQLite ────────────────────────────────────────────────
    # Dates are stored as ISO strings (YYYY-MM-DD) for portability.
    # if_exists="replace" ensures re-ingestion overwrites stale data.
    print(f"💾 Saving clean corporate files for {ticker} into local SQL storage...")
    conn = sqlite3.connect(DATABASE_PATH)

    for df, name in zip([income_df, balance_df, cashflow_df], ['income', 'balance', 'cashflow']):
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        df.to_sql(f"{ticker}_{name}", conn, if_exists="replace", index=False)
    conn.close()

    return income_df, balance_df, cashflow_df


def execute_valuation_audit_pipeline():
    """
    Standalone entry point: ingest data and print a human-readable audit table.

    Prompts the user for a ticker symbol, runs the ingestion pipeline, and
    displays a formatted summary of historical financials with all monetary
    values scaled to billions (÷ 1e9) for readability — mimicking the style
    of investment banking research reports.

    This function is only invoked when dcf_ingestion.py is run as __main__;
    it is NOT called by the forecasting or valuation modules.
    """
    print("==================================================================")
    user_ticker = input("📥 ENTER ANY TICKER YOU WANT TO VALUATE (e.g. AAPL, TSLA, AMD, INTC): ").strip().upper()
    print("==================================================================")

    income, balance, cashflow = load_or_ingest_asset_data(user_ticker)

    if income.empty:
        return

    print("\n✅ DATA PIPELINE BRIDGE CONNECTED SUCCESSFULLY.")
    print(f"Analyzing a {len(income)}-Year Historical Accounting Horizon.")
    print("------------------------------------------------------------------")

    # ── Scale raw absolute values to billions for human readability ───────
    # Corporate financials are typically in the range of $10^9 – $10^12.
    # Dividing by 1e9 converts them to a familiar "$X.XX Billion" format.
    print(f"📊 HISTORICAL AUDIT SUMMARY LEDGER FOR {user_ticker}:")

    fiscal_years = pd.to_datetime(income['date']).dt.year

    audit_dataframe = pd.DataFrame({
        "Fiscal Year": fiscal_years,
        "Total Revenue ($B)": pd.to_numeric(income['revenue']) / 1e9,
        "Operating Income / EBIT ($B)": pd.to_numeric(income['operatingIncome']) / 1e9,
        "Total Debt ($B)": pd.to_numeric(balance['totalDebt']) / 1e9,
        "Depreciation & Amort. ($B)": pd.to_numeric(cashflow['depreciationAndAmortization']) / 1e9,
        # CapEx is negated (* -1) because Yahoo reports it as negative (cash outflow),
        # but we want to display it as a positive expenditure for readability.
        "Capital Expenditures / CapEx ($B)": (pd.to_numeric(cashflow['capitalExpenditure']) * -1) / 1e9
    })

    print(audit_dataframe.to_string(index=False))
    print("==================================================================")
    print("Historical accounting fields verified. Ready to build the valuation forecasting models!")


if __name__ == "__main__":
    execute_valuation_audit_pipeline()