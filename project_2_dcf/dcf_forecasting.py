"""
dcf_forecasting.py — Stage 2 of the DCF Valuation Pipeline
============================================================

This module builds the forecasting engine that projects future Free Cash Flow
to Firm (FCFF) based on historical financial data. It sits between the data
ingestion layer (dcf_ingestion.py) and the valuation layer (dcf_valuation.py).

Responsibilities
----------------
  1. Load the merged historical financial data from the SQLite cache.
  2. Compute key operational drivers from the historical data:
     - Revenue CAGR (Compound Annual Growth Rate)
     - Average operating margin (EBIT / Revenue)
     - Average D&A-to-revenue ratio
     - Average CapEx-to-revenue ratio
  3. Project 5 years of future FCFF using the formula:
         FCFF = NOPAT + D&A − CapEx − ΔNWC

Financial Theory
----------------
FCFF (Free Cash Flow to Firm) represents the cash available to ALL capital
providers (both equity and debt holders) after the company has:
  - Paid its operating expenses (→ EBIT)
  - Paid its taxes (→ NOPAT = EBIT × (1 − Tax))
  - Added back non-cash charges (→ + D&A)
  - Reinvested in the business (→ − CapEx)
  - Funded working capital needs (→ − ΔNWC)

FCFF is the fundamental input to the DCF valuation; it's what gets discounted
back to present value in Stage 3.

Dependencies
------------
    - sqlite3 : Standard library; reads cached data from local database.
    - pandas  : DataFrame manipulation and SQL I/O.

Usage
-----
    # Called programmatically by dcf_valuation.py:
    forecast_df, cagr, margin = generate_fcff_projections("AAPL")
"""

import sys
import sqlite3
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ticker_utils import validate_ticker, quoted_table_name

# ─── MODEL CONFIGURATION ─────────────────────────────────────────────────────
# Resolved relative to this file, not the cwd — see dcf_ingestion.py.
BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "corporate_data.db"
TAX_RATE = 0.21               # Default tax rate (US federal, 21% since 2017 Tax Cuts and Jobs Act) used
                               # ONLY when generate_fcff_projections() is called without an explicit
                               # tax_rate — dcf_valuation.py always passes a jurisdiction-resolved rate.
FORECAST_YEARS = 5            # Number of explicit forecast years before terminal value takes over

# Every field the model actually consumes downstream (CAGR, margins, WACC's
# debt lookup). A fiscal year missing any of these is not usable.
REQUIRED_HISTORICAL_COLUMNS = [
    'revenue', 'operatingIncome', 'totalDebt',
    'depreciationAndAmortization', 'capitalExpenditure',
]
MIN_USABLE_YEARS = 3  # Below this, a CAGR/margin estimate isn't reliable enough to project from.


def _drop_incomplete_years(df):
    """
    Drop fiscal years with missing (NaN) data in any field the model depends on.

    Yahoo Finance frequently has gaps in the oldest year of its reporting
    window — e.g. TATASTEEL.NS reports no FY2022 Total Revenue at all, just
    a NaN. Left alone, that NaN flows straight into calculate_advanced_drivers():
    CAGR reads revenue.iloc[0] as revenue_start, which becomes NaN, so CAGR
    becomes NaN, and every downstream number (FCFF, PV, terminal value,
    intrinsic share price) becomes NaN too — silently. No exception is ever
    raised; the pipeline "succeeds" and prints "$nan" or similar as the
    answer. This function makes that failure loud and early instead.

    Parameters
    ----------
    df : pd.DataFrame
        The merged historical DataFrame, before use in calculate_advanced_drivers().
        Must already have a 'year' column.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with incomplete-year rows removed and the index reset.

    Raises
    ------
    ValueError
        If fewer than MIN_USABLE_YEARS rows remain after dropping incomplete
        years — not enough history left for a meaningful CAGR/margin estimate.
    """
    for col in REQUIRED_HISTORICAL_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    incomplete_mask = df[REQUIRED_HISTORICAL_COLUMNS].isna().any(axis=1)

    if incomplete_mask.any():
        dropped_years = df.loc[incomplete_mask, 'year'].tolist()
        missing_by_year = {
            int(row['year']): [c for c in REQUIRED_HISTORICAL_COLUMNS if pd.isna(row[c])]
            for _, row in df.loc[incomplete_mask].iterrows()
        }
        print(f"⚠️  Dropping {len(dropped_years)} incomplete fiscal year(s) from the historical "
              f"window — missing data: {missing_by_year}. These would otherwise silently poison "
              f"CAGR and every downstream projection with NaN.")
        df = df.loc[~incomplete_mask].reset_index(drop=True)

    if len(df) < MIN_USABLE_YEARS:
        raise ValueError(
            f"Only {len(df)} usable fiscal year(s) of historical data remain after dropping "
            f"incomplete years (minimum {MIN_USABLE_YEARS} required for a reliable CAGR/margin "
            f"estimate). Try a ticker with more complete reporting history."
        )

    return df


def load_historical_cache(ticker):
    """
    Load and merge the three historical financial tables from the local SQLite cache.

    Reads the income, balance, and cashflow tables that were previously saved
    by dcf_ingestion.py, then performs an inner merge on the 'date' column to
    produce a single unified DataFrame with all metrics aligned by fiscal year.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol whose cached data to load (e.g., "AAPL").

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
          date, revenue, operatingIncome, totalDebt,
          depreciationAndAmortization, capitalExpenditure, year
        Fiscal years missing any of these fields (Yahoo Finance often has
        gaps in the oldest reported year) are dropped — see
        _drop_incomplete_years() — with a warning printed for each one.

    Raises
    ------
    ValueError
        If `ticker` fails validate_ticker() (see Security below), or if
        fewer than MIN_USABLE_YEARS fiscal years remain after dropping
        incomplete years.
    sqlite3.OperationalError
        If the tables for this ticker do not exist in the database (i.e.,
        the ingestion step was not run first).

    Security
    --------
    `ticker` is validated via validate_ticker() before it's used to build
    any SQL. It comes from user input (dcf_valuation.py's ticker prompt),
    so building SQL directly from it would be a SQL injection point. Table
    names are quoted via quoted_table_name() so ticker suffixes containing
    "." (e.g. "TATASTEEL.NS") aren't misparsed as a schema.table separator.
    """
    ticker = validate_ticker(ticker)

    conn = sqlite3.connect(DATABASE_PATH)

    # Read each of the three cached tables into separate DataFrames.
    inc = pd.read_sql(f"SELECT * FROM {quoted_table_name(ticker, 'income')}", conn)
    bal = pd.read_sql(f"SELECT * FROM {quoted_table_name(ticker, 'balance')}", conn)
    cf = pd.read_sql(f"SELECT * FROM {quoted_table_name(ticker, 'cashflow')}", conn)
    conn.close()

    # Merge all three on the 'date' column so every fiscal year has
    # revenue, EBIT, debt, D&A, and CapEx in a single row.
    #
    # income_df already carries every field (dcf_ingestion.py's
    # fetch_from_yahoo_finance() populates balance_df/cashflow_df from the
    # same enriched row dicts as income_df, so totalDebt,
    # depreciationAndAmortization, and capitalExpenditure are exact
    # duplicates across all three tables). Merging on the full bal/cf
    # frames would collide on those overlapping non-key columns and pandas
    # would silently rename them to totalDebt_x/totalDebt_y etc., which
    # then KeyErrors downstream on df['totalDebt']. Merging on just 'date'
    # from bal/cf keeps the merge as a fiscal-year alignment check without
    # the collision.
    df = inc.merge(bal[['date']], on='date').merge(cf[['date']], on='date')

    # Extract the year as an integer for display and reference purposes.
    df['year'] = pd.to_datetime(df['date']).dt.year

    df = _drop_incomplete_years(df)

    return df


def calculate_advanced_drivers(df):
    """
    Compute the four key financial drivers from historical data.

    These drivers characterise the company's historical performance and are
    used as assumptions for projecting future years. They answer:
      1. How fast is the company growing? (CAGR)
      2. How profitable is it operationally? (operating margin)
      3. How much non-cash depreciation does it record? (D&A ratio)
      4. How much does it reinvest in assets? (CapEx ratio)

    Parameters
    ----------
    df : pd.DataFrame
        Merged historical DataFrame from load_historical_cache().

    Returns
    -------
    tuple[float, float, float, float]
        (cagr_growth, avg_margin, avg_dna_pct, avg_capex_pct)
        All values are decimal fractions (e.g., 0.08 = 8%).

    Notes
    -----
    CAGR vs. Simple Average Growth:
        CAGR = (End / Start)^(1/N) − 1

        CAGR captures the geometric compounding effect and is more appropriate
        than a simple average of year-over-year changes. Example:
            Revenue: $100 → $80 → $120 → $130
            Simple avg YoY: (-20% + 50% + 8.3%) / 3 = 12.8%  ← misleading
            CAGR: (130/100)^(1/3) − 1 = 9.1%                 ← realistic

    The other three drivers use arithmetic means of annual ratios, which is
    standard practice for projecting relatively stable operational metrics.
    """
    # ── CAGR: Smoothed geometric growth rate ─────────────────────────────
    revenue_start = float(df['revenue'].iloc[0])    # First historical year
    revenue_end = float(df['revenue'].iloc[-1])      # Last historical year
    intervals = len(df) - 1                          # Number of time steps

    # CAGR Formula: (End / Start) ^ (1 / N) − 1
    cagr_growth = (revenue_end / revenue_start) ** (1 / intervals) - 1

    # ── Operating Margin: Average EBIT-to-Revenue ratio ──────────────────
    # Measures how much of each revenue dollar becomes operating profit.
    avg_margin = (pd.to_numeric(df['operatingIncome']) / pd.to_numeric(df['revenue'])).mean()

    # ── D&A Ratio: Average Depreciation & Amortization as % of Revenue ──
    # D&A is a non-cash expense; it represents the gradual expensing of
    # long-lived assets (buildings, equipment, patents).
    avg_dna_pct = (pd.to_numeric(df['depreciationAndAmortization']) / pd.to_numeric(df['revenue'])).mean()

    # ── CapEx Ratio: Average Capital Expenditure as % of Revenue ─────────
    # CapEx is the cash spent on acquiring or maintaining physical assets.
    # We use abs() because Yahoo Finance reports CapEx as negative (outflow).
    avg_capex_pct = (abs(pd.to_numeric(df['capitalExpenditure'])) / pd.to_numeric(df['revenue'])).mean()

    return cagr_growth, avg_margin, avg_dna_pct, avg_capex_pct


def generate_fcff_projections(ticker, tax_rate=TAX_RATE):
    """
    Generate 5-year Free Cash Flow to Firm (FCFF) projections for a given ticker.

    This is the core forecasting function. It:
      1. Loads the merged historical data from the cache.
      2. Computes operational drivers (CAGR, margins, ratios).
      3. Iteratively compounds revenue forward for FORECAST_YEARS years.
      4. For each projected year, computes FCFF using:
             FCFF = NOPAT + D&A − CapEx − ΔNWC

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL").
    tax_rate : float, optional
        Corporate tax rate used for NOPAT = EBIT × (1 − tax_rate). Defaults
        to the module-level TAX_RATE (21%, a US assumption) for standalone/
        manual use, but callers valuing a non-US company should pass the
        jurisdiction-appropriate rate explicitly — dcf_valuation.py resolves
        this per-ticker via resolve_jurisdiction_defaults() and always
        passes it explicitly rather than relying on this default.

    Returns
    -------
    tuple[pd.DataFrame, float, float]
        - forecast_df : DataFrame with columns:
              Forecast Year, Projected Revenue ($B), Expected Future FCFF ($B)
          NOTE: the "$B" column labels are misleading in two ways this
          function has no way to fix — (1) despite the name, values are in
          RAW units, not billions (dcf_valuation.py divides by 1e9 for
          display); (2) despite the "$", values are in whatever currency
          Yahoo reports this company's financials in, which this module
          never looks up (it only reads the cached numbers, not the
          ticker's metadata). dcf_valuation.py fetches the real currency
          via fetch_market_assumptions() and relabels these values with it
          before ever displaying them to a user — nothing with this
          module's raw "$B" labels reaches the terminal as-is.
        - cagr_growth : float — the computed revenue CAGR (decimal fraction).
        - avg_margin  : float — the computed average operating margin.

    Notes
    -----
    - Revenue compounds geometrically: each year's revenue = previous year × (1 + CAGR).
    - ΔNWC (Change in Net Working Capital) is simplified to a flat 1% of revenue.
      In a production model, this would be derived from historical changes in
      current assets and current liabilities.
    """
    hist_df = load_historical_cache(ticker)

    # ── Compute the four operational drivers from history ─────────────────
    cagr_growth, avg_margin, avg_dna_pct, avg_capex_pct = calculate_advanced_drivers(hist_df)

    # ── Establish the "jump-off point" — the last observed year ──────────
    latest_rev = float(hist_df['revenue'].iloc[-1])   # Last historical revenue
    latest_year = hist_df['year'].iloc[-1]             # Last historical year number

    forecast_records = []
    running_rev = latest_rev  # This variable compounds forward each iteration

    # ── THE CONVEYOR BELT FORECAST LOOP ──────────────────────────────────
    # Each iteration represents one future fiscal year (Year 1 through Year 5).
    # Revenue grows by CAGR each step; all other metrics are derived as
    # fixed percentages of that projected revenue.
    for step in range(1, FORECAST_YEARS + 1):
        future_year = latest_year + step

        # 1. REVENUE: Compound forward by CAGR.
        #    Year N revenue = Year (N-1) revenue × (1 + CAGR)
        running_rev = running_rev * (1 + cagr_growth)

        # 2. NOPAT (Net Operating Profit After Tax):
        #    EBIT = Revenue × Avg Operating Margin
        #    NOPAT = EBIT × (1 − Tax Rate)
        #    This represents the after-tax cash profit from operations,
        #    ignoring how the company is financed (debt vs equity).
        projected_ebit = running_rev * avg_margin
        nopat = projected_ebit * (1 - tax_rate)

        # 3. NON-CASH AND REINVESTMENT ADJUSTMENTS:
        #    D&A:   Added back because it reduced EBIT but no cash actually left.
        #    CapEx: Subtracted because it represents real cash spent on assets.
        #    ΔNWC:  Cash trapped in day-to-day operations (inventory, receivables).
        #           Simplified here to 1% of revenue as a stable operational estimate.
        projected_dna = running_rev * avg_dna_pct
        projected_capex = running_rev * avg_capex_pct
        projected_nwc_change = running_rev * 0.01  # Stable 1% NWC operational trap

        # 4. FCFF: The Master Formula
        #    FCFF = NOPAT + D&A − CapEx − ΔNWC
        #    This is the cash flow available to ALL investors (equity + debt).
        fcff = nopat + projected_dna - projected_capex - projected_nwc_change

        forecast_records.append({
            "Forecast Year": future_year,
            "Projected Revenue ($B)": running_rev,
            "Expected Future FCFF ($B)": fcff
        })

    return pd.DataFrame(forecast_records), cagr_growth, avg_margin