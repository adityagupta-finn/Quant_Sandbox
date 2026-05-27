import sqlite3
import pandas as pd

DATABASE_PATH = "project_2_dcf/corporate_data.db"
TAX_RATE = 0.21              # Standard corporate tax rate (21%)
FORECAST_YEARS = 5           # How many years into the future we project

def load_historical_cache(ticker):
    """Loads our clean historical tables from the local database."""
    conn = sqlite3.connect(DATABASE_PATH)
    inc = pd.read_sql(f"SELECT * FROM {ticker}_income", conn)
    bal = pd.read_sql(f"SELECT * FROM {ticker}_balance", conn)
    cf = pd.read_sql(f"SELECT * FROM {ticker}_cashflow", conn)
    conn.close()
    
    # Merge the sheets together on matching dates
    df = inc.merge(bal, on='date').merge(cf, on='date')
    df['year'] = pd.to_datetime(df['date']).dt.year
    return df

def calculate_advanced_drivers(df):
    """
    Tracks the geometric compound annual growth rate (CAGR) 
    to map a smooth, realistic top-line growth trajectory.
    """
    revenue_start = float(df['revenue'].iloc[0])   # First historical year (2021)
    revenue_end = float(df['revenue'].iloc[-1])     # Last historical year (2025)
    intervals = len(df) - 1                        # Time steps in between (4)
    
    # CAGR Formula: (End / Start) ^ (1/N) - 1
    cagr_growth = (revenue_end / revenue_start) ** (1 / intervals) - 1
    
    # Margins and reinvestment rates use clean column averages
    avg_margin = (pd.to_numeric(df['operatingIncome']) / pd.to_numeric(df['revenue'])).mean()
    avg_dna_pct = (pd.to_numeric(df['depreciationAndAmortization']) / pd.to_numeric(df['revenue'])).mean()
    avg_capex_pct = (abs(pd.to_numeric(df['capitalExpenditure'])) / pd.to_numeric(df['revenue'])).mean()
    
    return cagr_growth, avg_margin, avg_dna_pct, avg_capex_pct

def generate_fcff_projections(ticker):
    """Calculates 5 years of expected raw Free Cash Flow to Firm (FCFF)."""
    hist_df = load_historical_cache(ticker)
    
    # Fetch our advanced operational metrics
    cagr_growth, avg_margin, avg_dna_pct, avg_capex_pct = calculate_advanced_drivers(hist_df)
    
    # Establish our jump-off coordinates from the last available year
    latest_rev = float(hist_df['revenue'].iloc[-1])
    latest_year = hist_df['year'].iloc[-1]
    
    forecast_records = []
    running_rev = latest_rev  
    
    # --- THE CONVEYOR BELT FORECAST LOOP ---
    for step in range(1, FORECAST_YEARS + 1):
        future_year = latest_year + step
        
        # 1. Compound revenue smoothly forward
        running_rev = running_rev * (1 + cagr_growth)
        
        # 2. Extract operating profit and deduct corporate taxes
        projected_ebit = running_rev * avg_margin
        nopat = projected_ebit * (1 - TAX_RATE)
        
        # 3. Reconcile non-cash elements
        projected_dna = running_rev * avg_dna_pct
        projected_capex = running_rev * avg_capex_pct
        projected_nwc_change = running_rev * 0.01  # Stable 1% NWC operational trap
        
        # 4. Master Financial Formula: FCFF = NOPAT + D&A - CapEx - DeltaNWC
        fcff = nopat + projected_dna - projected_capex - projected_nwc_change
        
        forecast_records.append({
            "Forecast Year": future_year,
            "Projected Revenue ($B)": running_rev,
            "Expected Future FCFF ($B)": fcff
        })
        
    return pd.DataFrame(forecast_records), cagr_growth, avg_margin