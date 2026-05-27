import sqlite3
import pandas as pd
# Import our projection engine straight from our forecasting file
from dcf_forecasting import generate_fcff_projections, load_historical_cache

# --- WACC & ENVIRONMENT MARKET SETTINGS ---
RISK_FREE_RATE = 0.04        # 4.0% (Return on safe 10-year Treasury Bonds)
EQUITY_RISK_PREMIUM = 0.05   # 5.0% (Extra return demanded by stock investors)
ASSUMED_BETA = 1.20          # Apple's market volatility factor (β)
MARKET_CAP_PROXY = 2800e9    # Assume a standard $2.8 Trillion capitalization scale
ASSUMED_COST_OF_DEBT = 0.045 # 4.5% pre-tax interest rate on debt
TAX_RATE = 0.21

# --- TERMINAL INFINITY MATH CONSTANTS ---
PERPETUAL_GROWTH_RATE = 0.025 # 2.5% (Speed the company grows forever after Year 5)
OUTSTANDING_SHARES = 15.4e9   # Assume 15.4 Billion outstanding shares for Apple

def calculate_live_wacc(ticker):
    """Blends the Cost of Equity and Debt to find our model's discount gravity."""
    hist_df = load_historical_cache(ticker)
    cost_of_equity = RISK_FREE_RATE + (ASSUMED_BETA * EQUITY_RISK_PREMIUM)
    latest_debt = float(hist_df['totalDebt'].iloc[-1])
    
    total_capital = MARKET_CAP_PROXY + latest_debt
    weight_of_equity = MARKET_CAP_PROXY / total_capital
    weight_of_debt = latest_debt / total_capital
    
    after_tax_cost_of_debt = ASSUMED_COST_OF_DEBT * (1 - TAX_RATE)
    wacc = (weight_of_equity * cost_of_equity) + (weight_of_debt * after_tax_cost_of_debt)
    return wacc, latest_debt

def run_master_valuation_app():
    print("==================================================================")
    user_ticker = input("📥 ENTER TARGET TICKER FOR FULL DCF VALUATION: ").strip().upper()
    print("==================================================================")
    
    try:
        # 1. Fire our isolated forecasting layer module
        forecast_df, cagr, margin = generate_fcff_projections(user_ticker)
        
        # 2. Compute the dynamic discount rate gravity and pull latest debt
        wacc_rate, latest_debt_raw = calculate_live_wacc(user_ticker)
        
        print(f"\n📈 INITIALIZING VALUATION FOR STRUCTURE: {user_ticker}")
        print(f" -> Computed Geometric Revenue CAGR: {cagr*100:.2f}%")
        print(f" -> Computed Baseline Operating Margin: {margin*100:.2f}%")
        print(f" -> Computed Cost of Capital (WACC): {wacc_rate*100:.2f}%")
        print("------------------------------------------------------------------\n")
        
        valuation_rows = []
        
        # --- THE MASTER TIME-VALUE DISCOUNTING LOOP ---
        for idx, row in forecast_df.iterrows():
            f_year = int(row['Forecast Year'])
            raw_rev = float(row['Projected Revenue ($B)'])
            raw_fcff = float(row['Expected Future FCFF ($B)'])
            
            time_step = idx + 1 # t = 1, 2, 3, 4, 5
            
            # Present Value Formula: PV = Future Cash Flow / (1 + WACC)^t
            discount_factor = (1 + wacc_rate) ** time_step
            present_value_of_cash = raw_fcff / discount_factor
            
            valuation_rows.append({
                "Year": f_year,
                "Time Step (t)": time_step,
                "Projected Revenue ($B)": raw_rev / 1e9,
                "Expected Future FCFF ($B)": raw_fcff / 1e9,
                "Present Value (PV) ($B)": present_value_of_cash / 1e9
            })
            
        summary_df = pd.DataFrame(valuation_rows)
        print(summary_df.to_string(index=False))
        print("------------------------------------------------------------------")
        
        # --- THE INFINITY CAPSTONE CALCULATIONS ---
        # A. Sum up the present values of our 5-year explicit window
        explicit_pv_sum = summary_df['Present Value (PV) ($B)'].sum()
        
        # B. Calculate Terminal Value at Year 5 using Gordon Growth Model
        final_year_fcff = valuation_rows[-1]["Expected Future FCFF ($B)"] * 1e9
        terminal_value_year_5 = (final_year_fcff * (1 + PERPETUAL_GROWTH_RATE)) / (wacc_rate - PERPETUAL_GROWTH_RATE)
        
        # C. Discount Terminal Value back to Year 0 (Today)
        # It happens at Year 5, so we discount it by (1 + WACC)^5
        pv_of_terminal_value = (terminal_value_year_5 / 1e9) / ((1 + wacc_rate) ** 5)
        
        # D. Enterprise Value (Value of the whole operational machine)
        enterprise_value = explicit_pv_sum + pv_of_terminal_value
        
        # E. Equity Value (Value belonging strictly to shareholders)
        # Equity Value = Enterprise Value - Total Debt
        latest_debt_billions = latest_debt_raw / 1e9
        equity_value = enterprise_value - latest_debt_billions
        
        # F. Intrinsic Share Price = Equity Value / Outstanding Shares
        intrinsic_share_price = (equity_value * 1e9) / OUTSTANDING_SHARES
        
        # --- DISPLAY THE ULTIMATE INVESTMENT METRICS ---
        print(f"💰 PV of Explicit 5-Year Cash Flows : ${explicit_pv_sum:.2f} Billion")
        print(f"🔮 PV of Terminal Value (Infinity)   : ${pv_of_terminal_value:.2f} Billion")
        print(f"🏢 Total Enterprise Value            : ${enterprise_value:.2f} Billion")
        print(f"🧾 Less: Outstanding Corporate Debt   : -${latest_debt_billions:.2f} Billion")
        print(f"🎯 Total Net Equity Value            : ${equity_value:.2f} Billion")
        print("------------------------------------------------------------------")
        print(f"💎 TARGET INTRINSIC VALUE PER SHARE   : ${intrinsic_share_price:.2f}")
        print("==================================================================")
        
    except Exception as error:
        print(f"❌ Valuation aborted: Ensure the stock ticker exists inside your local data cache. Error details: {error}")

if __name__ == "__main__":
    run_master_valuation_app()