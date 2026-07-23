"""
dcf_valuation.py — Stage 3 of the DCF Valuation Pipeline
==========================================================

This is the final and most important module in the DCF engine. It takes the
FCFF projections from Stage 2 (dcf_forecasting.py) and converts them into a
single number: the **intrinsic value per share**.

The Valuation Methodology
-------------------------
The process follows the textbook DCF framework used by investment banks:

  1. **Discount the explicit forecast period** (Years 1-5):
         PV = FCFF_t / (1 + WACC)^t
     Each future cash flow is "shrunk" back to today's value.

  2. **Calculate Terminal Value** (Year 6 to infinity):
     Using the Gordon Growth Model:
         TV = (FCFF_5 × (1 + g)) / (WACC − g)
     Then discount it back to today:
         PV(TV) = TV / (1 + WACC)^5

  3. **Sum everything** to get Enterprise Value:
         EV = Σ PV(FCFF_t) + PV(TV)

  4. **Subtract Debt** to get Equity Value (what shareholders own):
         Equity Value = EV − Total Debt

  5. **Divide by shares** to get the intrinsic price:
         Share Price = Equity Value / Shares Outstanding

Key Financial Concepts
----------------------
  WACC (Weighted Average Cost of Capital):
      The blended rate of return demanded by all capital providers. It's the
      "discount rate" or "hurdle rate" — the rate at which future cash flows
      lose value when brought back to the present. Computed via:
          WACC = (E/V × Ke) + (D/V × Kd × (1 − Tax))
      Where Ke (Cost of Equity) comes from the CAPM model.

  CAPM (Capital Asset Pricing Model):
      Ke = Risk-Free Rate + β × Equity Risk Premium
      The Risk-Free Rate is the 10-year Treasury yield; β measures the
      stock's volatility relative to the market; ERP is the excess return
      investors demand for holding stocks over risk-free bonds.

  Gordon Growth Model:
      Assumes the company grows at a constant rate 'g' forever after the
      explicit forecast period. This is a simplification — no company truly
      grows forever — but it's mathematically tractable and widely used.

Dependencies
------------
    - sqlite3 : Standard library; reads cached data.
    - pandas  : DataFrame manipulation.
    - dcf_forecasting : The Stage 2 module providing FCFF projections.

Usage
-----
    $ python dcf_valuation.py
    → Prompts for a ticker, then outputs the full valuation.
"""

import sqlite3
import pandas as pd
# Import the forecasting engine and data loader from Stage 2.
from dcf_forecasting import generate_fcff_projections, load_historical_cache

# ─── WACC & MARKET ENVIRONMENT PARAMETERS ─────────────────────────────────────
# These constants define the capital markets assumptions for the valuation.
# They are currently calibrated for an Apple (AAPL)-like company.
RISK_FREE_RATE = 0.04        # 4.0%  — Yield on 10-year US Treasury bonds (the "zero-risk" baseline)
EQUITY_RISK_PREMIUM = 0.05   # 5.0%  — Extra return stock investors demand over Treasuries
ASSUMED_BETA = 1.20          # 1.20  — Stock's volatility vs. market (β > 1 = more volatile than S&P 500)
MARKET_CAP_PROXY = 2800e9    # $2.8T — Assumed market capitalisation (used as the equity weight in WACC)
ASSUMED_COST_OF_DEBT = 0.045 # 4.5%  — Pre-tax interest rate the company pays on its borrowings
TAX_RATE = 0.21              # 21%   — US federal corporate tax rate

# ─── TERMINAL VALUE PARAMETERS ────────────────────────────────────────────────
# These control the "infinity" portion of the valuation.
PERPETUAL_GROWTH_RATE = 0.025 # 2.5% — Rate at which FCFF grows forever after Year 5
                               #        Typically set near long-term GDP or inflation (2-3%)
OUTSTANDING_SHARES = 15.4e9   # 15.4B — Number of shares outstanding for Apple


def calculate_live_wacc(ticker):
    """
    Compute the Weighted Average Cost of Capital (WACC) for the given ticker.

    WACC blends the Cost of Equity (from CAPM) and the after-tax Cost of Debt,
    weighted by their proportions in the company's capital structure.

    Formula
    -------
        Cost of Equity (Ke) = Rf + β × ERP                     (CAPM)
        After-Tax Cost of Debt = Kd × (1 − Tax)                (Tax shield)
        WACC = (E/V × Ke) + (D/V × Kd_aftertax)               (Blend)

    Where:
        Rf  = Risk-Free Rate (10-year Treasury)
        β   = Beta (stock volatility factor)
        ERP = Equity Risk Premium
        Kd  = Pre-tax Cost of Debt
        E   = Market value of equity (market cap)
        D   = Market value of debt
        V   = E + D (total capital)

    Parameters
    ----------
    ticker : str
        Stock ticker symbol whose debt data to load from the cache.

    Returns
    -------
    tuple[float, float]
        (wacc, latest_debt_raw)
        - wacc : float — The computed WACC as a decimal fraction (e.g., 0.09 = 9%).
        - latest_debt_raw : float — The most recent total debt in raw dollars.

    Notes
    -----
    - The equity weight uses the hardcoded MARKET_CAP_PROXY, not a live market cap.
      This means the model is calibrated for Apple; using it on a $50B company
      with a $2.8T equity weight will produce an inaccurate WACC.
    - The debt weight IS semi-dynamic — it reads the latest totalDebt from
      the cached database, so it reflects the actual debt reported by Yahoo.
    """
    hist_df = load_historical_cache(ticker)

    # ── Step 1: Cost of Equity via CAPM ──────────────────────────────────
    # Ke = Rf + β × ERP
    # Example: 0.04 + 1.20 × 0.05 = 0.10 (10%)
    # This says: "Equity investors demand at least a 10% return to hold this stock."
    cost_of_equity = RISK_FREE_RATE + (ASSUMED_BETA * EQUITY_RISK_PREMIUM)

    # ── Step 2: Read the latest total debt from the database ─────────────
    latest_debt = float(hist_df['totalDebt'].iloc[-1])

    # ── Step 3: Compute capital structure weights ────────────────────────
    # V = E + D (total value of all capital)
    total_capital = MARKET_CAP_PROXY + latest_debt
    weight_of_equity = MARKET_CAP_PROXY / total_capital  # E/V — typically ~95% for Apple
    weight_of_debt = latest_debt / total_capital          # D/V — typically ~5% for Apple

    # ── Step 4: After-tax Cost of Debt ───────────────────────────────────
    # Interest payments are tax-deductible, creating a "tax shield".
    # After-tax Kd = Kd × (1 − Tax) = 0.045 × 0.79 = 0.03555 (3.555%)
    after_tax_cost_of_debt = ASSUMED_COST_OF_DEBT * (1 - TAX_RATE)

    # ── Step 5: Blend into WACC ──────────────────────────────────────────
    # WACC = (E/V × Ke) + (D/V × Kd_aftertax)
    # This is the "discount gravity" — the rate at which future cash flows
    # lose value when pulled back to the present.
    wacc = (weight_of_equity * cost_of_equity) + (weight_of_debt * after_tax_cost_of_debt)
    return wacc, latest_debt


def run_master_valuation_app():
    """
    Main orchestrator: run the full DCF valuation and print the intrinsic share price.

    This function ties together all three pipeline stages:
      1. Calls generate_fcff_projections() to get 5-year FCFF forecasts.
      2. Calls calculate_live_wacc() to get the discount rate.
      3. Discounts each year's FCFF to present value.
      4. Computes Terminal Value using the Gordon Growth Model.
      5. Sums PV(FCFF) + PV(TV) to get Enterprise Value.
      6. Subtracts debt to get Equity Value.
      7. Divides by outstanding shares to get the intrinsic share price.

    The final output is the model's estimate of what the stock SHOULD be worth,
    independent of what the market currently prices it at.
    """
    print("==================================================================")
    user_ticker = input("📥 ENTER TARGET TICKER FOR FULL DCF VALUATION: ").strip().upper()
    print("==================================================================")

    try:
        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Generate FCFF projections (calls Stage 2)
        # ═══════════════════════════════════════════════════════════════════
        forecast_df, cagr, margin = generate_fcff_projections(user_ticker)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Compute WACC (the discount rate)
        # ═══════════════════════════════════════════════════════════════════
        wacc_rate, latest_debt_raw = calculate_live_wacc(user_ticker)

        # ── Print the computed model parameters ──────────────────────────
        print(f"\n📈 INITIALIZING VALUATION FOR STRUCTURE: {user_ticker}")
        print(f" -> Computed Geometric Revenue CAGR: {cagr*100:.2f}%")
        print(f" -> Computed Baseline Operating Margin: {margin*100:.2f}%")
        print(f" -> Computed Cost of Capital (WACC): {wacc_rate*100:.2f}%")
        print("------------------------------------------------------------------\n")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 3: Discount each year's FCFF to present value
        # ═══════════════════════════════════════════════════════════════════
        # The Time Value of Money principle: $100 received in Year 3 is worth
        # less than $100 today, because you could invest $100 today and earn
        # returns. The further in the future, the less it's worth now.
        #
        # Formula: PV = FV / (1 + WACC)^t
        #   where t = number of years from now.
        valuation_rows = []

        for idx, row in forecast_df.iterrows():
            f_year = int(row['Forecast Year'])
            raw_rev = float(row['Projected Revenue ($B)'])
            raw_fcff = float(row['Expected Future FCFF ($B)'])

            time_step = idx + 1  # t = 1, 2, 3, 4, 5

            # Discount factor = (1 + WACC)^t
            # As t increases, this grows exponentially, making the PV smaller.
            discount_factor = (1 + wacc_rate) ** time_step

            # Present Value = Future Cash Flow ÷ Discount Factor
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

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 4: Terminal Value — the "infinity" calculation
        # ═══════════════════════════════════════════════════════════════════
        # After the 5-year explicit forecast, we need to capture ALL future
        # value from Year 6 to infinity. We use the Gordon Growth Model:
        #
        #   TV = (FCFF_Year5 × (1 + g)) / (WACC − g)
        #
        # This formula assumes FCFF grows at a constant rate 'g' forever.
        # CRITICAL: g MUST be less than WACC, otherwise the formula produces
        # a negative or infinite value (which is economically nonsensical).

        # A. Sum the present values of the 5-year explicit window.
        explicit_pv_sum = summary_df['Present Value (PV) ($B)'].sum()

        # B. Terminal Value at Year 5 using the Gordon Growth Model.
        #    We take the LAST projected FCFF, grow it one more year by 'g',
        #    then capitalise it into a perpetuity.
        final_year_fcff = valuation_rows[-1]["Expected Future FCFF ($B)"] * 1e9  # Convert back to raw $
        terminal_value_year_5 = (final_year_fcff * (1 + PERPETUAL_GROWTH_RATE)) / (wacc_rate - PERPETUAL_GROWTH_RATE)

        # C. Discount Terminal Value back to today (Year 0).
        #    The terminal value occurs at Year 5, so we discount by (1 + WACC)^5.
        pv_of_terminal_value = (terminal_value_year_5 / 1e9) / ((1 + wacc_rate) ** 5)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 5: Enterprise Value → Equity Value → Share Price
        # ═══════════════════════════════════════════════════════════════════

        # D. Enterprise Value = PV(explicit period) + PV(terminal value)
        #    This represents the total value of the OPERATING business,
        #    owned jointly by equity holders and debt holders.
        enterprise_value = explicit_pv_sum + pv_of_terminal_value

        # E. Equity Value = Enterprise Value − Total Debt
        #    Debt holders get paid first; whatever remains belongs to
        #    equity shareholders.
        latest_debt_billions = latest_debt_raw / 1e9
        equity_value = enterprise_value - latest_debt_billions

        # F. Intrinsic Share Price = Equity Value / Outstanding Shares
        #    This is the model's answer to "what should one share be worth?"
        intrinsic_share_price = (equity_value * 1e9) / OUTSTANDING_SHARES

        # ═══════════════════════════════════════════════════════════════════
        # DISPLAY: The ultimate investment verdict
        # ═══════════════════════════════════════════════════════════════════
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