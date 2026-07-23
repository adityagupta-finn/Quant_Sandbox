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

import sys
import sqlite3
from pathlib import Path
import pandas as pd
import yfinance as yf

# Make sure dcf_forecasting resolves regardless of cwd or how this script
# was launched — relying on Python's implicit "script dir on sys.path[0]"
# behavior is fragile (e.g. it doesn't apply under `python -m`).
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the forecasting engine and data loader from Stage 2.
from dcf_forecasting import generate_fcff_projections, load_historical_cache

# ─── WACC & MARKET ENVIRONMENT PARAMETERS ─────────────────────────────────────
# Beta, market cap, and shares outstanding are now pulled live per-ticker via
# fetch_market_assumptions() below, instead of being hardcoded to Apple's
# numbers regardless of which company is being valued. DEFAULT_BETA is the
# only fallback *value* here — market cap and shares outstanding have no
# defensible universal fallback (a $50B company and a $2.8T company are not
# interchangeable), so fetch_market_assumptions() either derives them from
# other Yahoo fields or raises a clear error; it never invents a number.
DEFAULT_BETA = 1.0           # Market-neutral fallback when Yahoo reports no beta.
RISK_FREE_RATE = 0.04        # 4.0%  — Yield on 10-year US Treasury bonds (the "zero-risk" baseline)
EQUITY_RISK_PREMIUM = 0.05   # 5.0%  — Extra return stock investors demand over Treasuries
ASSUMED_COST_OF_DEBT = 0.045 # 4.5%  — Pre-tax interest rate the company pays on its borrowings
TAX_RATE = 0.21              # 21%   — US federal corporate tax rate

# ─── TERMINAL VALUE PARAMETERS ────────────────────────────────────────────────
# These control the "infinity" portion of the valuation.
PERPETUAL_GROWTH_RATE = 0.025 # 2.5% — Rate at which FCFF grows forever after Year 5
                               #        Typically set near long-term GDP or inflation (2-3%)


def fetch_market_assumptions(ticker):
    """
    Pull company-specific beta, market cap, and shares outstanding from
    yf.Ticker(ticker).info.

    Replaces the old hardcoded ASSUMED_BETA / MARKET_CAP_PROXY /
    OUTSTANDING_SHARES constants, which were silently calibrated to Apple
    and applied to every ticker regardless of company size — e.g. a $50B
    company would get Apple's $2.8T equity weight in its WACC.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., "AAPL", "TATASTEEL.NS").

    Returns
    -------
    dict
        Keys: beta, market_cap, shares_outstanding, currency, country.

    Raises
    ------
    ValueError
        If shares_outstanding can't be determined by any means. There is no
        defensible numeric fallback for a company's share count (unlike
        beta, where "assume market-average" is a standard convention) —
        without it there is no way to compute a per-share price, so this
        fails loudly instead of dividing by a fabricated number.

    Notes
    -----
    - beta: falls back to DEFAULT_BETA (market-neutral) with a warning if
      Yahoo doesn't report one.
    - market_cap: falls back to sharesOutstanding × currentPrice if Yahoo's
      marketCap field is missing; raises if neither is available.
    - shares_outstanding: Yahoo's `sharesOutstanding` field is frequently
      None even for well-covered tickers, so this tries
      impliedSharesOutstanding and floatShares as alternates before falling
      back to marketCap ÷ currentPrice; raises only if all of those fail.
    """
    stock = yf.Ticker(ticker)
    info = stock.info or {}

    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    # ── Beta ───────────────────────────────────────────────────────────────
    beta = info.get("beta")
    if beta is None:
        print(f"⚠️  {ticker}: Yahoo reports no beta. Falling back to DEFAULT_BETA={DEFAULT_BETA} (market-neutral).")
        beta = DEFAULT_BETA

    # ── Market Cap (equity weight in WACC) ───────────────────────────────────
    market_cap = info.get("marketCap")
    if market_cap is None:
        fallback_shares = info.get("sharesOutstanding")
        if fallback_shares and current_price:
            market_cap = fallback_shares * current_price
            print(f"⚠️  {ticker}: Yahoo reports no marketCap. Derived {market_cap:,.0f} from "
                  f"sharesOutstanding × currentPrice instead.")
        else:
            raise ValueError(
                f"{ticker}: Yahoo reports no marketCap, and there isn't enough data "
                f"(sharesOutstanding, currentPrice) to derive one. Cannot compute the "
                f"equity weight for WACC without it."
            )

    # ── Shares Outstanding (denominator for per-share price) ────────────────
    # sharesOutstanding is frequently None on Yahoo even when other fields
    # are populated — hence the fallback chain rather than a single lookup.
    shares_outstanding = (
        info.get("sharesOutstanding")
        or info.get("impliedSharesOutstanding")
        or info.get("floatShares")
    )
    if shares_outstanding is None and market_cap and current_price:
        shares_outstanding = market_cap / current_price
        print(f"⚠️  {ticker}: Yahoo reports no sharesOutstanding, impliedSharesOutstanding, or "
              f"floatShares. Derived {shares_outstanding:,.0f} from marketCap ÷ currentPrice instead.")
    if shares_outstanding is None:
        raise ValueError(
            f"{ticker}: Yahoo reports no sharesOutstanding, impliedSharesOutstanding, or "
            f"floatShares, and there isn't enough data (marketCap, currentPrice) to derive one. "
            f"Cannot compute intrinsic value per share without a share count."
        )

    return {
        "beta": beta,
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
        "currency": info.get("financialCurrency") or info.get("currency"),
        "country": info.get("country"),
    }


def calculate_live_wacc(ticker, market_assumptions):
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
    market_assumptions : dict
        Output of fetch_market_assumptions(ticker) — supplies beta and
        market_cap. Passed in rather than fetched again here so
        run_master_valuation_app() only hits the yfinance .info endpoint once.

    Returns
    -------
    tuple[float, float]
        (wacc, latest_debt_raw)
        - wacc : float — The computed WACC as a decimal fraction (e.g., 0.09 = 9%).
        - latest_debt_raw : float — The most recent total debt, in the
          company's own reporting currency (see market_assumptions["currency"]).

    Notes
    -----
    - The equity weight uses market_assumptions["market_cap"], pulled live
      from Yahoo per-ticker — see fetch_market_assumptions() for how it's
      derived and what happens when Yahoo doesn't report it directly.
    - The debt weight reads the latest totalDebt from the cached database,
      so it reflects the actual debt reported by Yahoo for this company.
    - Market cap and total debt are assumed to be in the same currency
      (both come from Yahoo's data for this ticker) — WACC itself is a
      dimensionless blend of rates, so currency doesn't affect it directly,
      but the caller must not mix market_assumptions from one ticker with
      hist_df from another.
    """
    hist_df = load_historical_cache(ticker)

    # ── Step 1: Cost of Equity via CAPM ──────────────────────────────────
    # Ke = Rf + β × ERP
    cost_of_equity = RISK_FREE_RATE + (market_assumptions["beta"] * EQUITY_RISK_PREMIUM)

    # ── Step 2: Read the latest total debt from the database ─────────────
    latest_debt = float(hist_df['totalDebt'].iloc[-1])

    # ── Step 3: Compute capital structure weights ────────────────────────
    # V = E + D (total value of all capital)
    market_cap = market_assumptions["market_cap"]
    total_capital = market_cap + latest_debt
    weight_of_equity = market_cap / total_capital  # E/V
    weight_of_debt = latest_debt / total_capital     # D/V

    # ── Step 4: After-tax Cost of Debt ───────────────────────────────────
    # Interest payments are tax-deductible, creating a "tax shield".
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
        # PHASE 0: Pull company-specific market assumptions (beta, market
        # cap, shares outstanding) — replaces the old Apple-hardcoded
        # constants. See fetch_market_assumptions()'s docstring for the
        # fallback chain and when it raises instead of guessing.
        # ═══════════════════════════════════════════════════════════════════
        market_assumptions = fetch_market_assumptions(user_ticker)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: Generate FCFF projections (calls Stage 2)
        # ═══════════════════════════════════════════════════════════════════
        forecast_df, cagr, margin = generate_fcff_projections(user_ticker)

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: Compute WACC (the discount rate)
        # ═══════════════════════════════════════════════════════════════════
        wacc_rate, latest_debt_raw = calculate_live_wacc(user_ticker, market_assumptions)

        # ── Print the computed model parameters ──────────────────────────
        print(f"\n📈 INITIALIZING VALUATION FOR STRUCTURE: {user_ticker}")
        print(f" -> Beta: {market_assumptions['beta']:.2f}  |  Market Cap: {market_assumptions['market_cap']:,.0f}  "
              f"|  Shares Outstanding: {market_assumptions['shares_outstanding']:,.0f}")
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
        intrinsic_share_price = (equity_value * 1e9) / market_assumptions["shares_outstanding"]

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