# Quantitative Finance Sandbox

Three independent personal projects: a pairs-trading signal generator
(Python + C++), a DCF valuation tool (Python), and a project finance
debt-sizing model (Python). All are learning projects, not production
trading, investment, or lending systems — see the Limitations section
under each before trusting any output.

---

## Project 1: Pairs Trading Signal Generator (`/project_1_pairs`)

Computes a rolling OLS hedge ratio between two equity tickers, tests
whether the resulting spread is cointegrated (Augmented Dickey-Fuller
test), and generates a mean-reversion signal from the spread's Z-score.

- **Data ingestion** (`alpaca_ingestion.py`): pulls 2 years of daily
  closes for two tickers from the Alpaca API into a local SQLite database.
- **Signal calculation** (`zscore_calculator.py`): rolling OLS beta
  (`statsmodels.OLS`, 60-day window) between the pair, the resulting
  spread, an ADF cointegration test on that spread, and a rolling Z-score:
  `Z = (spread − rolling_mean) / rolling_std`. If the ADF test doesn't
  reject the unit-root null (p ≥ 0.05), the pipeline halts and does not
  write a signal table — a non-cointegrated pair has no statistical basis
  for a mean-reversion trade.
- **Execution engine** (`engine.cpp`): reads the latest Z-score from
  SQLite and prints LONG/SHORT/HOLD text based on a Z-score threshold
  (default ±2.0, overridable as a second command-line argument — the
  dashboard's slider drives this directly). It does not place orders,
  track positions, or compute P&L — it prints a recommendation.
- **Dashboard** (`dashboard.py`): Streamlit UI showing the spread chart,
  hedge ratio, ADF result, and a button to run the C++ engine.

The ticker pair defaults to **GLD/GDX** (gold bullion vs. gold miners)
and is configurable via the `PAIRS_TICKER_A` / `PAIRS_TICKER_B`
environment variables.

**Why GLD/GDX, and why it can still halt.** Of AAPL/MSFT plus six classic
candidates (KO/PEP, XOM/CVX, V/MA, HD/LOW, UPS/FDX, GLD/GDX) tested
against real 2-year daily data, GLD/GDX is the only one with genuine
full-history cointegration (Engle-Granger ADF on the full 2-year spread:
p=0.0040 — GDX's gold-miner earnings are structurally tied to the gold
price GLD tracks). The others show no meaningful relationship at
full history either (all p > 0.11).

But `zscore_calculator.py`'s gate doesn't test full history — it
re-estimates beta fresh in every 60-day window and tests only the single
most recent one, which is a stricter, noisier test than full-history
cointegration. A pair can be soundly cointegrated over 2 years and still
have its latest 60-day slice come up non-stationary. At the time this was
written, GLD/GDX itself fails that specific test (p=0.8203 on the latest
window) — it was chosen as the default because it's a real candidate that
will sometimes pass, not because it passes on every run.

**A halt is intended behaviour, not a failure.** If `zscore_calculator.py`
prints "Cointegration test failed" and stops, the ADF test did its job:
the current window's spread isn't statistically mean-reverting, so there's
no valid basis for a signal right now. AAPL/MSFT — kept as the worked
counter-example of two large-caps with no structural relationship at all —
halts even more decisively (rolling: p≈0.88, full-history: p≈0.85).

### Limitations

- **No backtest.** There is no historical simulation of entries, exits,
  or returns anywhere in this project. All of the above operates on the
  latest data point only.
- **No P&L or position tracking.** The C++ engine prints a signal; it
  does not simulate or track what a position following that signal would
  have earned or lost.
- **Single-snapshot only.** Every run evaluates the current state of the
  data, not a track record across time.

---

## Project 2: DCF Valuation Pipeline (`/project_2_dcf`)

Pulls a company's income statement, balance sheet, and cash flow
statement via `yfinance`, projects 5 years of Free Cash Flow to Firm from
historical averages, computes a WACC from live per-ticker market data,
and discounts the projection plus a terminal value to an intrinsic
share price.

- **Ingestion** (`dcf_ingestion.py`): fetches and caches the three
  statements in SQLite (`corporate_data.db`), one row per fiscal year.
- **Forecasting** (`dcf_forecasting.py`): computes revenue CAGR and
  average operating margin / D&A / CapEx ratios from history, then
  projects `FCFF = NOPAT + D&A − CapEx − ΔNWC` forward 5 years. ΔNWC is a
  flat 1% of revenue assumption, not derived from actual working-capital
  changes. Fiscal years with missing data (Yahoo Finance commonly has
  gaps in the oldest reported year) are dropped before the CAGR
  calculation, with a warning printed for each one dropped.
- **Valuation** (`dcf_valuation.py`): computes WACC via CAPM using the
  ticker's live beta, market cap, and debt from `yfinance`, resolves
  tax rate / risk-free rate / equity risk premium from the ticker's own
  jurisdiction (`United States` and `India` currently have dedicated
  defaults; anything else falls back to the US numbers with a loud
  warning), discounts the 5-year forecast plus a Gordon Growth terminal
  value, and divides by shares outstanding. All monetary output is
  labeled with the company's actual reporting currency, not assumed USD.

### Limitations

- **Flat historical assumptions.** CAGR and operating margin are single
  averages from the last few years of history, projected forward
  unchanged. This fits a stable business reasonably well and fits a
  cyclical or volatile one (commodity producers, for example) poorly —
  a down-cycle in the historical window will project as permanent
  decline, and vice versa.
- **Single-point WACC, no sensitivity analysis.** The model produces one
  WACC and one intrinsic share price from one set of assumptions. There
  is no sensitivity table across WACC or terminal growth rate, and no
  EV/Revenue or other multiple-based cross-check.
- **Only two jurisdictions have tax/risk-free/ERP defaults.** Every other
  country falls back to US-calibrated numbers (with a warning) unless you
  pass `tax_rate` / `risk_free_rate` / `equity_risk_premium` explicitly.
- **Depends entirely on yfinance data quality.** Missing years, renamed
  line items, and inconsistent coverage across tickers are real and
  observed (not hypothetical) failure modes; see `dcf_forecasting.py`'s
  handling of incomplete fiscal years.

---

## Project 3: Solar SPV Debt-Sizing Model (`/project_3_projectfinance`)

Sizes a senior term loan for a 200MW solar SPV in India from deal
assumptions alone — no market data, no `yfinance`, nothing pulled live.
Given a tariff, capex range, generation profile, and target DSCR, it works
out how large a loan the project's own cash flows can support, shapes a
repayment schedule that holds DSCR flat at target rather than
straight-line, and reports both the unlevered Project IRR and the levered
Equity IRR.

- **Assumptions** (`pf_assumptions.py`): every locked input constant —
  capacity, tariff, capex/MW, CUF, D:E ratio, target DSCR, tenor,
  moratorium, tax rate, depreciation rate, O&M, degradation, loan rate —
  in one place, imported by every other module.
- **Construction** (`pf_construction.py`): monthly capex drawdown over the
  build period with interest during construction (IDC) capitalized into
  the loan balance. The debt/equity split of each month's draw is solved
  by bisection so the *final* debt balance (draws plus capitalized IDC)
  comes out to exactly 75% of total project cost, not just 75% of the
  draws themselves.
- **Operations** (`pf_operations.py`): the debt-independent annual series
  for all 15 years post-COD — generation (with panel degradation),
  revenue, opex (with escalation), and WDV tax depreciation.
- **Debt sizing** (`pf_debt_sizing.py`): bisection search on loan principal
  so a sculpted (DSCR-flat, not straight-line) repayment schedule fully
  amortizes by year 15 at exactly the target DSCR. That DSCR-based cap is
  compared against the 75% leverage cap from construction; whichever is
  lower governs the actual loan.
- **Returns** (`pf_returns.py`): Project IRR (unlevered, post-tax, no
  interest shield) and Equity IRR (post-debt-service), each solved two
  independent ways (Newton-Raphson and bisection) and cross-checked to
  agree.
- **Sensitivity** (`pf_sensitivity.py`): one-at-a-time tornado grid over
  CUF, tariff, capex overrun, and COD delay, re-evaluated against the
  *already-sized* base-case debt schedule rather than resizing debt per
  scenario.
- **Orchestrator** (`pf_model.py`): runs the base case end-to-end and
  prints every stage — construction convergence, debt-sizing convergence,
  the annual schedule, returns, and the sensitivity tornado.

**Base case result.** Sanctioned debt ₹499.4 Cr — the DSCR cap binds, not
the 75% leverage cap (₹570.1 Cr), meaning the deal is more conservatively
geared (~65.7%) than the headline 75:25 target actually allows. Project
IRR 7.07% (unlevered), Equity IRR 3.46% (levered) — a **−3.61 percentage
point leverage effect**. Leverage is negative here because the 10% loan
rate sits above the project's own unlevered return; the interest tax
shield helps equity but isn't enough to flip the sign.

Run it (no credentials needed, no cached data to build first):

```bash
python project_3_projectfinance/pf_model.py
```

### Limitations

- **Moratorium is modeled as a full interest-only year, not a literal 6
  months.** Every cash flow in this model runs on annual periods; the
  actual deal term is a 6-month grace. Approximated as "no principal due
  in Year 1" rather than introducing a semi-annual convention for just
  one year — slightly more generous to the SPV than the literal term.
- **No half-year depreciation convention.** Section 32 halves first-year
  depreciation to 20% if COD falls after October 3rd of that fiscal year,
  and allows an additional 20% first-year allowance for power-generating
  companies — both real, both confirmed still current, neither modeled.
  Depreciation is a flat 40% WDV every year regardless of COD timing,
  which understates the tax shield in a delayed-COD scenario.
- **Loan rate is a single flat number, not a rate path.** 10% for the
  life of the loan; no benchmark-linked or floating-rate modeling, so the
  model can't represent a loan that reprices with policy rates over its
  15-year tenor.
- **Sensitivity assumes cost overruns are 100% equity-funded.** The
  tornado grid freezes the base-case debt schedule and re-evaluates DSCR
  and Equity IRR against it; a capex overrun or COD delay raises total
  project cost, but since the loan doesn't grow with it, that entire
  excess is funded by additional equity. This is why capex overrun and
  COD delay move only Equity IRR in the sensitivity output and never
  DSCR — a deliberate design choice (debt service doesn't get worse under
  a fixed-size sanctioned loan), not a gap in the model.

---

## Tech Stack

- **Languages:** Python, C++
- **Python:** pandas, numpy, statsmodels, yfinance, streamlit, plotly,
  alpaca-py, python-dotenv — see `requirements.txt` for pinned versions.
- **Storage:** SQLite (projects 1 and 2 cache to a local `.db` file);
  project 3 has no persistence layer — it's a pure computation over deal
  assumptions, nothing to cache.
- **Requires:** Python 3.12+ (developed on 3.14).

---

## Build & Run (macOS)

```bash
# One-time setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Project 1's C++ engine
clang++ -std=c++17 project_1_pairs/engine.cpp -o project_1_pairs/engine -lsqlite3
```

**Project 1** needs Alpaca API credentials in `project_1_pairs/.env`:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
```

Then either run `./run_system.sh` (builds the engine if needed, runs the
signal calculation, and launches the dashboard), or step through it
manually:

```bash
python project_1_pairs/alpaca_ingestion.py
python project_1_pairs/zscore_calculator.py
streamlit run project_1_pairs/dashboard.py
```

Override the default AAPL/MSFT pair with `PAIRS_TICKER_A=GOOGL
PAIRS_TICKER_B=AMZN python project_1_pairs/alpaca_ingestion.py` (and the
same env vars for `zscore_calculator.py`).

**Project 2** needs no credentials (yfinance is unauthenticated):

```bash
python project_2_dcf/dcf_ingestion.py    # prompts for a ticker, prints an audit table
python project_2_dcf/dcf_valuation.py    # prompts for a ticker, prints the full valuation
```

**Project 3** needs no credentials and no prior ingestion step — every
assumption is a module-level constant in `pf_assumptions.py`:

```bash
python project_3_projectfinance/pf_model.py    # runs the base case + sensitivity, prints the full report
```

---

### Developed By
**Aditya Gupta**
B.E. (Hons.) Electrical & Electronics Engineering & M.Sc. (Hons.) Economics
Birla Institute of Technology & Science (BITS), Pilani - Goa Campus
