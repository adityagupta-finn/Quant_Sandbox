# Quantitative Finance Sandbox

Two independent personal projects: a pairs-trading signal generator
(Python + C++) and a DCF valuation tool (Python). Both are learning
projects, not production trading or investment systems — see the
Limitations section under each before trusting any output.

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

## Tech Stack

- **Languages:** Python, C++
- **Python:** pandas, numpy, statsmodels, yfinance, streamlit, plotly,
  alpaca-py, python-dotenv — see `requirements.txt` for pinned versions.
- **Storage:** SQLite (both projects cache to a local `.db` file).
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

---

### Developed By
**Aditya Gupta**
B.E. (Hons.) Electrical & Electronics Engineering & M.Sc. (Hons.) Economics
Birla Institute of Technology & Science (BITS), Pilani - Goa Campus
