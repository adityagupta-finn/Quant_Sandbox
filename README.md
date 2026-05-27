# Quantitative Finance & Algorithmic Trading Sandbox

This repository serves as a monorepo for advanced quantitative finance applications, bridging automated data engineering pipelines with low-latency statistical arbitrage frameworks. 

The architecture leverages a hybrid **Python/C++** stack to optimize both exploratory data analysis (EDA) and computationally intensive financial modeling.

---

## 📂 Project 1: Quantitative Pairs Trading Engine (`/project_1_pairs`)
A dual-language statistical arbitrage system designed for deterministic spread computation, signal generation, and parameterized backtesting across equity universes.

### System Architecture
* **Analytics & Visualization Layer:** Python (Pandas, `yfinance`, Streamlit, Plotly)
* **Execution Backend:** Compiled C++ invoked via Python `subprocess` orchestration to ensure low-latency spread computation.

### Mathematical Framework
1. **Dynamic Hedge Ratios:** Computes time-varying Beta ($\beta$) coefficients between equity pairs using rolling Ordinary Least Squares (OLS) regressions across configurable lookback windows.
2. **Cointegration Diagnostics:** Enforces mean-reversion confirmation by applying the Augmented Dickey-Fuller (ADF) unit root test on the residual spread series, gating signal generation at strict stationarity thresholds ($p < 0.05$).
3. **Signal Generation:** Normalizes the spread series into a Z-score and renders it on an interactive dashboard. Entry and exit signals are mathematically triggered at dynamic standard deviation thresholds:
   
   $$ Z = \frac{X - \mu}{\sigma} $$
   *(Positions are entered at $\pm2\sigma$ and exited upon mean reversion to $0\sigma$)*

---

## 📂 Project 2: Automated Intrinsic Valuation (DCF) Pipeline (`/project_2_dcf`)
An automated 3-statement financial modeling pipeline built to programmatically scrape, cache, and execute Discounted Cash Flow (DCF) valuations at scale.

### System Architecture
* **Data Ingestion:** Python (`yfinance`) with a multi-layer structural validation framework.
* **Caching Layer:** Integrated SQLite database to persist historical Income Statement, Balance Sheet, and Cash Flow data, minimizing redundant API calls and mitigating valuation latency.

### Mathematical Framework
1. **Free Cash Flow to Firm (FCFF):** Programmatically projects the core operating cash flows across a 5-year forward horizon:
   
   $$ FCFF = NOPAT + D\&A - CapEx - \Delta NWC $$

2. **Cost of Capital (WACC & CAPM):** Derives dynamic discount rates by computing the Weighted Average Cost of Capital (WACC), utilizing the Capital Asset Pricing Model (CAPM) to determine the baseline cost of equity:
   
   $$ K_e = R_f + \beta(R_m - R_f) $$

3. **Terminal Value Estimation:** Calculates the continuing value of the asset beyond the explicit forecast period using the Gordon Growth Model:
   
   $$ TV = \frac{FCFF_{t+1}}{WACC - g} $$

4. **Scenario Analysis:** Supports dynamic stress-testing of implied market premiums/discounts via EV/Revenue sensitivity scenarios (e.g., modeling M&A sponsor exit valuations).

---
*Developed by Aditya Gupta | B.E. Electrical & Electronics Engineering & M.Sc. Economics*
