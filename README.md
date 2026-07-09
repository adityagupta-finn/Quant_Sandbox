# Quantitative Finance & Algorithmic Trading Sandbox

This monorepo serves as an advanced quantitative finance environment, bridging automated data engineering pipelines with low-latency statistical arbitrage frameworks. The architecture leverages a hybrid Python/C++ execution model to optimize both rapid exploratory data analysis (EDA) and computationally intensive financial modeling.

---

## 📂 Project 1: Quantitative Pairs Trading Engine (`/project_1_pairs`)

A dual-language statistical arbitrage engine engineered for deterministic spread computation, real-time signal generation, and parameterized backtesting across equity universes.

### 🏗️ System Architecture
*   **Analytics & Visualization Layer:** Python (`Pandas`, `yfinance`, `Streamlit`, `Plotly`) for real-time dashboard rendering, pipeline orchestration, and interactive charting.
*   **Execution Backend:** Native compiled C++ binaries invoked via Python `subprocess` orchestration to optimize latency and accelerate vector math operations during rolling spread calculations.

### 𝖬 Math & Algorithmic Framework
*   **Dynamic Hedge Ratios:** Computes time-varying Beta ($\beta$) coefficients between equity pairs using rolling Ordinary Least Squares (OLS) regressions across configurable historical lookback windows.
*   **Cointegration Diagnostics:** Enforces mean-reversion confirmation by executing the Augmented Dickey-Fuller (ADF) unit root test on residual spread series, gating signal generation at strict stationarity thresholds ($p < 0.05$).
*   **Signal Generation:** Normalizes the tracking spread into a running $Z$-score, dynamically triggering entry signals at $\pm2\sigma$ and exit parameters upon mean reversion to $0\sigma$:
  $$Z = \frac{X - \mu}{\sigma}$$

---

## 📂 Project 2: Automated Intrinsic Valuation (DCF) Pipeline (`/project_2_dcf`)

An end-to-end, automated 3-statement financial modeling pipeline built to programmatically scrape raw financials, cache structured historical data, and execute Discounted Cash Flow (DCF) corporate valuations at scale.

### 🏗️ System Architecture
*   **Data Ingestion & Verification:** Python pipeline utilizing `yfinance` with a structured payload-validation layer to parse asymmetric financial statements.
*   **Caching Layer:** Local `SQLite` storage architecture designed to persist historical Income Statements, Balance Sheets, and Cash Flow Statements—drastically reducing external API network dependency and mitigating evaluation runtime latency.

### 𝖬 Math & Valuation Framework
*   **Free Cash Flow to Firm (FCFF):** Programmatically projects core operating cash flows across a 5-year explicit forward horizon using normalized historical drivers:
  $$FCFF = NOPAT + \text{D \amp A} - CapEx - \Delta NWC$$
*   **Cost of Capital (WACC & CAPM):** Derives dynamic discount rates by calculating the Weighted Average Cost of Capital (WACC), applying the Capital Asset Pricing Model (CAPM) to evaluate equity risk premiums:
  $$K_e = R_f + \beta(R_m - R_f)$$
*   **Terminal Value Estimation:** Computes the continuing enterprise value beyond the explicit forecast period using the Gordon Growth Model:
  $$TV = \frac{FCFF_{t+1}}{WACC - g}$$
*   **Scenario Stress-Testing:** Implements discrete sensitivity matrices across target valuation assumptions (e.g., WACC variations vs. terminal growth rates) alongside EV/Revenue exit multiples to model M&A sponsor returns.

---

## 🛠️ Monorepo Tech Stack

*   **Languages:** Python, C++
*   **Data Science & Visualization:** Pandas, NumPy, SciPy, Streamlit, Plotly
*   **Storage & Ingestion:** SQLite, yfinance, Requests, BeautifulSoup

---

### 🧑‍💻 Developed By
**Aditya Gupta**  
*B.E. (Hons.) Electrical & Electronics Engineering & M.Sc. (Hons.) Economics*  
Birla Institute of Technology & Science (BITS), Pilani - Goa Campus
