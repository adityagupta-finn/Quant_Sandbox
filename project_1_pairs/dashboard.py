import streamlit as st
import sqlite3
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import subprocess

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "market_data.db"
ENGINE_PATH = BASE_DIR / "engine"

# --- 1. ARCHITECTURAL CONFIGURATION ---
st.set_page_config(page_title="Pairs Arbitrage & Risk Engine", layout="wide")

# Institutional Slate-Dark Aesthetic via Custom CSS Injection
st.markdown("""
    <style>
        .reportview-container { background: #0b0f19; }
        div[data-testid="stMetricValue"] { font-size: 32px; font-weight: 600; color: #10b981; font-family: 'Courier New', monospace; }
        .stButton>button { background-color: #2563eb; color: white; font-weight: 500; width: 100%; border-radius: 4px; border: none; }
        .stButton>button:hover { background-color: #1d4ed8; color: white; }
        code { background-color: #1e293b !important; color: #f8fafc !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Cross-Asset Pairs Arbitrage & Risk Engine (CAPRE)")
st.caption("Low-latency time-series analysis engine and compiled bare-metal execution interface.")
st.markdown("---")

# --- 2. RISK SPECIFICATION & TELEMETRY SIDEBAR ---
st.sidebar.header("🔧 Risk Parameters & Constraints")

# Operational Inputs
z_threshold = st.sidebar.slider("Z-Score Entry Threshold (σ)", min_value=1.5, max_value=3.0, value=2.0, step=0.1)
enforce_coin = st.sidebar.toggle("Enforce Cointegration Constraints", value=True)

# Establish connection to the local database cluster
conn = sqlite3.connect(DB_PATH)
try:
    # Fetch time-series array ordered chronologically
    df = pd.read_sql("SELECT * FROM pairs_data ORDER BY Date ASC", conn)
    latest_row = df.iloc[-1]
    
    # Render Telemetry inside Sidebar Panel
    st.sidebar.markdown("---")
    st.sidebar.header("📈 Model Statistics")
    st.sidebar.metric(label="Dynamic OLS Hedge Ratio (β)", value=f"{latest_row['Beta']:.4f}")
    
    # Statistical Significance Monitor
    p_val = latest_row['ADF_P_Value']
    if p_val < 0.05:
        st.sidebar.success(f"ADF p-value: {p_val:.4f}\n[Stationary Regime]")
    else:
        st.sidebar.error(f"ADF p-value: {p_val:.4f}\n[Non-Stationary Warning]")
except Exception as e:
    st.sidebar.warning("Awaiting market data population...")
    df = pd.DataFrame()

# --- 3. CORE METRICS DISPLAY ---
if not df.empty:
    m_col1, m_col2, m_col3 = st.columns(3)
    
    with m_col1:
        st.metric(label="Current Residual Spread Value", value=f"${latest_row['Spread']:.4f}")
    with m_col2:
        z_val = latest_row['Z_Score']
        if abs(z_val) >= z_threshold:
            st.metric(label="Standardized Z-Score", value=f"{z_val:.4f}", delta="BOUNDARY BREACH", delta_color="inverse")
        else:
            st.metric(label="Standardized Z-Score", value=f"{z_val:.4f}", delta="Within Bounds", delta_color="normal")
    with m_col3:
        st.metric(label="Historical Sample Size (T)", value=f"{len(df)} Days")

    st.markdown("---")

    # --- 4. INTERACTIVE DISPERSION & REGIME CHARTING ---
    st.subheader("Linear Regression Residual Spread & Boundary Conditions")
    
    fig = go.Figure()

    # Dynamic boundary vectors based on user-defined threshold multiplier
    df['Upper_Boundary'] = df['Rolling_Mean'] + (z_threshold * df['Rolling_Std'])
    df['Lower_Boundary'] = df['Rolling_Mean'] - (z_threshold * df['Rolling_Std'])

    # Time-series plots
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Spread'], name='OLS Residual Spread (ε)', line=dict(color='#2563eb', width=2)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper_Boundary'], name='Upper Execution Boundary (+σ)', line=dict(color='#ef4444', width=1.5, dash='dash')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower_Boundary'], name='Lower Execution Boundary (-σ)', line=dict(color='#06b6d4', width=1.5, dash='dash')))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#111827",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(showgrid=True, gridcolor='#1f2937'),
        yaxis=dict(showgrid=True, gridcolor='#1f2937')
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 5. AUDITABLE RECENT DATA LEDGER ---
    st.subheader("Data Audit Ledger (Most Recent Observations)")
    # Render the last 5 rows of data in a clean data grid
    audit_df = df.tail(5)[['Date', 'AAPL_Close', 'MSFT_Close', 'Beta', 'Spread', 'Z_Score', 'ADF_P_Value']]
    st.dataframe(audit_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    # --- 6. SUBPROCESS EXECUTION CONSOLE ---
    st.subheader("Compiled Core Execution Pipeline")
    
    btn_col, spacer_col = st.columns([1, 3])
    with btn_col:
        trigger_scan = st.button("🔧 EXECUTE RISK ENGINE")
        
    if trigger_scan:
        st.markdown("**Core Binary Run-time Output Stream:**")
        
        # Invoke compiled C++ module with string-vector arguments (argv)
        process = subprocess.Popen(
            [str(ENGINE_PATH), str(DB_PATH), str(z_threshold), str(enforce_coin)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if stdout:
            st.code(stdout, language="text")
        if stderr:
            st.error(stderr)

else:
    st.info("System database table 'pairs_data' is unpopulated. Run calculation script to initialize.")

if 'conn' in locals():
    conn.close()