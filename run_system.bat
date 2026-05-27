@echo off
title Pairs Arbitrage Engine Launcher
cd /d "C:\Users\adity\Desktop\New folder\Quant_Sandbox"

echo [1/2] Running Statistical Calculation Engine...
.\venv\Scripts\python.exe project_1_pairs/zscore_calculator.py

echo [2/2] Launching Interactive Control Interface...
.\venv\Scripts\python.exe -m streamlit run project_1_pairs/dashboard.py
pause