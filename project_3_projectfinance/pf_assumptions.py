"""
pf_assumptions.py - deal assumptions for the 200MW solar SPV. Single
source of truth for every other module; see docs/project_3_projectfinance.md
for reasoning behind the non-obvious values.
"""

# --- Plant ---
CAPACITY_MW = 200.0
CUF_P90 = 0.24               # P90 capacity utilization factor
DEGRADATION_RATE = 0.005     # annual panel output degradation

# --- Revenue ---
TARIFF_PER_KWH = 2.86        # INR/kWh, flat PPA tariff, no escalation assumed

# --- Capex ---
# Quoted market range for 2026 fixed-tilt utility-scale EPC is INR 3.5-4.0
# Cr/MW. Base case uses 3.65 Cr/MW (lower-middle of the range - a 200MW
# project is large enough to see some scale discount off the top of the
# band). Sensitivity flexes this with an overrun percentage on top.
CAPEX_PER_MW_CR = 3.65
CONSTRUCTION_MONTHS = 12

# --- Opex ---
# INR 5-10 lakh/MW/year is the observed range for Indian utility-scale
# O&M; 8 lakh/MW/year is used as a deliberately conservative (higher-cost)
# base case rather than the range's midpoint.
OM_COST_PER_MW_LAKH = 8.0
OM_ESCALATION = 0.03          # annual O&M cost escalation

# --- Capital structure ---
DEBT_RATIO = 0.75
EQUITY_RATIO = 0.25

# --- Debt terms ---
LOAN_INTEREST_RATE = 0.10    # flat all-in rate, senior rupee term loan
TENOR_YEARS = 15             # full loan life from COD, includes moratorium
# Deal term is a 6-month moratorium. Operating cash flows are modeled on
# annual periods, so the moratorium is approximated as a full interest-only
# Year 1 rather than splitting Year 1 into two six-month debt-service
# periods. Sculpted principal repayment then runs Years 2-15 (14 years).
MORATORIUM_YEARS = 1
TARGET_MIN_DSCR = 1.30

# --- Tax ---
TAX_RATE = 0.2517             # effective corporate rate: 22% base + surcharge + cess
DEPRECIATION_RATE_WDV = 0.40  # Section 32 flat WDV rate, no half-year convention applied

# --- Unit conversions ---
CR_TO_INR = 1e7
LAKH_TO_INR = 1e5
HOURS_PER_YEAR = 8760
