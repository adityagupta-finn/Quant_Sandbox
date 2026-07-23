"""
pair_config.py — Shared ticker-pair configuration for project_1_pairs.

Reads the two tickers to trade from environment variables
(PAIRS_TICKER_A, PAIRS_TICKER_B), defaulting to AAPL/MSFT. Validated
against a strict whitelist since these values get used to build SQL table
and column names (e.g. "AAPL_Close", table "AAPL") in zscore_calculator.py.

Alpaca (this project's data source) only covers US equities, so the
allowed format is simpler than project_2_dcf's ticker_utils.py — no '.'
exchange suffixes or '^' index prefixes needed here.
"""

import os
import re

TICKER_PATTERN = re.compile(r'^[A-Z0-9]{1,10}$')


def _resolve_ticker(env_var, default):
    ticker = os.getenv(env_var, default).strip().upper()
    if not TICKER_PATTERN.match(ticker):
        raise ValueError(
            f"Invalid ticker in {env_var}={ticker!r}: only letters and digits are allowed "
            f"(max 10 characters)."
        )
    return ticker


TICKER_A = _resolve_ticker("PAIRS_TICKER_A", "AAPL")
TICKER_B = _resolve_ticker("PAIRS_TICKER_B", "MSFT")
