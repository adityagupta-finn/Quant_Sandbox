"""
pair_config.py — Shared ticker-pair configuration for project_1_pairs.

Reads the two tickers to trade from environment variables
(PAIRS_TICKER_A, PAIRS_TICKER_B), defaulting to GLD/GDX. Validated
against a strict whitelist since these values get used to build SQL table
and column names (e.g. "GLD_Close", table "GLD") in zscore_calculator.py.

GLD/GDX (gold bullion vs. gold miners) was chosen as the default because
it's the one pair, of several tested against real 2-year daily data, with
genuine full-history cointegration (Engle-Granger ADF p=0.0040) rather
than an arbitrary pair with no structural relationship. It can still fail
the rolling 60-day gate on any given day, since that gate re-estimates
beta fresh each window and tests only the latest one — a stricter, noisier
test than full-history cointegration. See the README for the tradeoff.

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


TICKER_A = _resolve_ticker("PAIRS_TICKER_A", "GLD")
TICKER_B = _resolve_ticker("PAIRS_TICKER_B", "GDX")
