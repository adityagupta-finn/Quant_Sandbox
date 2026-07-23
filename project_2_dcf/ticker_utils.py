"""
ticker_utils.py — Shared ticker validation and SQL identifier quoting
=======================================================================

dcf_ingestion.py and dcf_forecasting.py both build per-ticker SQLite table
names (e.g. "TATASTEEL.NS_income") directly from user-supplied ticker
strings. Two independent problems fall out of that:

  1. Ticker symbols containing "." (the norm for non-US exchanges, e.g.
     "TATASTEEL.NS", "0700.HK") break unquoted SQL identifiers, because
     SQLite parses an unquoted "." as a schema.table separator rather than
     a literal character.
  2. Building SQL strings directly from user input (the ticker is typed in
     via input()) is a textbook SQL injection point — nothing currently
     stops a "ticker" like `AAPL"; DROP TABLE foo; --` from reaching a
     read_sql()/to_sql() call.

validate_ticker() closes the injection hole with a strict whitelist (only
characters that legitimately appear in real ticker symbols are accepted).
quoted_table_name() closes the dot-parsing bug by double-quoting the
resulting SQL identifier. Callers should always validate before quoting —
quoting alone is defense in depth, not a substitute for validation.
"""

import re

# Real ticker symbols across exchanges: letters, digits, '.' (exchange
# suffixes like ".NS", ".HK"), '-' (share classes like "BRK-B"), and '^'
# (indices like "^GSPC"). Nothing else is legitimate, so nothing else is
# allowed through. Max length is generous for the longest realistic
# exchange-qualified symbol.
TICKER_PATTERN = re.compile(r'^[A-Z0-9\^][A-Z0-9.\-\^]{0,14}$')


def validate_ticker(ticker):
    """
    Validate a ticker symbol before it's used to build a SQL identifier.

    Parameters
    ----------
    ticker : str

    Returns
    -------
    str
        The same ticker, unchanged, if it passes validation.

    Raises
    ------
    ValueError
        If the ticker contains any character outside the allowed set
        (letters, digits, '.', '-', '^') or exceeds 15 characters. This is
        the primary defense against SQL injection via the ticker field —
        nothing that fails this check ever reaches a SQL string.
    """
    if not isinstance(ticker, str) or not TICKER_PATTERN.match(ticker):
        raise ValueError(
            f"Invalid ticker symbol: {ticker!r}. Only letters, digits, '.', '-', "
            f"and '^' are allowed (max 15 characters, e.g. 'AAPL', 'TATASTEEL.NS', "
            f"'BRK-B', '^GSPC')."
        )
    return ticker


def quoted_table_name(ticker, suffix):
    """
    Build a safely double-quoted SQL identifier for a per-ticker table.

    Parameters
    ----------
    ticker : str
        Must already have passed validate_ticker() — this function only
        handles identifier quoting, not input validation.
    suffix : str
        One of "income", "balance", "cashflow".

    Returns
    -------
    str
        A double-quoted SQL identifier, e.g.
        quoted_table_name("TATASTEEL.NS", "income") -> '"TATASTEEL.NS_income"'.
        Any embedded double-quote is doubled per standard SQL identifier
        escaping (belt-and-suspenders — validate_ticker() already rejects
        '"', but this keeps the function safe even if called directly).
    """
    raw_name = f"{ticker}_{suffix}"
    return '"' + raw_name.replace('"', '""') + '"'
