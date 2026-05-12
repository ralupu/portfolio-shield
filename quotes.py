"""
quotes.py - yfinance price fetching with a 60-second in-memory cache.
"""

import logging
import time

import yfinance as yf

logger = logging.getLogger(__name__)

_cache: dict[str, dict] = {}
QUOTE_TTL = 60


def get_cached(key: str, ttl: int, fetch_fn):
    """Return a cached value if it is still fresh."""
    now = time.time()
    entry = _cache.get(key)
    if entry and now - entry["ts"] < ttl:
        return entry["value"]
    value = fetch_fn()
    _cache[key] = {"value": value, "ts": now}
    return value


def fetch_quote(ticker: str) -> dict:
    """Fetch a live quote for ticker via yfinance."""

    def _fetch():
        tk = yf.Ticker(ticker.upper())
        info = tk.fast_info
        price = getattr(info, "last_price", None)
        prev = getattr(info, "previous_close", None)

        if price is None:
            raise ValueError(f"No price data for ticker '{ticker}'")

        change = round(price - prev, 2) if prev else 0.0
        change_pct = round((change / prev) * 100, 2) if prev else 0.0
        return {
            "ticker": ticker.upper(),
            "price": round(price, 2),
            "change": change,
            "change_pct": change_pct,
            "last_updated": time.strftime("%H:%M ET"),
        }

    try:
        return get_cached(f"quote:{ticker.upper()}", QUOTE_TTL, _fetch)
    except ValueError:
        raise
    except Exception as exc:
        logger.error("Quote fetch failed for %s: %s", ticker, exc)
        raise RuntimeError(f"Quote service unavailable for '{ticker}': {exc}")


if __name__ == "__main__":
    import json
    import sys

    tickers = sys.argv[1:] or ["AAPL", "MSFT", "TSLA"]
    for ticker in tickers:
        try:
            print(json.dumps(fetch_quote(ticker), indent=2))
        except (ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
