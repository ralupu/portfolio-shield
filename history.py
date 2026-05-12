"""
history.py â€” Portfolio history evolution + beta calculation.

Uses concurrent fetching with ThreadPoolExecutor for performance.
Results cached for 30 minutes via the shared cache in quotes.py.
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import datetime

import numpy as np
import yfinance as yf

from quotes import get_cached

logger = logging.getLogger(__name__)

HISTORY_TTL = 1800  # 30 minutes
MAX_WORKERS = 5
TICKER_TIMEOUT = 10  # seconds per ticker


# ---------------------------------------------------------------------------
# Internal: concurrent history fetch
# ---------------------------------------------------------------------------

def _fetch_ticker_history(ticker: str, period: str = "1y") -> dict | None:
    """Fetch daily close prices for a single ticker. Returns {dates, prices}."""
    try:
        tk = yf.Ticker(ticker.upper())
        hist = tk.history(period=period)
        if hist.empty or len(hist) < 5:
            logger.warning("Insufficient history for %s", ticker)
            return None
        dates = [d.strftime("%Y-%m-%d") for d in hist.index]
        prices = hist["Close"].tolist()
        return {"dates": dates, "prices": prices}
    except Exception as e:
        logger.error("History fetch failed for %s: %s", ticker, e)
        return None


def _fetch_all_histories(tickers: list[str], period: str = "1y") -> dict:
    """Fetch histories for all tickers concurrently.

    Returns dict: {ticker: {dates, prices}} â€” excludes tickers that failed.
    Also returns list of tickers that timed out.
    """
    results = {}
    warnings = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(_fetch_ticker_history, t, period): t
            for t in tickers
        }
        for future in futures:
            ticker = futures[future]
            try:
                data = future.result(timeout=TICKER_TIMEOUT)
                if data:
                    results[ticker] = data
                else:
                    warnings.append(f"{ticker}: price history unavailable")
            except FuturesTimeout:
                warnings.append(f"{ticker}: market data request timed out")
                logger.warning("History fetch timed out for %s", ticker)
            except Exception as e:
                warnings.append(f"{ticker}: {e}")
                logger.error("History fetch error for %s: %s", ticker, e)

    return results, warnings

def get_price_histories(tickers: list[str], period: str = "1y") -> dict:
    """Public helper returning cached price histories by ticker."""
    tickers = [ticker.upper() for ticker in tickers if ticker]
    if not tickers:
        return {}

    cache_key = f"price_histories:{'|'.join(sorted(set(tickers)))}:{period}"

    def _fetch():
        return _fetch_all_histories(sorted(set(tickers)), period)

    histories, _warnings = get_cached(cache_key, HISTORY_TTL, _fetch)
    return histories


# ---------------------------------------------------------------------------
# Portfolio history evolution
# ---------------------------------------------------------------------------

def get_portfolio_history(positions: list[dict]) -> dict:
    """Calculate 1-year daily portfolio value.

    positions: list of {ticker, shares, price, avg_cost}

    Returns:
        {
            dates: [str],
            values: [float],
            cost_basis: float,
            min_value: float,
            max_value: float,
            current_value: float,
            warnings: [str],
        }
    """
    tickers = [p["ticker"] for p in positions]
    shares_map: dict[str, float] = {}
    for p in positions:
        shares_map[p["ticker"]] = shares_map.get(p["ticker"], 0) + p["shares"]
    cost_basis = sum(p["shares"] * p.get("avg_cost", p["price"]) for p in positions)

    def _fetch():
        return _fetch_all_histories(tickers, period="1y")

    cache_key = f"history:{'|'.join(sorted(tickers))}"
    histories, warnings = get_cached(cache_key, HISTORY_TTL, _fetch)

    if not histories:
        return {
            "dates": [],
            "values": [],
            "cost_basis": round(cost_basis, 2),
            "min_value": 0,
            "max_value": 0,
            "current_value": 0,
            "warnings": warnings or ["No historical data is available for this portfolio."],
        }

    # Build a unified date index from all tickers
    all_dates = set()
    for data in histories.values():
        all_dates.update(data["dates"])
    all_dates = sorted(all_dates)

    # Build price lookup per ticker with forward-fill
    price_lookup = {}
    for ticker, data in histories.items():
        date_price = dict(zip(data["dates"], data["prices"]))
        filled = {}
        last_price = None
        for d in all_dates:
            if d in date_price:
                last_price = date_price[d]
            filled[d] = last_price
        price_lookup[ticker] = filled

    # Calculate portfolio value per day
    dates_out = []
    values_out = []
    for d in all_dates:
        daily_val = 0.0
        valid = True
        for ticker in tickers:
            if ticker in price_lookup and price_lookup[ticker].get(d) is not None:
                daily_val += shares_map[ticker] * price_lookup[ticker][d]
            else:
                valid = False
                break
        if valid:
            dates_out.append(d)
            values_out.append(round(daily_val, 2))

    if not values_out:
        return {
            "dates": [],
            "values": [],
            "cost_basis": round(cost_basis, 2),
            "min_value": 0,
            "max_value": 0,
            "current_value": 0,
            "warnings": warnings or ["We could not build a clean price history for this portfolio."],
        }

    return {
        "dates": dates_out,
        "values": values_out,
        "cost_basis": round(cost_basis, 2),
        "min_value": round(min(values_out), 2),
        "max_value": round(max(values_out), 2),
        "current_value": round(values_out[-1], 2),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Beta calculation
# ---------------------------------------------------------------------------

def get_portfolio_beta(positions: list[dict], period: str = "1y") -> dict:
    """Calculate portfolio beta vs SPY.

    Returns:
        {
            portfolio_beta: float,
            position_betas: {ticker: float},
            spy_correlation: float,
            warnings: [str],
        }
    """
    tickers = [p["ticker"] for p in positions]
    total_value = sum(p["shares"] * p["price"] for p in positions)
    value_by_ticker: dict[str, float] = {}
    for p in positions:
        ticker = p["ticker"]
        value_by_ticker[ticker] = value_by_ticker.get(ticker, 0.0) + (p["shares"] * p["price"])
    weights = {ticker: value / total_value for ticker, value in value_by_ticker.items()} if total_value > 0 else {}

    # Fetch all tickers + SPY concurrently
    all_tickers = list(set(tickers + ["SPY"]))

    cache_key = f"beta_hist:{'|'.join(sorted(all_tickers))}:{period}"

    def _fetch():
        return _fetch_all_histories(all_tickers, period)

    histories, warnings = get_cached(cache_key, HISTORY_TTL, _fetch)

    if "SPY" not in histories:
        return {
            "portfolio_beta": None,
            "position_betas": {},
            "spy_correlation": None,
            "warnings": warnings + ["We could not load SPY benchmark data."],
        }

    # Build aligned daily returns
    spy_data = histories["SPY"]
    spy_date_price = dict(zip(spy_data["dates"], spy_data["prices"]))

    position_betas = {}
    weighted_beta = 0.0

    # SPY returns
    spy_prices = np.array(spy_data["prices"])
    spy_returns = np.diff(spy_prices) / spy_prices[:-1]
    spy_dates_set = set(spy_data["dates"])

    for ticker in tickers:
        if ticker not in histories:
            warnings.append(f"{ticker}: excluded from beta (no data)")
            continue

        tk_data = histories[ticker]
        tk_date_price = dict(zip(tk_data["dates"], tk_data["prices"]))

        # Align dates
        common_dates = sorted(set(tk_data["dates"]) & spy_dates_set)
        if len(common_dates) < 20:
            warnings.append(f"{ticker}: insufficient overlapping data for beta")
            continue

        tk_aligned = [tk_date_price[d] for d in common_dates]
        spy_aligned = [spy_date_price[d] for d in common_dates]

        tk_arr = np.array(tk_aligned)
        spy_arr = np.array(spy_aligned)

        tk_ret = np.diff(tk_arr) / tk_arr[:-1]
        spy_ret = np.diff(spy_arr) / spy_arr[:-1]

        # Beta = cov(r_i, r_m) / var(r_m)
        cov = np.cov(tk_ret, spy_ret)[0][1]
        var_spy = np.var(spy_ret, ddof=1)

        if var_spy > 0:
            beta = round(float(cov / var_spy), 3)
        else:
            beta = 1.0

        position_betas[ticker] = beta
        weighted_beta += weights.get(ticker, 0) * beta

    # Portfolio correlation with SPY
    spy_correlation = None
    if len(position_betas) > 0:
        shares_map: dict[str, float] = {}
        for p in positions:
            shares_map[p["ticker"]] = shares_map.get(p["ticker"], 0.0) + p["shares"]
        spy_dates_list = spy_data["dates"]
        port_values = []
        for d in spy_dates_list:
            val = 0.0
            all_found = True
            for ticker in tickers:
                if ticker in histories:
                    tk_dp = dict(zip(histories[ticker]["dates"], histories[ticker]["prices"]))
                    if d in tk_dp:
                        val += shares_map.get(ticker, 0) * tk_dp[d]
                    else:
                        all_found = False
                        break
                else:
                    all_found = False
                    break
            if all_found:
                port_values.append(val)
            else:
                port_values.append(None)

        clean_port = []
        clean_spy = []
        for pv, sp in zip(port_values, spy_data["prices"]):
            if pv is not None:
                clean_port.append(pv)
                clean_spy.append(sp)

        if len(clean_port) > 20:
            port_arr = np.array(clean_port)
            spy_arr = np.array(clean_spy)
            port_ret = np.diff(port_arr) / port_arr[:-1]
            spy_ret = np.diff(spy_arr) / spy_arr[:-1]
            corr_matrix = np.corrcoef(port_ret, spy_ret)
            spy_correlation = round(float(corr_matrix[0][1]) * 100, 1)

    return {
        "portfolio_beta": round(weighted_beta, 3) if position_betas else None,
        "position_betas": position_betas,
        "spy_correlation": spy_correlation,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Performance summary
# ---------------------------------------------------------------------------

def get_performance_summary(positions: list[dict]) -> dict:
    """Calculate P&L and weight breakdown.

    positions: list of {ticker, shares, price, avg_cost}

    Returns dict with cost_basis, current_value, total_pnl, total_pnl_pct,
    best/worst performers, and per-position details.
    """
    cost_basis = 0.0
    current_value = 0.0
    pos_details = []

    for p in positions:
        avg_cost = p.get("avg_cost", p["price"])
        shares = p["shares"]
        live_price = p["price"]

        pos_cost = shares * avg_cost
        pos_value = shares * live_price
        pnl = pos_value - pos_cost
        pnl_pct = ((live_price - avg_cost) / avg_cost * 100) if avg_cost > 0 else 0.0

        cost_basis += pos_cost
        current_value += pos_value

        pos_details.append({
            "ticker": p["ticker"],
            "shares": shares,
            "avg_cost": round(avg_cost, 2),
            "live_price": round(live_price, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 2),
            "cost_value": round(pos_cost, 2),
            "live_value": round(pos_value, 2),
        })

    # Compute live weights
    for p in pos_details:
        p["weight"] = round((p["live_value"] / current_value * 100), 1) if current_value > 0 else 0.0
        p["cost_weight"] = round((p["cost_value"] / cost_basis * 100), 1) if cost_basis > 0 else 0.0

    total_pnl = current_value - cost_basis
    total_pnl_pct = ((current_value - cost_basis) / cost_basis * 100) if cost_basis > 0 else 0.0

    # Best / worst
    best = max(pos_details, key=lambda x: x["pnl_pct"]) if pos_details else None
    worst = min(pos_details, key=lambda x: x["pnl_pct"]) if pos_details else None

    return {
        "cost_basis": round(cost_basis, 2),
        "current_value": round(current_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "best": best,
        "worst": worst,
        "positions": pos_details,
    }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    test_positions = [
        {"ticker": "AAPL", "shares": 100, "price": 0, "avg_cost": 150},
        {"ticker": "MSFT", "shares": 50, "price": 0, "avg_cost": 280},
        {"ticker": "NVDA", "shares": 30, "price": 0, "avg_cost": 400},
    ]

    # Fill live prices
    from quotes import fetch_quote

    for p in test_positions:
        q = fetch_quote(p["ticker"])
        p["price"] = q["price"]
        print(f"{p['ticker']}: ${q['price']}")

    print("\n=== PERFORMANCE ===")
    perf = get_performance_summary(test_positions)
    print(json.dumps(perf, indent=2))

    print("\n=== PORTFOLIO HISTORY ===")
    hist = get_portfolio_history(test_positions)
    print(f"  Dates: {len(hist['dates'])} days")
    print(f"  Cost basis: ${hist['cost_basis']:,.0f}")
    print(f"  Min: ${hist['min_value']:,.0f}")
    print(f"  Max: ${hist['max_value']:,.0f}")
    print(f"  Current: ${hist['current_value']:,.0f}")
    if hist["warnings"]:
        print(f"  Warnings: {hist['warnings']}")

    print("\n=== BETA ===")
    beta = get_portfolio_beta(test_positions)
    print(json.dumps(beta, indent=2))







