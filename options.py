"""
options.py - Options chain fetching, filtering, selection, and fallback pricing.
"""

import math
import logging
from datetime import datetime, timedelta

import yfinance as yf
from scipy.stats import norm

from quotes import get_cached

logger = logging.getLogger(__name__)

CHAIN_TTL = 300  # 5 minutes
DEFAULT_MIN_OI = 100
DEFAULT_MAX_SPREAD_PCT = 0.18

OTM_OFFSETS = {
    "light": 0.10,
    "moderate": 0.05,
    "full": 0.00,
}

HEDGE_PCTS = {
    "light": 0.25,
    "moderate": 0.50,
    "full": 1.00,
}


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """European put price via Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put delta via Black-Scholes (negative value)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return norm.cdf(d1) - 1.0


def fetch_chain(ticker: str) -> dict:
    """Fetch all put expiries from yfinance for the ticker."""

    def _fetch():
        tk = yf.Ticker(ticker.upper())
        expiries = tk.options
        chains = {}
        for exp in expiries:
            try:
                chain = tk.option_chain(exp)
                chains[exp] = chain.puts
            except Exception as exc:
                logger.warning("Could not fetch chain %s/%s: %s", ticker, exp, exc)
        return chains

    return get_cached(f"chain:{ticker.upper()}", CHAIN_TTL, _fetch)


def select_put(
    ticker: str,
    current_price: float,
    hedge_level: str,
    risk_free_rate: float = 0.05,
    target_dte: int = 45,
    min_open_interest: int = DEFAULT_MIN_OI,
    max_spread_pct: float = DEFAULT_MAX_SPREAD_PCT,
) -> dict:
    """Select a put option using retail-oriented liquidity and spread checks."""
    otm_offset = OTM_OFFSETS[hedge_level]
    target_strike = current_price * (1.0 - otm_offset)

    now = datetime.now()
    min_dte = max(21, target_dte - 20)
    max_dte = min(120, target_dte + 30)

    chains = fetch_chain(ticker)
    best = None
    best_score = float("inf")

    for exp_str, puts in chains.items():
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        dte = (exp_date - now).days
        if dte < min_dte or dte > max_dte:
            continue
        if puts.empty:
            continue

        liquid = puts.copy()
        if "openInterest" in liquid.columns:
            liquid = liquid[liquid["openInterest"].fillna(0) >= min_open_interest]
        if liquid.empty:
            continue

        for _, row in liquid.iterrows():
            strike = float(row.get("strike", 0.0) or 0.0)
            bid = float(row.get("bid", 0.0) or 0.0)
            ask = float(row.get("ask", 0.0) or 0.0)
            if strike <= 0 or ask <= 0:
                continue

            mid = (bid + ask) / 2.0 if bid > 0 else ask
            if mid <= 0.05:
                continue

            spread_pct = (ask - bid) / mid if mid > 0 else 1.0
            if spread_pct > max_spread_pct:
                continue

            iv = float(row.get("impliedVolatility", 0.0) or 0.0)
            if iv <= 0:
                iv = max(_estimate_iv(ticker), 0.18)

            T = dte / 365.0
            delta = bs_put_delta(current_price, strike, T, risk_free_rate, iv)
            volume = int(row.get("volume", 0) or 0)
            open_interest = int(row.get("openInterest", 0) or 0)

            strike_score = abs(strike - target_strike) / max(current_price, 0.01)
            dte_score = abs(dte - target_dte) / max(target_dte, 1)
            spread_score = spread_pct
            liquidity_bonus = 1.0 / max(open_interest + volume, 1)
            score = strike_score * 45 + dte_score * 25 + spread_score * 30 + liquidity_bonus * 200

            candidate = {
                "ticker": ticker.upper(),
                "strike": round(strike, 2),
                "expiry": exp_str,
                "dte": dte,
                "mid_price": round(mid, 2),
                "bid": round(bid, 2),
                "ask": round(ask, 2),
                "spread_pct": round(spread_pct * 100, 2),
                "delta": round(delta, 4),
                "iv": round(iv, 4),
                "volume": volume,
                "open_interest": open_interest,
                "target_strike": round(target_strike, 2),
                "is_fallback": False,
            }
            if score < best_score:
                best_score = score
                best = candidate

    if best is None:
        logger.warning("No liquid options for %s - using fallback pricing", ticker)
        T = target_dte / 365.0
        iv = _estimate_iv(ticker)
        strike = round(target_strike, 2)
        mid = round(bs_put_price(current_price, strike, T, risk_free_rate, iv), 2)
        delta = round(bs_put_delta(current_price, strike, T, risk_free_rate, iv), 4)
        best = {
            "ticker": ticker.upper(),
            "strike": strike,
            "expiry": (now + timedelta(days=target_dte)).strftime("%Y-%m-%d"),
            "dte": target_dte,
            "mid_price": mid,
            "bid": mid,
            "ask": mid,
            "spread_pct": 0.0,
            "delta": delta,
            "iv": round(iv, 4),
            "volume": 0,
            "open_interest": 0,
            "target_strike": strike,
            "is_fallback": True,
        }

    return best


def _estimate_iv(ticker: str) -> float:
    """Estimate IV from realized volatility over the last 3 months,
    plus a variance risk premium adjustment."""
    try:
        tk = yf.Ticker(ticker.upper())
        hist = tk.history(period="3mo")
        if hist.empty or len(hist) < 10:
            return 0.30
        closes = hist["Close"]
        log_returns = (closes / closes.shift(1)).apply(math.log).dropna()
        realized_vol = float(log_returns.std() * math.sqrt(252))
        realized_vol = max(realized_vol, 0.10)
        iv_estimate = realized_vol + 0.03
        return iv_estimate
    except Exception:
        return 0.30


