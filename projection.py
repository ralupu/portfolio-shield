"""
projection.py - Historical path-based forward range projections for hedged vs unhedged portfolios.
"""

from __future__ import annotations

import hashlib
import math
import random
from datetime import datetime, timedelta
from typing import Any

from history import get_price_histories
from options import bs_put_delta, bs_put_price

TRADING_DAYS_PER_MONTH = 21
DEFAULT_PATHS = 350
DEFAULT_BLOCK_SIZE = 5
OUTER_Q = 0.10
INNER_Q = 0.25
CI95_Q = 0.025
RISK_FREE_RATE = 0.05
VARIANCE_RISK_PREMIUM = 0.03


def build_future_fan_chart(
    positions: list[dict[str, Any]],
    recommendation: dict[str, Any],
    history: dict[str, Any],
    months: int = 12,
    horizon_days: int | None = None,
    num_paths: int = DEFAULT_PATHS,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> dict[str, Any]:
    actual_current_value = float(history.get("current_value") or recommendation.get("total_value") or 0.0)
    if actual_current_value <= 0 or not positions:
        return {"has_data": False, "warnings": ["Not enough portfolio data is available to build the forward range."]}

    tickers = _simulation_tickers(positions, recommendation)
    histories = get_price_histories(tickers, period="1y")
    aligned = _build_aligned_prices(histories, tickers)
    if not aligned["dates"] or len(aligned["dates"]) < 60:
        return {
            "has_data": False,
            "warnings": ["Not enough market history is available to build the forward range."],
        }

    return_matrix = _build_return_matrix(aligned["prices"], tickers)
    if len(return_matrix) < 20:
        return {
            "has_data": False,
            "warnings": ["The available market history is too short to simulate a reliable forward range."],
        }

    realized_vols = _realized_vols(return_matrix, tickers)
    start_prices = _starting_prices(positions, recommendation, aligned["prices"], tickers)
    position_sizes = _position_sizes(positions)
    contract_legs = recommendation.get("contracts", [])
    initial_option_value = _option_value(contract_legs, start_prices, 0, realized_vols)
    initial_cost = float(recommendation.get("total_cost", initial_option_value) or 0.0)
    roll_days = int(recommendation.get("policy_config", {}).get("min_days_to_roll", 21) or 21)

    if horizon_days is not None and horizon_days > 0:
        total_trading_days = max(1, round(horizon_days * 252 / 365))
        step_sizes = []
        remaining_days = total_trading_days
        while remaining_days > 0:
            chunk = min(TRADING_DAYS_PER_MONTH, remaining_days)
            step_sizes.append(chunk)
            remaining_days -= chunk
        dates = _projection_dates_from_steps(history, step_sizes)
        projected_horizon_days = (dates[-1] - dates[0]).days if len(dates) > 1 else horizon_days
    else:
        step_sizes = [TRADING_DAYS_PER_MONTH for _ in range(months)]
        dates = _projection_dates(history, months)
        projected_horizon_days = (dates[-1] - dates[0]).days if len(dates) > 1 else months * 30

    summary = _simulate_projection_summary(
        positions=positions,
        recommendation=recommendation,
        position_sizes=position_sizes,
        start_prices=start_prices,
        contract_legs=contract_legs,
        realized_vols=realized_vols,
        dates=dates,
        step_sizes=step_sizes,
        return_matrix=return_matrix,
        tickers=tickers,
        block_size=block_size,
        num_paths=num_paths,
        roll_days=roll_days,
        initial_cost=initial_cost,
        actual_current_value=actual_current_value,
        initial_market_shock=None,
    )
    diagnostics = {
        "paths": num_paths,
        "historical_days": len(return_matrix),
        "method": "Historical block bootstrap with option repricing; compares hold-current-hedge vs hedge rolling",
        "coverage_pct": round(float(recommendation.get("market_delta_coverage_pct", 0.0) or 0.0), 1),
        "initial_hedge_cost": round(initial_cost, 2),
        "realized_vol_pct": round(_portfolio_realized_vol(positions, aligned["prices"], tickers) * 100, 1),
        "horizon_days": projected_horizon_days,
    }
    return _build_svg_payload(summary, diagnostics, history)


def _simulate_projection_summary(
    *,
    positions: list[dict[str, Any]],
    recommendation: dict[str, Any],
    position_sizes: dict[str, float],
    start_prices: dict[str, float],
    contract_legs: list[dict[str, Any]],
    realized_vols: dict[str, float],
    dates: list[datetime],
    step_sizes: list[int],
    return_matrix: list[dict[str, float]],
    tickers: list[str],
    block_size: int,
    num_paths: int,
    roll_days: int,
    initial_cost: float,
    actual_current_value: float,
    initial_market_shock: float | None,
) -> dict[str, Any]:
    shocked_prices = _apply_initial_market_shock(positions, recommendation, start_prices, initial_market_shock)
    initial_option_book = _initialize_option_book(contract_legs, start_prices)

    scenario_unhedged_start = actual_current_value
    scenario_hold_start = actual_current_value
    scenario_rolling_start = actual_current_value
    if initial_market_shock is not None:
        scenario_unhedged_start = _portfolio_value(position_sizes, shocked_prices)
        hold_option_value = _option_book_value(initial_option_book, shocked_prices, dates[0], realized_vols)
        scenario_hold_start = round(scenario_unhedged_start - initial_cost + hold_option_value, 2)
        scenario_rolling_start = scenario_hold_start

    unhedged_paths = [[scenario_unhedged_start] for _ in range(num_paths)]
    hold_paths = [[scenario_hold_start] for _ in range(num_paths)]
    rolling_paths = [[scenario_rolling_start] for _ in range(num_paths)]

    rng = random.Random(_stable_seed(positions, recommendation, initial_market_shock))

    for path_idx in range(num_paths):
        simulated_prices = dict(shocked_prices)
        hold_cash = -initial_cost
        rolling_cash = -initial_cost
        hold_option_book = [dict(leg) for leg in initial_option_book]
        rolling_option_book = [dict(leg) for leg in initial_option_book]
        for step_idx, step_days in enumerate(step_sizes, start=1):
            simulated_prices = _advance_prices(
                simulated_prices,
                return_matrix,
                tickers,
                rng,
                steps=step_days,
                block_size=block_size,
            )
            current_date = dates[step_idx]
            current_portfolio_value = _portfolio_value(position_sizes, simulated_prices)
            hold_option_book, hold_cash = _rebalance_option_book(
                hold_option_book,
                simulated_prices,
                current_date,
                realized_vols,
                hold_cash,
                current_portfolio_value=current_portfolio_value,
                roll_days=0,
                allow_roll=False,
            )
            rolling_option_book, rolling_cash = _rebalance_option_book(
                rolling_option_book,
                simulated_prices,
                current_date,
                realized_vols,
                rolling_cash,
                current_portfolio_value=current_portfolio_value,
                roll_days=roll_days,
                allow_roll=step_idx < len(step_sizes),
            )
            unhedged_value = current_portfolio_value
            hold_hedge_value = _option_book_value(hold_option_book, simulated_prices, current_date, realized_vols)
            rolling_hedge_value = _option_book_value(rolling_option_book, simulated_prices, current_date, realized_vols)
            hold_value = unhedged_value + hold_cash + hold_hedge_value
            rolling_value = unhedged_value + rolling_cash + rolling_hedge_value
            unhedged_paths[path_idx].append(unhedged_value)
            hold_paths[path_idx].append(round(hold_value, 2))
            rolling_paths[path_idx].append(round(rolling_value, 2))

    summary = _summarize_paths(unhedged_paths, hold_paths, rolling_paths, dates, actual_current_value)
    summary["anchor_value"] = actual_current_value
    return summary


def _apply_initial_market_shock(
    positions: list[dict[str, Any]],
    recommendation: dict[str, Any],
    start_prices: dict[str, float],
    initial_market_shock: float | None,
) -> dict[str, float]:
    shocked = dict(start_prices)
    if initial_market_shock is None:
        return shocked

    position_betas = recommendation.get("portfolio_state", {}).get("position_betas", {})
    position_tickers = {str(position["ticker"]).upper() for position in positions}
    for position in positions:
        ticker = str(position["ticker"]).upper()
        beta = float(position_betas.get(ticker, 1.0) or 1.0)
        shocked[ticker] = max(0.01, float(position["price"]) * (1.0 + initial_market_shock * beta))

    for leg in recommendation.get("contracts", []):
        underlying = str(leg.get("underlying") or leg.get("ticker") or "").upper()
        if underlying and underlying not in position_tickers:
            base_price = float(leg.get("underlying_price") or shocked.get(underlying, 0.0) or 0.0)
            shocked[underlying] = max(0.01, base_price * (1.0 + initial_market_shock))

    return shocked

def _simulation_tickers(positions: list[dict[str, Any]], recommendation: dict[str, Any]) -> list[str]:
    tickers = {str(position["ticker"]).upper() for position in positions}
    for leg in recommendation.get("contracts", []):
        underlying = str(leg.get("underlying") or leg.get("ticker") or "").upper()
        if underlying:
            tickers.add(underlying)
    return sorted(tickers)


def _build_aligned_prices(histories: dict[str, dict], tickers: list[str]) -> dict[str, Any]:
    if not histories:
        return {"dates": [], "prices": {}}

    all_dates = sorted({date for series in histories.values() for date in series.get("dates", [])})
    if not all_dates:
        return {"dates": [], "prices": {}}

    aligned_prices: dict[str, list[float]] = {}
    for ticker in tickers:
        series = histories.get(ticker)
        if not series:
            return {"dates": [], "prices": {}}
        date_to_price = dict(zip(series.get("dates", []), series.get("prices", [])))
        filled = []
        last_price = None
        for date in all_dates:
            if date in date_to_price:
                last_price = float(date_to_price[date])
            filled.append(last_price)
        aligned_prices[ticker] = filled

    filtered_dates = []
    filtered_prices = {ticker: [] for ticker in tickers}
    for idx, date in enumerate(all_dates):
        row = [aligned_prices[ticker][idx] for ticker in tickers]
        if any(value is None or value <= 0 for value in row):
            continue
        filtered_dates.append(date)
        for ticker in tickers:
            filtered_prices[ticker].append(float(aligned_prices[ticker][idx]))

    return {"dates": filtered_dates, "prices": filtered_prices}


def _build_return_matrix(price_map: dict[str, list[float]], tickers: list[str]) -> list[dict[str, float]]:
    if not tickers:
        return []
    count = len(price_map[tickers[0]])
    rows = []
    for idx in range(1, count):
        row = {}
        valid = True
        for ticker in tickers:
            prev_price = price_map[ticker][idx - 1]
            current_price = price_map[ticker][idx]
            if prev_price <= 0 or current_price <= 0:
                valid = False
                break
            row[ticker] = math.log(current_price / prev_price)
        if valid:
            rows.append(row)
    return rows


def _realized_vols(return_matrix: list[dict[str, float]], tickers: list[str]) -> dict[str, float]:
    vols = {}
    for ticker in tickers:
        series = [row[ticker] for row in return_matrix]
        if len(series) < 2:
            vols[ticker] = 0.25
            continue
        mean_return = sum(series) / len(series)
        variance = sum((value - mean_return) ** 2 for value in series) / (len(series) - 1)
        vols[ticker] = max(0.10, min(math.sqrt(max(variance, 0.0)) * math.sqrt(252), 1.00))
    return vols


def _starting_prices(
    positions: list[dict[str, Any]],
    recommendation: dict[str, Any],
    aligned_prices: dict[str, list[float]],
    tickers: list[str],
) -> dict[str, float]:
    start_prices = {ticker: aligned_prices[ticker][-1] for ticker in tickers if aligned_prices.get(ticker)}
    for position in positions:
        start_prices[str(position["ticker"]).upper()] = float(position["price"])
    for leg in recommendation.get("contracts", []):
        underlying = str(leg.get("underlying") or leg.get("ticker") or "").upper()
        if underlying:
            start_prices[underlying] = float(leg.get("underlying_price") or start_prices.get(underlying, 0.0))
    return start_prices


def _position_sizes(positions: list[dict[str, Any]]) -> dict[str, float]:
    sizes: dict[str, float] = {}
    for position in positions:
        ticker = str(position["ticker"]).upper()
        sizes[ticker] = sizes.get(ticker, 0.0) + float(position.get("shares", 0.0) or 0.0)
    return sizes


def _projection_dates(history: dict[str, Any], months: int) -> list[datetime]:
    if history.get("dates"):
        try:
            anchor = datetime.strptime(history["dates"][-1], "%Y-%m-%d")
        except ValueError:
            anchor = datetime.now()
    else:
        anchor = datetime.now()
    return [anchor + timedelta(days=30 * offset) for offset in range(months + 1)]

def _projection_dates_from_steps(history: dict[str, Any], step_sizes: list[int]) -> list[datetime]:
    if history.get("dates"):
        try:
            anchor = datetime.strptime(history["dates"][-1], "%Y-%m-%d")
        except ValueError:
            anchor = datetime.now()
    else:
        anchor = datetime.now()

    dates = [anchor]
    elapsed_calendar_days = 0.0
    for step in step_sizes:
        elapsed_calendar_days += step * (365.0 / 252.0)
        dates.append(anchor + timedelta(days=round(elapsed_calendar_days)))
    return dates


def _advance_prices(
    simulated_prices: dict[str, float],
    return_matrix: list[dict[str, float]],
    tickers: list[str],
    rng: random.Random,
    steps: int,
    block_size: int,
) -> dict[str, float]:
    updated = dict(simulated_prices)
    max_start = max(0, len(return_matrix) - block_size)
    steps_done = 0
    while steps_done < steps:
        start_idx = rng.randint(0, max_start) if max_start > 0 else 0
        end_idx = min(start_idx + block_size, len(return_matrix))
        for row in return_matrix[start_idx:end_idx]:
            for ticker in tickers:
                updated[ticker] = max(0.01, updated[ticker] * math.exp(row[ticker]))
            steps_done += 1
            if steps_done >= steps:
                break
    return updated


def _portfolio_value(position_sizes: dict[str, float], simulated_prices: dict[str, float]) -> float:
    return round(sum(position_sizes[ticker] * simulated_prices[ticker] for ticker in position_sizes), 2)


def _initialize_option_book(contract_legs: list[dict[str, Any]], simulated_prices: dict[str, float]) -> list[dict[str, Any]]:
    book = []
    for leg in contract_legs:
        underlying = str(leg.get("underlying") or leg.get("ticker") or "").upper()
        strike = float(leg.get("strike", 0.0) or 0.0)
        contracts = int(leg.get("contracts", 0) or 0)
        dte = int(leg.get("dte", 0) or 0)
        if not underlying or strike <= 0 or contracts <= 0 or dte <= 0:
            continue
        base_spot = float(leg.get("underlying_price") or simulated_prices.get(underlying, 0.0) or 0.0)
        if base_spot <= 0:
            continue
        try:
            expiry_date = datetime.strptime(str(leg.get("expiry") or ""), "%Y-%m-%d")
        except ValueError:
            expiry_date = datetime.now() + timedelta(days=dte)
        book.append(
            {
                "underlying": underlying,
                "contracts": contracts,
                "strike": strike,
                "iv": float(leg.get("iv", 0.0) or 0.22),
                "expiry_date": expiry_date,
                "original_dte": dte,
                "strike_ratio": strike / max(base_spot, 0.01),
                "strategy_scope": str(leg.get("strategy_scope") or "single"),
                "coverage_ratio": float(leg.get("coverage_ratio", 0.0) or 0.0),
                "market_beta": float(leg.get("market_beta", 1.0) or 1.0),
                "position_shares": float(leg.get("position_shares", 0.0) or 0.0),
                "target_delta": float(leg.get("target_delta", 0.0) or 0.0),
            }
        )
    return book

def _rebalance_option_book(
    option_book: list[dict[str, Any]],
    simulated_prices: dict[str, float],
    current_date: datetime,
    realized_vols: dict[str, float],
    hedge_cash: float,
    current_portfolio_value: float,
    roll_days: int,
    allow_roll: bool,
) -> tuple[list[dict[str, Any]], float]:
    updated_book = []
    for leg in option_book:
        remaining_days = (leg["expiry_date"] - current_date).days
        if remaining_days <= roll_days:
            hedge_cash += _single_option_position_value(leg, simulated_prices, current_date, realized_vols)
            if allow_roll:
                rolled_leg, roll_cost = _roll_option_leg(
                    leg,
                    simulated_prices,
                    current_date,
                    realized_vols,
                    current_portfolio_value,
                )
                hedge_cash -= roll_cost
                updated_book.append(rolled_leg)
            continue
        updated_book.append(leg)
    return updated_book, round(hedge_cash, 2)

def _roll_option_leg(
    leg: dict[str, Any],
    simulated_prices: dict[str, float],
    current_date: datetime,
    realized_vols: dict[str, float],
    current_portfolio_value: float,
) -> tuple[dict[str, Any], float]:
    underlying = leg["underlying"]
    spot = max(float(simulated_prices.get(underlying, 0.01)), 0.01)
    original_dte = max(int(leg.get("original_dte", 45) or 45), 7)
    sigma = float(leg.get("iv", 0.0) or 0.0)
    if sigma <= 0:
        sigma = realized_vols.get(underlying, 0.22) + VARIANCE_RISK_PREMIUM
    strike = round(spot * float(leg.get("strike_ratio", 1.0) or 1.0), 2)
    target_delta = _roll_target_delta(leg, spot, current_portfolio_value)
    put_delta = abs(bs_put_delta(spot, strike, original_dte / 365.0, RISK_FREE_RATE, sigma))
    put_delta = max(put_delta, 0.001)
    contracts = int(leg.get("contracts", 0) or 0)
    if target_delta > 0:
        contracts = max(1, round(target_delta / (100 * put_delta)))
    premium = bs_put_price(spot, strike, original_dte / 365.0, RISK_FREE_RATE, sigma) * contracts * 100
    new_leg = dict(leg)
    new_leg["strike"] = strike
    new_leg["contracts"] = contracts
    new_leg["iv"] = sigma
    new_leg["expiry_date"] = current_date + timedelta(days=original_dte)
    if target_delta > 0:
        new_leg["target_delta"] = round(target_delta, 3)
    return new_leg, round(premium, 2)


def _roll_target_delta(leg: dict[str, Any], spot: float, current_portfolio_value: float) -> float:
    stored_target_delta = max(float(leg.get("target_delta", 0.0) or 0.0), 0.0)
    coverage_ratio = max(float(leg.get("coverage_ratio", 0.0) or 0.0), 0.0)
    scope = str(leg.get("strategy_scope") or "single").lower()

    if scope == "index" and coverage_ratio > 0 and current_portfolio_value > 0 and spot > 0:
        market_beta = max(float(leg.get("market_beta", 1.0) or 1.0), 0.0)
        return current_portfolio_value * market_beta / spot * coverage_ratio

    position_shares = max(float(leg.get("position_shares", 0.0) or 0.0), 0.0)
    if position_shares > 0 and coverage_ratio > 0:
        return position_shares * coverage_ratio

    return stored_target_delta

def _option_book_value(
    option_book: list[dict[str, Any]],
    simulated_prices: dict[str, float],
    current_date: datetime,
    realized_vols: dict[str, float],
) -> float:
    return round(sum(_single_option_position_value(leg, simulated_prices, current_date, realized_vols) for leg in option_book), 2)


def _single_option_position_value(
    leg: dict[str, Any],
    simulated_prices: dict[str, float],
    as_of: datetime,
    realized_vols: dict[str, float],
) -> float:
    underlying = str(leg.get("underlying") or "").upper()
    strike = float(leg.get("strike", 0.0) or 0.0)
    contracts = int(leg.get("contracts", 0) or 0)
    if not underlying or strike <= 0 or contracts <= 0:
        return 0.0
    spot = float(simulated_prices.get(underlying, 0.0) or 0.0)
    sigma = float(leg.get("iv", 0.0) or 0.0)
    if sigma <= 0:
        sigma = realized_vols.get(underlying, 0.22) + VARIANCE_RISK_PREMIUM
    remaining_days = (leg["expiry_date"] - as_of).days
    remaining_t = max(remaining_days, 0) / 365.0
    if remaining_t <= 0:
        # Expired or at expiry — settle at intrinsic (put payoff)
        option_price = max(strike - spot, 0.0)
    else:
        option_price = bs_put_price(max(spot, 0.01), strike, remaining_t, RISK_FREE_RATE, sigma)
    return option_price * contracts * 100

def _option_value(
    contract_legs: list[dict[str, Any]],
    simulated_prices: dict[str, float],
    as_of: datetime | int,
    realized_vols: dict[str, float],
) -> float:
    if isinstance(as_of, int):
        now = datetime.now()
        book = _initialize_option_book(contract_legs, simulated_prices)
        refreshed_book = [
            dict(leg, expiry_date=now + timedelta(days=int(leg.get("original_dte", 0) or 0)))
            for leg in book
        ]
        return _option_book_value(refreshed_book, simulated_prices, now, realized_vols)
    book = _initialize_option_book(contract_legs, simulated_prices)
    return _option_book_value(book, simulated_prices, as_of, realized_vols)


def _portfolio_realized_vol(positions: list[dict[str, Any]], aligned_prices: dict[str, list[float]], tickers: list[str]) -> float:
    weights = _portfolio_weights(positions)
    portfolio_values = []
    count = len(aligned_prices[tickers[0]]) if tickers else 0
    for idx in range(count):
        value = 0.0
        for ticker, weight in weights.items():
            if ticker in aligned_prices:
                price_series = aligned_prices[ticker]
                value += weight * price_series[idx] / max(price_series[0], 0.01)
        portfolio_values.append(value)
    if len(portfolio_values) < 3:
        return 0.22
    returns = [math.log(portfolio_values[idx] / portfolio_values[idx - 1]) for idx in range(1, len(portfolio_values)) if portfolio_values[idx - 1] > 0 and portfolio_values[idx] > 0]
    if len(returns) < 2:
        return 0.22
    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / (len(returns) - 1)
    return max(0.08, min(math.sqrt(max(variance, 0.0)) * math.sqrt(252), 1.00))


def _portfolio_weights(positions: list[dict[str, Any]]) -> dict[str, float]:
    total_value = sum(float(position.get("shares", 0.0) or 0.0) * float(position.get("price", 0.0) or 0.0) for position in positions)
    if total_value <= 0:
        return {}
    weights = {}
    for position in positions:
        ticker = str(position["ticker"]).upper()
        position_value = float(position.get("shares", 0.0) or 0.0) * float(position.get("price", 0.0) or 0.0)
        weights[ticker] = position_value / total_value
    return weights


def _summarize_paths(
    unhedged_paths: list[list[float]],
    hold_paths: list[list[float]],
    rolling_paths: list[list[float]],
    dates: list[datetime],
    current_value: float,
) -> dict[str, Any]:
    num_steps = len(dates)
    points = []
    for idx in range(num_steps):
        unhedged_at_idx = _distribution_summary(sorted(path[idx] for path in unhedged_paths))
        hold_at_idx = _distribution_summary(sorted(path[idx] for path in hold_paths))
        rolling_at_idx = _distribution_summary(sorted(path[idx] for path in rolling_paths))
        points.append(
            {
                "date": dates[idx],
                "label": "Now" if idx == 0 else dates[idx].strftime("%b %y"),
                "unhedged": unhedged_at_idx,
                "hedged_hold": hold_at_idx,
                "hedged_roll": rolling_at_idx,
            }
        )
    return {"points": points, "current_value": current_value}

def _distribution_summary(sorted_values: list[float]) -> dict[str, float]:
    return {
        "ci95_low": round(_quantile(sorted_values, CI95_Q), 2),
        "outer_low": round(_quantile(sorted_values, OUTER_Q), 2),
        "inner_low": round(_quantile(sorted_values, INNER_Q), 2),
        "median": round(_quantile(sorted_values, 0.50), 2),
        "inner_high": round(_quantile(sorted_values, 1.0 - INNER_Q), 2),
        "outer_high": round(_quantile(sorted_values, 1.0 - OUTER_Q), 2),
        "ci95_high": round(_quantile(sorted_values, 1.0 - CI95_Q), 2),
    }



def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _stable_seed(positions: list[dict[str, Any]], recommendation: dict[str, Any], initial_market_shock: float | None = None) -> int:
    payload = {
        "positions": [
            {
                "ticker": str(position.get("ticker", "")).upper(),
                "shares": float(position.get("shares", 0.0) or 0.0),
                "price": float(position.get("price", 0.0) or 0.0),
            }
            for position in positions
        ],
        "contracts": [
            {
                "underlying": str(leg.get("underlying", "")).upper(),
                "strike": float(leg.get("strike", 0.0) or 0.0),
                "expiry": str(leg.get("expiry", "")),
                "contracts": int(leg.get("contracts", 0) or 0),
            }
            for leg in recommendation.get("contracts", [])
        ],
        "strategy": recommendation.get("strategy"),
        "initial_market_shock": round(float(initial_market_shock or 0.0), 4),
    }
    digest = hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)
def _build_svg_payload(summary: dict[str, Any], diagnostics: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    points = summary["points"]
    current_value = summary["current_value"]
    final_point = points[-1]
    horizon_days = int(diagnostics.get("horizon_days", max(len(points) - 1, 1) * 30) or 0)
    horizon_months = max(1, round(horizon_days / 30))

    series = [
        {"label": "Unhedged", "color": "#ef7d57", "range": final_point["unhedged"]},
        {"label": "Hedged (hold current)", "color": "#6ea8ff", "range": final_point["hedged_hold"]},
        {"label": "Hedged (rolling)", "color": "#52d6be", "range": final_point["hedged_roll"]},
    ]

    chart_w, chart_h = 860, 240
    pad_l, pad_r, pad_t, pad_b = 170, 24, 26, 42
    plot_w = chart_w - pad_l - pad_r
    plot_h = chart_h - pad_t - pad_b

    all_values = [current_value]
    for item in series:
        all_values.extend(
            [
                item["range"]["outer_low"],
                item["range"]["inner_low"],
                item["range"]["median"],
                item["range"]["inner_high"],
                item["range"]["outer_high"],
            ]
        )
    value_min = min(all_values)
    value_max = max(all_values)
    value_range = value_max - value_min if value_max != value_min else max(value_max * 0.1, 1.0)
    value_min = max(0.0, value_min - value_range * 0.08)
    value_max = value_max + value_range * 0.08
    value_range = value_max - value_min if value_max != value_min else 1.0

    def x_pos(value: float) -> float:
        return pad_l + ((value - value_min) / value_range) * plot_w

    def row_y(idx: int) -> float:
        if len(series) == 1:
            return pad_t + (plot_h / 2)
        return pad_t + (idx / (len(series) - 1)) * plot_h

    axis_labels = []
    for idx in range(5):
        label_value = value_min + (value_range * idx / 4)
        axis_labels.append(
            {
                "x": f"{x_pos(label_value):.1f}",
                "label": f"${label_value:,.0f}",
            }
        )

    rows = []
    for idx, item in enumerate(series):
        distribution = item["range"]
        y = row_y(idx)
        rows.append(
            {
                "label": item["label"],
                "color": item["color"],
                "y": f"{y:.1f}",
                "label_x": f"{(pad_l - 14):.1f}",
                "outer_low_x": f"{x_pos(distribution['outer_low']):.1f}",
                "inner_low_x": f"{x_pos(distribution['inner_low']):.1f}",
                "median_x": f"{x_pos(distribution['median']):.1f}",
                "inner_high_x": f"{x_pos(distribution['inner_high']):.1f}",
                "outer_high_x": f"{x_pos(distribution['outer_high']):.1f}",
            }
        )

    hold_downside_delta = round(final_point["hedged_hold"]["outer_low"] - final_point["unhedged"]["outer_low"], 2)
    rolling_downside_delta = round(final_point["hedged_roll"]["outer_low"] - final_point["unhedged"]["outer_low"], 2)
    hold_median_drag = round(final_point["hedged_hold"]["median"] - final_point["unhedged"]["median"], 2)
    rolling_median_drag = round(final_point["hedged_roll"]["median"] - final_point["unhedged"]["median"], 2)

    return {
        "has_data": True,
        "warnings": [],
        "fan_chart": _build_confidence_fan_chart(summary, diagnostics, history),
        "plot_x": pad_l,
        "plot_y": pad_t,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "ch_w": chart_w,
        "ch_h": chart_h,
        "axis_labels": axis_labels,
        "rows": rows,
        "current_x": f"{x_pos(current_value):.1f}",
        "current_label": f"${current_value:,.0f}",
        "range_unhedged_low": f"${final_point['unhedged']['outer_low']:,.0f}",
        "range_unhedged_high": f"${final_point['unhedged']['outer_high']:,.0f}",
        "range_hold_low": f"${final_point['hedged_hold']['outer_low']:,.0f}",
        "range_hold_high": f"${final_point['hedged_hold']['outer_high']:,.0f}",
        "range_rolling_low": f"${final_point['hedged_roll']['outer_low']:,.0f}",
        "range_rolling_high": f"${final_point['hedged_roll']['outer_high']:,.0f}",
        "range_hedged_low": f"${final_point['hedged_roll']['outer_low']:,.0f}",
        "range_hedged_high": f"${final_point['hedged_roll']['outer_high']:,.0f}",
        "median_unhedged": f"${final_point['unhedged']['median']:,.0f}",
        "median_hold": f"${final_point['hedged_hold']['median']:,.0f}",
        "median_rolling": f"${final_point['hedged_roll']['median']:,.0f}",
        "median_hedged": f"${final_point['hedged_roll']['median']:,.0f}",
        "hold_downside_delta": f"${hold_downside_delta:,.0f}",
        "rolling_downside_delta": f"${rolling_downside_delta:,.0f}",
        "hold_median_drag": f"${hold_median_drag:,.0f}",
        "rolling_median_drag": f"${rolling_median_drag:,.0f}",
        "hold_downside_better": hold_downside_delta >= 0,
        "rolling_downside_better": rolling_downside_delta >= 0,
        "hold_median_better": hold_median_drag >= 0,
        "rolling_median_better": rolling_median_drag >= 0,
        "coverage_pct": diagnostics["coverage_pct"],
        "realized_vol_pct": diagnostics["realized_vol_pct"],
        "paths": diagnostics["paths"],
        "historical_days": diagnostics["historical_days"],
        "method": diagnostics["method"],
        "initial_hedge_cost": f"${diagnostics['initial_hedge_cost']:,.0f}",
        "horizon_label": points[-1]["label"],
        "horizon_months": horizon_months,
        "horizon_days": horizon_days,
    }


def _build_confidence_fan_chart(summary: dict[str, Any], diagnostics: dict[str, Any], history: dict[str, Any]) -> dict[str, Any]:
    points = summary["points"]
    if len(points) < 2:
        return {"has_data": False}

    chart_w, chart_h = 860, 320
    pad_l, pad_r, pad_t, pad_b = 72, 34, 28, 46
    plot_w = chart_w - pad_l - pad_r
    plot_h = chart_h - pad_t - pad_b

    series_meta = [
        {"key": "unhedged", "label": "Unhedged", "color": "#ef7d57", "fill_opacity": 0.15},
        {"key": "hedged_hold", "label": "Hedged", "color": "#52d6be", "fill_opacity": 0.18},
    ]

    history_dates = history.get("dates", []) or []
    history_values = history.get("values", []) or []
    history_window = min(len(history_values), 60)
    if history_window > 1:
        history_dates = history_dates[-history_window:]
        history_values = history_values[-history_window:]
    else:
        history_dates = [points[0]["label"]]
        history_values = [summary["current_value"]]

    all_values = [summary["current_value"]]
    all_values.extend(float(value) for value in history_values)
    for point in points:
        for meta in series_meta:
            dist = point[meta["key"]]
            all_values.extend([dist["ci95_low"], dist["median"], dist["ci95_high"]])

    value_min = min(all_values)
    value_max = max(all_values)
    value_range = value_max - value_min if value_max != value_min else max(value_max * 0.1, 1.0)
    value_min = max(0.0, value_min - value_range * 0.08)
    value_max = value_max + value_range * 0.08
    value_range = value_max - value_min if value_max != value_min else 1.0

    history_count = max(len(history_values), 1)
    total_points = history_count + len(points) - 1

    def x_pos(idx: int) -> float:
        if total_points <= 1:
            return pad_l + (plot_w / 2)
        return pad_l + (idx / (total_points - 1)) * plot_w

    def y_pos(value: float) -> float:
        return pad_t + plot_h - ((value - value_min) / value_range) * plot_h

    history_fraction = 0.22 if history_count > 1 else 0.0
    future_fraction = 1.0 - history_fraction

    def x_pos_history(idx: int) -> float:
        if history_count <= 1:
            return pad_l + (plot_w * history_fraction)
        return pad_l + ((idx / max(history_count - 1, 1)) * plot_w * history_fraction)

    def x_pos_future(idx: int) -> float:
        if len(points) <= 1:
            return pad_l + (plot_w * history_fraction)
        return pad_l + (plot_w * history_fraction) + ((idx / max(len(points) - 1, 1)) * plot_w * future_fraction)

    history_points = [
        f"{x_pos_history(idx):.1f},{y_pos(float(value)):.1f}"
        for idx, value in enumerate(history_values)
    ]
    history_points.append(f"{x_pos_future(0):.1f},{y_pos(summary['current_value']):.1f}")
    history_polyline = " ".join(history_points)
    current_y = y_pos(summary["current_value"])

    x_labels = []
    if history_count > 1:
        x_labels.append({"x": f"{x_pos_history(0):.1f}", "label": history_dates[0][5:] if len(history_dates[0]) >= 10 else history_dates[0]})
        mid_hist_idx = history_count // 2
        x_labels.append({"x": f"{x_pos_history(mid_hist_idx):.1f}", "label": history_dates[mid_hist_idx][5:] if len(history_dates[mid_hist_idx]) >= 10 else history_dates[mid_hist_idx]})
    x_labels.append({"x": f"{x_pos_future(0):.1f}", "label": "Now"})
    future_label_step = max((len(points) - 1) // 4, 1)
    for idx, point in enumerate(points[1:], start=1):
        if idx == len(points) - 1 or idx % future_label_step == 0:
            x_labels.append({"x": f"{x_pos_future(idx):.1f}", "label": point["label"]})

    y_labels = []
    for idx in range(5):
        label_value = value_min + (value_range * idx / 4)
        y_labels.append({"y": f"{y_pos(label_value):.1f}", "label": f"${label_value:,.0f}"})

    series = []
    for meta in series_meta:
        upper_points = []
        lower_points = []
        median_points = []
        for idx, point in enumerate(points):
            x_val = x_pos_future(idx)
            dist = point[meta["key"]]
            upper_points.append(f"{x_val:.1f},{y_pos(dist['ci95_high']):.1f}")
            lower_points.append(f"{x_val:.1f},{y_pos(dist['ci95_low']):.1f}")
            median_points.append(f"{x_val:.1f},{y_pos(dist['median']):.1f}")
        final_dist = points[-1][meta["key"]]
        series.append(
            {
                "label": meta["label"],
                "color": meta["color"],
                "fill_opacity": meta["fill_opacity"],
                "band_points": " ".join(upper_points + list(reversed(lower_points))),
                "upper_points": " ".join(upper_points),
                "lower_points": " ".join(lower_points),
                "median_points": " ".join(median_points),
                "final_x": f"{x_pos_future(len(points) - 1):.1f}",
                "final_median_y": f"{y_pos(final_dist['median']):.1f}",
                "final_median_label": f"${final_dist['median']:,.0f}",
                "final_low_label": f"${final_dist['ci95_low']:,.0f}",
                "final_high_label": f"${final_dist['ci95_high']:,.0f}",
            }
        )

    return {
        "has_data": True,
        "ch_w": chart_w,
        "ch_h": chart_h,
        "plot_x": pad_l,
        "plot_y": pad_t,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "history_points": history_polyline,
        "history_start_label": history_dates[0][5:] if history_dates and len(history_dates[0]) >= 10 else (history_dates[0] if history_dates else ""),
        "current_x": f"{x_pos_future(0):.1f}",
        "current_y": f"{current_y:.1f}",
        "series": series,
        "current_label": f"${summary['current_value']:,.0f}",
        "confidence_label": "95%",
        "horizon_days": int(diagnostics.get("horizon_days", 0) or 0),
    }
