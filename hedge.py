"""
hedge.py - Portfolio-level hedge recommendation engine.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from hedge_sizing import calculate_underlying_hedge_adjustment
from options import HEDGE_PCTS, select_put
from policy import HedgePolicyConfig, ThresholdPeriodicPolicy
from portfolio_state import PortfolioState, aggregate_portfolio_net_delta
from quotes import fetch_quote
from scenarios import build_scenarios

logger = logging.getLogger(__name__)

OBJECTIVE_LABELS = {
    "reduce_downside": "Reduce 1-3 month downside",
    "protect_gains": "Protect recent gains",
    "crash_hedge": "Crash hedge only",
    "partial_delta": "Partial delta hedge",
}

EXPERIENCE_WARNINGS = {
    "beginner": "This plan uses listed put options. Use limit orders and review your broker's options disclosure before trading.",
    "intermediate": "Monitor the hedge after large moves or if the underlying mix changes materially.",
    "advanced": "Option sensitivity and event risk can change quickly; use this as a review tool, not an automatic trade ticket.",
}


def calculate_hedge(
    ticker: str,
    shares: int,
    price: float,
    hedge_level: str,
    target_dte: int = 45,
) -> dict:
    """Calculate hedge details for a single position."""
    hedge_pct = HEDGE_PCTS[hedge_level]
    put = select_put(ticker, price, hedge_level, target_dte=target_dte)

    put_delta = abs(put["delta"])
    if put_delta < 0.001:
        put_delta = 0.001

    target_delta = shares * hedge_pct
    contracts = max(1, round(target_delta / (100 * put_delta)))
    cost = round(contracts * put["mid_price"] * 100, 2)
    position_value = round(shares * price, 2)
    coverage_notional = round(contracts * 100 * put_delta * price, 2)
    breakeven = round(put["strike"] - put["mid_price"], 2)
    cost_pct = round((cost / position_value) * 100, 2) if position_value > 0 else 0.0

    return {
        "ticker": ticker.upper(),
        "underlying": ticker.upper(),
        "strategy_scope": "single",
        "shares": shares,
        "price": round(price, 2),
        "underlying_price": round(price, 2),
        "position_value": position_value,
        "target_delta": round(target_delta, 3),
        "contracts": contracts,
        "strike": put["strike"],
        "expiry": put["expiry"],
        "dte": put["dte"],
        "mid_price": put["mid_price"],
        "bid": put["bid"],
        "ask": put["ask"],
        "spread_pct": put["spread_pct"],
        "cost": cost,
        "breakeven": breakeven,
        "breakeven_label": "Put profit below this price",
        "position_breakeven": round(price + (cost / shares), 2) if shares > 0 else price,
        "position_breakeven_label": "Stock + hedge cost basis",
        "cost_pct": cost_pct,
        "iv": put["iv"],
        "delta": put["delta"],
        "open_interest": put["open_interest"],
        "volume": put["volume"],
        "coverage_notional": coverage_notional,
        "hedge_delta": round(contracts * 100 * abs(put["delta"]), 3),
        "is_fallback": put["is_fallback"],
    }


def calculate_portfolio_hedge(
    portfolio: PortfolioState | list[dict],
    hedge_level: str,
    target_dte: int = 45,
) -> dict:
    """Build a single-name protective-put candidate for the full portfolio."""
    portfolio_state = _coerce_portfolio_state(portfolio)
    legs = []
    total_value = 0.0
    total_cost = 0.0
    total_coverage = 0.0
    total_hedge_delta = 0.0
    any_fallback = False
    errors = []

    for position in portfolio_state.list_positions():
        try:
            leg = calculate_hedge(
                ticker=position.ticker,
                shares=int(abs(position.signed_quantity)),
                price=position.market_price,
                hedge_level=hedge_level,
                target_dte=target_dte,
            )
            leg["position_id"] = position.position_id
            leg["direction"] = position.direction
            leg["instrument_type"] = position.metadata.instrument_type
            leg["signed_position_delta"] = round(position.signed_delta, 3)
            legs.append(leg)
            total_value += abs(position.market_value)
            total_cost += leg["cost"]
            total_coverage += leg["coverage_notional"]
            total_hedge_delta += leg["hedge_delta"]
            any_fallback = any_fallback or leg["is_fallback"]
        except Exception as exc:
            logger.error("Hedge calc failed for %s: %s", position.ticker, exc)
            errors.append(f"{position.ticker}: {exc}")

    net_delta = aggregate_portfolio_net_delta(portfolio_state)
    total_cost_pct = round((total_cost / total_value) * 100, 2) if total_value > 0 else 0.0
    average_spread = round(sum(leg["spread_pct"] for leg in legs) / len(legs), 2) if legs else 0.0
    delta_coverage_pct = round((total_hedge_delta / abs(net_delta)) * 100, 1) if abs(net_delta) > 0 else 0.0

    return {
        "strategy": "single_name",
        "strategy_label": "Protective puts on each stock",
        "strategy_scope": "single",
        "positions": legs,
        "contracts": [
            {
                "underlying": leg["underlying"],
                "underlying_price": leg["underlying_price"],
                "strategy_scope": "single",
                "coverage_ratio": round(HEDGE_PCTS[hedge_level], 4),
                "market_beta": 1.0,
                "position_shares": leg["shares"],
                "target_delta": leg["target_delta"],
                "contracts": leg["contracts"],
                "strike": leg["strike"],
                "expiry": leg["expiry"],
                "cost": leg["cost"],
                "delta": leg["delta"],
                "dte": leg["dte"],
                "open_interest": leg["open_interest"],
                "spread_pct": leg["spread_pct"],
                "iv": leg["iv"],
                "ticker": leg["ticker"],
                "hedge_delta": leg["hedge_delta"],
            }
            for leg in legs
        ],
        "portfolio_state": portfolio_state.to_dict(),
        "portfolio_net_delta": round(net_delta, 3),
        "hedge_target_delta": round(abs(net_delta) * HEDGE_PCTS[hedge_level], 3),
        "hedge_delta": round(total_hedge_delta, 3),
        "total_value": round(total_value, 2),
        "total_cost": round(total_cost, 2),
        "total_cost_pct": total_cost_pct,
        "coverage_notional": round(total_coverage, 2),
        "market_delta_coverage_pct": delta_coverage_pct,
        "average_spread_pct": average_spread,
        "net_protected": round(total_value - total_cost, 2),
        "protection_threshold": _weighted_threshold(legs),
        "any_fallback": any_fallback,
        "errors": errors,
    }


def calculate_index_hedge(
    portfolio: PortfolioState | list[dict],
    hedge_level: str,
    portfolio_beta: float | None,
    target_dte: int = 45,
    index_ticker: str = "SPY",
) -> dict:
    """Build an index-put hedge candidate based on beta-adjusted market exposure."""
    portfolio_state = _coerce_portfolio_state(portfolio)
    total_value = round(sum(abs(position.market_value) for position in portfolio_state.list_positions()), 2)
    if total_value <= 0:
        raise ValueError("Portfolio value must be positive")

    raw_net_delta = aggregate_portfolio_net_delta(portfolio_state)
    beta = portfolio_beta if portfolio_beta and portfolio_beta > 0 else 1.0
    hedge_pct = HEDGE_PCTS[hedge_level]
    quote = fetch_quote(index_ticker)
    price = quote["price"]
    put = select_put(index_ticker, price, hedge_level, target_dte=target_dte)

    delta = abs(put["delta"])
    if delta < 0.001:
        delta = 0.001

    market_exposure = round(total_value * beta, 2)
    index_equivalent_delta = market_exposure / max(price, 0.01)
    target_portfolio_delta = index_equivalent_delta * hedge_pct
    contracts = max(1, round(target_portfolio_delta / (100 * delta)))
    total_cost = round(contracts * put["mid_price"] * 100, 2)
    hedge_delta = round(contracts * 100 * delta, 3)
    coverage_notional = round(contracts * 100 * delta * price, 2)
    total_cost_pct = round((total_cost / total_value) * 100, 2)
    market_delta_coverage_pct = round((hedge_delta / max(index_equivalent_delta, 0.001)) * 100, 1) if index_equivalent_delta > 0 else 0.0

    contract = {
        "ticker": index_ticker,
        "underlying": index_ticker,
        "underlying_price": price,
        "strategy_scope": "index",
        "coverage_ratio": round(hedge_pct, 4),
        "market_beta": round(beta, 4),
        "position_shares": 0.0,
        "target_delta": round(target_portfolio_delta, 3),
        "contracts": contracts,
        "strike": put["strike"],
        "expiry": put["expiry"],
        "cost": total_cost,
        "mid_price": put["mid_price"],
        "delta": put["delta"],
        "dte": put["dte"],
        "open_interest": put["open_interest"],
        "volume": put["volume"],
        "spread_pct": put["spread_pct"],
        "iv": put["iv"],
        "coverage_notional": coverage_notional,
        "hedge_delta": hedge_delta,
        "is_fallback": put["is_fallback"],
    }

    return {
        "strategy": "index_put",
        "strategy_label": f"{index_ticker} protective puts",
        "strategy_scope": "index",
        "positions": [],
        "contracts": [contract],
        "portfolio_state": portfolio_state.to_dict(),
        "portfolio_net_delta": round(index_equivalent_delta, 3),
        "portfolio_raw_net_delta": round(raw_net_delta, 3),
        "hedge_target_delta": round(target_portfolio_delta, 3),
        "hedge_delta": hedge_delta,
        "total_value": total_value,
        "total_cost": total_cost,
        "total_cost_pct": total_cost_pct,
        "coverage_notional": coverage_notional,
        "market_exposure": market_exposure,
        "market_delta_coverage_pct": market_delta_coverage_pct,
        "average_spread_pct": put["spread_pct"],
        "net_protected": round(total_value - total_cost, 2),
        "protection_threshold": round((1 - put["strike"] / price) * 100, 1),
        "any_fallback": put["is_fallback"],
        "errors": [],
    }


def build_delta_advice(
    positions: PortfolioState | list[dict[str, Any]],
    hedge_level: str,
    profile: dict,
    portfolio_beta: float | None = None,
    position_betas: dict[str, float] | None = None,
) -> dict:
    """Compare hedge candidates and return one retail-facing recommendation."""
    portfolio_state = _coerce_portfolio_state(positions)
    target_dte = max(21, min(int(profile.get("horizon_days", 45) or 45), 90))
    budget = float(profile.get("max_budget", 0) or 0)
    objective = profile.get("objective", "reduce_downside")
    experience = profile.get("experience", "beginner")
    target_pct = HEDGE_PCTS[hedge_level] * 100

    candidates = [calculate_portfolio_hedge(portfolio_state, hedge_level, target_dte=target_dte)]
    if len(portfolio_state.list_positions()) > 1:
        try:
            candidates.append(
                calculate_index_hedge(
                    portfolio_state,
                    hedge_level,
                    portfolio_beta=portfolio_beta,
                    target_dte=target_dte,
                )
            )
        except Exception as exc:
            logger.warning("Index hedge unavailable: %s", exc)

    scored = []
    for candidate in candidates:
        score = _score_candidate(candidate, objective, budget, experience, target_pct)
        candidate["score"] = score
        candidate["rationale"] = _build_rationale(candidate, objective, budget)
        scored.append(candidate)

    recommendation = min(scored, key=lambda item: item["score"])
    policy_config = HedgePolicyConfig.from_profile(profile)
    policy = ThresholdPeriodicPolicy(policy_config)
    policy_decision = policy.evaluate(
        as_of=date.today(),
        target_dte=target_dte,
        portfolio_net_delta=recommendation["portfolio_net_delta"],
        hedge_delta=recommendation["hedge_delta"],
    )
    recommendation["objective_label"] = OBJECTIVE_LABELS.get(objective, OBJECTIVE_LABELS["reduce_downside"])
    recommendation["experience_warning"] = EXPERIENCE_WARNINGS.get(experience, EXPERIENCE_WARNINGS["beginner"])
    recommendation["review_date"] = policy_decision.next_review_date
    recommendation["review_window_days"] = policy_decision.review_window_days
    recommendation["rebalance_triggers"] = policy_decision.triggers
    recommendation["policy_summary"] = policy_decision.summary
    recommendation["policy_config"] = policy_decision.config
    recommendation["suitability_notes"] = _build_suitability_notes(recommendation, experience, budget)
    if position_betas:
        recommendation.setdefault("portfolio_state", {})["position_betas"] = {
            pos.ticker: float(position_betas.get(pos.ticker, portfolio_beta if portfolio_beta is not None else 1.0))
            for pos in portfolio_state.list_positions()
        }
    elif portfolio_beta is not None:
        recommendation.setdefault("portfolio_state", {})["position_betas"] = (
            {pos.ticker: portfolio_beta for pos in portfolio_state.list_positions()}
        )
    recommendation["scenarios"] = build_scenarios(
        _positions_for_scenarios(portfolio_state),
        recommendation,
        portfolio_beta,
    )
    recommendation["alternatives"] = [
        {
            "strategy": candidate["strategy_label"],
            "cost_pct": candidate["total_cost_pct"],
            "coverage_pct": candidate["market_delta_coverage_pct"],
            "score": round(candidate["score"], 1),
        }
        for candidate in sorted(scored, key=lambda item: item["score"])
        if candidate["strategy"] != recommendation["strategy"]
    ]
    sizing_underlying = str(profile.get("sizing_underlying") or ("SPY" if len(portfolio_state.list_positions()) > 1 else portfolio_state.list_positions()[0].ticker)).upper()
    sizing_result = calculate_underlying_hedge_adjustment(
        portfolio_state,
        hedge_underlying=sizing_underlying,
        hedge_delta_per_unit=1.0,
        contract_multiplier=1,
    )
    recommendation["neutralization_estimate"] = sizing_result.to_dict()
    recommendation["target_dte"] = target_dte
    recommendation["target_delta_reduction_pct"] = round(target_pct, 0)
    recommendation["residual_delta_pct"] = round(max(0.0, 100.0 - recommendation["market_delta_coverage_pct"]), 1)
    return recommendation


def _coerce_portfolio_state(portfolio: PortfolioState | list[dict[str, Any]]) -> PortfolioState:
    if isinstance(portfolio, PortfolioState):
        return portfolio
    return PortfolioState.from_equity_snapshot(portfolio)


def _positions_for_scenarios(portfolio_state: PortfolioState) -> list[dict[str, Any]]:
    return [
        {
            "ticker": position.ticker,
            "shares": abs(position.signed_quantity),
            "price": position.market_price,
            "avg_cost": position.avg_cost,
        }
        for position in portfolio_state.list_positions()
    ]


def _weighted_threshold(legs: list[dict]) -> float:
    if not legs:
        return 0.0
    total_value = sum(leg["position_value"] for leg in legs)
    if total_value <= 0:
        return 0.0
    weighted_otm = sum((1 - leg["strike"] / leg["price"]) * leg["position_value"] for leg in legs) / total_value
    return round(weighted_otm * 100, 1)


def _score_candidate(candidate: dict, objective: str, budget: float, experience: str, target_pct: float) -> float:
    score = candidate["total_cost_pct"] * 4
    score += abs(candidate["market_delta_coverage_pct"] - target_pct) * 0.7
    score += candidate.get("average_spread_pct", 0.0) * 0.6

    if budget and candidate["total_cost"] > budget:
        score += 35 + (candidate["total_cost"] - budget) / max(budget, 1) * 20

    if candidate.get("any_fallback"):
        score += 20

    if objective == "crash_hedge":
        if candidate["strategy"] == "index_put":
            score -= 8
        score += candidate["total_cost_pct"] * 2
    elif objective == "protect_gains":
        if candidate["strategy"] == "single_name":
            score -= 6
        score -= candidate["market_delta_coverage_pct"] * 0.08
    elif objective == "partial_delta":
        score += candidate["total_cost_pct"] * 4
    else:
        score -= candidate["market_delta_coverage_pct"] * 0.04

    if experience == "beginner" and candidate["average_spread_pct"] > 12:
        score += 8

    return round(score, 3)


def _build_rationale(candidate: dict, objective: str, budget: float) -> str:
    base = []
    if candidate["strategy"] == "index_put":
        base.append("This plan uses one liquid index hedge instead of several single-stock contracts")
    else:
        base.append("This plan keeps the protection tied directly to each stock in the portfolio")

    base.append(f"Estimated portfolio delta is about {candidate['portfolio_net_delta']:.1f}")
    base.append(f"target hedge delta is about {candidate['hedge_target_delta']:.1f}")
    base.append(f"selected contracts provide about {candidate['hedge_delta']:.1f} of hedge delta")
    base.append(f"estimated premium cost is {candidate['total_cost_pct']:.2f}% of portfolio value")

    if budget:
        if candidate["total_cost"] <= budget:
            base.append("and it stays within your stated budget")
        else:
            base.append("but it is above your current budget")

    if objective == "crash_hedge":
        base.append("which fits a lower-cost downside cushion")
    elif objective == "protect_gains":
        base.append("which fits a tighter protection goal")
    else:
        base.append("which offers a balanced mix of cost, coverage, and liquidity")

    return ". ".join(base) + "."


def _build_suitability_notes(candidate: dict, experience: str, budget: float) -> list[str]:
    notes = []
    if candidate["average_spread_pct"] > 10:
        notes.append("Some selected options have wider bid-ask spreads than ideal. Use limit orders and verify liquidity before acting.")
    if candidate.get("any_fallback"):
        notes.append("At least one contract uses estimated pricing because a liquid option chain was not available. Treat those figures with extra caution.")
    if budget and candidate["total_cost"] > budget:
        notes.append("This plan is above your stated budget, so consider wider strikes, lower target coverage, or a shorter hedge horizon.")
    if experience == "beginner":
        notes.append("If you are newer to options, avoid very short-dated contracts and confirm your account approval before trading.")
    return notes














