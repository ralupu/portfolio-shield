"""
hedge_sizing.py - Explicit net-delta hedge sizing logic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from portfolio_state import PortfolioPosition, PortfolioState, aggregate_portfolio_net_delta


@dataclass(slots=True)
class HedgeSizingResult:
    hedge_underlying: str
    current_net_delta: float
    existing_hedge_delta: float
    delta_to_neutralize: float
    hedge_delta_per_unit: float
    contract_multiplier: int
    required_units_exact: float
    required_units_rounded: int
    action: str
    added_hedge_delta_rounded: float
    estimated_post_trade_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_underlying_hedge_adjustment(
    portfolio_state: PortfolioState,
    hedge_underlying: str,
    hedge_delta_per_unit: float = 1.0,
    contract_multiplier: int = 1,
) -> HedgeSizingResult:
    if contract_multiplier <= 0:
        raise ValueError("contract_multiplier must be positive")
    if hedge_delta_per_unit == 0:
        raise ValueError("hedge_delta_per_unit must be non-zero")

    current_net_delta = aggregate_portfolio_net_delta(portfolio_state)
    existing_hedge_delta = round(sum(position.signed_delta for position in _existing_hedge_positions(portfolio_state)), 6)
    delta_to_neutralize = round(-current_net_delta, 6)
    unit_delta = hedge_delta_per_unit * contract_multiplier
    required_units_exact = delta_to_neutralize / unit_delta
    required_units_rounded = _round_away_from_zero(required_units_exact)
    added_hedge_delta_rounded = round(required_units_rounded * unit_delta, 6)
    estimated_post_trade_delta = round(current_net_delta + added_hedge_delta_rounded, 6)

    return HedgeSizingResult(
        hedge_underlying=hedge_underlying.upper(),
        current_net_delta=round(current_net_delta, 6),
        existing_hedge_delta=existing_hedge_delta,
        delta_to_neutralize=delta_to_neutralize,
        hedge_delta_per_unit=hedge_delta_per_unit,
        contract_multiplier=contract_multiplier,
        required_units_exact=round(required_units_exact, 6),
        required_units_rounded=required_units_rounded,
        action="buy" if required_units_rounded > 0 else ("sell" if required_units_rounded < 0 else "hold"),
        added_hedge_delta_rounded=added_hedge_delta_rounded,
        estimated_post_trade_delta=estimated_post_trade_delta,
    )


def _existing_hedge_positions(portfolio_state: PortfolioState) -> list[PortfolioPosition]:
    return [
        position
        for position in portfolio_state.list_positions()
        if bool(position.metadata.extra.get("is_hedge", False))
    ]


def _round_away_from_zero(value: float) -> int:
    if value > 0:
        return math.ceil(value)
    if value < 0:
        return math.floor(value)
    return 0
