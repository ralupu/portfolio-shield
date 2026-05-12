"""
policy.py - Configurable hedging policy interface.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta


@dataclass(slots=True)
class HedgePolicyConfig:
    move_threshold_pct: float = 5.0
    review_frequency_days: int = 14
    min_days_to_roll: int = 21
    enable_threshold_rule: bool = True
    enable_periodic_rule: bool = True

    @classmethod
    def from_profile(cls, profile: dict) -> "HedgePolicyConfig":
        return cls(
            move_threshold_pct=max(1.0, float(profile.get("move_threshold_pct", 5.0) or 5.0)),
            review_frequency_days=max(1, int(profile.get("review_frequency_days", 14) or 14)),
            min_days_to_roll=max(1, int(profile.get("min_days_to_roll", 21) or 21)),
            enable_threshold_rule=bool(profile.get("enable_threshold_rule", True)),
            enable_periodic_rule=bool(profile.get("enable_periodic_rule", True)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class HedgePolicyDecision:
    next_review_date: str
    review_window_days: int
    triggers: list[str]
    summary: str
    config: dict


class HedgePolicy:
    def __init__(self, config: HedgePolicyConfig):
        self.config = config

    def evaluate(self, *, as_of: date, target_dte: int, portfolio_net_delta: float, hedge_delta: float) -> HedgePolicyDecision:
        raise NotImplementedError


class ThresholdPeriodicPolicy(HedgePolicy):
    def evaluate(self, *, as_of: date, target_dte: int, portfolio_net_delta: float, hedge_delta: float) -> HedgePolicyDecision:
        review_window_days = min(self.config.review_frequency_days, max(target_dte, 1))
        next_review_date = as_of + timedelta(days=review_window_days)

        triggers: list[str] = []
        if self.config.enable_threshold_rule:
            triggers.append(
                f"Review the hedge if the portfolio moves by {self.config.move_threshold_pct:.1f}% or more from today's level."
            )
        if self.config.enable_periodic_rule:
            triggers.append(
                f"Check the hedge every {self.config.review_frequency_days} days even if markets are calm."
            )
        triggers.append(
            f"Consider rolling or replacing the hedge when time to expiry falls below {self.config.min_days_to_roll} days."
        )

        summary_bits = []
        if self.config.enable_threshold_rule:
            summary_bits.append(f"{self.config.move_threshold_pct:.1f}% move trigger")
        if self.config.enable_periodic_rule:
            summary_bits.append(f"{self.config.review_frequency_days}-day check-in")
        summary_bits.append(f"consider rolling below {self.config.min_days_to_roll} days to expiry")
        summary = "Review plan: " + ", ".join(summary_bits)

        if abs(portfolio_net_delta) > 0:
            coverage = abs(hedge_delta) / abs(portfolio_net_delta) * 100
            summary += f". The current hedge offsets about {coverage:.1f}% of portfolio sensitivity."

        return HedgePolicyDecision(
            next_review_date=next_review_date.isoformat(),
            review_window_days=review_window_days,
            triggers=triggers,
            summary=summary,
            config=self.config.to_dict(),
        )
