"""
portfolio_state.py - Internal portfolio position model and net-delta aggregation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4


VALID_DIRECTIONS = {"long", "short"}
VALID_INSTRUMENT_TYPES = {"equity", "option", "future", "etf", "index"}


@dataclass(slots=True)
class InstrumentMetadata:
    ticker: str
    instrument_type: str = "equity"
    contract_multiplier: int = 1
    currency: str = "USD"
    venue: str = "SMART"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PortfolioPosition:
    ticker: str
    quantity: float
    direction: str = "long"
    market_price: float = 0.0
    avg_cost: float | None = None
    delta_per_unit: float = 1.0
    metadata: InstrumentMetadata | None = None
    position_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        self.ticker = self.ticker.upper().strip()
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(f"Invalid direction '{self.direction}'")
        if self.quantity < 0:
            raise ValueError("Quantity must be non-negative")
        if self.metadata is None:
            self.metadata = InstrumentMetadata(ticker=self.ticker)
        if self.metadata.instrument_type not in VALID_INSTRUMENT_TYPES:
            raise ValueError(f"Invalid instrument type '{self.metadata.instrument_type}'")
        if self.metadata.contract_multiplier <= 0:
            raise ValueError("Contract multiplier must be positive")
        if self.avg_cost is None:
            self.avg_cost = self.market_price

    @property
    def signed_quantity(self) -> float:
        return self.quantity if self.direction == "long" else -self.quantity

    @property
    def signed_delta(self) -> float:
        return self.signed_quantity * self.metadata.contract_multiplier * self.delta_per_unit

    @property
    def market_value(self) -> float:
        return self.signed_quantity * self.market_price * self.metadata.contract_multiplier

    def update(self, **changes) -> None:
        metadata_updates = changes.pop("metadata", None)
        for key, value in changes.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if metadata_updates:
            for key, value in metadata_updates.items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "ticker": self.ticker,
            "quantity": self.quantity,
            "direction": self.direction,
            "market_price": self.market_price,
            "avg_cost": self.avg_cost,
            "delta_per_unit": self.delta_per_unit,
            "metadata": self.metadata.to_dict(),
            "signed_quantity": self.signed_quantity,
            "signed_delta": self.signed_delta,
            "market_value": self.market_value,
        }


class PortfolioState:
    """Mutable in-memory representation of portfolio positions."""

    def __init__(self, positions: list[PortfolioPosition] | None = None):
        self._positions: dict[str, PortfolioPosition] = {}
        if positions:
            for position in positions:
                self.add_position(position)

    def add_position(self, position: PortfolioPosition) -> str:
        self._positions[position.position_id] = position
        return position.position_id

    def remove_position(self, position_id: str) -> PortfolioPosition:
        if position_id not in self._positions:
            raise KeyError(position_id)
        return self._positions.pop(position_id)

    def update_position(self, position_id: str, **changes) -> PortfolioPosition:
        position = self.get_position(position_id)
        position.update(**changes)
        return position

    def get_position(self, position_id: str) -> PortfolioPosition:
        if position_id not in self._positions:
            raise KeyError(position_id)
        return self._positions[position_id]

    def list_positions(self) -> list[PortfolioPosition]:
        return list(self._positions.values())

    def replace_positions(self, positions: list[PortfolioPosition]) -> None:
        self._positions = {}
        for position in positions:
            self.add_position(position)

    def total_market_value(self) -> float:
        return sum(position.market_value for position in self.list_positions())

    def net_delta(self) -> float:
        return aggregate_portfolio_net_delta(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": [position.to_dict() for position in self.list_positions()],
            "total_market_value": self.total_market_value(),
            "net_delta": self.net_delta(),
        }

    @classmethod
    def from_equity_snapshot(cls, positions: list[dict[str, Any]]) -> "PortfolioState":
        state_positions = []
        for pos in positions:
            ticker = pos["ticker"].upper()
            shares = float(pos.get("shares", pos.get("quantity", 0)))
            direction = pos.get("direction", "long")
            state_positions.append(
                PortfolioPosition(
                    ticker=ticker,
                    quantity=abs(shares),
                    direction=direction if shares >= 0 else "short",
                    market_price=float(pos.get("price", pos.get("market_price", 0.0))),
                    avg_cost=float(pos.get("avg_cost", pos.get("price", 0.0))),
                    delta_per_unit=float(pos.get("delta_per_unit", 1.0)),
                    metadata=InstrumentMetadata(
                        ticker=ticker,
                        instrument_type=pos.get("instrument_type", "equity"),
                        contract_multiplier=int(pos.get("contract_multiplier", 1)),
                        extra={
                            "source": pos.get("source", "form"),
                        },
                    ),
                )
            )
        return cls(state_positions)


def aggregate_portfolio_net_delta(portfolio: PortfolioState | list[PortfolioPosition]) -> float:
    positions = portfolio.list_positions() if isinstance(portfolio, PortfolioState) else portfolio
    return round(sum(position.signed_delta for position in positions), 6)
