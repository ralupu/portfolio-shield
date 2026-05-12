import unittest

from hedge_sizing import calculate_underlying_hedge_adjustment
from portfolio_state import InstrumentMetadata, PortfolioPosition, PortfolioState


class TestHedgeSizing(unittest.TestCase):
    def test_neutralizes_simple_long_delta(self):
        state = PortfolioState([
            PortfolioPosition(ticker="AAPL", quantity=100, direction="long", market_price=190.0, delta_per_unit=1.0)
        ])
        result = calculate_underlying_hedge_adjustment(state, hedge_underlying="SPY")
        self.assertEqual(result.current_net_delta, 100.0)
        self.assertEqual(result.action, "sell")
        self.assertEqual(result.required_units_rounded, -100)
        self.assertEqual(result.estimated_post_trade_delta, 0.0)

    def test_handles_existing_hedge_positions(self):
        hedge_meta = InstrumentMetadata(
            ticker="SPY",
            instrument_type="etf",
            contract_multiplier=1,
            extra={"is_hedge": True},
        )
        state = PortfolioState([
            PortfolioPosition(ticker="AAPL", quantity=100, direction="long", market_price=190.0, delta_per_unit=1.0),
            PortfolioPosition(ticker="SPY", quantity=30, direction="short", market_price=510.0, delta_per_unit=1.0, metadata=hedge_meta),
        ])
        result = calculate_underlying_hedge_adjustment(state, hedge_underlying="SPY")
        self.assertEqual(result.current_net_delta, 70.0)
        self.assertEqual(result.existing_hedge_delta, -30.0)
        self.assertEqual(result.required_units_rounded, -70)
        self.assertEqual(result.estimated_post_trade_delta, 0.0)

    def test_handles_existing_option_hedge_position(self):
        hedge_meta = InstrumentMetadata(
            ticker="SPY",
            instrument_type="option",
            contract_multiplier=100,
            extra={"is_hedge": True},
        )
        state = PortfolioState([
            PortfolioPosition(ticker="AAPL", quantity=100, direction="long", market_price=190.0, delta_per_unit=1.0),
            PortfolioPosition(ticker="SPY", quantity=1, direction="long", market_price=6.2, delta_per_unit=-0.40, metadata=hedge_meta),
        ])
        result = calculate_underlying_hedge_adjustment(state, hedge_underlying="SPY")
        self.assertEqual(result.current_net_delta, 60.0)
        self.assertEqual(result.existing_hedge_delta, -40.0)
        self.assertEqual(result.required_units_rounded, -60)
        self.assertEqual(result.estimated_post_trade_delta, 0.0)


if __name__ == "__main__":
    unittest.main()
