import unittest

from portfolio_state import InstrumentMetadata, PortfolioPosition, PortfolioState, aggregate_portfolio_net_delta


class TestPortfolioState(unittest.TestCase):
    def test_add_update_remove_position(self):
        state = PortfolioState()
        position = PortfolioPosition(
            ticker="AAPL",
            quantity=100,
            direction="long",
            market_price=190.0,
        )
        position_id = state.add_position(position)
        self.assertEqual(len(state.list_positions()), 1)
        self.assertAlmostEqual(state.total_market_value(), 19000.0)

        updated = state.update_position(position_id, quantity=120, market_price=195.0)
        self.assertEqual(updated.quantity, 120)
        self.assertAlmostEqual(updated.market_price, 195.0)
        self.assertAlmostEqual(state.total_market_value(), 23400.0)

        removed = state.remove_position(position_id)
        self.assertEqual(removed.ticker, "AAPL")
        self.assertEqual(len(state.list_positions()), 0)

    def test_state_persists_across_steps(self):
        state = PortfolioState.from_equity_snapshot([
            {"ticker": "AAPL", "shares": 100, "price": 190.0},
            {"ticker": "MSFT", "shares": 50, "price": 410.0},
        ])
        self.assertEqual(len(state.list_positions()), 2)
        first = state.list_positions()[0]
        state.update_position(first.position_id, market_price=200.0)
        self.assertEqual(state.get_position(first.position_id).market_price, 200.0)


class TestNetDeltaAggregation(unittest.TestCase):
    def test_long_equity_delta_scaling(self):
        state = PortfolioState([
            PortfolioPosition(ticker="AAPL", quantity=100, direction="long", market_price=190.0, delta_per_unit=1.0)
        ])
        self.assertEqual(aggregate_portfolio_net_delta(state), 100.0)

    def test_short_equity_delta_sign(self):
        state = PortfolioState([
            PortfolioPosition(ticker="MSFT", quantity=40, direction="short", market_price=410.0, delta_per_unit=1.0)
        ])
        self.assertEqual(aggregate_portfolio_net_delta(state), -40.0)

    def test_option_delta_scaling(self):
        option_meta = InstrumentMetadata(ticker="SPY", instrument_type="option", contract_multiplier=100)
        state = PortfolioState([
            PortfolioPosition(
                ticker="SPY",
                quantity=2,
                direction="long",
                market_price=6.2,
                delta_per_unit=-0.35,
                metadata=option_meta,
            )
        ])
        self.assertEqual(aggregate_portfolio_net_delta(state), -70.0)

    def test_mixed_portfolio_net_delta(self):
        option_meta = InstrumentMetadata(ticker="SPY", instrument_type="option", contract_multiplier=100)
        state = PortfolioState([
            PortfolioPosition(ticker="AAPL", quantity=100, direction="long", market_price=190.0, delta_per_unit=1.0),
            PortfolioPosition(ticker="MSFT", quantity=25, direction="short", market_price=410.0, delta_per_unit=1.0),
            PortfolioPosition(ticker="SPY", quantity=1, direction="long", market_price=6.2, delta_per_unit=-0.40, metadata=option_meta),
        ])
        self.assertEqual(aggregate_portfolio_net_delta(state), 35.0)


if __name__ == "__main__":
    unittest.main()
