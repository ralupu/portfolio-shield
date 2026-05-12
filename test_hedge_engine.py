import unittest
from unittest.mock import patch

import hedge
from portfolio_state import PortfolioState


class TestIndexHedgeSizing(unittest.TestCase):
    @staticmethod
    def _stub_quote(_ticker: str):
        return {"ticker": "SPY", "price": 500.0}

    @staticmethod
    def _stub_put(_ticker: str, _price: float, _hedge_level: str, target_dte: int = 45):
        return {
            "ticker": "SPY",
            "strike": 475.0,
            "expiry": "2026-06-19",
            "dte": target_dte,
            "mid_price": 5.0,
            "bid": 4.9,
            "ask": 5.1,
            "spread_pct": 4.0,
            "delta": -0.40,
            "iv": 0.22,
            "volume": 1000,
            "open_interest": 5000,
            "is_fallback": False,
        }

    def test_index_hedge_uses_market_exposure_not_raw_share_count(self):
        portfolio = PortfolioState.from_equity_snapshot(
            [
                {"ticker": "AAPL", "shares": 1000, "price": 10.0},
                {"ticker": "MSFT", "shares": 1000, "price": 10.0},
            ]
        )
        with patch.object(hedge, "fetch_quote", side_effect=self._stub_quote), \
             patch.object(hedge, "select_put", side_effect=self._stub_put):
            result = hedge.calculate_index_hedge(portfolio, "moderate", portfolio_beta=1.0, target_dte=45)

        self.assertEqual(result["contracts"][0]["contracts"], 1)
        self.assertAlmostEqual(result["portfolio_net_delta"], 40.0, places=3)
        self.assertAlmostEqual(result["portfolio_raw_net_delta"], 2000.0, places=3)
        self.assertAlmostEqual(result["market_delta_coverage_pct"], 100.0, places=1)


class TestScenarioBetaThreading(unittest.TestCase):
    def test_build_delta_advice_threads_position_betas_into_recommendation(self):
        portfolio = PortfolioState.from_equity_snapshot(
            [
                {"ticker": "AAPL", "shares": 10, "price": 100.0},
                {"ticker": "NVDA", "shares": 5, "price": 200.0},
            ]
        )
        candidate = {
            "strategy": "single_name",
            "strategy_label": "Single-name protective puts",
            "portfolio_state": {},
            "portfolio_net_delta": 20.0,
            "hedge_delta": 10.0,
            "hedge_target_delta": 10.0,
            "market_delta_coverage_pct": 50.0,
            "total_cost": 100.0,
            "total_cost_pct": 1.0,
            "average_spread_pct": 2.0,
            "any_fallback": False,
            "contracts": [],
            "positions": [],
            "errors": [],
        }
        profile = {
            "objective": "reduce_downside",
            "experience": "beginner",
            "horizon_days": 45,
            "max_budget": 0,
            "move_threshold_pct": 5.0,
            "review_frequency_days": 14,
            "min_days_to_roll": 21,
            "sizing_underlying": "SPY",
        }

        with patch.object(hedge, "calculate_portfolio_hedge", return_value=dict(candidate)), \
             patch.object(hedge, "calculate_index_hedge", side_effect=RuntimeError("skip")), \
             patch.object(hedge, "build_scenarios", return_value=[]), \
             patch.object(hedge, "calculate_underlying_hedge_adjustment") as sizing_mock:
            sizing_mock.return_value.to_dict.return_value = {
                "current_net_delta": 20.0,
                "existing_hedge_delta": 0.0,
                "hedge_underlying": "SPY",
                "action": "sell",
                "required_units_rounded": 20,
                "estimated_post_trade_delta": 0.0,
            }
            result = hedge.build_delta_advice(
                portfolio,
                "moderate",
                profile,
                portfolio_beta=1.35,
                position_betas={"AAPL": 1.12, "NVDA": 1.65},
            )

        self.assertEqual(result["portfolio_state"]["position_betas"], {"AAPL": 1.12, "NVDA": 1.65})


if __name__ == "__main__":
    unittest.main()
