import unittest

import scenarios


class TestScenarioSnapshot(unittest.TestCase):
    def test_downside_hedge_uses_bs_valuation_with_time_value(self):
        """Deep ITM put after crash should show option BS value > intrinsic,
        and hedged value should NOT be artificially capped."""
        positions = [{"ticker": "AAPL", "shares": 100, "price": 100.0}]
        recommendation = {
            "total_value": 10000.0,
            "total_cost": 100.0,
            "portfolio_state": {"position_betas": {"AAPL": 1.0}},
            "contracts": [
                {
                    "underlying": "AAPL",
                    "underlying_price": 100.0,
                    "contracts": 1,
                    "strike": 200.0,
                    "dte": 45,
                    "iv": 0.30,
                }
            ],
        }

        rows = scenarios.build_scenarios(positions, recommendation)
        crash_row = next(row for row in rows if row["label"] == "Market -20%")

        self.assertEqual(crash_row["market_move_pct"], -20.0)
        self.assertEqual(crash_row["portfolio_move_pct"], -20.0)
        self.assertEqual(crash_row["unhedged_value"], 8000.0)
        # BS value of deep ITM put (strike 200 vs spot 80) includes time value
        # so hedged_value should be higher than intrinsic-only would give
        self.assertGreater(crash_row["option_value"], 11800.0)
        # Hedged value should exceed old capped value of 9900
        self.assertGreater(crash_row["hedged_value"], 9900.0)
        # hedge_net should be strongly positive in a crash
        self.assertGreater(crash_row["hedge_net"], 0)

    def test_upside_scenario_still_reflects_hedge_cost_drag(self):
        positions = [{"ticker": "AAPL", "shares": 100, "price": 100.0}]
        recommendation = {
            "total_value": 10000.0,
            "total_cost": 100.0,
            "portfolio_state": {"position_betas": {"AAPL": 1.0}},
            "contracts": [
                {
                    "underlying": "AAPL",
                    "underlying_price": 100.0,
                    "contracts": 1,
                    "strike": 95.0,
                    "dte": 45,
                    "iv": 0.25,
                }
            ],
        }

        rows = scenarios.build_scenarios(positions, recommendation)
        up_row = next(row for row in rows if row["label"] == "Market +5%")

        self.assertEqual(up_row["unhedged_value"], 10500.0)
        # Hedged value should be lower than unhedged (premium drag)
        # but higher than old intrinsic-only value because put still has time value
        self.assertLess(up_row["hedged_value"], up_row["unhedged_value"])
        self.assertGreater(up_row["option_value"], 0)
        # hedge_net should be negative (insurance cost in upside)
        self.assertLess(up_row["hedge_net"], 0)
        self.assertAlmostEqual(up_row["net_change_hedged"], up_row["hedged_value"] - 10000.0, places=0)

    def test_downside_shocks_are_scaled_by_beta_proxy(self):
        positions = [{"ticker": "AAPL", "shares": 100, "price": 100.0}]
        recommendation = {
            "total_value": 10000.0,
            "total_cost": 50.0,
            "portfolio_state": {"position_betas": {"AAPL": 1.5}},
            "contracts": [],
        }

        rows = scenarios.build_scenarios(positions, recommendation)
        crash_row = next(row for row in rows if row["label"] == "Market -20%")

        self.assertEqual(crash_row["market_move_pct"], -20.0)
        self.assertEqual(crash_row["portfolio_move_pct"], -30.0)
        self.assertEqual(crash_row["unhedged_value"], 7000.0)
        self.assertEqual(crash_row["hedged_value"], 6950.0)


if __name__ == "__main__":
    unittest.main()
