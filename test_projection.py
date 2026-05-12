import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import projection


class TestProjectionFanChart(unittest.TestCase):
    def setUp(self):
        base_prices_aapl = []
        base_prices_spy = []
        aapl = 100.0
        spy = 400.0
        for idx in range(140):
            aapl *= 1.0015 if idx % 7 not in (2, 5) else 0.992
            spy *= 1.001 if idx % 9 not in (3, 7) else 0.995
            base_prices_aapl.append(round(aapl, 2))
            base_prices_spy.append(round(spy, 2))

        start = datetime(2025, 8, 1)
        dates = [(start + timedelta(days=idx)).strftime("%Y-%m-%d") for idx in range(140)]
        self.histories = {
            "AAPL": {"dates": dates, "prices": base_prices_aapl},
            "SPY": {"dates": dates, "prices": base_prices_spy},
        }
        self.positions = [
            {"ticker": "AAPL", "shares": 100, "price": 118.0, "avg_cost": 95.0},
        ]
        self.recommendation = {
            "strategy": "single_name",
            "market_delta_coverage_pct": 55.0,
            "total_cost": 650.0,
            "contracts": [
                {
                    "underlying": "AAPL",
                    "underlying_price": 118.0,
                    "contracts": 2,
                    "strike": 112.0,
                    "expiry": "2026-06-19",
                    "dte": 100,
                    "iv": 0.24,
                }
            ],
        }
        self.history = {
            "dates": ["2026-01-15", "2026-02-15", "2026-03-11"],
            "values": [11200.0, 11650.0, 11800.0],
            "current_value": 11800.0,
        }

    def test_fan_chart_compares_hold_and_rolling_ranges(self):
        with patch.object(projection, "get_price_histories", return_value=self.histories):
            fan = projection.build_future_fan_chart(self.positions, self.recommendation, self.history, months=12, num_paths=180)

        self.assertTrue(fan["has_data"])
        self.assertEqual(fan["method"], "Historical block bootstrap with option repricing; compares hold-current-hedge vs hedge rolling")
        self.assertEqual(fan["paths"], 180)
        self.assertTrue(fan["range_hold_low"].startswith("$"))
        self.assertTrue(fan["range_rolling_low"].startswith("$"))
        self.assertTrue(fan["median_hold"].startswith("$"))
        self.assertTrue(fan["median_rolling"].startswith("$"))
        self.assertEqual(len(fan["rows"]), 3)
        self.assertEqual(fan["rows"][0]["label"], "Unhedged")
        self.assertTrue(fan["current_x"])
        self.assertEqual(len(fan["axis_labels"]), 5)
        self.assertIn("hold_downside_delta", fan)
        self.assertIn("rolling_median_drag", fan)
        self.assertTrue(fan["fan_chart"]["has_data"])
        self.assertEqual(fan["fan_chart"]["confidence_label"], "95%")
        self.assertEqual(len(fan["fan_chart"]["series"]), 2)
        self.assertEqual(fan["fan_chart"]["series"][0]["label"], "Unhedged")
        self.assertEqual(fan["fan_chart"]["series"][1]["label"], "Hedged")
        self.assertTrue(fan["fan_chart"]["history_points"])
        self.assertTrue(fan["fan_chart"]["current_x"])
        self.assertTrue(fan["fan_chart"]["current_y"])
        self.assertTrue(fan["fan_chart"]["series"][0]["upper_points"])
        self.assertTrue(fan["fan_chart"]["series"][0]["lower_points"])
        self.assertLess(float(fan["fan_chart"]["current_x"]), fan["fan_chart"]["plot_x"] + fan["fan_chart"]["plot_w"] * 0.35)
        self.assertTrue(
            fan["fan_chart"]["history_points"].endswith(
                f'{fan["fan_chart"]["current_x"]},{fan["fan_chart"]["current_y"]}'
            )
        )

    def test_fan_uses_actual_quantiles_per_step(self):
        summary = projection._summarize_paths(
            unhedged_paths=[[100.0, 90.0, 80.0], [100.0, 100.0, 100.0], [100.0, 115.0, 130.0]],
            hold_paths=[[100.0, 97.0, 95.0], [100.0, 100.0, 98.0], [100.0, 108.0, 120.0]],
            rolling_paths=[[100.0, 98.0, 96.0], [100.0, 101.0, 99.0], [100.0, 107.0, 118.0]],
            dates=[datetime(2026, 3, 11), datetime(2026, 4, 11), datetime(2026, 5, 11)],
            current_value=100.0,
        )

        mid_point = summary["points"][1]
        final_point = summary["points"][-1]

        self.assertEqual(mid_point["unhedged"]["median"], 100.0)
        self.assertEqual(mid_point["unhedged"]["ci95_low"], 90.5)
        self.assertAlmostEqual(mid_point["unhedged"]["ci95_high"], 114.25, places=2)
        self.assertEqual(mid_point["unhedged"]["outer_low"], 92.0)
        self.assertEqual(mid_point["hedged_hold"]["outer_low"], 97.6)
        self.assertEqual(mid_point["hedged_roll"]["median"], 101.0)
        self.assertEqual(final_point["unhedged"]["ci95_low"], 81.0)
        self.assertAlmostEqual(final_point["unhedged"]["ci95_high"], 128.5, places=2)
        self.assertAlmostEqual(final_point["hedged_hold"]["ci95_low"], 95.15, places=2)
        self.assertAlmostEqual(final_point["hedged_hold"]["ci95_high"], 118.9, places=2)
        self.assertEqual(final_point["unhedged"]["outer_low"], 84.0)
        self.assertEqual(final_point["hedged_hold"]["outer_low"], 95.6)
        self.assertEqual(final_point["hedged_roll"]["outer_low"], 96.6)
        self.assertLessEqual(final_point["hedged_roll"]["outer_low"], final_point["hedged_roll"]["median"])

    def test_option_value_int_branch_uses_original_dte_for_time_value(self):
        captured = {}

        def _fake_option_book_value(book, _simulated_prices, as_of, _realized_vols):
            captured["remaining_days"] = (book[0]["expiry_date"] - as_of).days
            return 123.45

        with patch.object(projection, "_option_book_value", side_effect=_fake_option_book_value):
            option_value = projection._option_value(
                contract_legs=[
                    {
                        "underlying": "AAPL",
                        "underlying_price": 100.0,
                        "contracts": 1,
                        "strike": 100.0,
                        "dte": 45,
                        "iv": 0.24,
                    }
                ],
                simulated_prices={"AAPL": 100.0},
                as_of=0,
                realized_vols={"AAPL": 0.24},
            )

        self.assertEqual(option_value, 123.45)
        self.assertGreaterEqual(captured["remaining_days"], 44)

    def test_horizon_days_uses_exact_hedge_window_not_full_month_ceiling(self):
        with patch.object(projection, "get_price_histories", return_value=self.histories):
            fan = projection.build_future_fan_chart(
                self.positions,
                self.recommendation,
                self.history,
                months=12,
                horizon_days=45,
                num_paths=60,
            )

        self.assertGreaterEqual(fan["horizon_days"], 40)
        self.assertLessEqual(fan["horizon_days"], 50)

    def test_returns_no_chart_when_history_is_too_short(self):
        short_histories = {"AAPL": {"dates": ["2026-03-01", "2026-03-02"], "prices": [100.0, 101.0]}}
        with patch.object(projection, "get_price_histories", return_value=short_histories):
            fan = projection.build_future_fan_chart(self.positions, self.recommendation, self.history)
        self.assertFalse(fan["has_data"])
        self.assertTrue(fan["warnings"])

    def test_index_roll_recalculates_contracts_from_current_portfolio_value(self):
        leg = {
            "underlying": "SPY",
            "contracts": 1,
            "strike": 475.0,
            "iv": 0.22,
            "expiry_date": datetime(2026, 4, 1),
            "original_dte": 45,
            "strike_ratio": 0.95,
            "strategy_scope": "index",
            "coverage_ratio": 0.5,
            "market_beta": 1.0,
            "position_shares": 0.0,
            "target_delta": 40.0,
        }

        rolled_leg, roll_cost = projection._roll_option_leg(
            leg,
            {"SPY": 500.0},
            datetime(2026, 3, 11),
            {"SPY": 0.22},
            current_portfolio_value=60000.0,
        )

        self.assertGreaterEqual(rolled_leg["contracts"], 2)
        self.assertGreater(roll_cost, 0.0)
        self.assertAlmostEqual(rolled_leg["target_delta"], 60.0, places=1)


if __name__ == "__main__":
    unittest.main()








