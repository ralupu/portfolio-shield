import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import projection


class TestFutureFanChart(unittest.TestCase):
    def test_build_future_fan_chart_returns_hold_and_roll_ranges(self):
        prices = []
        spot = 100.0
        for idx in range(140):
            spot *= 1.001 if idx % 8 else 0.994
            prices.append(round(spot, 2))
        start = datetime(2025, 8, 1)
        dates = [(start + timedelta(days=idx)).strftime("%Y-%m-%d") for idx in range(140)]
        histories = {"AAPL": {"dates": dates, "prices": prices}}
        positions = [{"ticker": "AAPL", "shares": 100, "price": 104.8, "avg_cost": 98.0}]
        recommendation = {
            "strategy": "single_name",
            "market_delta_coverage_pct": 55.0,
            "total_cost": 220.0,
            "contracts": [
                {
                    "underlying": "AAPL",
                    "underlying_price": 104.8,
                    "contracts": 1,
                    "strike": 100.0,
                    "expiry": "2026-06-19",
                    "dte": 90,
                    "iv": 0.24,
                }
            ],
        }
        history = {
            "dates": ["2026-01-01", "2026-02-01", "2026-03-11"],
            "values": [102500.0, 103600.0, 104800.0],
            "current_value": 104800.0,
        }

        with patch.object(projection, "get_price_histories", return_value=histories):
            fan = projection.build_future_fan_chart(positions, recommendation, history, months=12, num_paths=120)

        self.assertTrue(fan["has_data"])
        self.assertIn("range_hold_low", fan)
        self.assertIn("range_rolling_low", fan)
        self.assertIn("median_hold", fan)
        self.assertIn("median_rolling", fan)
        self.assertEqual(len(fan["rows"]), 3)
        self.assertEqual(len(fan["axis_labels"]), 5)


if __name__ == "__main__":
    unittest.main()



