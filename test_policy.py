import unittest
from datetime import date

from policy import HedgePolicyConfig, ThresholdPeriodicPolicy


class TestHedgePolicy(unittest.TestCase):
    def test_policy_config_from_profile(self):
        config = HedgePolicyConfig.from_profile(
            {
                "move_threshold_pct": 7.5,
                "review_frequency_days": 10,
                "min_days_to_roll": 18,
            }
        )
        self.assertEqual(config.move_threshold_pct, 7.5)
        self.assertEqual(config.review_frequency_days, 10)
        self.assertEqual(config.min_days_to_roll, 18)

    def test_threshold_periodic_policy_decision(self):
        policy = ThresholdPeriodicPolicy(
            HedgePolicyConfig(move_threshold_pct=6.0, review_frequency_days=12, min_days_to_roll=15)
        )
        decision = policy.evaluate(
            as_of=date(2026, 3, 11),
            target_dte=45,
            portfolio_net_delta=120.0,
            hedge_delta=60.0,
        )
        self.assertEqual(decision.next_review_date, "2026-03-23")
        self.assertEqual(decision.review_window_days, 12)
        self.assertTrue(any("6.0%" in trigger for trigger in decision.triggers))
        self.assertTrue(any("12 days" in trigger for trigger in decision.triggers))
        self.assertTrue(any("15 days" in trigger for trigger in decision.triggers))
        self.assertIn("offsets about 50.0% of portfolio sensitivity", decision.summary)


if __name__ == "__main__":
    unittest.main()
