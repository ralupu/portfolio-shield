import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import hedge
import main


class TestAppSmoke(unittest.TestCase):
    def setUp(self):
        self.quote_map = {
            "AAPL": {"ticker": "AAPL", "price": 190.0, "name": "Apple Inc."},
            "MSFT": {"ticker": "MSFT", "price": 410.0, "name": "Microsoft Corp."},
            "SPY": {"ticker": "SPY", "price": 520.0, "name": "SPDR S&P 500 ETF Trust"},
        }

    def _stub_fetch_quote(self, ticker: str):
        return self.quote_map.get(ticker.upper(), {"ticker": ticker.upper(), "price": 100.0, "name": ticker.upper()})

    @staticmethod
    def _stub_select_put(ticker: str, price: float, hedge_level: str, target_dte: int = 45):
        del hedge_level
        return {
            "ticker": ticker.upper(),
            "strike": round(price * 0.95, 2),
            "expiry": "2026-06-19",
            "dte": target_dte,
            "mid_price": 4.25,
            "bid": 4.10,
            "ask": 4.40,
            "spread_pct": 7.0,
            "iv": 0.24,
            "open_interest": 2500,
            "volume": 350,
            "delta": -0.38,
            "is_fallback": False,
        }

    @staticmethod
    def _stub_performance_summary(_positions):
        return {
            "cost_basis": 31000.0,
            "current_value": 39500.0,
            "positions": [
                {"ticker": "AAPL", "shares": 100, "avg_cost": 150.0, "live_price": 190.0, "pnl": 4000.0, "weight": 48.1, "cost_weight": 48.4},
                {"ticker": "MSFT", "shares": 50, "avg_cost": 320.0, "live_price": 410.0, "pnl": 4500.0, "weight": 51.9, "cost_weight": 51.6},
            ],
        }

    @staticmethod
    def _stub_portfolio_history(_positions):
        return {
            "dates": ["2026-03-01", "2026-03-05", "2026-03-11"],
            "values": [38200.0, 39150.0, 39500.0],
            "cost_basis": 31000.0,
            "min_value": 38200.0,
            "max_value": 39500.0,
            "current_value": 39500.0,
            "warnings": [],
        }

    @staticmethod
    def _stub_portfolio_beta(_positions):
        return {
            "portfolio_beta": 1.08,
            "position_betas": {"AAPL": 1.12, "MSFT": 1.03},
            "spy_correlation": 0.87,
            "warnings": [],
        }

    @staticmethod
    def _stub_svg(_positions, _performance, _portfolio_history, _recommendation):
        return {
            "pie_live": [],
            "pie_initial": [],
            "palette": ["#d3a44a", "#52d6be"],
            "chart": {"has_data": False},
            "scenario_chart": {
                "has_data": True,
                "ch_w": 860,
                "ch_h": 300,
                "plot_x": 90,
                "plot_y": 30,
                "plot_w": 680,
                "plot_h": 228,
                "current_x": "450.0",
                "current_label": "$39,500",
                "total_cost": "$1,200",
                "axis_labels": [
                    {"x": "90", "label": "$28,000"},
                    {"x": "270", "label": "$33,000"},
                    {"x": "450", "label": "$38,000"},
                    {"x": "630", "label": "$43,000"},
                    {"x": "770", "label": "$48,000"},
                ],
                "rows": [
                    {
                        "label": "Market -30%", "cy": "58.5", "bar_h": "34.2",
                        "unhedged_bar_x": "130.0", "unhedged_bar_w": "320.0", "unhedged_bar_y": "40.0",
                        "hedged_bar_x": "220.0", "hedged_bar_w": "230.0", "hedged_bar_y": "78.0",
                        "unhedged_end_x": "130.0", "hedged_end_x": "220.0",
                        "unhedged_label": "$27,650", "hedged_label": "$31,400",
                        "unhedged_label_x": "124.0", "hedged_label_x": "214.0",
                        "unhedged_label_anchor": "end", "hedged_label_anchor": "end",
                        "hedge_net": 3750, "hedge_net_label": "+$3,750", "hedge_net_positive": True,
                        "is_downside": True,
                    },
                    {
                        "label": "Market +5%", "cy": "210.5", "bar_h": "34.2",
                        "unhedged_bar_x": "450.0", "unhedged_bar_w": "60.0", "unhedged_bar_y": "192.0",
                        "hedged_bar_x": "450.0", "hedged_bar_w": "45.0", "hedged_bar_y": "230.0",
                        "unhedged_end_x": "510.0", "hedged_end_x": "495.0",
                        "unhedged_label": "$41,475", "hedged_label": "$40,600",
                        "unhedged_label_x": "516.0", "hedged_label_x": "501.0",
                        "unhedged_label_anchor": "start", "hedged_label_anchor": "start",
                        "hedge_net": -875, "hedge_net_label": "-$875", "hedge_net_positive": False,
                        "is_downside": False,
                    },
                ],
            },
            "fan": {
                "has_data": True,
                "fan_chart": {
                    "has_data": True,
                    "confidence_label": "95%",
                    "current_label": "$39,500",
                    "current_x": "323",
                    "current_y": "176",
                    "history_points": "72.0,210.0 155.0,198.0 239.0,186.0 323.0,176.0",
                    "ch_w": 860,
                    "ch_h": 320,
                    "plot_x": 72,
                    "plot_y": 28,
                    "plot_w": 754,
                    "plot_h": 246,
                    "x_labels": [
                        {"x": "72", "label": "Now"},
                        {"x": "323", "label": "Apr 26"},
                        {"x": "575", "label": "May 26"},
                        {"x": "826", "label": "Jun 26"},
                    ],
                    "y_labels": [
                        {"y": "274", "label": "$30,000"},
                        {"y": "212", "label": "$35,000"},
                        {"y": "151", "label": "$40,000"},
                        {"y": "89", "label": "$45,000"},
                        {"y": "28", "label": "$50,000"},
                    ],
                    "series": [
                        {
                            "label": "Unhedged",
                            "color": "#ef7d57",
                            "fill_opacity": 0.15,
                            "band_points": "72.0,180.0 323.0,160.0 575.0,130.0 826.0,110.0 826.0,230.0 575.0,220.0 323.0,210.0 72.0,200.0",
                            "upper_points": "323.0,160.0 575.0,130.0 826.0,110.0",
                            "lower_points": "323.0,210.0 575.0,220.0 826.0,230.0",
                            "median_points": "72.0,190.0 323.0,185.0 575.0,170.0 826.0,160.0",
                            "final_x": "826.0",
                            "final_median_y": "160.0",
                            "final_median_label": "$41,000",
                            "final_low_label": "$31,000",
                            "final_high_label": "$49,000",
                        },
                        {
                            "label": "Hedged",
                            "color": "#52d6be",
                            "fill_opacity": 0.18,
                            "band_points": "72.0,188.0 323.0,170.0 575.0,150.0 826.0,138.0 826.0,214.0 575.0,205.0 323.0,198.0 72.0,200.0",
                            "upper_points": "323.0,170.0 575.0,150.0 826.0,138.0",
                            "lower_points": "323.0,198.0 575.0,205.0 826.0,214.0",
                            "median_points": "72.0,190.0 323.0,182.0 575.0,168.0 826.0,164.0",
                            "final_x": "826.0",
                            "final_median_y": "164.0",
                            "final_median_label": "$40,900",
                            "final_low_label": "$35,500",
                            "final_high_label": "$47,500",
                        },
                    ],
                },
                "scenario_tunnels": [
                    {
                        "key": "current",
                        "label": "Current market",
                        "description": "No immediate market shock. Tunnel starts from the current portfolio state.",
                        "fan_chart": {
                            "has_data": True,
                            "confidence_label": "95%",
                            "current_label": "$39,500",
                            "current_x": "323",
                            "current_y": "176",
                            "history_points": "72.0,210.0 155.0,198.0 239.0,186.0 323.0,176.0",
                            "ch_w": 860,
                            "ch_h": 320,
                            "plot_x": 72,
                            "plot_y": 28,
                            "plot_w": 754,
                            "plot_h": 246,
                            "x_labels": [
                                {"x": "72", "label": "Jan 26"},
                                {"x": "198", "label": "Feb 26"},
                                {"x": "323", "label": "Now"},
                                {"x": "575", "label": "May 26"},
                                {"x": "826", "label": "Jun 26"},
                            ],
                            "y_labels": [
                                {"y": "274", "label": "$30,000"},
                                {"y": "212", "label": "$35,000"},
                                {"y": "151", "label": "$40,000"},
                                {"y": "89", "label": "$45,000"},
                                {"y": "28", "label": "$50,000"},
                            ],
                            "series": [
                                {
                                    "label": "Fara hedge",
                                    "color": "#ef7d57",
                                    "fill_opacity": 0.15,
                                    "band_points": "323.0,160.0 575.0,130.0 826.0,110.0 826.0,230.0 575.0,220.0 323.0,210.0",
                                    "upper_points": "323.0,160.0 575.0,130.0 826.0,110.0",
                                    "lower_points": "323.0,210.0 575.0,220.0 826.0,230.0",
                                    "median_points": "323.0,185.0 575.0,170.0 826.0,160.0",
                                    "final_x": "826.0",
                                    "final_median_y": "160.0",
                                    "final_median_label": "$41,000",
                                    "final_low_label": "$31,000",
                                    "final_high_label": "$49,000",
                                },
                                {
                                    "label": "Cu hedge",
                                    "color": "#52d6be",
                                    "fill_opacity": 0.18,
                                    "band_points": "323.0,170.0 575.0,150.0 826.0,138.0 826.0,214.0 575.0,205.0 323.0,198.0",
                                    "upper_points": "323.0,170.0 575.0,150.0 826.0,138.0",
                                    "lower_points": "323.0,198.0 575.0,205.0 826.0,214.0",
                                    "median_points": "323.0,182.0 575.0,168.0 826.0,164.0",
                                    "final_x": "826.0",
                                    "final_median_y": "164.0",
                                    "final_median_label": "$40,900",
                                    "final_low_label": "$35,500",
                                    "final_high_label": "$47,500",
                                },
                            ],
                        },
                    },
                    {
                        "key": "mkt_20",
                        "label": "Mkt -20%",
                        "description": "Applies an immediate Mkt -20% shock now, then projects the next 45 days from that stressed base.",
                        "fan_chart": {
                            "has_data": True,
                            "confidence_label": "95%",
                            "current_label": "$39,500",
                            "current_x": "323",
                            "current_y": "176",
                            "history_points": "72.0,210.0 155.0,198.0 239.0,186.0 323.0,176.0",
                            "ch_w": 860,
                            "ch_h": 320,
                            "plot_x": 72,
                            "plot_y": 28,
                            "plot_w": 754,
                            "plot_h": 246,
                            "x_labels": [
                                {"x": "72", "label": "Jan 26"},
                                {"x": "198", "label": "Feb 26"},
                                {"x": "323", "label": "Now"},
                                {"x": "575", "label": "May 26"},
                                {"x": "826", "label": "Jun 26"},
                            ],
                            "y_labels": [
                                {"y": "274", "label": "$24,000"},
                                {"y": "212", "label": "$28,000"},
                                {"y": "151", "label": "$32,000"},
                                {"y": "89", "label": "$36,000"},
                                {"y": "28", "label": "$40,000"},
                            ],
                            "series": [
                                {
                                    "label": "Fara hedge",
                                    "color": "#ef7d57",
                                    "fill_opacity": 0.15,
                                    "band_points": "323.0,214.0 575.0,198.0 826.0,186.0 826.0,270.0 575.0,258.0 323.0,242.0",
                                    "upper_points": "323.0,214.0 575.0,198.0 826.0,186.0",
                                    "lower_points": "323.0,242.0 575.0,258.0 826.0,270.0",
                                    "median_points": "323.0,226.0 575.0,220.0 826.0,214.0",
                                    "final_x": "826.0",
                                    "final_median_y": "214.0",
                                    "final_median_label": "$31,500",
                                    "final_low_label": "$24,600",
                                    "final_high_label": "$34,800",
                                },
                                {
                                    "label": "Cu hedge",
                                    "color": "#52d6be",
                                    "fill_opacity": 0.18,
                                    "band_points": "323.0,188.0 575.0,176.0 826.0,168.0 826.0,240.0 575.0,232.0 323.0,220.0",
                                    "upper_points": "323.0,188.0 575.0,176.0 826.0,168.0",
                                    "lower_points": "323.0,220.0 575.0,232.0 826.0,240.0",
                                    "median_points": "323.0,202.0 575.0,198.0 826.0,194.0",
                                    "final_x": "826.0",
                                    "final_median_y": "194.0",
                                    "final_median_label": "$33,200",
                                    "final_low_label": "$27,000",
                                    "final_high_label": "$36,300",
                                },
                            ],
                        },
                    },
                ],
                "current_label": "$39,500",
                "current_x": "420",
                "range_unhedged_low": "$31,000",
                "range_unhedged_high": "$52,000",
                "range_hold_low": "$35,500",
                "range_hold_high": "$48,500",
                "range_rolling_low": "$34,500",
                "range_rolling_high": "$47,000",
                "range_hedged_low": "$34,500",
                "range_hedged_high": "$47,000",
                "median_unhedged": "$41,000",
                "median_hold": "$40,900",
                "median_rolling": "$40,500",
                "median_hedged": "$40,500",
                "hold_downside_delta": "$4,500",
                "rolling_downside_delta": "$3,500",
                "hold_median_drag": "$-100",
                "rolling_median_drag": "$-500",
                "hold_downside_better": True,
                "rolling_downside_better": True,
                "hold_median_better": False,
                "rolling_median_better": False,
                "coverage_pct": 50.0,
                "realized_vol_pct": 22.0,
                "paths": 350,
                "historical_days": 252,
                "method": "Historical block bootstrap with option repricing; compares hold-current-hedge vs hedge rolling",
                "initial_hedge_cost": "$1,200",
                "ch_w": 860,
                "ch_h": 240,
                "plot_x": 170,
                "plot_y": 26,
                "plot_w": 666,
                "plot_h": 172,
                "axis_labels": [
                    {"x": "170", "label": "$30,000"},
                    {"x": "336", "label": "$36,000"},
                    {"x": "503", "label": "$42,000"},
                    {"x": "669", "label": "$48,000"},
                    {"x": "836", "label": "$54,000"}
                ],
                "rows": [
                    {"label": "Unhedged", "color": "#ef7d57", "y": "26.0", "label_x": "156.0", "outer_low_x": "200.0", "inner_low_x": "260.0", "median_x": "470.0", "inner_high_x": "630.0", "outer_high_x": "760.0"},
                    {"label": "Hold current hedge", "color": "#6ea8ff", "y": "112.0", "label_x": "156.0", "outer_low_x": "290.0", "inner_low_x": "340.0", "median_x": "468.0", "inner_high_x": "610.0", "outer_high_x": "700.0"},
                    {"label": "Rolling hedge", "color": "#52d6be", "y": "198.0", "label_x": "156.0", "outer_low_x": "280.0", "inner_low_x": "330.0", "median_x": "460.0", "inner_high_x": "590.0", "outer_high_x": "680.0"}
                ],
                "horizon_months": 2,
                "horizon_days": 45,
            },
        }

    def test_index_and_analyze_render_with_policy_and_neutralization(self):
        with patch.object(main, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(main, "get_performance_summary", side_effect=self._stub_performance_summary), \
             patch.object(main, "get_portfolio_history", side_effect=self._stub_portfolio_history), \
             patch.object(main, "get_portfolio_beta", side_effect=self._stub_portfolio_beta), \
             patch.object(main, "save_recommendation", return_value=42), \
             patch.object(main, "_build_svg_data", side_effect=self._stub_svg), \
             patch.object(hedge, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(hedge, "select_put", side_effect=self._stub_select_put):
            with TestClient(main.app) as client:
                index_response = client.get("/")
                self.assertEqual(index_response.status_code, 200)
                self.assertIn("Portfolio Holdings", index_response.text)
                self.assertIn("Reference ETF for sizing", index_response.text)
                self.assertIn("Load Demo Portfolio", index_response.text)
                self.assertIn("Clear Portfolio", index_response.text)
                self.assertIn('class="info-inline"', index_response.text)
                self.assertIn("The stock symbol you own, such as AAPL or MSFT.", index_response.text)

                analyze_response = client.post(
                    "/analyze",
                    data={
                        "ticker": ["AAPL", "MSFT"],
                        "shares": ["100", "50"],
                        "avg_cost": ["150", "320"],
                        "hedge_level": "moderate",
                        "objective": "reduce_downside",
                        "experience": "beginner",
                        "options_approval": "yes",
                        "horizon_days": "45",
                        "max_budget": "1200",
                        "move_threshold_pct": "6.5",
                        "review_frequency_days": "10",
                        "min_days_to_roll": "18",
                        "sizing_underlying": "SPY",
                        "ack_advisory": "yes",
                        "ack_options_risk": "yes",
                    },
                )

                self.assertEqual(analyze_response.status_code, 200)
                text = analyze_response.text
                self.assertIn("Recommendation #42", text)
                self.assertIn("Recommended Contracts", text)
                self.assertIn("Reference Hedge Estimate", text)
                self.assertIn("Review Triggers", text)
                self.assertIn("6.5% / 10d", text)
                self.assertIn("Review plan: 6.5% move trigger, 10-day check-in, consider rolling below 18 days to expiry", text)
                self.assertIn("Suggested action in", text)
                self.assertIn("Scenario Impact: Unhedged vs Hedged", text)
                self.assertIn("Scenario Breakdown", text)
                self.assertIn("Hedge Net", text)
                self.assertIn("Portfolio Tunnel Chart (45 Days)", text)
                self.assertIn("95%", text)
                self.assertIn("forward 95% confidence range starting exactly from the current portfolio value", text)
                self.assertIn("the projected ranges begin from that exact same point", text)
                self.assertIn("Unhedged", text)
                self.assertIn("Hedged", text)
                self.assertNotIn("Future Range Comparison (45 Days)", text)

    def test_quote_api_returns_expected_status_codes(self):
        def _quote_side_effect(ticker: str):
            ticker = ticker.upper()
            if ticker == "AAPL":
                return {"ticker": "AAPL", "price": 190.0, "change": 1.2, "change_pct": 0.6}
            if ticker == "BAD":
                raise ValueError("Ticker not recognized")
            if ticker == "DOWN":
                raise RuntimeError("Quote service unavailable")
            raise Exception("boom")

        with patch.object(main, "fetch_quote", side_effect=_quote_side_effect):
            with TestClient(main.app) as client:
                ok_response = client.get("/api/quote/AAPL")
                bad_response = client.get("/api/quote/BAD")
                down_response = client.get("/api/quote/DOWN")
                boom_response = client.get("/api/quote/BOOM")

        self.assertEqual(ok_response.status_code, 200)
        self.assertEqual(ok_response.json()["ticker"], "AAPL")
        self.assertEqual(bad_response.status_code, 400)
        self.assertEqual(down_response.status_code, 503)
        self.assertEqual(boom_response.status_code, 500)

    def test_analyze_shows_form_validation_errors_for_missing_acknowledgements(self):
        with patch.object(main, "fetch_quote", side_effect=self._stub_fetch_quote):
            with TestClient(main.app) as client:
                analyze_response = client.post(
                    "/analyze",
                    data={
                        "ticker": ["AAPL"],
                        "shares": ["100"],
                        "avg_cost": ["150"],
                        "hedge_level": "moderate",
                        "objective": "reduce_downside",
                        "experience": "beginner",
                        "options_approval": "yes",
                        "horizon_days": "45",
                        "max_budget": "1200",
                        "move_threshold_pct": "6.5",
                        "review_frequency_days": "10",
                        "min_days_to_roll": "18",
                        "sizing_underlying": "SPY",
                    },
                )

        self.assertEqual(analyze_response.status_code, 200)
        text = analyze_response.text
        self.assertIn("Please confirm that this app is advisory only and does not execute trades.", text)
        self.assertIn("Please confirm that you understand options can lose value and protection is not guaranteed.", text)

    def test_analyze_renders_even_if_recommendation_save_fails(self):
        with patch.object(main, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(main, "get_performance_summary", side_effect=self._stub_performance_summary), \
             patch.object(main, "get_portfolio_history", side_effect=self._stub_portfolio_history), \
             patch.object(main, "get_portfolio_beta", side_effect=self._stub_portfolio_beta), \
             patch.object(main, "save_recommendation", side_effect=RuntimeError("database is locked")), \
             patch.object(main, "_build_svg_data", side_effect=self._stub_svg), \
             patch.object(hedge, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(hedge, "select_put", side_effect=self._stub_select_put):
            with TestClient(main.app) as client:
                analyze_response = client.post(
                    "/analyze",
                    data={
                        "ticker": ["AAPL", "MSFT"],
                        "shares": ["100", "50"],
                        "avg_cost": ["150", "320"],
                        "hedge_level": "moderate",
                        "objective": "reduce_downside",
                        "experience": "beginner",
                        "options_approval": "yes",
                        "horizon_days": "45",
                        "max_budget": "1200",
                        "move_threshold_pct": "6.5",
                        "review_frequency_days": "10",
                        "min_days_to_roll": "18",
                        "sizing_underlying": "SPY",
                        "ack_advisory": "yes",
                        "ack_options_risk": "yes",
                    },
                )

                self.assertEqual(analyze_response.status_code, 200)
                text = analyze_response.text
                self.assertIn("Analysis generated at", text)
                self.assertIn("Your analysis wasn&#39;t saved, but the results below are still available.", text)
                self.assertIn("Recommended Contracts", text)

    def test_analyze_renders_when_future_range_projection_is_unavailable(self):
        svg_without_fan = self._stub_svg([], {}, {}, {})
        svg_without_fan["fan"] = {
            "has_data": False,
            "warnings": ["Not enough market history is available to build the forward range."],
        }
        with patch.object(main, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(main, "get_performance_summary", side_effect=self._stub_performance_summary), \
             patch.object(main, "get_portfolio_history", side_effect=self._stub_portfolio_history), \
             patch.object(main, "get_portfolio_beta", side_effect=self._stub_portfolio_beta), \
             patch.object(main, "save_recommendation", return_value=42), \
             patch.object(main, "_build_svg_data", return_value=svg_without_fan), \
             patch.object(hedge, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(hedge, "select_put", side_effect=self._stub_select_put):
            with TestClient(main.app) as client:
                analyze_response = client.post(
                    "/analyze",
                    data={
                        "ticker": ["AAPL", "MSFT"],
                        "shares": ["100", "50"],
                        "avg_cost": ["150", "320"],
                        "hedge_level": "moderate",
                        "objective": "reduce_downside",
                        "experience": "beginner",
                        "options_approval": "yes",
                        "horizon_days": "45",
                        "max_budget": "1200",
                        "move_threshold_pct": "6.5",
                        "review_frequency_days": "10",
                        "min_days_to_roll": "18",
                        "sizing_underlying": "SPY",
                        "ack_advisory": "yes",
                        "ack_options_risk": "yes",
                    },
                )

                self.assertEqual(analyze_response.status_code, 200)
                text = analyze_response.text
                self.assertIn("Recommendation #42", text)
                self.assertIn("Not enough market history is available to build the forward range.", text)
                self.assertNotIn("Future Range Comparison (45 Days)", text)

    def test_analyze_falls_back_to_form_on_unexpected_error(self):
        with patch.object(main, "fetch_quote", side_effect=self._stub_fetch_quote), \
             patch.object(main, "get_performance_summary", side_effect=RuntimeError("forced performance failure")):
            with TestClient(main.app) as client:
                analyze_response = client.post(
                    "/analyze",
                    data={
                        "ticker": ["AAPL"],
                        "shares": ["100"],
                        "avg_cost": ["150"],
                        "hedge_level": "moderate",
                        "objective": "reduce_downside",
                        "experience": "beginner",
                        "options_approval": "yes",
                        "horizon_days": "45",
                        "max_budget": "1200",
                        "move_threshold_pct": "6.5",
                        "review_frequency_days": "10",
                        "min_days_to_roll": "18",
                        "sizing_underlying": "SPY",
                        "ack_advisory": "yes",
                        "ack_options_risk": "yes",
                    },
                )

                self.assertEqual(analyze_response.status_code, 200)
                text = analyze_response.text
                self.assertIn("We couldn&#39;t complete the analysis right now.", text)
                self.assertIn("Please try again. If the issue continues, refresh the app and retry.", text)


if __name__ == "__main__":
    unittest.main()
