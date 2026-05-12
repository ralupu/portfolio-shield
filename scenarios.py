"""
scenarios.py - Downside and upside scenario snapshots for hedge recommendations.

Uses Black-Scholes valuation for option positions after a shock so that
time value is included, not just intrinsic.  This gives a realistic
picture of what the hedge is worth if the move happens today while the
options still have their full DTE remaining.
"""

from options import bs_put_price

SCENARIO_SHOCKS = (-0.30, -0.20, -0.10, -0.05, 0.05, 0.10)
RISK_FREE_RATE = 0.05


def build_scenarios(
    positions: list[dict],
    recommendation: dict,
    portfolio_beta: float | None = None,
) -> list[dict]:
    """Build scenario rows using Black-Scholes option valuation.

    Shocks represent market-level moves. Individual stock moves are
    scaled by their beta relative to the market (default 1.0).
    Option positions are repriced with BS using the post-shock spot
    and the original IV / DTE so that time value is preserved.
    """
    base_value = float(recommendation.get("total_value", 0.0) or 0.0)
    total_cost = float(recommendation.get("total_cost", 0.0) or 0.0)
    contracts = recommendation.get("contracts", [])
    position_betas = recommendation.get("portfolio_state", {}).get("position_betas", {})
    fallback_beta = float(portfolio_beta) if portfolio_beta is not None else 1.0

    rows = []
    for shock in SCENARIO_SHOCKS:
        scenario_prices = {}
        for pos in positions:
            ticker = pos["ticker"]
            beta = float(position_betas.get(ticker, fallback_beta))
            stock_shock = shock * beta
            scenario_prices[ticker] = max(pos["price"] * (1.0 + stock_shock), 0.01)

        unhedged_value = round(
            sum(pos["shares"] * scenario_prices[pos["ticker"]] for pos in positions),
            2,
        )

        option_value = 0.0
        for leg in contracts:
            underlying = leg["underlying"]
            shocked_price = scenario_prices.get(
                underlying,
                max(float(leg.get("underlying_price", 0.0) or 0.0) * (1.0 + shock), 0.01),
            )
            strike = float(leg["strike"])
            num_contracts = int(leg["contracts"])
            dte = int(leg.get("dte", 0) or 0)
            iv = float(leg.get("iv", 0.0) or 0.0)
            if iv <= 0:
                iv = 0.25
            T = max(dte, 1) / 365.0
            put_price = bs_put_price(max(shocked_price, 0.01), strike, T, RISK_FREE_RATE, iv)
            option_value += put_price * num_contracts * 100

        hedged_value = round(unhedged_value + option_value - total_cost, 2)

        portfolio_move_pct = round(((unhedged_value / base_value) - 1.0) * 100, 1) if base_value > 0 else round(shock * 100, 1)
        hedge_net = round(hedged_value - unhedged_value, 2)
        rows.append(
            {
                "label": f"Market {shock * 100:+.0f}%",
                "market_move_pct": round(shock * 100, 1),
                "portfolio_move_pct": portfolio_move_pct,
                "unhedged_value": unhedged_value,
                "hedged_value": hedged_value,
                "option_value": round(option_value, 2),
                "premium_paid": round(total_cost, 2),
                "hedge_net": hedge_net,
                "net_change_unhedged": round(unhedged_value - base_value, 2),
                "net_change_hedged": round(hedged_value - base_value, 2),
            }
        )

    return rows



