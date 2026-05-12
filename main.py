"""
main.py - FastAPI app, routes, and startup.
"""

from contextlib import asynccontextmanager
import logging
import math
from pathlib import Path
import time

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from hedge import build_delta_advice
from history import get_performance_summary, get_portfolio_beta, get_portfolio_history
from portfolio_state import PortfolioState
from projection import build_future_fan_chart
from quotes import fetch_quote
from storage import init_storage, save_recommendation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_storage()
    yield


app = FastAPI(title="Portfolio Shield", version="2.1.0", lifespan=lifespan)
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/quote/{ticker}")
async def api_quote(ticker: str):
    try:
        quote = fetch_quote(ticker)
        return JSONResponse(quote)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except RuntimeError as exc:
        logger.error("Quote API service error for %s: %s", ticker, exc)
        return JSONResponse({"error": str(exc)}, status_code=503)
    except Exception as exc:
        logger.error("Quote API unexpected error for %s: %s", ticker, exc)
        return JSONResponse({"error": "Unexpected quote error."}, status_code=500)


@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request):
    try:
        return await _analyze_impl(request)
    except Exception as exc:
        logger.error("Unhandled analyze request error: %s", exc, exc_info=True)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "errors": [
                    "We couldn't complete the analysis right now.",
                    "Please try again. If the issue continues, refresh the app and retry.",
                ],
            },
        )


async def _analyze_impl(request: Request):
    form = await request.form()

    tickers = form.getlist("ticker")
    shares_list = form.getlist("shares")
    avg_cost_list = form.getlist("avg_cost")
    hedge_level = form.get("hedge_level", "moderate")
    objective = form.get("objective", "reduce_downside")
    experience = form.get("experience", "beginner")
    horizon_days = _safe_int(form.get("horizon_days"), default=45, minimum=21, maximum=90)
    max_budget = _safe_float(form.get("max_budget"), default=0.0, minimum=0.0)
    move_threshold_pct = _safe_float(form.get("move_threshold_pct"), default=5.0, minimum=1.0)
    review_frequency_days = _safe_int(form.get("review_frequency_days"), default=14, minimum=1, maximum=60)
    min_days_to_roll = _safe_int(form.get("min_days_to_roll"), default=21, minimum=1, maximum=60)
    sizing_underlying = str(form.get("sizing_underlying", "SPY") or "SPY").upper()
    options_approval = form.get("options_approval", "unsure")
    ack_advisory = form.get("ack_advisory") == "yes"
    ack_options_risk = form.get("ack_options_risk") == "yes"

    if hedge_level not in ("light", "moderate", "full"):
        hedge_level = "moderate"
    if objective not in ("reduce_downside", "protect_gains", "crash_hedge", "partial_delta"):
        objective = "reduce_downside"
    if experience not in ("beginner", "intermediate", "advanced"):
        experience = "beginner"
    if options_approval not in ("yes", "no", "unsure"):
        options_approval = "unsure"

    positions = []
    errors = []

    if not ack_advisory:
        errors.append("Please confirm that this app is advisory only and does not execute trades.")
    if not ack_options_risk:
        errors.append("Please confirm that you understand options can lose value and protection is not guaranteed.")

    for idx, ticker in enumerate(tickers):
        ticker = ticker.strip().upper()
        if not ticker:
            continue

        shares_str = shares_list[idx] if idx < len(shares_list) else ""
        avg_cost_str = avg_cost_list[idx] if idx < len(avg_cost_list) else ""

        try:
            shares = int(shares_str)
            if shares <= 0:
                raise ValueError
        except (ValueError, TypeError):
            errors.append(f"Enter a valid share count for {ticker}.")
            continue

        avg_cost = None
        if avg_cost_str and avg_cost_str.strip():
            try:
                avg_cost = float(avg_cost_str)
                if avg_cost <= 0:
                    avg_cost = None
            except (ValueError, TypeError):
                avg_cost = None

        try:
            quote = fetch_quote(ticker)
            positions.append(
                {
                    "ticker": ticker,
                    "shares": shares,
                    "price": quote["price"],
                    "avg_cost": avg_cost if avg_cost else quote["price"],
                }
            )
        except ValueError:
            errors.append(f"Ticker not recognized: {ticker}.")
        except RuntimeError:
            errors.append(f"Market data is unavailable for {ticker} right now. Please try again.")
        except Exception:
            errors.append(f"We couldn't load market data for {ticker}.")

    if (not ack_advisory) or (not ack_options_risk):
        return templates.TemplateResponse(request, "index.html", {"errors": errors})

    if errors and not positions:
        return templates.TemplateResponse(request, "index.html", {"errors": errors})

    if not positions:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"errors": ["Enter at least one valid position."]},
        )

    portfolio_state = PortfolioState.from_equity_snapshot(positions)
    performance = get_performance_summary(positions)

    portfolio_history = {"dates": [], "values": [], "cost_basis": 0, "warnings": []}
    try:
        portfolio_history = get_portfolio_history(positions)
    except Exception as exc:
        logger.error("History fetch failed: %s", exc)
        portfolio_history["warnings"] = [str(exc)]

    portfolio_beta = {"portfolio_beta": None, "position_betas": {}, "spy_correlation": None, "warnings": []}
    try:
        portfolio_beta = get_portfolio_beta(positions)
    except Exception as exc:
        logger.error("Beta calc failed: %s", exc)
        portfolio_beta["warnings"] = [str(exc)]

    profile = {
        "objective": objective,
        "experience": experience,
        "horizon_days": horizon_days,
        "max_budget": max_budget,
        "move_threshold_pct": move_threshold_pct,
        "review_frequency_days": review_frequency_days,
        "min_days_to_roll": min_days_to_roll,
        "sizing_underlying": sizing_underlying,
        "options_approval": options_approval,
        "ack_advisory": ack_advisory,
        "ack_options_risk": ack_options_risk,
    }

    try:
        recommendation = build_delta_advice(
            portfolio_state,
            hedge_level,
            profile,
            portfolio_beta=portfolio_beta.get("portfolio_beta"),
            position_betas=portfolio_beta.get("position_betas"),
        )
        recommendation["timestamp"] = time.strftime("%H:%M ET")
        recommendation["errors"] = errors
        recommendation["gate_warnings"] = []
        recommendation["readiness_label"] = "Ready to Review" if options_approval == "yes" else "Account Setup Needs Review"
        if options_approval != "yes":
            recommendation["gate_warnings"].append(
                "Options approval was not confirmed. Treat this plan as research until your account permissions are verified."
            )
        if experience == "beginner":
            recommendation["gate_warnings"].append(
                "Beginner profile selected. Review premium cost, liquidity, and expiry carefully before placing any trade."
            )
        if recommendation["gate_warnings"]:
            recommendation["suitability_notes"] = recommendation.get("suitability_notes", []) + recommendation["gate_warnings"]
    except Exception as exc:
        logger.error("Advisory calculation failed: %s", exc)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"errors": ["We couldn't build the protection plan right now."]},
        )

    try:
        svg_data = _build_svg_data(positions, performance, portfolio_history, recommendation)
    except Exception as exc:
        logger.error("SVG data build failed: %s", exc, exc_info=True)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"errors": ["We couldn't render the charts for this analysis."]},
        )

    payload = {
        "profile": profile,
        "positions": positions,
        "portfolio_state": portfolio_state.to_dict(),
        "performance": performance,
        "portfolio_beta": portfolio_beta,
        "recommendation": recommendation,
    }
    recommendation_id = None
    try:
        recommendation_id = save_recommendation(payload)
    except Exception as exc:
        logger.error("Recommendation persistence failed: %s", exc, exc_info=True)
        recommendation["errors"] = recommendation.get("errors", []) + [
            "Your analysis wasn't saved, but the results below are still available."
        ]

    try:
        return templates.TemplateResponse(
            request,
            "results.html",
            {
                "profile": profile,
                "performance": performance,
                "portfolio_history": portfolio_history,
                "portfolio_beta": portfolio_beta,
                "recommendation": recommendation,
                "recommendation_id": recommendation_id,
                "svg": svg_data,
            },
        )
    except Exception as exc:
        logger.error("Template rendering failed: %s", exc, exc_info=True)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"errors": ["We couldn't display the analysis results."]},
        )


PALETTE = ["#d3a44a", "#45c4b0", "#ef7d57", "#7b8cff", "#f0b35b", "#f87171", "#4fd1c5", "#f6ad55"]


def _build_pie_slices(weights: list[dict], key: str) -> list[dict]:
    slices = []
    cumulative = 0.0
    for idx, position in enumerate(weights):
        pct = position[key] / 100.0
        if pct <= 0:
            continue
        start = cumulative * 2 * math.pi
        end = (cumulative + pct) * 2 * math.pi
        large = 1 if pct > 0.5 else 0

        start_x, start_y = math.cos(start), math.sin(start)
        end_x, end_y = math.cos(end), math.sin(end)
        mid = (cumulative + pct / 2) * 2 * math.pi
        label_x, label_y = 0.65 * math.cos(mid), 0.65 * math.sin(mid)

        slices.append(
            {
                "path": f"M 0 0 L {start_x:.4f} {start_y:.4f} A 1 1 0 {large} 1 {end_x:.4f} {end_y:.4f} Z",
                "color": PALETTE[idx % len(PALETTE)],
                "ticker": position["ticker"],
                "pct": position[key],
                "label_x": f"{label_x:.4f}",
                "label_y": f"{label_y:.4f}",
                "show_label": pct > 0.06,
            }
        )
        cumulative += pct
    return slices


def _build_chart_points(history: dict) -> dict:
    dates = history.get("dates", [])
    values = history.get("values", [])
    if len(values) < 5:
        return {
            "points": "",
            "hover_points": [],
            "x_labels": [],
            "y_labels": [],
            "cb_y": 0,
            "last_x": 0,
            "last_y": 0,
            "min_x": 0,
            "min_y_pos": 0,
            "max_x": 0,
            "has_data": False,
        }

    chart_w, chart_h = 860, 320
    pad_l, pad_r, pad_t, pad_b = 70, 20, 30, 50
    plot_w = chart_w - pad_l - pad_r
    plot_h = chart_h - pad_t - pad_b

    value_min = history["min_value"]
    value_max = history["max_value"]
    value_range = value_max - value_min if value_max != value_min else 1
    cost_basis = history["cost_basis"]
    count = len(values)

    points = []
    for idx, value in enumerate(values):
        x_pos = pad_l + (idx / (count - 1)) * plot_w
        y_pos = pad_t + plot_h - ((value - value_min) / value_range * plot_h)
        points.append(f"{x_pos:.1f},{y_pos:.1f}")
    points_str = " ".join(points)

    cb_y = pad_t + plot_h - ((cost_basis - value_min) / value_range * plot_h)
    cb_in_range = value_min <= cost_basis <= value_max
    last_x = pad_l + plot_w
    last_y = pad_t + plot_h - ((values[-1] - value_min) / value_range * plot_h)

    min_idx = values.index(value_min)
    max_idx = values.index(value_max)
    min_x = pad_l + (min_idx / (count - 1)) * plot_w
    max_x = pad_l + (max_idx / (count - 1)) * plot_w

    hover_points = []
    for idx, value in enumerate(values):
        if idx % 3 == 0 or idx == count - 1:
            x_pos = pad_l + (idx / (count - 1)) * plot_w
            y_pos = pad_t + plot_h - ((value - value_min) / value_range * plot_h)
            hover_points.append({"x": f"{x_pos:.1f}", "y": f"{y_pos:.1f}", "date": dates[idx], "value": f"{value:,.0f}"})

    x_labels = []
    step = max(count // 6, 1)
    for idx in range(0, count, step):
        x_pos = pad_l + (idx / (count - 1)) * plot_w
        x_labels.append({"x": f"{x_pos:.1f}", "label": dates[idx][5:]})

    y_labels = []
    for idx in range(5):
        label_value = value_min + (value_range * idx / 4)
        y_pos = pad_t + plot_h - (plot_h * idx / 4)
        y_labels.append({"y": f"{y_pos:.1f}", "label": f"${label_value:,.0f}"})

    close_left = f"{pad_l:.1f},{cb_y:.1f}"
    close_right = f"{pad_l + plot_w:.1f},{cb_y:.1f}"
    area_points = points_str + f" {close_right} {close_left}"

    return {
        "has_data": True,
        "points": points_str,
        "area_points": area_points,
        "hover_points": hover_points,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "cb_y": f"{cb_y:.1f}",
        "cb_in_range": cb_in_range,
        "last_x": f"{last_x:.1f}",
        "last_y": f"{last_y:.1f}",
        "min_x": f"{min_x:.1f}",
        "min_y_pos": f"{(pad_t + plot_h - 5):.1f}",
        "max_x": f"{max_x:.1f}",
        "max_y_pos": f"{(pad_t + 12):.1f}",
        "plot_x": pad_l,
        "plot_y": pad_t,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "ch_w": chart_w,
        "ch_h": chart_h,
    }


def _build_svg_data(positions: list[dict], performance: dict, history: dict, recommendation: dict) -> dict:
    contracts = recommendation.get("contracts", [])
    max_dte = max((int(c.get("dte", 0) or 0) for c in contracts), default=0)
    fan_months = max(2, math.ceil(max_dte / 30)) if max_dte > 0 else 3

    return {
        "pie_initial": _build_pie_slices(performance["positions"], "cost_weight"),
        "pie_live": _build_pie_slices(performance["positions"], "weight"),
        "chart": _build_chart_points(history),
        "fan": build_future_fan_chart(positions, recommendation, history, months=fan_months, horizon_days=max_dte),
        "scenario_chart": _build_scenario_chart(recommendation),
        "palette": PALETTE,
    }


def _build_scenario_chart(recommendation: dict) -> dict:
    """Build SVG data for a scenario comparison bar chart.

    Shows unhedged vs hedged portfolio values at each market shock level,
    making the crash protection visually obvious.
    """
    scenarios = recommendation.get("scenarios", [])
    current_value = float(recommendation.get("total_value", 0.0) or 0.0)
    total_cost = float(recommendation.get("total_cost", 0.0) or 0.0)
    if not scenarios or current_value <= 0:
        return {"has_data": False}

    chart_w, chart_h = 860, 300
    pad_l, pad_r, pad_t, pad_b = 90, 90, 30, 42
    plot_w = chart_w - pad_l - pad_r
    plot_h = chart_h - pad_t - pad_b

    all_values = [current_value]
    for row in scenarios:
        all_values.extend([row["unhedged_value"], row["hedged_value"]])
    value_min = min(all_values)
    value_max = max(all_values)
    value_range = value_max - value_min if value_max != value_min else max(value_max * 0.1, 1.0)
    value_min = max(0.0, value_min - value_range * 0.06)
    value_max = value_max + value_range * 0.06
    value_range = value_max - value_min if value_max != value_min else 1.0

    def x_pos(v: float) -> float:
        return pad_l + ((v - value_min) / value_range) * plot_w

    num_rows = len(scenarios)
    row_height = plot_h / num_rows
    bar_h = row_height * 0.30
    gap = row_height * 0.04

    current_x = x_pos(current_value)

    rows = []
    for idx, sc in enumerate(scenarios):
        cy = pad_t + (idx + 0.5) * row_height
        ux = x_pos(sc["unhedged_value"])
        hx = x_pos(sc["hedged_value"])

        hedge_net = sc.get("hedge_net", 0)
        rows.append({
            "label": sc["label"],
            "cy": f"{cy:.1f}",
            "unhedged_bar_x": f"{min(ux, current_x):.1f}",
            "unhedged_bar_w": f"{abs(ux - current_x):.1f}",
            "unhedged_bar_y": f"{cy - bar_h - gap:.1f}",
            "hedged_bar_x": f"{min(hx, current_x):.1f}",
            "hedged_bar_w": f"{abs(hx - current_x):.1f}",
            "hedged_bar_y": f"{cy + gap:.1f}",
            "bar_h": f"{bar_h:.1f}",
            "unhedged_end_x": f"{ux:.1f}",
            "hedged_end_x": f"{hx:.1f}",
            "unhedged_label": f"${sc['unhedged_value']:,.0f}",
            "hedged_label": f"${sc['hedged_value']:,.0f}",
            "hedge_net": hedge_net,
            "hedge_net_label": f"{'+'if hedge_net >= 0 else ''}${hedge_net:,.0f}",
            "hedge_net_positive": hedge_net >= 0,
            "is_downside": sc["market_move_pct"] < 0,
            "unhedged_label_anchor": "end" if sc["market_move_pct"] < 0 else "start",
            "hedged_label_anchor": "end" if sc["market_move_pct"] < 0 else "start",
            "unhedged_label_x": f"{ux - 6 if sc['market_move_pct'] < 0 else ux + 6:.1f}",
            "hedged_label_x": f"{hx - 6 if sc['market_move_pct'] < 0 else hx + 6:.1f}",
        })

    axis_labels = []
    for i in range(6):
        v = value_min + (value_range * i / 5)
        axis_labels.append({"x": f"{x_pos(v):.1f}", "label": f"${v:,.0f}"})

    return {
        "has_data": True,
        "ch_w": chart_w,
        "ch_h": chart_h,
        "plot_x": pad_l,
        "plot_y": pad_t,
        "plot_w": plot_w,
        "plot_h": plot_h,
        "rows": rows,
        "current_x": f"{current_x:.1f}",
        "current_label": f"${current_value:,.0f}",
        "axis_labels": axis_labels,
        "total_cost": f"${total_cost:,.0f}",
    }

def _safe_int(value, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _safe_float(value, default: float, minimum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)




