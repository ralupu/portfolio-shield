"""
storage.py - Minimal SQLite persistence for hedge recommendations.
"""

from dataclasses import asdict, is_dataclass
from datetime import date, datetime
import json
import math
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).with_name("portfolio_shield.db")


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_json_value(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _normalize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item) for item in value]
    return value


def init_storage() -> None:
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                payload_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def save_recommendation(payload: dict) -> int:
    serialized = json.dumps(
        _normalize_json_value(payload),
        default=_json_default,
        allow_nan=False,
    )
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        cur = conn.execute(
            "INSERT INTO recommendations (payload_json) VALUES (?)",
            (serialized,),
        )
        conn.commit()
        return int(cur.lastrowid)
