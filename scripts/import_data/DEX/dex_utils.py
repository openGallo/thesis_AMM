"""
Shared utilities for Uniswap V3 DEX import scripts.

Imported by:
    fetch_all_monthly.py
    fetch_uniswap_extra_events.py
    fetch_uniswap_pool_timeseries.py
    fetch_uniswap_tick_snapshots.py
    fetch_uniswap_positions.py
"""

from __future__ import annotations

import calendar
import os
import time
from datetime import datetime
from typing import Optional

import requests

# ── Pool and subgraph ─────────────────────────────────────────────────────────

POOL     = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
SUBGRAPH = "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# Pool inception to thesis data cut-off.
START_TS = 1620172800   # 2021-05-05 00:00:00 UTC
END_TS   = 1777593599   # 2026-04-30 23:59:59 UTC

# ── Query tuning ──────────────────────────────────────────────────────────────

BATCH       = 1000   # rows per query (The Graph maximum)
SLEEP       = 0.35   # seconds between queries
MAX_RETRIES = 10
RETRY_WAIT  = 15     # base wait (seconds); doubles each attempt (exponential backoff)

# ── API setup ─────────────────────────────────────────────────────────────────

def _load_api_key() -> str:
    key = os.getenv("THEGRAPH_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Missing THEGRAPH_API_KEY. Set it with:\n"
            '    setx THEGRAPH_API_KEY "your_key_here"\n'
            "Then restart your terminal and re-run the script."
        )
    return key


API_KEY = _load_api_key()
URL = f"https://gateway.thegraph.com/api/{API_KEY}/subgraphs/id/{SUBGRAPH}"

# ── HTTP helper ───────────────────────────────────────────────────────────────

def run_query(query: str, variables: Optional[dict] = None) -> dict:
    payload: dict = {"query": query}
    if variables is not None:
        payload["variables"] = variables

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(URL, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            if "errors" in result:
                raise RuntimeError(result["errors"])
            return result["data"]
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            wait = min(RETRY_WAIT * (2 ** (attempt - 1)), 120)
            print(f"[Retry {attempt}/{MAX_RETRIES}] {exc}. Waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("run_query exhausted retries without raising")  # unreachable

# ── Type-safe coercions ───────────────────────────────────────────────────────

def safe_int(x) -> Optional[int]:
    if x is None or x == "":
        return None
    return int(x)


def safe_float(x) -> Optional[float]:
    if x is None or x == "":
        return None
    return float(x)

# ── Month-range helpers ───────────────────────────────────────────────────────

def month_start_ts(year: int, month: int) -> int:
    return calendar.timegm(datetime(year, month, 1, 0, 0, 0).timetuple())


def next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def build_month_ranges(
    start_ts: int = START_TS,
    end_ts: int = END_TS,
) -> list[tuple[str, int, int]]:
    """
    Return (label, chunk_start, chunk_end) for each calendar month within
    [start_ts, end_ts]. Label format: "YYYY_MM".
    """
    ranges: list[tuple[str, int, int]] = []

    dt_start = datetime.utcfromtimestamp(start_ts)
    year, month = dt_start.year, dt_start.month

    dt_end = datetime.utcfromtimestamp(end_ts)
    end_ym = (dt_end.year, dt_end.month)

    while (year, month) <= end_ym:
        ny, nm = next_month(year, month)
        m_start = month_start_ts(year, month)
        m_end   = month_start_ts(ny, nm) - 1

        chunk_start = max(m_start, start_ts)
        chunk_end   = min(m_end, end_ts)

        if chunk_start <= chunk_end:
            ranges.append((f"{year}_{month:02d}", chunk_start, chunk_end))

        year, month = ny, nm

    return ranges
