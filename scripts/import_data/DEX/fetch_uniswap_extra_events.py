"""
Uniswap V3 USDC/WETH 0.05% — Extra Event Downloader
====================================================

Fetches additional Uniswap v3 pool events from The Graph:

    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\collects_YYYY_MM.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\flashes_YYYY_MM.csv

These complement your existing:

    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\swaps_YYYY_MM.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\mints_YYYY_MM.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\burns_YYYY_MM.csv

Important:
    - API key is read from THEGRAPH_API_KEY.
    - No API key is hard-coded.
    - Pagination uses id_gt inside each month to avoid timestamp-cursor skipping.

Setup:
    C:\\Interpreters\\python.exe -m pip install requests pandas
    setx THEGRAPH_API_KEY "your_new_api_key_here"
    Restart terminal.
    C:\\Interpreters\\python.exe C:\\Courses\\thesis_AMM\\scripts\\import_data\\DEX\\fetch_uniswap_extra_events.py
"""

import os
import time
import calendar
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

API_KEY = os.getenv("THEGRAPH_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Missing THEGRAPH_API_KEY. Set it with:\n"
        '    setx THEGRAPH_API_KEY "your_new_api_key_here"\n'
        "Then restart your terminal."
    )

POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640".lower()

URL = (
    f"https://gateway.thegraph.com/api/{API_KEY}"
    f"/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
)

START_TS = 1620172800   # 2021-05-05 00:00:00 UTC
END_TS   = 1777593599   # 2026-04-30 23:59:59 UTC

BATCH       = 1000
SLEEP       = 0.35
MAX_RETRIES = 5
RETRY_WAIT  = 10


# ─────────────────────────────────────────────────────────────────────────────
# GraphQL queries
# ─────────────────────────────────────────────────────────────────────────────

COLLECTS_QUERY = """
query($pool: String!, $start: Int!, $end: Int!, $first: Int!, $lastId: String!) {
  collects(
    first: $first
    orderBy: id
    orderDirection: asc
    where: {
      pool: $pool
      timestamp_gte: $start
      timestamp_lte: $end
      id_gt: $lastId
    }
  ) {
    id
    timestamp
    owner
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
    logIndex
    transaction {
      id
      blockNumber
      timestamp
      gasUsed
      gasPrice
    }
    pool {
      id
      feeTier
    }
  }
}
"""

FLASHES_QUERY = """
query($pool: String!, $start: Int!, $end: Int!, $first: Int!, $lastId: String!) {
  flashes(
    first: $first
    orderBy: id
    orderDirection: asc
    where: {
      pool: $pool
      timestamp_gte: $start
      timestamp_lte: $end
      id_gt: $lastId
    }
  ) {
    id
    timestamp
    sender
    recipient
    amount0
    amount1
    amountUSD
    amount0Paid
    amount1Paid
    logIndex
    transaction {
      id
      blockNumber
      timestamp
      gasUsed
      gasPrice
    }
    pool {
      id
      feeTier
    }
  }
}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_query(query, variables):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                URL,
                json={"query": query, "variables": variables},
                timeout=90,
            )
            response.raise_for_status()
            result = response.json()

            if "errors" in result:
                raise RuntimeError(result["errors"])

            return result["data"]

        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"[Retry {attempt}/{MAX_RETRIES}] {exc}. Waiting {RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)


def safe_int(x):
    if x is None or x == "":
        return None
    return int(x)


def safe_float(x):
    if x is None or x == "":
        return None
    return float(x)


def month_start_ts(year, month):
    return calendar.timegm(datetime(year, month, 1, 0, 0, 0).timetuple())


def next_month(year, month):
    if month == 12:
        return year + 1, 1
    return year, month + 1


def build_month_ranges():
    ranges = []
    year, month = 2021, 5

    while True:
        ny, nm = next_month(year, month)

        month_start = month_start_ts(year, month)
        month_end = month_start_ts(ny, nm) - 1

        start_ts = max(month_start, START_TS)
        end_ts = min(month_end, END_TS)

        if start_ts <= end_ts:
            ranges.append((f"{year}_{month:02d}", start_ts, end_ts))

        if year == 2026 and month == 4:
            break

        year, month = ny, nm

    return ranges


def fetch_entity(query, entity_name, start_ts, end_ts):
    rows = []
    last_id = ""
    query_n = 0

    while True:
        variables = {
            "pool": POOL,
            "start": start_ts,
            "end": end_ts,
            "first": BATCH,
            "lastId": last_id,
        }

        data = run_query(query, variables)
        batch = data[entity_name]
        query_n += 1

        if not batch:
            break

        rows.extend(batch)
        last_id = batch[-1]["id"]

        if len(rows) % 10000 < BATCH:
            print(f"  {entity_name}: {len(rows):>8,} rows | {query_n:>5} queries")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    return rows


def flatten_collects(rows):
    out = []

    for r in rows:
        tx = r.get("transaction") or {}
        pool = r.get("pool") or {}

        out.append({
            "collect_id": r["id"],
            "tx_hash": tx.get("id"),
            "block_number": safe_int(tx.get("blockNumber")),
            "log_index": safe_int(r.get("logIndex")),
            "timestamp": pd.to_datetime(safe_int(r["timestamp"]), unit="s", utc=True),

            "owner": r.get("owner"),

            "amount0_usdc": safe_float(r.get("amount0")),
            "amount1_eth": safe_float(r.get("amount1")),
            "amount_usd": safe_float(r.get("amountUSD")),

            "tick_lower": safe_int(r.get("tickLower")),
            "tick_upper": safe_int(r.get("tickUpper")),

            "gas_used": safe_int(tx.get("gasUsed")),
            "gas_price_wei": safe_int(tx.get("gasPrice")),
            "gas_cost_eth": (
                safe_int(tx.get("gasUsed")) * safe_int(tx.get("gasPrice")) / 1e18
                if tx.get("gasUsed") is not None and tx.get("gasPrice") is not None
                else None
            ),

            "pool_id": pool.get("id"),
            "pool_fee_tier": pool.get("feeTier"),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["timestamp", "block_number", "log_index", "collect_id"])
    return df


def flatten_flashes(rows):
    out = []

    for r in rows:
        tx = r.get("transaction") or {}
        pool = r.get("pool") or {}

        out.append({
            "flash_id": r["id"],
            "tx_hash": tx.get("id"),
            "block_number": safe_int(tx.get("blockNumber")),
            "log_index": safe_int(r.get("logIndex")),
            "timestamp": pd.to_datetime(safe_int(r["timestamp"]), unit="s", utc=True),

            "sender": r.get("sender"),
            "recipient": r.get("recipient"),

            "amount0_usdc": safe_float(r.get("amount0")),
            "amount1_eth": safe_float(r.get("amount1")),
            "amount_usd": safe_float(r.get("amountUSD")),
            "amount0_paid_usdc": safe_float(r.get("amount0Paid")),
            "amount1_paid_eth": safe_float(r.get("amount1Paid")),

            "gas_used": safe_int(tx.get("gasUsed")),
            "gas_price_wei": safe_int(tx.get("gasPrice")),
            "gas_cost_eth": (
                safe_int(tx.get("gasUsed")) * safe_int(tx.get("gasPrice")) / 1e18
                if tx.get("gasUsed") is not None and tx.get("gasPrice") is not None
                else None
            ),

            "pool_id": pool.get("id"),
            "pool_fee_tier": pool.get("feeTier"),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["timestamp", "block_number", "log_index", "flash_id"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(r"C:\Courses\thesis_AMM")
    data_dir = project_root / "data_raw" / "DEX"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Uniswap V3 USDC/WETH 0.05% — Collects and Flashes")
    print("=" * 70)
    print(f"Pool: {POOL}")
    print(f"Output folder: {data_dir}")

    for month_label, start_ts, end_ts in build_month_ranges():
        start_str = datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d")
        end_str = datetime.utcfromtimestamp(end_ts).strftime("%Y-%m-%d")

        print("\n" + "=" * 70)
        print(f"MONTH {month_label}: {start_str} → {end_str}")
        print("=" * 70)

        # Collects
        collects_rows = fetch_entity(COLLECTS_QUERY, "collects", start_ts, end_ts)
        df_collects = flatten_collects(collects_rows)
        collects_file = os.path.join(data_dir, f"collects_{month_label}.csv")
        df_collects.to_csv(collects_file, index=False)
        print(f"Saved {collects_file} ({len(df_collects):,} rows)")

        # Flashes
        flashes_rows = fetch_entity(FLASHES_QUERY, "flashes", start_ts, end_ts)
        df_flashes = flatten_flashes(flashes_rows)
        flashes_file = os.path.join(data_dir, f"flashes_{month_label}.csv")
        df_flashes.to_csv(flashes_file, index=False)
        print(f"Saved {flashes_file} ({len(df_flashes):,} rows)")

    print("\nDONE")


if __name__ == "__main__":
    main()