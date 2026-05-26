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

Pagination uses id_gt inside each month to avoid timestamp-cursor skipping.
Partial saves happen every 50,000 rows; existing month files are skipped.

Setup:
    C:\\Interpreters\\python.exe -m pip install requests pandas
    setx THEGRAPH_API_KEY "your_key_here"
    Restart terminal.
    C:\\Interpreters\\python.exe C:\\Courses\\thesis_AMM\\scripts\\import_data\\DEX\\fetch_uniswap_extra_events.py
"""

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from dex_utils import (
    POOL, BATCH, SLEEP,
    run_query, safe_int, safe_float, build_month_ranges,
)


SAVE_EVERY = 50_000
LOG_EVERY  = 10_000
OVERWRITE  = False   # set True to re-fetch months whose output files already exist


# ── GraphQL queries ───────────────────────────────────────────────────────────

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


# ── Flatten nested JSON → flat rows ──────────────────────────────────────────

def flatten_collects(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        tx        = r.get("transaction") or {}
        pool      = r.get("pool") or {}
        gas_used  = safe_int(tx.get("gasUsed"))
        gas_price = safe_int(tx.get("gasPrice"))
        out.append({
            "collect_id":    r["id"],
            "tx_hash":       tx.get("id"),
            "block_number":  safe_int(tx.get("blockNumber")),
            "log_index":     safe_int(r.get("logIndex")),
            "timestamp":     pd.to_datetime(safe_int(r["timestamp"]), unit="s", utc=True),
            "owner":         r.get("owner"),
            "amount0_usdc":  safe_float(r.get("amount0")),
            "amount1_eth":   safe_float(r.get("amount1")),
            "amount_usd":    safe_float(r.get("amountUSD")),
            "tick_lower":    safe_int(r.get("tickLower")),
            "tick_upper":    safe_int(r.get("tickUpper")),
            "gas_used":      gas_used,
            "gas_price_wei": gas_price,
            "gas_cost_eth":  gas_used * gas_price / 1e18 if gas_used and gas_price else None,
            "pool_id":       pool.get("id"),
            "pool_fee_tier": pool.get("feeTier"),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["timestamp", "block_number", "log_index", "collect_id"])
    return df


def flatten_flashes(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        tx        = r.get("transaction") or {}
        pool      = r.get("pool") or {}
        gas_used  = safe_int(tx.get("gasUsed"))
        gas_price = safe_int(tx.get("gasPrice"))
        out.append({
            "flash_id":          r["id"],
            "tx_hash":           tx.get("id"),
            "block_number":      safe_int(tx.get("blockNumber")),
            "log_index":         safe_int(r.get("logIndex")),
            "timestamp":         pd.to_datetime(safe_int(r["timestamp"]), unit="s", utc=True),
            "sender":            r.get("sender"),
            "recipient":         r.get("recipient"),
            "amount0_usdc":      safe_float(r.get("amount0")),
            "amount1_eth":       safe_float(r.get("amount1")),
            "amount_usd":        safe_float(r.get("amountUSD")),
            "amount0_paid_usdc": safe_float(r.get("amount0Paid")),
            "amount1_paid_eth":  safe_float(r.get("amount1Paid")),
            "gas_used":          gas_used,
            "gas_price_wei":     gas_price,
            "gas_cost_eth":      gas_used * gas_price / 1e18 if gas_used and gas_price else None,
            "pool_id":           pool.get("id"),
            "pool_fee_tier":     pool.get("feeTier"),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["timestamp", "block_number", "log_index", "flash_id"])
    return df


# ── Pagination engine ─────────────────────────────────────────────────────────

def fetch_entity(
    query: str,
    entity_name: str,
    start_ts: int,
    end_ts: int,
    output_file: str,
    flatten_fn: Callable[[list], pd.DataFrame],
) -> list:
    """
    Paginate entity within [start_ts, end_ts] using id_gt cursor.
    Saves a partial CSV every SAVE_EVERY rows.
    """
    rows: list = []
    last_id    = ""
    query_n    = 0
    start_time = time.time()

    print(f"\n{'─'*60}")
    print(f"Fetching {entity_name}...")
    print(f"{'─'*60}")

    while True:
        variables = {
            "pool":   POOL,
            "start":  start_ts,
            "end":    end_ts,
            "first":  BATCH,
            "lastId": last_id,
        }

        data    = run_query(query, variables)
        batch   = data[entity_name]
        query_n += 1

        if not batch:
            break

        rows.extend(batch)
        last_id = batch[-1]["id"]

        n = len(rows)
        if n % LOG_EVERY < BATCH:
            elapsed = time.time() - start_time
            rate    = n / elapsed if elapsed > 0 else 0
            print(f"  {n:>8,} rows | {query_n:>5} queries | "
                  f"{rate:,.0f} rows/s | {elapsed/60:.1f} min elapsed")

        if n % SAVE_EVERY < BATCH and n > 0:
            tmp = output_file + ".partial"
            flatten_fn(rows).to_csv(tmp, index=False)
            print(f"  [Auto-saved {n:,} rows → {tmp}]")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    elapsed = time.time() - start_time
    print(f"\n  Completed: {len(rows):,} {entity_name} in "
          f"{query_n} queries ({elapsed/60:.1f} min)")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir     = project_root / "data_raw" / "DEX"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Uniswap V3 USDC/WETH 0.05% — Collects and Flashes")
    print("=" * 70)
    print(f"Pool:   {POOL}")
    print(f"Output: {data_dir}")
    print(f"Note:   existing month files are {'overwritten' if OVERWRITE else 'skipped'}")
    print("=" * 70)

    for month_label, start_ts, end_ts in build_month_ranges():
        start_str = datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d")
        end_str   = datetime.utcfromtimestamp(end_ts).strftime("%Y-%m-%d")

        collects_file = str(data_dir / f"collects_{month_label}.csv")
        flashes_file  = str(data_dir / f"flashes_{month_label}.csv")

        print("\n" + "=" * 70)
        print(f"MONTH {month_label}: {start_str} → {end_str}")
        print("=" * 70)

        if not OVERWRITE and all(
            os.path.exists(p) and os.path.getsize(p) > 0
            for p in (collects_file, flashes_file)
        ):
            print("  All files exist — skipping. Set OVERWRITE = True to re-fetch.")
            continue

        # Collects
        collects_rows = fetch_entity(
            COLLECTS_QUERY, "collects", start_ts, end_ts,
            collects_file, flatten_collects,
        )
        df_collects = flatten_collects(collects_rows)
        df_collects.to_csv(collects_file, index=False)
        if os.path.exists(collects_file + ".partial"):
            os.remove(collects_file + ".partial")
        print(f"  Saved {collects_file} ({len(df_collects):,} rows)")

        # Flashes
        flashes_rows = fetch_entity(
            FLASHES_QUERY, "flashes", start_ts, end_ts,
            flashes_file, flatten_flashes,
        )
        df_flashes = flatten_flashes(flashes_rows)
        df_flashes.to_csv(flashes_file, index=False)
        if os.path.exists(flashes_file + ".partial"):
            os.remove(flashes_file + ".partial")
        print(f"  Saved {flashes_file} ({len(df_flashes):,} rows)")

    print("\nDONE")


if __name__ == "__main__":
    main()
