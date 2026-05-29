"""
Uniswap V3 USDC/WETH 0.05% — Position Snapshot Downloader
==========================================================

Fetches historical position-state snapshots from The Graph's
positionSnapshots entity.

A positionSnapshot is recorded each time an LP position changes state —
on every Mint, Burn, or Collect event.  Each row captures:

    - Cumulative deposited / withdrawn token amounts
    - Cumulative collected fees (token0/1)
    - feeGrowthInsideLastX128 accumulators for precise fee attribution
    - Position tick range (for in-range analysis)

This data enables:
    - Individual LP P&L tracking over the full position lifecycle
    - In-range fraction analysis: position is active when
          tick_lower <= pool_tick <= tick_upper
    - Precise fee income per position via fee-growth arithmetic:
          uncollected_fee0 = liquidity × (feeGrowthInside0Upper - feeGrowthInside0LastX128) / 2^128
    - Entry/exit timing and deposit/withdrawal history per LP

Output (monthly CSVs):
    data_raw/DEX/position_snapshots_YYYY_MM.csv

Columns:
    snapshot_id                 — "{position_id}#{blockNumber}" (unique)
    position_id                 — links to positions_current.csv
    owner                       — LP wallet address
    pool_id / pool_fee_tier
    block_number / timestamp
    tick_lower / tick_upper     — LP range for in-range check
    liquidity                   — raw liquidity units at this snapshot
    deposited_token0_usdc       — CUMULATIVE deposited USDC
    deposited_token1_weth       — CUMULATIVE deposited WETH
    withdrawn_token0_usdc       — CUMULATIVE withdrawn USDC
    withdrawn_token1_weth       — CUMULATIVE withdrawn WETH
    collected_fees_token0_usdc  — CUMULATIVE collected fees in USDC
    collected_fees_token1_weth  — CUMULATIVE collected fees in WETH
    fee_growth_inside0_last_x128 — fee growth accumulator (X128) token0
    fee_growth_inside1_last_x128 — fee growth accumulator (X128) token1
    tx_hash / gas_used / gas_price_wei / gas_cost_eth

Note:
    All deposited/withdrawn/fee columns are CUMULATIVE lifetime values,
    not deltas.  To get incremental amounts between two snapshots for the
    same position, subtract consecutive rows.

Estimated total rows: ~500,000 – 1,000,000 over the study period
(proportional to total mint + burn + collect events: ~440K).

Setup:
    setx THEGRAPH_API_KEY "your_key_here"
    Restart terminal.
    python scripts/import_data/DEX/fetch_uniswap_position_snapshots.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from dex_utils import (
    POOL, BATCH, SLEEP,
    run_query, safe_int, safe_float, build_month_ranges,
)


SAVE_EVERY = 50_000
LOG_EVERY  = 10_000
OVERWRITE  = False   # set True to re-fetch months whose output files already exist


# ── GraphQL query ─────────────────────────────────────────────────────────────

POSITION_SNAPSHOTS_QUERY = """
query($pool: String!, $lastTs: Int!, $first: Int!, $end: Int!) {
  positionSnapshots(
    first: $first
    orderBy: timestamp
    orderDirection: asc
    where: {
      pool: $pool
      timestamp_gt: $lastTs
      timestamp_lte: $end
    }
  ) {
    id
    owner
    blockNumber
    timestamp
    liquidity
    depositedToken0
    depositedToken1
    withdrawnToken0
    withdrawnToken1
    collectedFeesToken0
    collectedFeesToken1
    feeGrowthInside0LastX128
    feeGrowthInside1LastX128
    pool {
      id
      feeTier
    }
    position {
      id
      tickLower { tickIdx }
      tickUpper { tickIdx }
    }
    transaction {
      id
      blockNumber
      timestamp
      gasUsed
      gasPrice
    }
  }
}
"""


# ── Flatten nested JSON → flat rows ──────────────────────────────────────────

def flatten_snapshots(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        pool     = r.get("pool") or {}
        position = r.get("position") or {}
        tl       = position.get("tickLower") or {}
        tu       = position.get("tickUpper") or {}
        tx       = r.get("transaction") or {}
        gas_used  = safe_int(tx.get("gasUsed"))
        gas_price = safe_int(tx.get("gasPrice"))
        out.append({
            "snapshot_id":                  r["id"],
            "position_id":                  position.get("id"),
            "owner":                        r.get("owner"),
            "pool_id":                      pool.get("id"),
            "pool_fee_tier":                safe_int(pool.get("feeTier")),
            "block_number":                 safe_int(r.get("blockNumber")),
            "timestamp":                    pd.to_datetime(
                                                safe_int(r["timestamp"]),
                                                unit="s", utc=True,
                                            ),
            # LP range (from parent position — constant per position)
            "tick_lower":                   safe_int(tl.get("tickIdx")),
            "tick_upper":                   safe_int(tu.get("tickIdx")),
            # Current state at this snapshot
            "liquidity":                    r.get("liquidity"),
            # Cumulative amounts (token0 = USDC, token1 = WETH)
            "deposited_token0_usdc":        safe_float(r.get("depositedToken0")),
            "deposited_token1_weth":        safe_float(r.get("depositedToken1")),
            "withdrawn_token0_usdc":        safe_float(r.get("withdrawnToken0")),
            "withdrawn_token1_weth":        safe_float(r.get("withdrawnToken1")),
            "collected_fees_token0_usdc":   safe_float(r.get("collectedFeesToken0")),
            "collected_fees_token1_weth":   safe_float(r.get("collectedFeesToken1")),
            # Fee-growth accumulators (X128 fixed-point integers)
            "fee_growth_inside0_last_x128": r.get("feeGrowthInside0LastX128"),
            "fee_growth_inside1_last_x128": r.get("feeGrowthInside1LastX128"),
            # Transaction / gas
            "tx_hash":                      tx.get("id"),
            "gas_used":                     gas_used,
            "gas_price_wei":                gas_price,
            "gas_cost_eth":                 (
                gas_used * gas_price / 1e18
                if gas_used and gas_price else None
            ),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["timestamp", "block_number", "snapshot_id"])
    return df


# ── Pagination engine ─────────────────────────────────────────────────────────

def fetch_month(start_ts: int, end_ts: int, output_file: str) -> list:
    """
    Paginate positionSnapshots within [start_ts, end_ts] using timestamp cursor.
    Auto-saves a partial CSV every SAVE_EVERY rows.
    """
    all_rows   = []
    last_ts    = start_ts - 1
    query_n    = 0
    start_time = time.time()

    while True:
        variables = {
            "pool":   POOL,
            "lastTs": last_ts,
            "first":  BATCH,
            "end":    end_ts,
        }

        data    = run_query(POSITION_SNAPSHOTS_QUERY, variables)
        batch   = data["positionSnapshots"]
        query_n += 1

        if not batch:
            break

        all_rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])

        n = len(all_rows)
        if n % LOG_EVERY < BATCH:
            elapsed = time.time() - start_time
            rate    = n / elapsed if elapsed > 0 else 0
            last_dt = datetime.utcfromtimestamp(last_ts).strftime("%Y-%m-%d")
            print(f"  {n:>8,} rows | {query_n:>5} queries | "
                  f"up to {last_dt} | {rate:,.0f} rows/s | "
                  f"{elapsed/60:.1f} min elapsed")

        if n % SAVE_EVERY < BATCH and n > 0:
            tmp = output_file + ".partial"
            flatten_snapshots(all_rows).to_csv(tmp, index=False)
            print(f"  [Auto-saved {n:,} rows -> {tmp}]")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    elapsed = time.time() - start_time
    print(f"\n  Completed: {len(all_rows):,} snapshots in "
          f"{query_n} queries ({elapsed/60:.1f} min)")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir     = project_root / "data_raw" / "DEX"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Uniswap V3 USDC/WETH 0.05% — Position Snapshots")
    print("=" * 70)
    print(f"Pool:   {POOL}")
    print(f"Period: 2021-05-05  ->  2026-04-30")
    print(f"Output: {data_dir}")
    print(f"Files:  position_snapshots_YYYY_MM.csv  (60 months)")
    print(f"Skip:   existing files are {'overwritten' if OVERWRITE else 'kept (set OVERWRITE=True to re-fetch)'}")
    print("=" * 70)

    grand_total = 0

    for month_label, start_ts, end_ts in build_month_ranges():
        start_str = datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d")
        end_str   = datetime.utcfromtimestamp(end_ts).strftime("%Y-%m-%d")
        out_file  = str(data_dir / f"position_snapshots_{month_label}.csv")

        print("\n" + "=" * 70)
        print(f"MONTH {month_label}: {start_str} -> {end_str}")
        print("=" * 70)

        if not OVERWRITE and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            existing_rows = sum(1 for _ in open(out_file)) - 1
            print(f"  Exists ({existing_rows:,} rows) — skipping.")
            grand_total += max(existing_rows, 0)
            continue

        rows = fetch_month(start_ts, end_ts, out_file)
        df   = flatten_snapshots(rows)
        df.to_csv(out_file, index=False)
        if os.path.exists(out_file + ".partial"):
            os.remove(out_file + ".partial")

        grand_total += len(df)
        size_kb = os.path.getsize(out_file) / 1e3
        print(f"  Saved {out_file}  ({len(df):,} rows, {size_kb:.0f} KB)")

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Total position snapshots: {grand_total:,}")
    print(f"  Output dir: {data_dir}")
    print("=" * 70)
    print("\nNext step: run scripts/import_data/DEX/fetch_uniswap_tick_snapshots.py")


if __name__ == "__main__":
    main()
