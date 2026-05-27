"""
Uniswap V3 USDC/WETH 0.05% — Flash Loan Downloader
====================================================

Fetches Flash events from The Graph.

Uniswap v3 Flash events are ERC-3156-style flash loans: a caller borrows
token0 and/or token1 within a single transaction and must repay the
principal plus a fee in the same tx.  The fee equals the regular swap
fee on the borrowed amount (0.05% for this pool).

Output (monthly CSVs):
    data_raw/DEX/flashes_YYYY_MM.csv

Columns:
    flash_id                    — TheGraph ID (unique)
    tx_hash / block_number / log_index / timestamp
    sender / recipient          — borrower addresses
    amount0_usdc                — USDC borrowed (may be 0)
    amount1_weth                — WETH borrowed (may be 0)
    amount_usd                  — USD value of borrowed amount (TheGraph estimate)
    amount0_paid_usdc           — USDC repaid (principal + fee)
    amount1_paid_weth           — WETH repaid (principal + fee)
    fee_token0_usdc             — amount0_paid - amount0  (USDC fee, exact)
    fee_token1_weth             — amount1_paid - amount1  (WETH fee, exact)
    pool_id / pool_fee_tier
    gas_used / gas_price_wei / gas_cost_eth

Note:
    fetch_uniswap_extra_events.py also fetches flash events but is coupled
    with collects (which returns 0 results for this pool), causing skip-logic
    problems.  This dedicated script handles flash events independently.

    Expected total rows: a few hundred to a few thousand — flash loans are
    rare relative to regular swaps on this pool.

Setup:
    setx THEGRAPH_API_KEY "your_key_here"
    Restart terminal.
    python scripts/import_data/DEX/fetch_uniswap_flashes.py
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


OVERWRITE = False   # set True to re-fetch months whose output files already exist


# ── GraphQL query ─────────────────────────────────────────────────────────────

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

def flatten_flashes(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        tx        = r.get("transaction") or {}
        pool      = r.get("pool") or {}
        gas_used  = safe_int(tx.get("gasUsed"))
        gas_price = safe_int(tx.get("gasPrice"))

        amt0      = safe_float(r.get("amount0"))
        amt1      = safe_float(r.get("amount1"))
        paid0     = safe_float(r.get("amount0Paid"))
        paid1     = safe_float(r.get("amount1Paid"))

        out.append({
            "flash_id":          r["id"],
            "tx_hash":           tx.get("id"),
            "block_number":      safe_int(tx.get("blockNumber")),
            "log_index":         safe_int(r.get("logIndex")),
            "timestamp":         pd.to_datetime(
                                     safe_int(r["timestamp"]),
                                     unit="s", utc=True,
                                 ),
            "sender":            r.get("sender"),
            "recipient":         r.get("recipient"),
            # Borrowed amounts
            "amount0_usdc":      amt0,
            "amount1_weth":      amt1,
            "amount_usd":        safe_float(r.get("amountUSD")),
            # Repaid amounts (principal + fee)
            "amount0_paid_usdc": paid0,
            "amount1_paid_weth": paid1,
            # Fees earned by LPs (exact, in native tokens)
            "fee_token0_usdc":   (paid0 - amt0) if (paid0 is not None and amt0 is not None) else None,
            "fee_token1_weth":   (paid1 - amt1) if (paid1 is not None and amt1 is not None) else None,
            # Pool
            "pool_id":           pool.get("id"),
            "pool_fee_tier":     safe_int(pool.get("feeTier")),
            # Gas
            "gas_used":          gas_used,
            "gas_price_wei":     gas_price,
            "gas_cost_eth":      (
                gas_used * gas_price / 1e18
                if gas_used and gas_price else None
            ),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values(["timestamp", "block_number", "log_index"])
    return df


# ── Pagination engine ─────────────────────────────────────────────────────────

def fetch_month(start_ts: int, end_ts: int) -> list:
    """
    Paginate flashes within [start_ts, end_ts] using id_gt cursor.
    id_gt is safe for flashes because their IDs are ordered by TheGraph
    insertion order (monotonically increasing).
    """
    rows    = []
    last_id = ""
    query_n = 0

    while True:
        variables = {
            "pool":   POOL,
            "start":  start_ts,
            "end":    end_ts,
            "first":  BATCH,
            "lastId": last_id,
        }

        data    = run_query(FLASHES_QUERY, variables)
        batch   = data["flashes"]
        query_n += 1

        if not batch:
            break

        rows.extend(batch)
        last_id = batch[-1]["id"]

        print(f"  flashes: {len(rows):>6,} rows | {query_n:>4} queries | last_id={last_id}")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    print(f"  Completed: {len(rows):,} flash events in {query_n} queries")
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir     = project_root / "data_raw" / "DEX"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Uniswap V3 USDC/WETH 0.05% — Flash Events")
    print("=" * 70)
    print(f"Pool:   {POOL}")
    print(f"Period: 2021-05-05  →  2026-04-30")
    print(f"Output: {data_dir}")
    print(f"Files:  flashes_YYYY_MM.csv  (60 months)")
    print(f"Skip:   existing files are {'overwritten' if OVERWRITE else 'kept (set OVERWRITE=True to re-fetch)'}")
    print("=" * 70)

    grand_total = 0

    for month_label, start_ts, end_ts in build_month_ranges():
        start_str = datetime.utcfromtimestamp(start_ts).strftime("%Y-%m-%d")
        end_str   = datetime.utcfromtimestamp(end_ts).strftime("%Y-%m-%d")
        out_file  = str(data_dir / f"flashes_{month_label}.csv")

        print(f"\nMONTH {month_label}: {start_str} → {end_str}")

        if not OVERWRITE and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            existing_rows = sum(1 for _ in open(out_file)) - 1
            print(f"  Exists ({existing_rows:,} rows) — skipping.")
            grand_total += max(existing_rows, 0)
            continue

        rows = fetch_month(start_ts, end_ts)
        df   = flatten_flashes(rows)
        df.to_csv(out_file, index=False)

        grand_total += len(df)
        print(f"  Saved: {out_file}  ({len(df):,} rows)")

    print("\n" + "=" * 70)
    print("DONE")
    print(f"  Total flash events fetched: {grand_total:,}")
    print("=" * 70)
    print("\nNext step: run fetch_uniswap_position_snapshots.py")


if __name__ == "__main__":
    main()
