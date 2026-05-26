"""
Uniswap V3 USDC/ETH — Since Inception to End of April 2026 Dataset
===================================================================
Fetches all swaps, mints, and burns from Uniswap V3 inception up to
2026-04-30 23:59:59 UTC.

Uses timestamp-cursor pagination — no row limit, no skip limit.

Pool:   USDC/ETH 0.05% on Ethereum mainnet
Period: 2021-05-05 00:00:00 UTC → 2026-04-30 23:59:59 UTC

Output files, saved in /data/ relative to this script:
    data/swaps_YYYY_MM.csv
    data/mints_YYYY_MM.csv
    data/burns_YYYY_MM.csv

Setup (VS Code terminal — Windows):
    C:\\Interpreters\\python.exe -m pip install requests pandas
    setx THEGRAPH_API_KEY "your_key_here"
    Restart terminal.
    C:\\Interpreters\\python.exe fetch_all_monthly.py

Runtime: very long. Leave it running — progress is printed every 10,000 rows and
partial results are saved automatically every 50,000 rows. Months that already
have output files are skipped; set OVERWRITE = True to re-fetch them.
"""

import os
import time
from datetime import datetime

import pandas as pd

from dex_utils import (
    POOL, BATCH, SLEEP,
    run_query, safe_int, safe_float, build_month_ranges,
)


SAVE_EVERY = 50_000
LOG_EVERY  = 10_000
OVERWRITE  = False   # set True to re-fetch months whose output files already exist


# ── GraphQL queries ───────────────────────────────────────────────────────────

SWAPS_QUERY = """
query($pool: String!, $lastTs: Int!, $first: Int!, $end: Int!) {
  swaps(
    first: $first
    where: { pool: $pool, timestamp_gt: $lastTs, timestamp_lte: $end }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    timestamp
    origin
    sender
    recipient
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
    logIndex
    transaction { id blockNumber timestamp gasUsed gasPrice }
    pool {
      id feeTier liquidity sqrtPrice tick
      token0Price token1Price
      volumeUSD feesUSD txCount
      totalValueLockedUSD totalValueLockedToken0 totalValueLockedToken1
    }
    token0 { id symbol decimals derivedETH }
    token1 { id symbol decimals derivedETH }
  }
}
"""

MINTS_QUERY = """
query($pool: String!, $lastTs: Int!, $first: Int!, $end: Int!) {
  mints(
    first: $first
    where: { pool: $pool, timestamp_gt: $lastTs, timestamp_lte: $end }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    timestamp
    origin
    owner
    amount
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
    logIndex
    transaction { id blockNumber timestamp gasUsed gasPrice }
    pool {
      id feeTier liquidity sqrtPrice tick
      token0Price token1Price totalValueLockedUSD
    }
    token0 { id symbol decimals }
    token1 { id symbol decimals }
  }
}
"""

BURNS_QUERY = """
query($pool: String!, $lastTs: Int!, $first: Int!, $end: Int!) {
  burns(
    first: $first
    where: { pool: $pool, timestamp_gt: $lastTs, timestamp_lte: $end }
    orderBy: timestamp
    orderDirection: asc
  ) {
    id
    timestamp
    origin
    owner
    amount
    amount0
    amount1
    amountUSD
    tickLower
    tickUpper
    logIndex
    transaction { id blockNumber timestamp gasUsed gasPrice }
    pool {
      id feeTier liquidity sqrtPrice tick
      token0Price token1Price totalValueLockedUSD
    }
    token0 { id symbol decimals }
    token1 { id symbol decimals }
  }
}
"""


# ── Flatten nested JSON → flat rows ──────────────────────────────────────────

def flatten_swaps(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        tx = r["transaction"]
        pl = r["pool"]
        t0 = r["token0"]
        t1 = r["token1"]
        fee_rate = safe_int(pl["feeTier"]) / 1_000_000
        out.append({
            # Identifiers
            "swap_id":              r["id"],
            "tx_hash":              tx["id"],
            "block_number":         int(tx["blockNumber"]),
            "log_index":            r["logIndex"],
            # Time
            "timestamp":            pd.to_datetime(int(r["timestamp"]), unit="s", utc=True),
            # Participants
            "origin":               r["origin"],
            "sender":               r["sender"],
            "recipient":            r["recipient"],
            # Swap amounts
            "amount0_usdc":         float(r["amount0"]),
            "amount1_eth":          float(r["amount1"]),
            "amount_usd":           float(r["amountUSD"]),
            "fee_usd":              float(r["amountUSD"]) * fee_rate,
            # Direction
            "buy_eth":              float(r["amount0"]) > 0,
            # Price (raw)
            "sqrt_price_x96":       r["sqrtPriceX96"],
            "tick_at_swap":         r["tick"],
            # Gas
            "gas_used":             int(tx["gasUsed"]),
            "gas_price_wei":        int(tx["gasPrice"]),
            "gas_cost_eth":         int(tx["gasUsed"]) * int(tx["gasPrice"]) / 1e18,
            # Pool state
            "pool_liquidity":       pl["liquidity"],
            "pool_sqrt_price":      pl["sqrtPrice"],
            "pool_tick":            pl["tick"],
            "pool_token0_price":    float(pl["token0Price"]),
            "pool_token1_price":    float(pl["token1Price"]),
            "pool_fee_tier":        pl["feeTier"],
            "pool_volume_usd":      float(pl["volumeUSD"]),
            "pool_fees_usd":        float(pl["feesUSD"]),
            "pool_tx_count":        int(pl["txCount"]),
            "pool_tvl_usd":         float(pl["totalValueLockedUSD"]),
            "pool_tvl_token0_usdc": float(pl["totalValueLockedToken0"]),
            "pool_tvl_token1_eth":  float(pl["totalValueLockedToken1"]),
            # Token metadata
            "token0_symbol":        t0["symbol"],
            "token0_decimals":      t0["decimals"],
            "token0_derived_eth":   float(t0["derivedETH"]),
            "token1_symbol":        t1["symbol"],
            "token1_decimals":      t1["decimals"],
            "token1_derived_eth":   float(t1["derivedETH"]),
        })
    return pd.DataFrame(out)


def flatten_mints_burns(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        tx = r["transaction"]
        pl = r["pool"]
        t0 = r["token0"]
        t1 = r["token1"]
        out.append({
            # Identifiers
            "id":                   r["id"],
            "tx_hash":              tx["id"],
            "block_number":         int(tx["blockNumber"]),
            "log_index":            r["logIndex"],
            # Time
            "timestamp":            pd.to_datetime(int(r["timestamp"]), unit="s", utc=True),
            # Participants
            "origin":               r["origin"],
            "owner":                r["owner"],
            # Liquidity
            "liquidity_units":      float(r["amount"]),
            "amount0_usdc":         float(r["amount0"]),
            "amount1_eth":          float(r["amount1"]),
            "amount_usd":           float(r["amountUSD"]),
            # LP range
            "tick_lower":           r["tickLower"],
            "tick_upper":           r["tickUpper"],
            # Gas
            "gas_used":             int(tx["gasUsed"]),
            "gas_price_wei":        int(tx["gasPrice"]),
            "gas_cost_eth":         int(tx["gasUsed"]) * int(tx["gasPrice"]) / 1e18,
            # Pool state
            "pool_liquidity":       pl["liquidity"],
            "pool_sqrt_price":      pl["sqrtPrice"],
            "pool_tick":            pl["tick"],
            "pool_token0_price":    float(pl["token0Price"]),
            "pool_token1_price":    float(pl["token1Price"]),
            "pool_fee_tier":        pl["feeTier"],
            "pool_tvl_usd":         float(pl["totalValueLockedUSD"]),
            # Token metadata
            "token0_symbol":        t0["symbol"],
            "token0_decimals":      t0["decimals"],
            "token1_symbol":        t1["symbol"],
            "token1_decimals":      t1["decimals"],
        })
    return pd.DataFrame(out)


# ── Pagination engine ─────────────────────────────────────────────────────────

def _save_partial(rows: list, entity: str, output_file: str) -> None:
    tmp = output_file + ".partial"
    df = flatten_swaps(rows) if entity == "swaps" else flatten_mints_burns(rows)
    df.to_csv(tmp, index=False)
    print(f"  [Auto-saved {len(rows):,} rows → {tmp}]")


def fetch_all(query: str, entity: str, output_file: str, start_ts: int, end_ts: int) -> list:
    """
    Paginate using timestamp cursor.
    Saves partial CSV every SAVE_EVERY rows so you don't lose progress
    if the script is interrupted.
    """
    all_rows   = []
    last_ts    = start_ts - 1
    query_n    = 0
    start_time = time.time()

    print(f"\n{'─'*60}")
    print(f"Fetching {entity}...")
    print(f"{'─'*60}")

    while True:
        variables = {
            "pool":   POOL,
            "lastTs": last_ts,
            "first":  BATCH,
            "end":    end_ts,
        }

        data    = run_query(query, variables)
        batch   = data[entity]
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
            _save_partial(all_rows, entity, output_file)

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    elapsed = time.time() - start_time
    print(f"\n  Completed: {len(all_rows):,} {entity} in "
          f"{query_n} queries ({elapsed/60:.1f} min)")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("Uniswap V3 USDC/ETH — Since Inception to End of April 2026")
    print("=" * 60)
    print(f"Pool:   {POOL}")
    print(f"Period: 2021-05-05  →  2026-04-30")
    print(f"Output: monthly files in {data_dir}")
    print(f"Note:   partial saves every {SAVE_EVERY:,} rows")
    print(f"Note:   existing month files are {'overwritten' if OVERWRITE else 'skipped'}")
    print("=" * 60)

    monthly_ranges    = build_month_ranges()
    grand_total_swaps = 0
    grand_total_mints = 0
    grand_total_burns = 0
    all_output_files  = []

    for month_label, month_start, month_end in monthly_ranges:
        month_start_str = datetime.utcfromtimestamp(month_start).strftime("%Y-%m-%d")
        month_end_str   = datetime.utcfromtimestamp(month_end).strftime("%Y-%m-%d")

        swaps_file = os.path.join(data_dir, f"swaps_{month_label}.csv")
        mints_file = os.path.join(data_dir, f"mints_{month_label}.csv")
        burns_file = os.path.join(data_dir, f"burns_{month_label}.csv")

        print("\n" + "=" * 60)
        print(f"MONTH: {month_label}")
        print(f"Period: {month_start_str}  →  {month_end_str}")
        print("=" * 60)

        if not OVERWRITE and all(
            os.path.exists(p) and os.path.getsize(p) > 0
            for p in (swaps_file, mints_file, burns_file)
        ):
            print("  All files exist — skipping. Set OVERWRITE = True to re-fetch.")
            all_output_files.extend([swaps_file, mints_file, burns_file])
            continue

        # ── Swaps ──
        raw_swaps = fetch_all(SWAPS_QUERY, "swaps", swaps_file, month_start, month_end)
        df_swaps  = flatten_swaps(raw_swaps)
        df_swaps.to_csv(swaps_file, index=False)
        if os.path.exists(swaps_file + ".partial"):
            os.remove(swaps_file + ".partial")
        print(f"  Saved: {swaps_file}  ({len(df_swaps):,} rows, "
              f"{os.path.getsize(swaps_file)/1e6:.1f} MB)")

        # ── Mints ──
        raw_mints = fetch_all(MINTS_QUERY, "mints", mints_file, month_start, month_end)
        df_mints  = flatten_mints_burns(raw_mints)
        df_mints.to_csv(mints_file, index=False)
        if os.path.exists(mints_file + ".partial"):
            os.remove(mints_file + ".partial")
        print(f"  Saved: {mints_file}  ({len(df_mints):,} rows, "
              f"{os.path.getsize(mints_file)/1e6:.1f} MB)")

        # ── Burns ──
        raw_burns = fetch_all(BURNS_QUERY, "burns", burns_file, month_start, month_end)
        df_burns  = flatten_mints_burns(raw_burns)
        df_burns.to_csv(burns_file, index=False)
        if os.path.exists(burns_file + ".partial"):
            os.remove(burns_file + ".partial")
        print(f"  Saved: {burns_file}  ({len(df_burns):,} rows, "
              f"{os.path.getsize(burns_file)/1e6:.1f} MB)")

        grand_total_swaps += len(df_swaps)
        grand_total_mints += len(df_mints)
        grand_total_burns += len(df_burns)
        all_output_files.extend([swaps_file, mints_file, burns_file])

        month_mb = (
            os.path.getsize(swaps_file)
            + os.path.getsize(mints_file)
            + os.path.getsize(burns_file)
        ) / 1e6

        print("\n" + "-" * 60)
        print(f"MONTH DONE: {month_label}")
        print(f"  swaps_{month_label}.csv  — {len(df_swaps):>8,} rows")
        print(f"  mints_{month_label}.csv  — {len(df_mints):>8,} rows")
        print(f"  burns_{month_label}.csv  — {len(df_burns):>8,} rows")
        print(f"  Month size              — {month_mb:.1f} MB")
        print("-" * 60)

    total_mb = sum(
        os.path.getsize(f) for f in all_output_files if os.path.exists(f)
    ) / 1e6

    print("\n" + "=" * 60)
    print("DONE")
    print(f"  Monthly files saved in: {data_dir}")
    print(f"  swaps total  — {grand_total_swaps:>8,} rows")
    print(f"  mints total  — {grand_total_mints:>8,} rows")
    print(f"  burns total  — {grand_total_burns:>8,} rows")
    print(f"  Total size   — {total_mb:.1f} MB")
    print("=" * 60)
    print("\nNext step: fetch CEX prices from Binance and merge on timestamp.")


if __name__ == "__main__":
    main()
