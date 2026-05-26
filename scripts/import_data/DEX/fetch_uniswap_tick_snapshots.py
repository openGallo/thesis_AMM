"""
Uniswap V3 USDC/WETH 0.05% — Tick Snapshot Downloader
======================================================

Fetches tick-level liquidity snapshots from The Graph.

Outputs:
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\tick_snapshots\\ticks_current.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\tick_snapshots\\pool_state_current.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\tick_snapshots\\ticks_YYYY_MM_block_BLOCKNUMBER.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\tick_snapshots\\pool_state_YYYY_MM_block_BLOCKNUMBER.csv
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\tick_snapshots\\month_end_blocks_from_existing_files.csv

How month-end blocks are selected:
    This script scans your existing monthly event CSVs:

        swaps_YYYY_MM.csv
        mints_YYYY_MM.csv
        burns_YYYY_MM.csv
        collects_YYYY_MM.csv
        flashes_YYYY_MM.csv

    Then it takes the maximum block_number available for each month.

This is good enough for monthly snapshots based on your Uniswap activity data.
For exact calendar month-end blocks, you need a separate block-by-timestamp source.

Setup:
    C:\\Interpreters\\python.exe -m pip install requests pandas
    setx THEGRAPH_API_KEY "your_key_here"
    Restart terminal.
    C:\\Interpreters\\python.exe C:\\Courses\\thesis_AMM\\scripts\\import_data\\DEX\\fetch_uniswap_tick_snapshots.py
"""

import glob
import os
import re
import time
from pathlib import Path

import pandas as pd

from dex_utils import POOL, BATCH, SLEEP, run_query, safe_int, safe_float


# Only fetch ticks with nonzero liquidityNet — enough for active-liquidity
# reconstruction because liquidity only changes where liquidityNet != 0.
LIQUIDITY_NET_ONLY = True


# ── GraphQL queries ───────────────────────────────────────────────────────────

POOL_STATE_CURRENT_QUERY = """
query($pool: ID!) {
  pool(id: $pool) {
    id
    feeTier
    liquidity
    sqrtPrice
    tick
    token0Price
    token1Price
    totalValueLockedUSD
    totalValueLockedToken0
    totalValueLockedToken1
    token0 { id symbol decimals }
    token1 { id symbol decimals }
  }
}
"""

POOL_STATE_BLOCK_QUERY = """
query($pool: ID!, $blockNumber: Int!) {
  pool(id: $pool, block: { number: $blockNumber }) {
    id
    feeTier
    liquidity
    sqrtPrice
    tick
    token0Price
    token1Price
    totalValueLockedUSD
    totalValueLockedToken0
    totalValueLockedToken1
    token0 { id symbol decimals }
    token1 { id symbol decimals }
  }
}
"""

TICKS_CURRENT_QUERY_NET_ONLY = """
query($pool: String!, $first: Int!, $lastTick: Int!) {
  ticks(
    first: $first
    orderBy: tickIdx
    orderDirection: asc
    where: {
      poolAddress: $pool
      liquidityNet_not: "0"
      tickIdx_gt: $lastTick
    }
  ) {
    id
    poolAddress
    tickIdx
    liquidityGross
    liquidityNet
    price0
    price1
    createdAtTimestamp
    createdAtBlockNumber
  }
}
"""

TICKS_BLOCK_QUERY_NET_ONLY = """
query($pool: String!, $first: Int!, $lastTick: Int!, $blockNumber: Int!) {
  ticks(
    first: $first
    orderBy: tickIdx
    orderDirection: asc
    block: { number: $blockNumber }
    where: {
      poolAddress: $pool
      liquidityNet_not: "0"
      tickIdx_gt: $lastTick
    }
  ) {
    id
    poolAddress
    tickIdx
    liquidityGross
    liquidityNet
    price0
    price1
    createdAtTimestamp
    createdAtBlockNumber
  }
}
"""

TICKS_CURRENT_QUERY_ALL = """
query($pool: String!, $first: Int!, $lastTick: Int!) {
  ticks(
    first: $first
    orderBy: tickIdx
    orderDirection: asc
    where: {
      poolAddress: $pool
      tickIdx_gt: $lastTick
    }
  ) {
    id
    poolAddress
    tickIdx
    liquidityGross
    liquidityNet
    price0
    price1
    createdAtTimestamp
    createdAtBlockNumber
  }
}
"""

TICKS_BLOCK_QUERY_ALL = """
query($pool: String!, $first: Int!, $lastTick: Int!, $blockNumber: Int!) {
  ticks(
    first: $first
    orderBy: tickIdx
    orderDirection: asc
    block: { number: $blockNumber }
    where: {
      poolAddress: $pool
      tickIdx_gt: $lastTick
    }
  ) {
    id
    poolAddress
    tickIdx
    liquidityGross
    liquidityNet
    price0
    price1
    createdAtTimestamp
    createdAtBlockNumber
  }
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def flatten_pool_state(pool: dict, label: str, block_number=None) -> pd.DataFrame:
    if pool is None:
        return pd.DataFrame()

    token0 = pool.get("token0") or {}
    token1 = pool.get("token1") or {}

    row = {
        "snapshot_label":  label,
        "block_number":    block_number,
        "pool_id":         pool.get("id"),
        "fee_tier":        safe_int(pool.get("feeTier")),
        "liquidity":       pool.get("liquidity"),
        "sqrt_price":      pool.get("sqrtPrice"),
        "tick":            safe_int(pool.get("tick")),
        "token0_price":    safe_float(pool.get("token0Price")),
        "token1_price":    safe_float(pool.get("token1Price")),
        "tvl_usd":         safe_float(pool.get("totalValueLockedUSD")),
        "tvl_token0":      safe_float(pool.get("totalValueLockedToken0")),
        "tvl_token1":      safe_float(pool.get("totalValueLockedToken1")),
        "token0_id":       token0.get("id"),
        "token0_symbol":   token0.get("symbol"),
        "token0_decimals": safe_int(token0.get("decimals")),
        "token1_id":       token1.get("id"),
        "token1_symbol":   token1.get("symbol"),
        "token1_decimals": safe_int(token1.get("decimals")),
    }

    return pd.DataFrame([row])


def flatten_ticks(rows: list, label: str, block_number=None) -> pd.DataFrame:
    out = []
    for r in rows:
        out.append({
            "snapshot_label":          label,
            "block_number":            block_number,
            "tick_id":                 r.get("id"),
            "pool_address":            r.get("poolAddress"),
            "tick_idx":                safe_int(r.get("tickIdx")),
            "liquidity_gross":         r.get("liquidityGross"),
            "liquidity_net":           r.get("liquidityNet"),
            "price0":                  safe_float(r.get("price0")),
            "price1":                  safe_float(r.get("price1")),
            "created_at_timestamp":    safe_int(r.get("createdAtTimestamp")),
            "created_at_block_number": safe_int(r.get("createdAtBlockNumber")),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("tick_idx")
    return df


def fetch_ticks(block_number=None) -> list:
    rows      = []
    last_tick = -887273
    query_n   = 0

    if block_number is None:
        query = TICKS_CURRENT_QUERY_NET_ONLY if LIQUIDITY_NET_ONLY else TICKS_CURRENT_QUERY_ALL
    else:
        query = TICKS_BLOCK_QUERY_NET_ONLY if LIQUIDITY_NET_ONLY else TICKS_BLOCK_QUERY_ALL

    while True:
        variables: dict = {
            "pool":      POOL,
            "first":     BATCH,
            "lastTick":  last_tick,
        }
        if block_number is not None:
            variables["blockNumber"] = int(block_number)

        data    = run_query(query, variables)
        batch   = data["ticks"]
        query_n += 1

        if not batch:
            break

        rows.extend(batch)
        last_tick = int(batch[-1]["tickIdx"])

        print(f"  ticks: {len(rows):>8,} rows | {query_n:>5} queries | last_tick={last_tick}")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    return rows


def fetch_pool_state(block_number=None) -> dict:
    if block_number is None:
        data = run_query(POOL_STATE_CURRENT_QUERY, {"pool": POOL})
    else:
        data = run_query(POOL_STATE_BLOCK_QUERY, {
            "pool":        POOL,
            "blockNumber": int(block_number),
        })
    return data["pool"]


def find_month_end_blocks(data_dir: str) -> pd.DataFrame:
    """
    Infer one block per month from already-downloaded CSVs.
    Uses the maximum block_number found in each monthly file.
    """
    regex = re.compile(
        r"(swaps|mints|burns|collects|flashes)_(\d{4})_(\d{2})\.csv$",
        re.IGNORECASE,
    )

    month_to_blocks: dict[str, int] = {}

    for file in glob.glob(os.path.join(data_dir, "*.csv")):
        name  = os.path.basename(file)
        match = regex.match(name)
        if not match:
            continue

        label = f"{match.group(2)}_{match.group(3)}"

        try:
            df = pd.read_csv(file, usecols=["block_number"])
        except Exception:
            continue

        if df.empty or "block_number" not in df.columns:
            continue

        blocks = pd.to_numeric(df["block_number"], errors="coerce").dropna()
        if blocks.empty:
            continue

        max_block = int(blocks.max())
        month_to_blocks[label] = max(month_to_blocks.get(label, 0), max_block)

    return pd.DataFrame([
        {"month_label": label, "block_number": block}
        for label, block in sorted(month_to_blocks.items())
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir     = project_root / "data_raw" / "DEX"
    snapshot_dir = data_dir / "tick_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Uniswap V3 USDC/WETH 0.05% — Tick Snapshots")
    print("=" * 70)

    # Current snapshot
    print("\nFetching current pool state...")
    pool_current    = fetch_pool_state()
    df_pool_current = flatten_pool_state(pool_current, "current", None)
    pool_current_file = snapshot_dir / "pool_state_current.csv"
    df_pool_current.to_csv(pool_current_file, index=False)
    print(f"Saved {pool_current_file}")

    print("\nFetching current ticks...")
    ticks_current    = fetch_ticks()
    df_ticks_current = flatten_ticks(ticks_current, "current", None)
    ticks_current_file = snapshot_dir / "ticks_current.csv"
    df_ticks_current.to_csv(ticks_current_file, index=False)
    print(f"Saved {ticks_current_file} ({len(df_ticks_current):,} rows)")

    # Month-end inferred blocks
    print("\nScanning existing monthly event files for month-end blocks...")
    df_blocks   = find_month_end_blocks(str(data_dir))
    blocks_file = snapshot_dir / "month_end_blocks_from_existing_files.csv"
    df_blocks.to_csv(blocks_file, index=False)
    print(f"Saved {blocks_file} ({len(df_blocks):,} months)")

    if df_blocks.empty:
        print(f"No monthly event files found in {data_dir}.")
        return

    # Historical tick snapshots
    for _, row in df_blocks.iterrows():
        month_label  = row["month_label"]
        block_number = int(row["block_number"])

        print("\n" + "=" * 70)
        print(f"Snapshot {month_label} at block {block_number}")
        print("=" * 70)

        pool_state = fetch_pool_state(block_number)
        df_pool    = flatten_pool_state(pool_state, month_label, block_number)
        pool_file  = snapshot_dir / f"pool_state_{month_label}_block_{block_number}.csv"
        df_pool.to_csv(pool_file, index=False)
        print(f"Saved {pool_file}")

        ticks    = fetch_ticks(block_number)
        df_ticks = flatten_ticks(ticks, month_label, block_number)
        ticks_file = snapshot_dir / f"ticks_{month_label}_block_{block_number}.csv"
        df_ticks.to_csv(ticks_file, index=False)
        print(f"Saved {ticks_file} ({len(df_ticks):,} rows)")

    print("\nDONE")


if __name__ == "__main__":
    main()
