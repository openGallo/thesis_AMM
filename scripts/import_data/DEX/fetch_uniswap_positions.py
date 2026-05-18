"""
Uniswap V3 USDC/WETH 0.05% — Position Downloader
=================================================

Fetches current/lifetime position-level data from The Graph.

Output:
    C:\\Courses\\thesis_AMM\\data_raw\\DEX\\positions_current.csv

Important:
    This is not a full historical position panel.
    It gives the position entity state currently indexed by the subgraph.

For historical LP activity, use:
    mints_YYYY_MM.csv
    burns_YYYY_MM.csv
    collects_YYYY_MM.csv

Setup:
    C:\\Interpreters\\python.exe -m pip install requests pandas
    setx THEGRAPH_API_KEY "your_new_api_key_here"
    Restart terminal.
    C:\\Interpreters\\python.exe C:\\Courses\\thesis_AMM\\scripts\\import_data\\DEX\\fetch_uniswap_positions.py
"""

import os
import time
import requests
import pandas as pd
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

BATCH       = 1000
SLEEP       = 0.35
MAX_RETRIES = 5
RETRY_WAIT  = 10


# ─────────────────────────────────────────────────────────────────────────────
# GraphQL query
# ─────────────────────────────────────────────────────────────────────────────

POSITIONS_QUERY = """
query($pool: String!, $first: Int!, $lastId: String!) {
  positions(
    first: $first
    orderBy: id
    orderDirection: asc
    where: {
      pool: $pool
      id_gt: $lastId
    }
  ) {
    id
    owner
    liquidity

    depositedToken0
    depositedToken1
    withdrawnToken0
    withdrawnToken1

    collectedFeesToken0
    collectedFeesToken1
    collectedFeesUSD

    feeGrowthInside0LastX128
    feeGrowthInside1LastX128

    pool {
      id
      feeTier
    }

    token0 {
      id
      symbol
      decimals
    }

    token1 {
      id
      symbol
      decimals
    }

    tickLower {
      tickIdx
      price0
      price1
    }

    tickUpper {
      tickIdx
      price0
      price1
    }

    transaction {
      id
      blockNumber
      timestamp
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
                timeout=120,
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


def fetch_positions():
    rows = []
    last_id = ""
    query_n = 0

    while True:
        variables = {
            "pool": POOL,
            "first": BATCH,
            "lastId": last_id,
        }

        data = run_query(POSITIONS_QUERY, variables)
        batch = data["positions"]
        query_n += 1

        if not batch:
            break

        rows.extend(batch)
        last_id = batch[-1]["id"]

        print(f"  positions: {len(rows):>8,} rows | {query_n:>5} queries | last_id={last_id}")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    return rows


def flatten_positions(rows):
    out = []

    for r in rows:
        pool = r.get("pool") or {}
        token0 = r.get("token0") or {}
        token1 = r.get("token1") or {}
        tick_lower = r.get("tickLower") or {}
        tick_upper = r.get("tickUpper") or {}
        tx = r.get("transaction") or {}

        out.append({
            "position_id": r.get("id"),
            "owner": r.get("owner"),

            "pool_id": pool.get("id"),
            "pool_fee_tier": safe_int(pool.get("feeTier")),

            "liquidity": r.get("liquidity"),

            "deposited_token0": safe_float(r.get("depositedToken0")),
            "deposited_token1": safe_float(r.get("depositedToken1")),
            "withdrawn_token0": safe_float(r.get("withdrawnToken0")),
            "withdrawn_token1": safe_float(r.get("withdrawnToken1")),

            "collected_fees_token0": safe_float(r.get("collectedFeesToken0")),
            "collected_fees_token1": safe_float(r.get("collectedFeesToken1")),
            "collected_fees_usd": safe_float(r.get("collectedFeesUSD")),

            "fee_growth_inside0_last_x128": r.get("feeGrowthInside0LastX128"),
            "fee_growth_inside1_last_x128": r.get("feeGrowthInside1LastX128"),

            "tick_lower": safe_int(tick_lower.get("tickIdx")),
            "tick_lower_price0": safe_float(tick_lower.get("price0")),
            "tick_lower_price1": safe_float(tick_lower.get("price1")),

            "tick_upper": safe_int(tick_upper.get("tickIdx")),
            "tick_upper_price0": safe_float(tick_upper.get("price0")),
            "tick_upper_price1": safe_float(tick_upper.get("price1")),

            "token0_id": token0.get("id"),
            "token0_symbol": token0.get("symbol"),
            "token0_decimals": safe_int(token0.get("decimals")),

            "token1_id": token1.get("id"),
            "token1_symbol": token1.get("symbol"),
            "token1_decimals": safe_int(token1.get("decimals")),

            "creation_tx_hash": tx.get("id"),
            "creation_block_number": safe_int(tx.get("blockNumber")),
            "creation_timestamp": (
                pd.to_datetime(safe_int(tx.get("timestamp")), unit="s", utc=True)
                if tx.get("timestamp") is not None
                else None
            ),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("position_id")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    project_root = Path(r"C:\Courses\thesis_AMM")
    data_dir = project_root / "data_raw" / "DEX"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Uniswap V3 USDC/WETH 0.05% — Positions")
    print("=" * 70)

    rows = fetch_positions()
    df = flatten_positions(rows)

    output_file = os.path.join(data_dir, "positions_current.csv")
    df.to_csv(output_file, index=False)

    print(f"\nSaved {output_file} ({len(df):,} rows)")
    print("DONE")


if __name__ == "__main__":
    main()