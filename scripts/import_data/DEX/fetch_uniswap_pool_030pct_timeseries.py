"""
Uniswap V3 USDC/WETH 0.30% — Pool Time-Series Downloader
=========================================================

Fetches from The Graph:

    data_raw/multitier/fee_3000/pool_hour_data.csv
    data_raw/multitier/fee_3000/pool_day_data.csv
    data_raw/multitier/fee_3000/pool_metadata_current.csv
    data_raw/multitier/fee_3000/bundle_current.csv

Pool:      USDC/WETH 0.30%  (fee tier 3000)
Address:   0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8
Subgraph:  Uniswap v3 (same as 0.05% pool)

This script is a direct sibling of fetch_uniswap_pool_timeseries.py.
It reuses all API infrastructure from dex_utils.py unchanged.

Setup:
    setx THEGRAPH_API_KEY "your_key_here"   (Windows; restart terminal after)
    py -3 scripts/import_data/DEX/fetch_uniswap_pool_030pct_timeseries.py
"""

import time
from pathlib import Path

import pandas as pd

from dex_utils import START_TS, END_TS, BATCH, SLEEP, run_query, safe_int, safe_float

POOL     = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"   # USDC/WETH 0.30%
FEE_TIER = 3000
FEE_RATE = 0.003

_START_TS: int = int(START_TS)
_END_TS:   int = int(END_TS)


# ── GraphQL queries (identical to fetch_uniswap_pool_timeseries.py) ───────────

POOL_HOUR_QUERY = """
query($pool: String!, $first: Int!, $lastId: String!) {
  poolHourDatas(
    first: $first
    orderBy: id
    orderDirection: asc
    where: {
      pool: $pool
      id_gt: $lastId
    }
  ) {
    id
    periodStartUnix
    liquidity
    sqrtPrice
    token0Price
    token1Price
    tick
    tvlUSD
    volumeToken0
    volumeToken1
    volumeUSD
    feesUSD
    txCount
    open
    high
    low
    close
  }
}
"""

POOL_DAY_QUERY = """
query($pool: String!, $first: Int!, $lastId: String!) {
  poolDayDatas(
    first: $first
    orderBy: id
    orderDirection: asc
    where: {
      pool: $pool
      id_gt: $lastId
    }
  ) {
    id
    date
    liquidity
    sqrtPrice
    token0Price
    token1Price
    tick
    tvlUSD
    volumeToken0
    volumeToken1
    volumeUSD
    feesUSD
    txCount
    open
    high
    low
    close
  }
}
"""

POOL_METADATA_QUERY = """
query($pool: ID!) {
  pool(id: $pool) {
    id
    createdAtTimestamp
    createdAtBlockNumber
    feeTier
    liquidity
    sqrtPrice
    tick
    token0Price
    token1Price
    volumeUSD
    feesUSD
    txCount
    totalValueLockedUSD
    totalValueLockedToken0
    totalValueLockedToken1
    token0 { id symbol name decimals derivedETH }
    token1 { id symbol name decimals derivedETH }
  }
}
"""

BUNDLE_QUERY = """
{
  bundle(id: "1") {
    id
    ethPriceUSD
  }
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_paginated(query: str, entity_name: str) -> list:
    rows    = []
    last_id = ""
    query_n = 0

    while True:
        variables = {"pool": POOL, "first": BATCH, "lastId": last_id}
        data    = run_query(query, variables)
        batch   = data[entity_name]
        query_n += 1

        if not batch:
            break

        rows.extend(batch)
        last_id = batch[-1]["id"]
        print(f"  {entity_name}: {len(rows):>8,} rows | {query_n:>5} queries")

        if len(batch) < BATCH:
            break

        time.sleep(SLEEP)

    return rows


def flatten_pool_hour(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        ts = safe_int(r["periodStartUnix"])
        out.append({
            "id":                r["id"],
            "period_start_unix": ts,
            "datetime":          pd.to_datetime(ts, unit="s", utc=True),
            "liquidity":         r.get("liquidity"),
            "sqrt_price":        r.get("sqrtPrice"),
            "tick":              safe_int(r.get("tick")),
            "token0_price":      safe_float(r.get("token0Price")),
            "token1_price":      safe_float(r.get("token1Price")),
            "tvl_usd":           safe_float(r.get("tvlUSD")),
            "volume_token0":     safe_float(r.get("volumeToken0")),
            "volume_token1":     safe_float(r.get("volumeToken1")),
            "volume_usd":        safe_float(r.get("volumeUSD")),
            "fees_usd":          safe_float(r.get("feesUSD")),
            "tx_count":          safe_int(r.get("txCount")),
            "open":              safe_float(r.get("open")),
            "high":              safe_float(r.get("high")),
            "low":               safe_float(r.get("low")),
            "close":             safe_float(r.get("close")),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("period_start_unix")
        df = df[(df["period_start_unix"] >= _START_TS) & (df["period_start_unix"] <= _END_TS)]
    return df


def flatten_pool_day(rows: list) -> pd.DataFrame:
    out = []
    for r in rows:
        ts = safe_int(r["date"])
        out.append({
            "id":            r["id"],
            "date_unix":     ts,
            "date":          pd.to_datetime(ts, unit="s", utc=True).date(),
            "liquidity":     r.get("liquidity"),
            "sqrt_price":    r.get("sqrtPrice"),
            "tick":          safe_int(r.get("tick")),
            "token0_price":  safe_float(r.get("token0Price")),
            "token1_price":  safe_float(r.get("token1Price")),
            "tvl_usd":       safe_float(r.get("tvlUSD")),
            "volume_token0": safe_float(r.get("volumeToken0")),
            "volume_token1": safe_float(r.get("volumeToken1")),
            "volume_usd":    safe_float(r.get("volumeUSD")),
            "fees_usd":      safe_float(r.get("feesUSD")),
            "tx_count":      safe_int(r.get("txCount")),
            "open":          safe_float(r.get("open")),
            "high":          safe_float(r.get("high")),
            "low":           safe_float(r.get("low")),
            "close":         safe_float(r.get("close")),
        })
    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("date_unix")
        df = df[(df["date_unix"] >= _START_TS) & (df["date_unix"] <= _END_TS)]
    return df


def flatten_pool_metadata(pool: dict) -> pd.DataFrame:
    if not pool:
        return pd.DataFrame()
    token0 = pool.get("token0") or {}
    token1 = pool.get("token1") or {}
    row = {
        "pool_id":                  pool.get("id"),
        "created_at_timestamp":     safe_int(pool.get("createdAtTimestamp")),
        "created_at_datetime":      pd.to_datetime(
            safe_int(pool.get("createdAtTimestamp")), unit="s", utc=True
        ) if pool.get("createdAtTimestamp") is not None else None,
        "created_at_block_number":  safe_int(pool.get("createdAtBlockNumber")),
        "fee_tier":                 safe_int(pool.get("feeTier")),
        "liquidity":                pool.get("liquidity"),
        "sqrt_price":               pool.get("sqrtPrice"),
        "tick":                     safe_int(pool.get("tick")),
        "token0_price":             safe_float(pool.get("token0Price")),
        "token1_price":             safe_float(pool.get("token1Price")),
        "volume_usd":               safe_float(pool.get("volumeUSD")),
        "fees_usd":                 safe_float(pool.get("feesUSD")),
        "tx_count":                 safe_int(pool.get("txCount")),
        "tvl_usd":                  safe_float(pool.get("totalValueLockedUSD")),
        "tvl_token0":               safe_float(pool.get("totalValueLockedToken0")),
        "tvl_token1":               safe_float(pool.get("totalValueLockedToken1")),
        "token0_id":                token0.get("id"),
        "token0_symbol":            token0.get("symbol"),
        "token0_name":              token0.get("name"),
        "token0_decimals":          safe_int(token0.get("decimals")),
        "token0_derived_eth":       safe_float(token0.get("derivedETH")),
        "token1_id":                token1.get("id"),
        "token1_symbol":            token1.get("symbol"),
        "token1_name":              token1.get("name"),
        "token1_decimals":          safe_int(token1.get("decimals")),
        "token1_derived_eth":       safe_float(token1.get("derivedETH")),
    }
    return pd.DataFrame([row])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    data_dir     = project_root / "data_raw" / "multitier" / "fee_3000"
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Uniswap V3 USDC/WETH 0.30%  (fee tier {FEE_TIER})  —  Pool Time Series")
    print(f"Pool: {POOL}")
    print("=" * 70)

    # Pool metadata
    metadata  = run_query(POOL_METADATA_QUERY, {"pool": POOL})
    df_meta   = flatten_pool_metadata(metadata["pool"])
    meta_file = data_dir / "pool_metadata_current.csv"
    df_meta.to_csv(meta_file, index=False)
    print(f"Saved {meta_file}")

    # Current ETH/USD bundle
    bundle    = run_query(BUNDLE_QUERY)
    df_bundle = pd.DataFrame([{
        "id":                bundle["bundle"]["id"],
        "eth_price_usd":     safe_float(bundle["bundle"]["ethPriceUSD"]),
        "downloaded_at_utc": pd.Timestamp.now("UTC"),
    }])
    bundle_file = data_dir / "bundle_current.csv"
    df_bundle.to_csv(bundle_file, index=False)
    print(f"Saved {bundle_file}")

    # Hourly data
    print("\nFetching poolHourDatas...")
    hour_rows = fetch_paginated(POOL_HOUR_QUERY, "poolHourDatas")
    df_hour   = flatten_pool_hour(hour_rows)
    hour_file = data_dir / "pool_hour_data.csv"
    df_hour.to_csv(hour_file, index=False)
    print(f"Saved {hour_file} ({len(df_hour):,} rows)")

    # Daily data
    print("\nFetching poolDayDatas...")
    day_rows = fetch_paginated(POOL_DAY_QUERY, "poolDayDatas")
    df_day   = flatten_pool_day(day_rows)
    day_file = data_dir / "pool_day_data.csv"
    df_day.to_csv(day_file, index=False)
    print(f"Saved {day_file} ({len(df_day):,} rows)")

    print("\nDONE")


if __name__ == "__main__":
    main()
