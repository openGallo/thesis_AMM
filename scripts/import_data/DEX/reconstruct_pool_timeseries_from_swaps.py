"""
Reconstruct pool_hour_data.csv and pool_day_data.csv from the local swap CSVs.

This is an alternative to fetch_uniswap_pool_timeseries.py that works purely
from already-downloaded data — no TheGraph API calls required.

For each calendar hour from pool inception to study end-date:
  - OHLC   : open/high/low/close of pool_token0_price across swaps in the hour
  - volume  : sum of amount_usd for swaps in the hour
  - fees    : sum of fee_usd  (same as volume × 0.0005)
  - tx_count: number of swaps
  - tvl_usd : pool_tvl_usd at the LAST swap in the hour
  - liquidity, sqrt_price, tick : pool state at last swap

Hours with zero swaps are filled:
  - volume_usd = fees_usd = tx_count = 0
  - price + state columns forward-filled from previous hour

Daily data: resample hourly data to calendar day (UTC).

Usage:
    py scripts/import_data/DEX/reconstruct_pool_timeseries_from_swaps.py

Outputs:
    data_raw/DEX/pool_hour_data.csv
    data_raw/DEX/pool_day_data.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data_raw" / "DEX"
SWAP_DIR     = DATA_RAW
OUT_HOUR     = DATA_RAW / "pool_hour_data.csv"
OUT_DAY      = DATA_RAW / "pool_day_data.csv"

START_TS = 1620172800   # 2021-05-05 00:00:00 UTC  (pool inception)
END_TS   = 1777593599   # 2026-04-30 23:59:59 UTC  (study end)


# ── Load all swap CSVs ────────────────────────────────────────────────────────

SWAP_COLS = [
    "timestamp",
    "amount_usd",
    "fee_usd",
    "sqrt_price_x96",        # per-swap on-chain price (accurate)
    "pool_token0_price",     # subgraph derived (stale/unreliable — kept for reference)
    "pool_token1_price",
    "pool_tvl_usd",
    "pool_liquidity",
    "pool_sqrt_price",       # pool state sqrt price (same derivation as sqrt_price_x96)
    "pool_tick",
]

# Price from sqrtPriceX96:  USDC per WETH = 10^12 / (sqrtPriceX96 / 2^96)^2
# Uses 10^12 = 10^(decimals_WETH - decimals_USDC) = 10^(18-6)
_X96     = 2 ** 96
_DEC_ADJ = 10 ** 12


def price_from_sqrt_x96(series: pd.Series) -> pd.Series:
    """Convert sqrtPriceX96 -> USDC per WETH (ETH price in USDC)."""
    ratio = pd.to_numeric(series, errors="coerce").astype(float) / _X96
    with np.errstate(divide="ignore", invalid="ignore"):
        return _DEC_ADJ / (ratio ** 2)


def load_all_swaps() -> pd.DataFrame:
    files = sorted(SWAP_DIR.glob("swaps_*.csv"))
    if not files:
        print("[ERROR] No swap CSVs found in", SWAP_DIR, file=sys.stderr)
        sys.exit(1)

    chunks = []
    for f in files:
        print(f"  Loading {f.name}...", end="\r", flush=True)
        # Only read the columns we need to keep memory manageable
        df = pd.read_csv(f, usecols=SWAP_COLS, low_memory=False)
        chunks.append(df)
    print(f"\n  Loaded {len(files)} swap files -> {sum(len(c) for c in chunks):,} rows")

    swaps = pd.concat(chunks, ignore_index=True)

    # Parse timestamps
    swaps["timestamp"] = pd.to_datetime(swaps["timestamp"], utc=True)

    # Convert numeric columns (some may be strings)
    for col in SWAP_COLS[1:]:
        swaps[col] = pd.to_numeric(swaps[col], errors="coerce")

    # Drop rows outside study window
    ts = (swaps["timestamp"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().astype("int64")
    swaps = swaps[(ts >= START_TS) & (ts <= END_TS)].copy()

    swaps = swaps.sort_values("timestamp").reset_index(drop=True)

    # Compute accurate ETH price from on-chain sqrtPriceX96 (preferred over
    # pool_token0_price which is a stale subgraph-cached value).
    swaps["eth_price_x96"] = price_from_sqrt_x96(swaps["sqrt_price_x96"])

    return swaps


# ── Aggregate to hourly ───────────────────────────────────────────────────────

def build_hourly(swaps: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating to hourly...", flush=True)

    swaps["hour"] = swaps["timestamp"].dt.floor("h")

    grp = swaps.groupby("hour")

    # Price OHLC from sqrtPriceX96-derived price (accurate on-chain price).
    # pool_token0_price is a stale subgraph cached value — do NOT use for OHLC.
    ohlc = grp["eth_price_x96"].agg(
        open="first", high="max", low="min", close="last"
    )

    # Volume and fees
    vol_fee = grp.agg(
        volume_usd=("amount_usd",       "sum"),
        fees_usd  =("fee_usd",          "sum"),
        tx_count  =("amount_usd",       "count"),
    )

    # Last pool state (TVL etc. as-of end-of-hour)
    state = grp.agg(
        tvl_usd      =("pool_tvl_usd",    "last"),
        liquidity    =("pool_liquidity",  "last"),
        sqrt_price   =("pool_sqrt_price", "last"),
        tick         =("pool_tick",       "last"),
        token0_price =("eth_price_x96",   "last"),   # x96-derived price (accurate)
        token1_price =("pool_token1_price","last"),  # kept for reference
    )

    hour_df = pd.concat([ohlc, vol_fee, state], axis=1)

    # ── Fill full hourly index (include zero-swap hours) ──────────────────────
    full_idx = pd.date_range(
        start=pd.to_datetime(START_TS, unit="s", utc=True).floor("h"),
        end  =pd.to_datetime(END_TS,   unit="s", utc=True).floor("h"),
        freq ="h",
    )
    hour_df = hour_df.reindex(full_idx)

    # Volume / fees / tx_count: 0 for no-swap hours
    hour_df[["volume_usd", "fees_usd", "tx_count"]] = (
        hour_df[["volume_usd", "fees_usd", "tx_count"]].fillna(0)
    )

    # Price + state: forward-fill from last observed swap
    price_state_cols = [
        "open", "high", "low", "close",
        "tvl_usd", "liquidity", "sqrt_price", "tick",
        "token0_price", "token1_price",
    ]
    hour_df[price_state_cols] = (
        hour_df[price_state_cols].ffill()
    )

    # For no-swap hours the OHLC should be a flat candle (open=close=prev_close)
    # ffill already set close; set open=high=low=close for zero-volume hours
    no_swap = hour_df["tx_count"] == 0
    hour_df.loc[no_swap, "open"]  = hour_df.loc[no_swap, "close"]
    hour_df.loc[no_swap, "high"]  = hour_df.loc[no_swap, "close"]
    hour_df.loc[no_swap, "low"]   = hour_df.loc[no_swap, "close"]

    hour_df.index.name = "timestamp_utc"
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    hour_df["period_start_unix"] = (
        (hour_df.index - _epoch).total_seconds().astype("int64")
    )

    # Build id column matching TheGraph convention: {pool}-{hour_number}
    POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    hour_df["id"] = POOL + "-" + (hour_df["period_start_unix"] // 3600).astype(str)

    return hour_df.reset_index()


# ── Aggregate to daily ────────────────────────────────────────────────────────

def build_daily(hour_df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating to daily...", flush=True)
    h = hour_df.set_index("timestamp_utc").sort_index()
    h.index = pd.to_datetime(h.index, utc=True)

    day_grp = h.groupby(h.index.floor("D"))

    # Aggregate OHLC properly from hourly OHLC columns (first open / max high / min low / last close).
    day_ohlc2 = day_grp.agg(
        open        = ("open",        "first"),
        high        = ("high",        "max"),
        low         = ("low",         "min"),
        close       = ("close",       "last"),
        volume_usd  = ("volume_usd",  "sum"),
        fees_usd    = ("fees_usd",    "sum"),
        tx_count    = ("tx_count",    "sum"),
        tvl_usd     = ("tvl_usd",     "last"),
        liquidity   = ("liquidity",   "last"),
        sqrt_price  = ("sqrt_price",  "last"),
        tick        = ("tick",        "last"),
        token0_price= ("token0_price","last"),
        token1_price= ("token1_price","last"),
    )

    day_df = day_ohlc2.copy()
    day_df.index.name = "timestamp_utc"
    _epoch = pd.Timestamp("1970-01-01", tz="UTC")
    day_df["date_unix"] = (day_df.index - _epoch).total_seconds().astype("int64")
    day_df["date"]      = day_df.index.date

    POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
    day_df["id"] = POOL + "-" + (day_df["date_unix"] // 86400).astype(str)

    return day_df.reset_index()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("Pool Time-Series Reconstruction from Swap Data")
    print("=" * 70)

    swaps = load_all_swaps()
    print(f"  Swap date range: {swaps['timestamp'].min()}  ->  {swaps['timestamp'].max()}")
    print(f"  Total swaps: {len(swaps):,}", flush=True)

    # Hourly
    hour_df = build_hourly(swaps)
    hour_df.to_csv(OUT_HOUR, index=False)
    print(f"\nSaved {OUT_HOUR}  ({len(hour_df):,} rows)", flush=True)
    print(f"  Period: {hour_df['timestamp_utc'].min()} -> {hour_df['timestamp_utc'].max()}")
    print(f"  Zero-swap hours: {(hour_df['tx_count'] == 0).sum():,}")

    # Daily
    day_df = build_daily(hour_df)
    day_df.to_csv(OUT_DAY, index=False)
    print(f"\nSaved {OUT_DAY}  ({len(day_df):,} rows)", flush=True)

    print("\nDONE")


if __name__ == "__main__":
    main()
