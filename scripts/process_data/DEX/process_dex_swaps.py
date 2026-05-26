"""
Build the analytical DEX swap panel from raw monthly CSVs.

Inputs:
    data_raw/DEX/swaps_YYYY_MM.csv  (all months, produced by fetch_all_monthly.py)

Key variables derived:
    eth_usdc_price     — pool_token0_price (USDC per WETH; direct from subgraph)
    eth_usdc_price_x96 — independent price from sqrtPriceX96 (cross-check)
                         formula: 10^12 / (sqrtPriceX96 / 2^96)^2
    direction          — "buy_eth" | "sell_eth"
    gas_cost_usd       — gas_cost_eth * eth_usdc_price
    log_price_change   — log(price_n / price_{n-1}), sorted chronologically
    trade_size_bucket  — USD notional bin: <1k / 1k-10k / 10k-100k / >100k

Output:
    data_processed/DEX/dex_swaps.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data_raw" / "DEX"
DATA_OUT     = PROJECT_ROOT / "data_processed" / "DEX"

_X96        = 2 ** 96
_DEC_ADJ    = 10 ** 12   # 10^(decimals_WETH - decimals_USDC) = 10^(18-6)


def price_from_sqrt(sqrt_price_x96: pd.Series) -> pd.Series:
    """USDC per WETH from raw sqrtPriceX96 integer (cross-check column)."""
    ratio = pd.to_numeric(sqrt_price_x96, errors="coerce").astype(float) / _X96
    with np.errstate(divide="ignore", invalid="ignore"):
        return _DEC_ADJ / (ratio ** 2)


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DEX Swap Panel Processing")
    print("=" * 60)

    files = sorted(DATA_RAW.glob("swaps_*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No swaps_*.csv found in {DATA_RAW}\n"
            "Run fetch_all_monthly.py first."
        )
    print(f"Loading {len(files)} monthly swap files...")

    frames = []
    for f in files:
        print(f"  {f.name}")
        frames.append(pd.read_csv(f, low_memory=False))
    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal: {len(df):,} swaps")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Primary DEX price
    df["eth_usdc_price"] = pd.to_numeric(df["pool_token0_price"], errors="coerce")

    # Independent price cross-check from sqrtPriceX96
    df["eth_usdc_price_x96"] = price_from_sqrt(df["sqrt_price_x96"])

    # Direction label
    df["direction"] = df["buy_eth"].map({True: "buy_eth", False: "sell_eth", 1: "buy_eth", 0: "sell_eth"})

    # Gas cost in USD
    df["gas_cost_usd"] = (
        pd.to_numeric(df["gas_cost_eth"], errors="coerce") * df["eth_usdc_price"]
    )

    # Log price change between consecutive swaps
    df["log_price_change"] = np.log(df["eth_usdc_price"] / df["eth_usdc_price"].shift(1))

    # Trade size bucket
    df["trade_size_bucket"] = pd.cut(
        pd.to_numeric(df["amount_usd"], errors="coerce").abs(),
        bins=[0, 1_000, 10_000, 100_000, float("inf")],
        labels=["<1k", "1k-10k", "10k-100k", ">100k"],
        right=True,
    )

    out_cols = [
        "swap_id", "tx_hash", "block_number", "log_index", "timestamp",
        "origin", "sender", "recipient",
        "eth_usdc_price", "eth_usdc_price_x96",
        "amount0_usdc", "amount1_eth", "amount_usd", "fee_usd",
        "direction", "log_price_change",
        "gas_used", "gas_price_wei", "gas_cost_eth", "gas_cost_usd",
        "trade_size_bucket",
        "sqrt_price_x96", "tick_at_swap",
        "pool_liquidity", "pool_tick", "pool_tvl_usd",
        "pool_token0_price", "pool_token1_price",
        "pool_fee_tier", "pool_volume_usd", "pool_fees_usd", "pool_tx_count",
        "pool_tvl_token0_usdc", "pool_tvl_token1_eth",
        "token0_symbol", "token1_symbol",
    ]
    out_cols = [c for c in out_cols if c in df.columns]

    out = DATA_OUT / "dex_swaps.csv"
    df[out_cols].to_csv(out, index=False)
    print(f"\nSaved {out}")
    print(f"  {len(df):,} rows | {out.stat().st_size / 1e6:.1f} MB")
    print(f"  Period: {df['timestamp'].min()} → {df['timestamp'].max()}")

    # Direction breakdown
    if "direction" in df.columns:
        counts = df["direction"].value_counts()
        for k, v in counts.items():
            print(f"  {k}: {v:,} ({v / len(df) * 100:.1f}%)")

    print("\nDONE")


if __name__ == "__main__":
    main()
