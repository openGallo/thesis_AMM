"""
Merge DEX pool hourly data with CEX price panel into one analytical frame.

Prerequisite: run process_dex_pool_hourly.py and process_cex_price.py first.

Inputs:
    data_processed/DEX/dex_pool_hourly.csv
    data_processed/CEX/cex_price_hourly.csv

Key variables derived:
    dex_cex_basis_bps  - (dex_price - cex_price) / cex_price * 10_000
                         Positive = DEX trading at a premium to CEX.
    arbitrage_flag     - |basis_bps| > 5  (exceeds the 0.05% pool fee tier;
                         sufficient to cover a round-trip arbitrage trade)
    dex_cex_vol_ratio  - DEX volume (USD) / CEX volume (USD-equivalent)
                         proxy for relative market share

Output:
    data_processed/merged/merged_hourly.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROC    = PROJECT_ROOT / "data_processed"
DATA_OUT     = DATA_PROC / "merged"

# Columns to carry forward from each dataset (prefix applied after selection)
DEX_COLS = [
    "eth_usdc_open", "eth_usdc_high", "eth_usdc_low", "eth_usdc_close",
    "eth_usdc_price", "volume_usd", "fees_usd", "tvl_usd", "tx_count",
    "fee_rate", "fee_apr_ann", "vol_over_tvl", "log_return_1h",
    "liquidity", "tick",
]
CEX_COLS = [
    "eth_usdc_open", "eth_usdc_high", "eth_usdc_low", "eth_usdc_close",
    "log_return_1h",
    "realized_vol_1h_ann", "realized_vol_24h_ann", "realized_vol_7d_ann",
    "vol_base_1h_ethusdt", "n_trades_1h_ethusdt",
]


def load_csv(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path}\n-> Run {name} first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index.name = "timestamp_utc"
    return df.sort_index()


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Merged DEX + CEX Hourly Panel")
    print("=" * 60)

    dex = load_csv(DATA_PROC / "DEX" / "dex_pool_hourly.csv",   "process_dex_pool_hourly.py")
    cex = load_csv(DATA_PROC / "CEX" / "cex_price_hourly.csv",  "process_cex_price.py")

    print(f"DEX: {len(dex):,} hourly rows  ({dex.index.min()} -> {dex.index.max()})")
    print(f"CEX: {len(cex):,} hourly rows  ({cex.index.min()} -> {cex.index.max()})")

    # Select columns that actually exist
    dex_sel = dex[[c for c in DEX_COLS if c in dex.columns]]
    cex_sel = cex[[c for c in CEX_COLS if c in cex.columns]]

    # Prefix to avoid collisions after join
    dex_sel = dex_sel.add_prefix("dex_")
    cex_sel = cex_sel.add_prefix("cex_")

    merged = dex_sel.join(cex_sel, how="outer")
    merged.index.name = "timestamp_utc"

    # ── DEX-CEX price basis ───────────────────────────────────────────────────
    dex_price = merged.get("dex_eth_usdc_close")
    cex_price = merged.get("cex_eth_usdc_close")

    if dex_price is not None and cex_price is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            merged["dex_cex_basis_bps"] = (
                (dex_price - cex_price) / cex_price * 10_000
            )
        merged["arbitrage_flag"] = merged["dex_cex_basis_bps"].abs() > 5

    # ── DEX / CEX volume ratio ────────────────────────────────────────────────
    vol_dex = merged.get("dex_volume_usd")
    vol_cex = merged.get("cex_vol_base_1h_ethusdt")

    if vol_dex is not None and vol_cex is not None:
        # Convert CEX ETH volume -> USD using CEX close price for comparability
        cex_usd = vol_cex * cex_price if cex_price is not None else vol_cex
        with np.errstate(divide="ignore", invalid="ignore"):
            merged["dex_cex_vol_ratio"] = np.where(
                cex_usd > 0, vol_dex / cex_usd, np.nan
            )

    out = DATA_OUT / "merged_hourly.csv"
    merged.to_csv(out)

    overlap_mask = merged.get("dex_eth_usdc_close")
    if overlap_mask is not None:
        n_overlap = merged[["dex_eth_usdc_close", "cex_eth_usdc_close"]].dropna().shape[0]
    else:
        n_overlap = 0

    print(f"\nSaved {out}")
    print(f"  Total rows : {len(merged):,}")
    print(f"  DEX+CEX overlap: {n_overlap:,} hours")

    if "dex_cex_basis_bps" in merged.columns:
        b = merged["dex_cex_basis_bps"].dropna()
        print(f"  Basis bps  : mean={b.mean():.2f}  median={b.median():.2f}  "
              f"p5={b.quantile(0.05):.2f}  p95={b.quantile(0.95):.2f}")

    if "arbitrage_flag" in merged.columns:
        arb_pct = merged["arbitrage_flag"].mean() * 100
        print(f"  Arb flags  : {arb_pct:.1f}% of hours have |basis| > 5 bps")

    print("\nDONE")


if __name__ == "__main__":
    main()
