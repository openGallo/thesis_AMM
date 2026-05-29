"""
Compute Loss-Versus-Rebalancing (LVR) from hourly pool data and CEX realized volatility.

Theory (Milionis, Moallemi, Roughgarden, Zhang 2024):
    dLVR_t = (1/2) σ_t² p_t² |x'(p_t)| dt

For v3 with active liquidity L at current price p (active-range approximation):
    x(p) = L/√p  ->  |x'(p)| = L/(2p^{3/2})
    dLVR_t = (σ_t² L_t √p_t / 4) dt

TVL approximation (balanced pool: x_t · p_t ≈ TVL_t / 2):
    dLVR_t ≈ (TVL_t / 8) σ_t² dt

Annualized LVR rate:
    LVR_rate_ann = σ_ann² / 8   (fraction of TVL per year)

Note: σ from cex_price_hourly is end-of-hour rolling realized vol (annualized).
      LVR is expressed in USD and as a fraction of TVL.

Inputs:
    data_processed/DEX/dex_pool_hourly.csv
    data_processed/CEX/cex_price_hourly.csv     (for σ_t)

Output:
    data_processed/DEX/dex_lvr_hourly.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROC    = PROJECT_ROOT / "data_processed"
DATA_OUT     = DATA_PROC / "DEX"

_HOURS_PER_YEAR = 365.25 * 24


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DEX LVR (Loss-Versus-Rebalancing) Computation")
    print("=" * 60)

    dex_path = DATA_PROC / "DEX" / "dex_pool_hourly.csv"
    cex_path = DATA_PROC / "CEX" / "cex_price_hourly.csv"

    if not dex_path.exists():
        raise FileNotFoundError(f"{dex_path}\n-> Run process_dex_pool_hourly.py first.")
    if not cex_path.exists():
        raise FileNotFoundError(f"{cex_path}\n-> Run process_cex_price.py first.")

    dex = pd.read_csv(dex_path, index_col=0, parse_dates=True)
    dex.index = pd.to_datetime(dex.index, utc=True)

    cex = pd.read_csv(cex_path, index_col=0, parse_dates=True)
    cex.index = pd.to_datetime(cex.index, utc=True)

    # Align on common hourly UTC index
    df = dex[["tvl_usd", "eth_usdc_price", "liquidity"]].join(
        cex[["realized_vol_24h_ann"]], how="inner"
    )
    df = df.dropna(subset=["tvl_usd", "realized_vol_24h_ann"])
    df.index.name = "timestamp_utc"

    # σ² (annualized variance)
    sigma2 = df["realized_vol_24h_ann"] ** 2

    # dt in annual units: one hour = 1/8760 year
    dt = 1.0 / _HOURS_PER_YEAR

    # ── Method 1: TVL approximation ───────────────────────────────────────────
    # dLVR = (TVL/8) * σ² * dt
    df["lvr_usd_tvl_approx"] = (df["tvl_usd"] / 8.0) * sigma2 * dt
    df["lvr_rate_ann"]        = sigma2 / 8.0          # fraction of TVL per year

    # ── Method 2: Active-liquidity formula (v3) ───────────────────────────────
    # dLVR = (σ² * L * √p / 4) * dt
    # Units: L is raw pool liquidity (dimensionless in subgraph), p is USDC/WETH.
    # L * √p has mixed units; this produces a relative LVR that needs TVL scaling.
    # Use only when pool liquidity is available.
    if "liquidity" in df.columns:
        p = df["eth_usdc_price"]
        L = pd.to_numeric(df["liquidity"], errors="coerce")
        # Normalise: L/√p gives WETH-equivalent active depth; multiply by p gives USD
        active_depth_usd = L / np.sqrt(p) * p   # = L * √p  (in raw pool units * price)
        # Scale to TVL fraction (L is in raw units, not directly USD)
        tvl_nonzero = df["tvl_usd"].replace(0, np.nan)
        # Relative LVR rate via active liquidity (normalised to TVL for interpretability)
        df["lvr_rate_ann_v3_approx"] = (sigma2 / 4.0) * (active_depth_usd / tvl_nonzero)

    # ── Cumulative LVR ────────────────────────────────────────────────────────
    df["lvr_usd_cumulative"] = df["lvr_usd_tvl_approx"].cumsum()

    # ── LVR as a fraction of hourly fee income ────────────────────────────────
    if "fees_usd" in dex.columns:
        fees = dex["fees_usd"].reindex(df.index)
        df["fees_usd"] = fees
        df["lvr_to_fee_ratio"] = np.where(
            fees > 0, df["lvr_usd_tvl_approx"] / fees, np.nan
        )

    out = DATA_OUT / "dex_lvr_hourly.csv"
    df.to_csv(out)
    print(f"Saved {out}  ({len(df):,} rows)")

    # ── Summary statistics ────────────────────────────────────────────────────
    print(f"\n  LVR rate (ann, mean):   {df['lvr_rate_ann'].mean():.4%} of TVL per year")
    print(f"  LVR USD (hourly, mean): ${df['lvr_usd_tvl_approx'].mean():,.0f}/hour")
    print(f"  LVR USD (total):        ${df['lvr_usd_cumulative'].iloc[-1]:,.0f}")
    if "lvr_to_fee_ratio" in df.columns:
        r = df["lvr_to_fee_ratio"].dropna()
        print(f"  LVR / fee income:       mean={r.mean():.2f}  median={r.median():.2f}")
    print("\nDONE")


if __name__ == "__main__":
    main()
