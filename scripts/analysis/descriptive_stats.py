"""
Descriptive statistics for all key variables.

Outputs:
    output/tables/desc_pool.csv/.tex         DEX pool hourly stats
    output/tables/desc_cex.csv/.tex          CEX price hourly stats
    output/tables/desc_swaps.csv/.tex        Swap-level stats
    output/tables/desc_lp_positions.csv/.tex LP position stats
    output/tables/desc_lvr.csv/.tex          LVR hourly stats
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis_utils import DATA_PROC, load, savetable

PCTS = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]


def describe(df: pd.DataFrame, cols: list[str], labels: dict[str, str] | None = None) -> pd.DataFrame:
    df = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        row = {
            "N":    len(s),
            "Mean": s.mean(),
            "Std":  s.std(),
            "Min":  s.min(),
            "P1":   s.quantile(0.01),
            "P25":  s.quantile(0.25),
            "P50":  s.quantile(0.50),
            "P75":  s.quantile(0.75),
            "P95":  s.quantile(0.95),
            "Max":  s.max(),
        }
        name = (labels or {}).get(col, col)
        rows.append({"Variable": name, **row})
    return pd.DataFrame(rows).set_index("Variable")


def main() -> None:
    print("=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)

    # ── DEX pool hourly ───────────────────────────────────────────
    dex = load("DEX/dex_pool_hourly.csv")
    if dex is not None:
        cols = ["eth_usdc_price", "volume_usd", "fees_usd", "tvl_usd",
                "tx_count", "fee_apr_ann", "vol_over_tvl", "log_return_1h"]
        labels = {
            "eth_usdc_price":  "ETH/USDC price (DEX)",
            "volume_usd":      "Volume USD (hourly)",
            "fees_usd":        "Fees USD (hourly)",
            "tvl_usd":         "TVL USD",
            "tx_count":        "Swap count (hourly)",
            "fee_apr_ann":     "Fee APR (annualized)",
            "vol_over_tvl":    "Volume / TVL",
            "log_return_1h":   "Log return (1h DEX)",
        }
        tab = describe(dex, cols, labels)
        savetable(tab, "desc_pool")
        print(f"  Pool hourly: {len(dex):,} rows")

    # ── CEX price hourly ─────────────────────────────────────────
    cex = load("CEX/cex_price_hourly.csv")
    if cex is not None:
        cols = ["eth_usdc_close", "log_return_1h", "realized_vol_1h_ann",
                "realized_vol_24h_ann", "realized_vol_7d_ann",
                "vol_base_1h_ethusdt", "n_trades_1h_ethusdt"]
        labels = {
            "eth_usdc_close":       "ETH/USDC close (CEX)",
            "log_return_1h":        "Log return (1h CEX)",
            "realized_vol_1h_ann":  "Realized vol 1h (ann.)",
            "realized_vol_24h_ann": "Realized vol 24h (ann.)",
            "realized_vol_7d_ann":  "Realized vol 7d (ann.)",
            "vol_base_1h_ethusdt":  "CEX volume ETH (1h)",
            "n_trades_1h_ethusdt":  "CEX trade count (1h)",
        }
        tab = describe(cex, cols, labels)
        savetable(tab, "desc_cex")
        print(f"  CEX hourly: {len(cex):,} rows")

    # ── Swaps ────────────────────────────────────────────────────
    swaps_path = DATA_PROC / "DEX" / "dex_swaps.csv"
    if swaps_path.exists():
        print("  Loading swaps (may be large)...")
        swaps = pd.read_csv(swaps_path, low_memory=False,
                            usecols=lambda c: c in [
                                "amount_usd", "gas_cost_usd", "gas_price_wei",
                                "log_price_change"])
        cols = ["amount_usd", "gas_cost_usd", "gas_price_wei", "log_price_change"]
        labels = {
            "amount_usd":       "Swap size USD",
            "gas_cost_usd":     "Gas cost USD",
            "gas_price_wei":    "Gas price (Wei)",
            "log_price_change": "Log price change per swap",
        }
        swaps["gas_price_wei"] = pd.to_numeric(swaps.get("gas_price_wei"), errors="coerce") / 1e9
        labels["gas_price_wei"] = "Gas price (Gwei)"
        tab = describe(swaps, cols, labels)
        savetable(tab, "desc_swaps")
        print(f"  Swaps: {len(swaps):,} rows")
    else:
        print("  [SKIP] dex_swaps.csv not found")

    # ── LP positions ─────────────────────────────────────────────
    lp = load("DEX/dex_lp_positions.csv", index_col=None)
    if lp is not None:
        cols = ["total_minted_usd", "total_burned_usd", "total_collected_usd",
                "net_pnl_usd", "fee_income_usd", "range_width_pct", "duration_days"]
        labels = {
            "total_minted_usd":    "Total minted USD",
            "total_burned_usd":    "Total burned USD",
            "total_collected_usd": "Total collected USD",
            "net_pnl_usd":        "Net P&L USD",
            "fee_income_usd":     "Fee income USD",
            "range_width_pct":    "Range width (%)",
            "duration_days":      "Position duration (days)",
        }
        tab = describe(lp, cols, labels)
        savetable(tab, "desc_lp_positions")

        if "position_type" in lp.columns:
            pnl_by_type = (
                lp.groupby("position_type", observed=True)["net_pnl_usd"]
                .agg(["count", "mean", "median", "std"])
                .rename(columns={"count": "N", "mean": "Mean P&L",
                                 "median": "Median P&L", "std": "Std P&L"})
            )
            savetable(pnl_by_type, "desc_lp_by_type")
        print(f"  LP positions: {len(lp):,}")

    # ── LVR ──────────────────────────────────────────────────────
    lvr = load("DEX/dex_lvr_hourly.csv")
    if lvr is not None:
        cols = ["lvr_usd_tvl_approx", "lvr_rate_ann", "lvr_to_fee_ratio"]
        labels = {
            "lvr_usd_tvl_approx": "LVR USD (hourly, TVL approx.)",
            "lvr_rate_ann":       "LVR rate (annualized, % of TVL)",
            "lvr_to_fee_ratio":   "LVR / fee income ratio",
        }
        tab = describe(lvr, cols, labels)
        savetable(tab, "desc_lvr")
        print(f"  LVR hourly: {len(lvr):,} rows")

    # ── Merged: basis ────────────────────────────────────────────
    merged = load("merged/merged_hourly.csv")
    if merged is not None:
        cols = ["dex_cex_basis_bps", "dex_cex_vol_ratio"]
        labels = {
            "dex_cex_basis_bps":  "DEX-CEX basis (bps)",
            "dex_cex_vol_ratio":  "DEX/CEX volume ratio",
        }
        tab = describe(merged, cols, labels)
        savetable(tab, "desc_basis")
        if "arbitrage_flag" in merged.columns:
            arb_pct = merged["arbitrage_flag"].mean() * 100
            print(f"  Arbitrage flag: {arb_pct:.1f}% of hours")

    print("\nDONE")


if __name__ == "__main__":
    main()
