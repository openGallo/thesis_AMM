"""
Loss-Versus-Rebalancing (LVR) analysis.

Figures:
    lvr_fee_comparison          Hourly LVR vs fee income time series
    lvr_to_fee_ratio            LVR/fee ratio distribution
    lvr_cumulative              Cumulative LVR vs cumulative fee income
    lvr_rate_series             Annualized LVR rate time series

Tables:
    lvr_summary                 LVR summary statistics
    lvr_by_regime               LVR and fee income by volatility regime
    lvr_fee_welfare             Annual LVR rate vs fee APR comparison
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analysis_utils import COLORS, load, savefig, savetable, vol_regime


def main() -> None:
    print("=" * 60)
    print("LVR Analysis")
    print("=" * 60)

    lvr = load("DEX/dex_lvr_hourly.csv")
    dex = load("DEX/dex_pool_hourly.csv")
    cex = load("CEX/cex_price_hourly.csv")

    if lvr is None:
        print("  dex_lvr_hourly.csv not found — run process_dex_lvr.py first")
        return

    lvr_usd   = pd.to_numeric(lvr.get("lvr_usd_tvl_approx"), errors="coerce")
    lvr_rate  = pd.to_numeric(lvr.get("lvr_rate_ann"),       errors="coerce")
    fees_usd  = pd.to_numeric(lvr.get("fees_usd"),           errors="coerce") if "fees_usd" in lvr.columns else None
    ratio_col = pd.to_numeric(lvr.get("lvr_to_fee_ratio"),   errors="coerce") if "lvr_to_fee_ratio" in lvr.columns else None

    print(f"  LVR hourly obs: {lvr_usd.dropna().__len__():,}")
    print(f"  Mean LVR rate (ann): {lvr_rate.mean():.4%} of TVL")
    if fees_usd is not None:
        print(f"  Mean hourly LVR USD: ${lvr_usd.mean():,.0f}")
        print(f"  Mean hourly fee USD: ${fees_usd.mean():,.0f}")

    # ── Figure 1: LVR vs fee income time series ───────────────────
    if fees_usd is not None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

        roll = 24 * 7  # 7-day rolling
        axes[0].plot(lvr.index, lvr_usd.rolling(roll).mean() / 1e3,
                     color=COLORS[1], lw=0.9, label="LVR (USD k, 7d MA)")
        axes[0].plot(lvr.index, fees_usd.rolling(roll).mean() / 1e3,
                     color=COLORS[2], lw=0.9, label="Fee income (USD k, 7d MA)")
        axes[0].set_ylabel("USD thousands")
        axes[0].legend()
        axes[0].set_title("Hourly LVR vs Fee Income (7-day moving average)")

        if ratio_col is not None:
            r_clipped = ratio_col.clip(0, ratio_col.quantile(0.99))
            axes[1].plot(lvr.index, r_clipped.rolling(roll).median(),
                         color=COLORS[0], lw=0.9)
            axes[1].axhline(1, color="black", lw=0.8, ls="--",
                            label="LVR = fee income")
            axes[1].set_ylabel("LVR / fee ratio")
            axes[1].legend()
        savefig("lvr_fee_comparison")

    # ── Figure 2: LVR/fee ratio distribution ─────────────────────
    if ratio_col is not None:
        r = ratio_col.dropna()
        r = r[(r > 0) & (r < r.quantile(0.99))]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(r, bins=80, color=COLORS[0], alpha=0.7, density=True)
        ax.axvline(1,          color=COLORS[1], lw=1.5, ls="--", label="LVR = fees")
        ax.axvline(r.median(), color=COLORS[2], lw=1.5, ls="-",
                   label=f"Median = {r.median():.2f}")
        ax.set_xlabel("LVR / fee income ratio")
        ax.set_ylabel("Density")
        ax.set_title("Distribution of LVR-to-Fee Ratio")
        ax.legend()
        savefig("lvr_to_fee_ratio")

    # ── Figure 3: Cumulative LVR vs fees ─────────────────────────
    if fees_usd is not None and "lvr_usd_cumulative" in lvr.columns:
        cum_lvr  = pd.to_numeric(lvr["lvr_usd_cumulative"], errors="coerce")
        cum_fees = fees_usd.cumsum()
        fig, ax  = plt.subplots(figsize=(9, 3.5))
        ax.plot(lvr.index, cum_lvr  / 1e6, color=COLORS[1], lw=1.2, label="Cumulative LVR (USD M)")
        ax.plot(lvr.index, cum_fees / 1e6, color=COLORS[2], lw=1.2, label="Cumulative fees (USD M)")
        ax.set_ylabel("USD millions")
        ax.set_title("Cumulative LVR vs Cumulative Fee Income")
        ax.legend()
        savefig("lvr_cumulative")

    # ── Figure 4: LVR rate annualized series ─────────────────────
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(lvr.index, lvr_rate * 100, color=COLORS[0], lw=0.5, alpha=0.5)
    ax.plot(lvr.index, lvr_rate.rolling(24 * 7).mean() * 100,
            color=COLORS[1], lw=1.2, label="7-day MA")
    ax.set_ylabel("LVR rate (% of TVL per year)")
    ax.set_title("Annualized LVR Rate (= $\\sigma^2/8$)")
    ax.legend()
    savefig("lvr_rate_series")

    # ── Table: summary ────────────────────────────────────────────
    rows = {
        "LVR rate ann. mean":       f"{lvr_rate.mean():.4%}",
        "LVR rate ann. median":     f"{lvr_rate.median():.4%}",
        "LVR USD hourly mean":      f"${lvr_usd.mean():,.0f}",
    }
    if fees_usd is not None:
        rows["Fee income hourly mean"] = f"${fees_usd.mean():,.0f}"
    if ratio_col is not None:
        r = ratio_col[(ratio_col > 0) & (ratio_col < ratio_col.quantile(0.99))].dropna()
        rows["LVR/fee ratio mean"]   = f"{r.mean():.3f}"
        rows["LVR/fee ratio median"] = f"{r.median():.3f}"
        rows["Pct hours LVR > fees"] = f"{(ratio_col.dropna() > 1).mean()*100:.1f}%"

    if dex is not None and "fee_apr_ann" in dex.columns:
        apr = pd.to_numeric(dex["fee_apr_ann"], errors="coerce").mean()
        rows["Fee APR (ann., mean)"] = f"{apr:.4%}"
        rows["LVR rate / fee APR"]   = f"{lvr_rate.mean() / apr:.3f}" if apr > 0 else "--"

    tab = pd.DataFrame.from_dict(rows, orient="index", columns=["Value"])
    savetable(tab, "lvr_summary")

    # ── Table: by vol regime ──────────────────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol    = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        regime = vol_regime(vol)
        df     = pd.DataFrame({
            "lvr_rate_ann": lvr_rate,
            "lvr_usd":      lvr_usd,
            "regime":       regime.reindex(lvr_rate.index),
        }).dropna(subset=["regime"])
        if fees_usd is not None:
            df["fees_usd"] = fees_usd.reindex(lvr_rate.index)

        agg = {"lvr_rate_ann": ["mean", "median"],
               "lvr_usd":      ["mean"]}
        if "fees_usd" in df.columns:
            agg["fees_usd"] = ["mean"]

        grp = df.groupby("regime").agg(agg).round(6)
        grp.columns = ["_".join(c).strip() for c in grp.columns]
        savetable(grp, "lvr_by_regime")

    print("\nDONE")


if __name__ == "__main__":
    main()
