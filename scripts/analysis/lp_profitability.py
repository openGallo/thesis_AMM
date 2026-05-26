"""
LP position profitability analysis.

Figures:
    lp_pnl_distribution         Net P&L distribution by position type
    lp_fee_vs_pnl               Fee income vs net P&L scatter
    lp_pnl_by_range_width       Median P&L and fee income vs range width
    fee_apr_series              Fee APR time series with TVL overlay
    lp_duration_vs_pnl          Position duration vs net P&L (log scale)

Tables:
    lp_summary_by_type          P&L summary by position type (narrow/medium/wide)
    lp_profitable_fraction      Fraction profitable by type and vol regime
    fee_apr_by_regime           Fee APR statistics by volatility regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from analysis_utils import COLORS, load, savefig, savetable, vol_regime


def main() -> None:
    print("=" * 60)
    print("LP Profitability Analysis")
    print("=" * 60)

    lp  = load("DEX/dex_lp_positions.csv", index_col=None)
    dex = load("DEX/dex_pool_hourly.csv")
    cex = load("CEX/cex_price_hourly.csv")

    # ── LP positions ──────────────────────────────────────────────
    if lp is not None:
        for col in ["total_minted_usd", "net_pnl_usd", "fee_income_usd",
                    "range_width_pct", "duration_days"]:
            if col in lp.columns:
                lp[col] = pd.to_numeric(lp[col], errors="coerce")

        # Keep positions with meaningful capital deployed
        lp_valid = lp[lp["total_minted_usd"].fillna(0) > 100].copy()
        print(f"  Positions with >$100 minted: {len(lp_valid):,} / {len(lp):,}")

        # Summary by type
        if "position_type" in lp_valid.columns:
            grp = lp_valid.groupby("position_type", observed=True)
            summary = grp.agg(
                N=("net_pnl_usd", "count"),
                mean_pnl=("net_pnl_usd", "mean"),
                median_pnl=("net_pnl_usd", "median"),
                pct_profitable=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
                mean_fee=("fee_income_usd", "mean"),
                mean_range=("range_width_pct", "mean"),
                mean_duration=("duration_days", "mean"),
            ).round(2)
            savetable(summary, "lp_summary_by_type")

        # ── Figure 1: P&L distribution by type ───────────────────
        if "position_type" in lp_valid.columns:
            types = lp_valid["position_type"].dropna().unique()
            fig, axes = plt.subplots(1, len(types), figsize=(4 * len(types), 3.5),
                                     sharey=False)
            if len(types) == 1:
                axes = [axes]
            for ax, (pt, col) in zip(axes, zip(types, COLORS)):
                sub = lp_valid[lp_valid["position_type"] == pt]["net_pnl_usd"].dropna()
                clip = sub.quantile([0.02, 0.98])
                sub  = sub.clip(*clip)
                ax.hist(sub, bins=60, color=col, alpha=0.75, density=True)
                ax.axvline(0, color="black", lw=1, ls="--")
                ax.axvline(sub.median(), color="red", lw=1.2, ls="-",
                           label=f"Median: ${sub.median():,.0f}")
                ax.set_title(f"{pt.capitalize()} positions")
                ax.set_xlabel("Net P&L (USD)")
                ax.legend(fontsize=8)
            fig.suptitle("LP Net P&L Distribution by Position Type")
            savefig("lp_pnl_distribution")

        # ── Figure 2: Fee income vs net P&L ──────────────────────
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        fee = lp_valid["fee_income_usd"].dropna()
        pnl = lp_valid["net_pnl_usd"].dropna()
        idx = fee.index.intersection(pnl.index)
        ax.scatter(fee.loc[idx].clip(upper=fee.quantile(0.99)),
                   pnl.loc[idx].clip(*pnl.quantile([0.01, 0.99])),
                   s=4, alpha=0.25, color=COLORS[0])
        lim = max(abs(ax.get_xlim()[1]), abs(ax.get_ylim()[1]))
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.plot([0, lim], [0, lim], color=COLORS[1], lw=1, ls="--",
                label="Break-even (P&L = fee income)")
        ax.set_xlabel("Fee income (USD)")
        ax.set_ylabel("Net P&L (USD)")
        ax.set_title("Fee Income vs Net P&L per LP Position")
        ax.legend()
        savefig("lp_fee_vs_pnl")

        # ── Figure 3: Median P&L and fee by range width ───────────
        if "range_width_pct" in lp_valid.columns:
            lp_valid["range_bin"] = pd.cut(
                lp_valid["range_width_pct"],
                bins=[0, 2, 5, 10, 20, 50, np.inf],
                labels=["0-2%", "2-5%", "5-10%", "10-20%", "20-50%", ">50%"],
            )
            grp2 = lp_valid.groupby("range_bin", observed=True)[
                ["net_pnl_usd", "fee_income_usd"]].median()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            x = np.arange(len(grp2))
            w = 0.35
            ax.bar(x - w/2, grp2["net_pnl_usd"],    w, label="Median net P&L",    color=COLORS[0])
            ax.bar(x + w/2, grp2["fee_income_usd"],  w, label="Median fee income", color=COLORS[2])
            ax.set_xticks(x)
            ax.set_xticklabels(grp2.index, rotation=30)
            ax.set_xlabel("Range width")
            ax.set_ylabel("USD")
            ax.set_title("Median P&L and Fee Income by Range Width")
            ax.axhline(0, color="black", lw=0.8)
            ax.legend()
            savefig("lp_pnl_by_range_width")

        # ── Figure 4: Duration vs P&L ─────────────────────────────
        if "duration_days" in lp_valid.columns:
            dur = lp_valid["duration_days"].clip(lower=0.01)
            pnl = lp_valid["net_pnl_usd"]
            idx = dur.dropna().index.intersection(pnl.dropna().index)
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            colors_arr = [COLORS[2] if v > 0 else COLORS[1]
                          for v in pnl.loc[idx]]
            ax.scatter(dur.loc[idx], pnl.loc[idx].clip(*pnl.quantile([0.02, 0.98])),
                       s=3, alpha=0.3, c=colors_arr)
            ax.set_xscale("log")
            ax.set_xlabel("Position duration (days, log scale)")
            ax.set_ylabel("Net P&L (USD)")
            ax.set_title("Duration vs Net P&L")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            savefig("lp_duration_vs_pnl")

    # ── Fee APR time series ───────────────────────────────────────
    if dex is not None and "fee_apr_ann" in dex.columns:
        apr = pd.to_numeric(dex["fee_apr_ann"], errors="coerce")
        tvl = pd.to_numeric(dex.get("tvl_usd"), errors="coerce") if "tvl_usd" in dex.columns else None

        fig, ax1 = plt.subplots(figsize=(9, 3.5))
        apr_pct = apr.clip(0, apr.quantile(0.99)) * 100
        ax1.plot(dex.index, apr_pct.rolling(24).mean(), color=COLORS[0], lw=0.9,
                 label="Fee APR (24h MA, %)")
        ax1.set_ylabel("Fee APR (%)")
        ax1.set_xlabel("")

        if tvl is not None:
            ax2 = ax1.twinx()
            ax2.fill_between(dex.index, tvl / 1e6, alpha=0.12, color=COLORS[3])
            ax2.plot(dex.index, tvl / 1e6, color=COLORS[3], lw=0.5,
                     label="TVL (USD M)", alpha=0.6)
            ax2.set_ylabel("TVL (USD M)")
            ax2.spines["right"].set_visible(True)

        ax1.set_title("Hourly Fee APR and Pool TVL")
        ax1.legend(loc="upper left")
        savefig("fee_apr_series")

        # Fee APR by vol regime
        if cex is not None and "realized_vol_24h_ann" in cex.columns:
            vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
            regime = vol_regime(vol)
            aligned = apr.reindex(regime.index)
            tab = pd.DataFrame({"fee_apr_ann": aligned, "regime": regime}).dropna()
            grp = tab.groupby("regime")["fee_apr_ann"].agg(
                N="count", mean="mean", median="median", std="std").round(4)
            savetable(grp, "fee_apr_by_regime")

    print("\nDONE")


if __name__ == "__main__":
    main()
