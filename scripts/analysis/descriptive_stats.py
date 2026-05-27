"""
Descriptive statistics for all key variables.

Per-variable statistics:
    N, Mean [95% block-bootstrap CI], Std, Min, P1, P25, Median, P75, P95, P99, Max
    Skewness, Excess kurtosis, Jarque-Bera normality test
    ADF + KPSS stationarity tests (time-series variables)

Additional outputs:
    data_availability       Coverage summary for all processed files
    corr_matrix / corr_pvals  Pearson + Spearman correlation matrices
    corr_heatmap              Figure: annotated Pearson correlation heatmap
    desc_pool, desc_cex, desc_swaps, desc_lp_positions, desc_lvr, desc_basis
    panel_timeline            Figure: 4-panel time series of key metrics
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats as sp_stats

from analysis_utils import (
    DATA_PROC, FIG_DIR, load, load_swaps, savefig, savetable,
    stars, block_bootstrap_ci, stationarity_tests, COLORS,
)

warnings.filterwarnings("ignore")


# ── Extended describe ─────────────────────────────────────────────────────────

def describe(
    df: pd.DataFrame,
    cols: list[str],
    labels: dict[str, str] | None = None,
    is_timeseries: bool = True,
    block_size: int = 24,
) -> pd.DataFrame:
    """
    Extended descriptive statistics table.
    Uses block bootstrap CI for autocorrelated time series (block_size hours).
    """
    df_num = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce")
    rows = []
    for col in df_num.columns:
        s = df_num[col].dropna()
        if len(s) < 5:
            continue

        jb_stat, jb_p = sp_stats.jarque_bera(s)
        if is_timeseries:
            ci_lo, ci_hi = block_bootstrap_ci(s, func=np.mean, block_size=block_size)
        else:
            ci_lo, ci_hi = block_bootstrap_ci(s, func=np.mean, block_size=1)

        row: dict = {
            "N":        len(s),
            "Mean":     round(float(s.mean()), 6),
            "CI95_lo":  round(ci_lo, 6),
            "CI95_hi":  round(ci_hi, 6),
            "Std":      round(float(s.std()), 6),
            "Min":      round(float(s.min()), 6),
            "P1":       round(float(s.quantile(0.01)), 6),
            "P25":      round(float(s.quantile(0.25)), 6),
            "Median":   round(float(s.quantile(0.50)), 6),
            "P75":      round(float(s.quantile(0.75)), 6),
            "P95":      round(float(s.quantile(0.95)), 6),
            "P99":      round(float(s.quantile(0.99)), 6),
            "Max":      round(float(s.max()), 6),
            "Skewness": round(float(sp_stats.skew(s)), 4),
            "Ex.Kurt":  round(float(sp_stats.kurtosis(s)), 4),
            "JB_stat":  round(float(jb_stat), 4),
            "JB_pval":  round(float(jb_p), 4),
            "JB_sig":   stars(float(jb_p)),
            # Large-N caveat: JB test has power -> 1 as N -> inf.
            # For N >> 1000, p=0 is expected even for negligible non-normality.
            # Use skewness and Ex.Kurt as economic descriptors instead.
            # Refs: D'Agostino & Stephens (1986); Thadewald & Buning (2007).
            "JB_large_N_note": (
                f"N={len(s):,}: JB power~1; p=0 expected for any deviation. "
                "Report Skewness and Ex.Kurt as economic descriptors. "
                "Ref: Thadewald & Buning (2007) J. Applied Statistics."
            ) if len(s) > 5_000 else None,
        }

        if is_timeseries and isinstance(df_num.index, pd.DatetimeIndex) and len(s) >= 20:
            res = stationarity_tests(s)
            row["ADF_pval"]     = res.get("ADF p-val")
            row["ADF_sig"]      = stars(res.get("ADF p-val", 1.0))
            row["KPSS_pval"]    = res.get("KPSS p-val")
            row["Stationarity"] = res.get("Conclusion", "")
        else:
            row["ADF_pval"] = row["ADF_sig"] = row["KPSS_pval"] = row["Stationarity"] = None

        name = (labels or {}).get(col, col)
        rows.append({"Variable": name, **row})

    return pd.DataFrame(rows).set_index("Variable")


def corr_matrices(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (pearson_corr, pearson_pvals, spearman_corr) using pairwise complete obs.
    """
    sub = df[[c for c in cols if c in df.columns]].apply(pd.to_numeric, errors="coerce").dropna()
    col_list = sub.columns.tolist()
    n = len(col_list)
    p_corr = pd.DataFrame(np.eye(n), index=col_list, columns=col_list)
    p_pval = pd.DataFrame(np.zeros((n, n)), index=col_list, columns=col_list)
    s_corr = pd.DataFrame(np.eye(n), index=col_list, columns=col_list)
    for i in range(n):
        for j in range(i + 1, n):
            r, p = sp_stats.pearsonr(sub.iloc[:, i], sub.iloc[:, j])
            p_corr.iloc[i, j] = p_corr.iloc[j, i] = round(r, 4)
            p_pval.iloc[i, j] = p_pval.iloc[j, i] = round(p, 4)
            rs, _ = sp_stats.spearmanr(sub.iloc[:, i], sub.iloc[:, j])
            s_corr.iloc[i, j] = s_corr.iloc[j, i] = round(rs, 4)
    return p_corr.round(4), p_pval.round(4), s_corr.round(4)


def plot_corr_heatmap(corr: pd.DataFrame, title: str, fname: str) -> None:
    """Annotated correlation heatmap with diverging colormap."""
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.65)))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Correlation")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    labels = [l.replace("_", "\n") for l in corr.columns]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    ax.set_title(title)
    savefig(fname)


def plot_data_availability(summary: pd.DataFrame) -> None:
    """Horizontal bar chart showing row counts and date coverage per dataset."""
    available = summary[summary["rows"] > 0].copy()
    if available.empty:
        return
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(available))))
    ys = range(len(available))
    bars = ax.barh(ys, available["rows"] / 1e3, color=COLORS[0], alpha=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(available["dataset"].tolist(), fontsize=8)
    ax.set_xlabel("Rows (thousands)")
    ax.set_title("Processed Data Availability")
    for bar, lbl in zip(bars, available["coverage"].tolist()):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                str(lbl), va="center", fontsize=7)
    savefig("data_availability")


def plot_panel_timeline(
    cex: pd.DataFrame | None,
    dex: pd.DataFrame | None,
    lvr: pd.DataFrame | None,
    merged: pd.DataFrame | None,
) -> None:
    """4-panel time series overview of the key pool metrics."""
    sources = [s for s in [cex, dex, lvr, merged] if s is not None]
    if not sources:
        return

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    # Panel 1: ETH/USDC price
    if cex is not None and "eth_usdc_close" in cex.columns:
        p = pd.to_numeric(cex["eth_usdc_close"], errors="coerce")
        axes[0].plot(cex.index, p, color=COLORS[0], lw=0.5, label="CEX price")
    if dex is not None and "eth_usdc_price" in dex.columns:
        p2 = pd.to_numeric(dex["eth_usdc_price"], errors="coerce")
        axes[0].plot(dex.index, p2, color=COLORS[1], lw=0.5, alpha=0.6, label="DEX price")
    axes[0].set_ylabel("ETH/USDC")
    axes[0].set_title("ETH/USDC Price")
    axes[0].legend(fontsize=8)

    # Panel 2: Realized volatility (24h annualized)
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        v = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce") * 100
        axes[1].fill_between(cex.index, 0, v, alpha=0.25, color=COLORS[2])
        axes[1].plot(cex.index, v, color=COLORS[2], lw=0.5)
    axes[1].set_ylabel("24h Realized Vol (%)")
    axes[1].set_title("Annualized Realized Volatility (24h window)")

    # Panel 3: Pool TVL and fee APR
    if dex is not None and "tvl_usd" in dex.columns:
        tvl = pd.to_numeric(dex["tvl_usd"], errors="coerce") / 1e6
        ax3b = axes[2].twinx()
        axes[2].fill_between(dex.index, 0, tvl, alpha=0.15, color=COLORS[3])
        axes[2].plot(dex.index, tvl, color=COLORS[3], lw=0.5, label="TVL (USD M)")
        axes[2].set_ylabel("TVL (USD M)")
        if "fee_apr_ann" in dex.columns:
            apr = pd.to_numeric(dex["fee_apr_ann"], errors="coerce") * 100
            ax3b.plot(dex.index, apr.rolling(24).mean(), color=COLORS[4], lw=0.8,
                      label="Fee APR (24h MA, %)")
            ax3b.set_ylabel("Fee APR (%)")
            ax3b.spines["right"].set_visible(True)
    axes[2].set_title("Pool TVL and Fee APR")

    # Panel 4: DEX-CEX basis
    if merged is not None and "dex_cex_basis_bps" in merged.columns:
        b = pd.to_numeric(merged["dex_cex_basis_bps"], errors="coerce")
        axes[3].plot(merged.index, b, color=COLORS[0], lw=0.4, alpha=0.6)
        axes[3].axhline(5,  color=COLORS[1], lw=0.8, ls="--", alpha=0.7)
        axes[3].axhline(-5, color=COLORS[1], lw=0.8, ls="--", alpha=0.7)
        axes[3].fill_between(merged.index, b, 0,
                             where=(b.abs() > 5), color=COLORS[1], alpha=0.12)
    elif dex is not None and "log_return_1h" in dex.columns:
        r = pd.to_numeric(dex["log_return_1h"], errors="coerce") * 100
        axes[3].plot(dex.index, r, color=COLORS[0], lw=0.4)
    axes[3].set_ylabel("DEX-CEX basis (bps)" if merged is not None else "Log return (%)")
    axes[3].set_title("DEX-CEX Price Basis (arbitrage opportunities shaded)")
    axes[3].set_xlabel("")

    for ax in axes:
        ax.grid(True, alpha=0.25)
    savefig("panel_timeline")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Descriptive Statistics")
    print("=" * 60)

    # ── Data availability summary ─────────────────────────────────
    avail_rows = []
    for rel, label in [
        ("DEX/dex_pool_hourly.csv",  "DEX pool hourly"),
        ("DEX/dex_swaps.csv",        "DEX swaps"),
        ("DEX/dex_lvr_hourly.csv",   "DEX LVR hourly"),
        ("DEX/dex_lp_positions.csv", "DEX LP positions"),
        ("CEX/cex_price_hourly.csv", "CEX price hourly"),
        ("merged/merged_hourly.csv", "Merged hourly panel"),
    ]:
        path = DATA_PROC / rel
        if path.exists():
            try:
                tmp = pd.read_csv(path, nrows=5, index_col=0, parse_dates=True,
                                  low_memory=False)
                full = pd.read_csv(path, index_col=0, low_memory=False,
                                   usecols=[0])
                n = len(full)
                try:
                    full.index = pd.to_datetime(full.index, utc=True, errors="coerce")
                    t0 = str(full.index.min().date())
                    t1 = str(full.index.max().date())
                    cov = f"{t0} to {t1}"
                except Exception:
                    cov = "date parse failed"
                avail_rows.append({"dataset": label, "rows": n, "coverage": cov,
                                   "cols": len(tmp.columns)})
            except Exception as exc:
                avail_rows.append({"dataset": label, "rows": 0, "coverage": str(exc), "cols": 0})
        else:
            avail_rows.append({"dataset": label, "rows": 0, "coverage": "NOT FOUND", "cols": 0})

    avail_df = pd.DataFrame(avail_rows)
    savetable(avail_df.set_index("dataset"), "data_availability")
    print("\nData availability:")
    for _, r in avail_df.iterrows():
        status = "OK" if r["rows"] > 0 else "MISSING"
        print(f"  [{status}] {r['dataset']}: {r['rows']:,} rows  {r['coverage']}")

    # ── Load data ─────────────────────────────────────────────────
    dex    = load("DEX/dex_pool_hourly.csv")
    cex    = load("CEX/cex_price_hourly.csv")
    lvr    = load("DEX/dex_lvr_hourly.csv")
    merged = load("merged/merged_hourly.csv")

    # ── DEX pool hourly ───────────────────────────────────────────
    if dex is not None:
        cols = [
            "eth_usdc_price", "volume_usd", "fees_usd", "tvl_usd",
            "tx_count", "fee_apr_ann", "vol_over_tvl", "log_return_1h",
            "liquidity",
        ]
        labels = {
            "eth_usdc_price": "ETH/USDC price (DEX)",
            "volume_usd":     "Volume USD (hourly)",
            "fees_usd":       "Fees USD (hourly)",
            "tvl_usd":        "TVL USD",
            "tx_count":       "Swap count (hourly)",
            "fee_apr_ann":    "Fee APR (annualized)",
            "vol_over_tvl":   "Volume / TVL",
            "log_return_1h":  "Log return (1h DEX)",
            "liquidity":      "Active liquidity (raw)",
        }
        tab = describe(dex, cols, labels, is_timeseries=True, block_size=24)
        savetable(tab, "desc_pool")
        print(f"\n  DEX pool hourly: {len(dex):,} rows "
              f"({dex.index.min().date()} to {dex.index.max().date()})")

    # ── CEX price hourly ─────────────────────────────────────────
    if cex is not None:
        cols = [
            "eth_usdc_close", "log_return_1h", "log_return_6h",
            "log_return_24h", "realized_vol_1h_ann", "realized_vol_24h_ann",
            "realized_vol_7d_ann", "ohlc_range_1h",
            "vol_base_1h_ethusdt", "n_trades_1h_ethusdt",
        ]
        labels = {
            "eth_usdc_close":       "ETH/USDC close (CEX)",
            "log_return_1h":        "Log return (1h CEX)",
            "log_return_6h":        "Log return (6h CEX)",
            "log_return_24h":       "Log return (24h CEX)",
            "realized_vol_1h_ann":  "Realized vol 1h (ann.)",
            "realized_vol_24h_ann": "Realized vol 24h (ann.)",
            "realized_vol_7d_ann":  "Realized vol 7d (ann.)",
            "ohlc_range_1h":        "OHLC range (1h)",
            "vol_base_1h_ethusdt":  "CEX volume ETH (1h)",
            "n_trades_1h_ethusdt":  "CEX trade count (1h)",
        }
        tab = describe(cex, cols, labels, is_timeseries=True, block_size=24)
        savetable(tab, "desc_cex")
        print(f"  CEX hourly: {len(cex):,} rows "
              f"({cex.index.min().date()} to {cex.index.max().date()})")

    # ── LVR hourly ────────────────────────────────────────────────
    if lvr is not None:
        cols   = ["lvr_usd_tvl_approx", "lvr_rate_ann", "lvr_to_fee_ratio"]
        labels = {
            "lvr_usd_tvl_approx": "LVR USD (hourly, TVL approx.)",
            "lvr_rate_ann":       "LVR rate (annualized, % of TVL)",
            "lvr_to_fee_ratio":   "LVR / fee income ratio",
        }
        tab = describe(lvr, cols, labels, is_timeseries=True, block_size=24)
        savetable(tab, "desc_lvr")
        print(f"  LVR hourly: {len(lvr):,} rows")

    # ── DEX-CEX basis ────────────────────────────────────────────
    if merged is not None:
        cols   = ["dex_cex_basis_bps", "dex_cex_vol_ratio"]
        labels = {
            "dex_cex_basis_bps": "DEX-CEX basis (bps)",
            "dex_cex_vol_ratio": "DEX/CEX volume ratio",
        }
        tab = describe(merged, cols, labels, is_timeseries=True, block_size=24)
        savetable(tab, "desc_basis")
        if "arbitrage_flag" in merged.columns:
            arb_pct = merged["arbitrage_flag"].mean() * 100
            print(f"  Arbitrage flag: {arb_pct:.1f}% of hours")

    # ── Swaps ─────────────────────────────────────────────────────
    swaps = load_swaps()
    if swaps is not None:
        swaps["amount_usd"] = swaps["amount_usd"].abs()
        cols   = ["amount_usd", "gas_cost_usd", "gas_gwei", "log_price_change"]
        labels = {
            "amount_usd":       "Swap size USD",
            "gas_cost_usd":     "Gas cost USD",
            "gas_gwei":         "Gas price (Gwei)",
            "log_price_change": "Log price change per swap",
        }
        tab = describe(swaps, cols, labels, is_timeseries=False)
        savetable(tab, "desc_swaps")
        print(f"  Swaps: {len(swaps):,} rows")

    # ── LP positions ─────────────────────────────────────────────
    lp_path = DATA_PROC / "DEX" / "dex_lp_positions.csv"
    lp = pd.read_csv(lp_path, low_memory=False) if lp_path.exists() else None
    if lp is not None:
        cols   = ["total_minted_usd", "total_burned_usd", "total_collected_usd",
                  "net_pnl_usd", "fee_income_usd", "range_width_pct", "duration_days"]
        labels = {
            "total_minted_usd":    "Total minted USD",
            "total_burned_usd":    "Total burned USD",
            "total_collected_usd": "Total collected USD",
            "net_pnl_usd":         "Net P&L USD",
            "fee_income_usd":      "Fee income USD",
            "range_width_pct":     "Range width (%)",
            "duration_days":       "Position duration (days)",
        }
        tab = describe(lp, cols, labels, is_timeseries=False)
        savetable(tab, "desc_lp_positions")

        # Wilcoxon signed-rank: H0: median net P&L = 0, by position type
        if "position_type" in lp.columns and "net_pnl_usd" in lp.columns:
            lp["net_pnl_usd"] = pd.to_numeric(lp["net_pnl_usd"], errors="coerce")
            wx_rows = []
            for pt, grp in lp.groupby("position_type", observed=True):
                pnl = grp["net_pnl_usd"].dropna()
                if len(pnl) < 10:
                    continue
                try:
                    stat, pval = sp_stats.wilcoxon(
                        pnl, zero_method="wilcox", alternative="two-sided")
                except ValueError:
                    continue
                ci_lo, ci_hi = block_bootstrap_ci(pnl, func=np.median, block_size=1)
                wx_rows.append({
                    "Type":       pt,
                    "N":          len(pnl),
                    "Median_PnL": round(float(pnl.median()), 2),
                    "CI95_lo":    round(ci_lo, 2),
                    "CI95_hi":    round(ci_hi, 2),
                    "W_stat":     round(float(stat), 2),
                    "p_val":      round(float(pval), 4),
                    "Sig":        stars(float(pval)),
                    "Reject_H0":  "Yes" if pval < 0.05 else "No",
                })
            if wx_rows:
                savetable(pd.DataFrame(wx_rows).set_index("Type"), "desc_lp_wilcoxon")
                print(f"  Wilcoxon tests: {len(wx_rows)} position types")
        print(f"  LP positions: {len(lp):,}")
    else:
        print("  [SKIP] dex_lp_positions.csv not found")

    # ── Correlation matrices (hourly panel) ───────────────────────
    if merged is not None:
        panel_cols = [c for c in [
            "log_return_1h", "realized_vol_24h_ann", "fee_apr_ann",
            "tx_count", "dex_cex_basis_bps", "lvr_rate_ann",
            "vol_over_tvl", "gas_gwei", "cex_vol_1h",
        ] if c in merged.columns]
        if len(panel_cols) >= 2:
            print(f"\n  Computing correlation matrices ({len(panel_cols)} variables)...")
            p_corr, p_pval, s_corr = corr_matrices(merged, panel_cols)
            savetable(p_corr, "corr_pearson")
            savetable(p_pval, "corr_pvals")
            savetable(s_corr, "corr_spearman")
            plot_corr_heatmap(p_corr, "Pearson Correlation Matrix (hourly panel)",
                              "corr_heatmap_pearson")
            plot_corr_heatmap(s_corr, "Spearman Correlation Matrix (hourly panel)",
                              "corr_heatmap_spearman")

    elif dex is not None:
        # Fallback: use DEX + CEX columns only
        combined = dex.copy()
        if cex is not None:
            for col in ["log_return_1h", "realized_vol_24h_ann"]:
                if col in cex.columns and col not in combined.columns:
                    combined[col] = cex[col].reindex(combined.index)
        panel_cols = [c for c in [
            "log_return_1h", "realized_vol_24h_ann", "fee_apr_ann",
            "tx_count", "vol_over_tvl",
        ] if c in combined.columns]
        if len(panel_cols) >= 2:
            p_corr, p_pval, s_corr = corr_matrices(combined, panel_cols)
            savetable(p_corr, "corr_pearson")
            savetable(p_pval, "corr_pvals")
            plot_corr_heatmap(p_corr, "Pearson Correlation Matrix",
                              "corr_heatmap_pearson")

    # ── 4-panel timeline figure ───────────────────────────────────
    print("\n  Plotting timeline figure...")
    plot_panel_timeline(cex, dex, lvr, merged)

    # ── Data availability figure ──────────────────────────────────
    plot_data_availability(avail_df)

    print("\nDONE")


if __name__ == "__main__":
    main()
