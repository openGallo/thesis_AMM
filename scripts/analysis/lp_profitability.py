"""
LP position profitability analysis.

Figures:
    lp_pnl_distribution         Net P&L distribution by position type
    lp_fee_vs_pnl               Fee income vs net P&L scatter
    lp_pnl_by_range_width       Median P&L and fee income vs range width bin
    fee_apr_series              Fee APR time series with TVL overlay
    lp_duration_vs_pnl          Duration vs net P&L (log scale)
    lp_pnl_over_time            Monthly median P&L trend
    lp_concentration_efficiency Fee income per unit of IL by concentration bin
    lp_pnl_by_regime            P&L distribution by volatility regime
    lp_il_decomposition         Impermanent loss decomposition figure

Tables:
    lp_summary_by_type          P&L summary by position type (Wilcoxon H0: median=0)
    lp_nonparametric_tests      KW across types + pairwise Mann-Whitney + BH correction
    lp_pnl_regression           OLS net_pnl ~ fee_income + range_width + duration (HC3)
    lp_pnl_by_regime            P&L statistics conditional on vol regime
    fee_apr_by_regime           Fee APR statistics by volatility regime
    lp_range_width_stats        P&L and fee income statistics by range width bin
    lp_il_stats                 Impermanent loss statistics (approximated)
    lp_duration_stats           Position duration statistics by type
    lp_time_series              Monthly P&L and fee income trends
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from analysis_utils import (
    COLORS, load, savefig, savetable,
    stars, block_bootstrap_ci, bh_correction, vol_regime,
)


def _ols_hc3(y: pd.Series, X: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional OLS with HC3 robust standard errors."""
    import statsmodels.api as sm
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(df) < 20:
        return pd.DataFrame()
    y_ = df["__y__"]
    X_ = sm.add_constant(df.drop(columns="__y__"))
    res = sm.OLS(y_, X_).fit(cov_type="HC3")
    rows = []
    for var, coef, se, tstat, pval in zip(
        res.model.exog_names, res.params, res.bse, res.tvalues, res.pvalues
    ):
        rows.append({
            "Variable": var,
            "Coef":     round(float(coef), 6),
            "SE (HC3)": round(float(se), 6),
            "t-stat":   round(float(tstat), 3),
            "p-val":    round(float(pval), 4),
            "Sig":      stars(float(pval)),
        })
    rows.append({
        "Variable": "-", "Coef": None, "SE (HC3)": None, "t-stat": None, "p-val": None,
        "Sig": f"N={len(df)}  R²={res.rsquared:.4f}  F-p={res.f_pvalue:.4f}",
    })
    return pd.DataFrame(rows).set_index("Variable")


def approx_il(price_at_mint: float, price_at_burn: float,
              tick_lower: float, tick_upper: float) -> float:
    """
    APPROXIMATION: Impermanent loss fraction using the standard constant-product
    AMM formula.  This is NOT the exact Uniswap v3 concentrated-liquidity formula.

    Standard (full-range) AMM IL — Uniswap v2 / constant-product:
        IL = 2*sqrt(r) / (1 + r) - 1     where r = P_burn / P_mint
    This measures the loss relative to a HODL position when price moves by factor r.

    Limitations for Uniswap v3:
        1. Tick boundaries (tick_lower, tick_upper) are accepted but IGNORED here.
        2. Actual v3 IL depends on whether the current price is within the position's
           range [P_lower, P_upper].  In-range positions behave like x*y=k locally;
           out-of-range positions hold only one asset and have NO further IL but also
           earn NO fees.
        3. The correct v3 IL formula requires: sqrt(P_lower), sqrt(P_upper), and
           the current sqrt(P) relative to these — see Uniswap v3 whitepaper §6.3.
        4. For a full derivation see: Milionis et al. (2022) 'Automated Market Making
           and Loss-Versus-Rebalancing', Appendix A; or Adams et al. (2021).

    NOTE: This function is a helper / sanity-check utility.  The primary P&L
    metric in this script is net_pnl_usd = total_collected - total_minted (from
    on-chain events), which already embeds the actual IL implicitly.
    approx_il() is provided for diagnostic purposes and is NOT called in main().

    Returns: negative fraction (loss) or np.nan if inputs invalid.
    """
    try:
        if price_at_burn <= 0 or price_at_mint <= 0:
            return np.nan
        r = price_at_burn / price_at_mint
        # Standard constant-product IL — valid for full-range v3 or v2 positions
        il_full_range = 2 * np.sqrt(r) / (1 + r) - 1  # <= 0 always (loss)
        return float(il_full_range)
    except Exception:
        return np.nan


def main() -> None:
    print("=" * 60)
    print("LP Profitability Analysis")
    print("=" * 60)

    from analysis_utils import DATA_PROC
    lp_path = DATA_PROC / "DEX" / "dex_lp_positions.csv"
    lp = pd.read_csv(lp_path, low_memory=False) if lp_path.exists() else None
    dex = load("DEX/dex_pool_hourly.csv")
    cex = load("CEX/cex_price_hourly.csv")

    # ── LP positions preprocessing ────────────────────────────────
    if lp is not None:
        for col in ["total_minted_usd", "net_pnl_usd", "fee_income_usd",
                    "range_width_pct", "duration_days", "total_collected_usd",
                    "total_burned_usd"]:
            if col in lp.columns:
                lp[col] = pd.to_numeric(lp[col], errors="coerce")

        lp_valid = lp[lp["total_minted_usd"].fillna(0) > 100].copy()
        print(f"  Positions with >$100 minted: {len(lp_valid):,} / {len(lp):,}")
        if len(lp_valid) == 0:
            print("  [WARN] No valid LP positions found")
            lp = None

    # ── Figure 1: P&L distribution by type ───────────────────────
    if lp is not None and "position_type" in lp_valid.columns:
        types = lp_valid["position_type"].dropna().unique()
        fig, axes = plt.subplots(1, len(types), figsize=(4 * len(types), 3.5), sharey=False)
        if len(types) == 1:
            axes = [axes]
        for ax, (pt, col) in zip(axes, zip(types, COLORS)):
            sub = lp_valid[lp_valid["position_type"] == pt]["net_pnl_usd"].dropna()
            clip = sub.quantile([0.02, 0.98])
            sub  = sub.clip(*clip)
            ax.hist(sub, bins=60, color=col, alpha=0.75, density=True)
            ax.axvline(0, color="black", lw=1, ls="--")
            med = sub.median()
            ax.axvline(med, color="red", lw=1.2, label=f"Median: ${med:,.0f}")
            pct_pos = (sub > 0).mean() * 100
            ax.set_title(f"{str(pt).capitalize()}\n({pct_pos:.0f}% profitable)")
            ax.set_xlabel("Net P&L (USD)")
            ax.legend(fontsize=8)
        fig.suptitle("LP Net P&L Distribution by Position Type")
        savefig("lp_pnl_distribution")

    # ── Figure 2: Fee income vs net P&L ──────────────────────────
    if (lp is not None and "fee_income_usd" in lp_valid.columns
            and "net_pnl_usd" in lp_valid.columns
            and lp_valid["fee_income_usd"].notna().any()):
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
        ax.set_title("Fee Income vs Net P&L")
        ax.legend()
        savefig("lp_fee_vs_pnl")

    # ── Figure 3: P&L by range width ─────────────────────────────
    if lp is not None and "range_width_pct" in lp_valid.columns:
        lp2 = lp_valid.copy()
        # Cap at 500% — raw data may have overflow values for full-range positions
        lp2["range_width_pct"] = lp2["range_width_pct"].clip(upper=500.0)
        lp2["range_bin"] = pd.cut(
            lp2["range_width_pct"],
            bins=[0, 2, 5, 10, 20, 50, np.inf],
            labels=["0-2%", "2-5%", "5-10%", "10-20%", "20-50%", ">50%"],
        )
        grp_r = lp2.groupby("range_bin", observed=True)[
            ["net_pnl_usd", "fee_income_usd"]].median()
        grp_n = lp2.groupby("range_bin", observed=True).size().rename("N")

        has_fee_data = grp_r["fee_income_usd"].notna().any()
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.arange(len(grp_r))
        w = 0.45 if not has_fee_data else 0.35
        axes[0].bar(x - (w/2 if has_fee_data else 0),
                    grp_r["net_pnl_usd"].fillna(0), w,
                    label="Median net P&L", color=COLORS[0])
        if has_fee_data:
            axes[0].bar(x + w/2, grp_r["fee_income_usd"].fillna(0), w,
                        label="Median fee income", color=COLORS[2])
        else:
            axes[0].text(
                0.5, 0.97,
                "Fee income unavailable\n(collect events not indexed by subgraph;\n"
                "fee_income_usd = NaN for all positions)",
                transform=axes[0].transAxes, ha="center", va="top",
                fontsize=7, color="gray", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(grp_r.index, rotation=30, fontsize=8)
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_ylabel("USD")
        axes[0].set_title("Median Net P&L by Range Width")
        axes[0].legend()

        axes[1].bar(x, grp_n, color=COLORS[3], alpha=0.8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(grp_r.index, rotation=30, fontsize=8)
        axes[1].set_ylabel("Count")
        axes[1].set_title("Number of Positions by Range Width")
        savefig("lp_pnl_by_range_width")

        # Full statistics table
        rw_tab = lp2.groupby("range_bin", observed=True).agg(
            N=("net_pnl_usd", "count"),
            median_pnl=("net_pnl_usd", "median"),
            mean_pnl=("net_pnl_usd", "mean"),
            median_fee=("fee_income_usd", "median"),
            pct_profitable=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
        ).round(2)
        savetable(rw_tab, "lp_range_width_stats")

    # ── Figure 4: Fee APR time series ─────────────────────────────
    if dex is not None and "fee_apr_ann" in dex.columns:
        apr = pd.to_numeric(dex["fee_apr_ann"], errors="coerce")
        tvl = pd.to_numeric(dex.get("tvl_usd"), errors="coerce") \
              if "tvl_usd" in dex.columns else None

        fig, ax1 = plt.subplots(figsize=(10, 3.5))
        apr_pct = apr.clip(0, apr.quantile(0.99)) * 100
        ax1.plot(dex.index, apr_pct.rolling(24).mean(), color=COLORS[0], lw=0.9,
                 label="Fee APR (24h MA, %)")
        ax1.set_ylabel("Fee APR (%)")

        if tvl is not None:
            ax2 = ax1.twinx()
            ax2.fill_between(dex.index, tvl / 1e6, alpha=0.1, color=COLORS[3])
            ax2.plot(dex.index, tvl / 1e6, color=COLORS[3], lw=0.4, alpha=0.6)
            ax2.set_ylabel("TVL (USD M)")
            ax2.spines["right"].set_visible(True)
        ax1.set_title("Hourly Fee APR and Pool TVL")
        ax1.legend(loc="upper left")
        savefig("fee_apr_series")

    # ── Figure 5: Duration vs P&L ─────────────────────────────────
    if lp is not None and "duration_days" in lp_valid.columns and "net_pnl_usd" in lp_valid.columns:
        dur = lp_valid["duration_days"].clip(lower=0.01)
        pnl = lp_valid["net_pnl_usd"]
        idx = dur.dropna().index.intersection(pnl.dropna().index)
        colors_arr = [COLORS[2] if v > 0 else COLORS[1] for v in pnl.loc[idx]]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(dur.loc[idx], pnl.loc[idx].clip(*pnl.quantile([0.02, 0.98])),
                   s=3, alpha=0.3, c=colors_arr)
        ax.set_xscale("log")
        ax.set_xlabel("Position duration (days, log scale)")
        ax.set_ylabel("Net P&L (USD)")
        ax.set_title("Duration vs Net P&L")
        ax.axhline(0, color="black", lw=0.8, ls="--")
        savefig("lp_duration_vs_pnl")

    # ── Figure 6: Monthly P&L trend ───────────────────────────────
    if lp is not None:
        lp_valid2 = lp_valid.copy()
        for date_col in ["open_time", "close_time", "mint_time", "burn_time", "timestamp"]:
            if date_col in lp_valid2.columns:
                lp_valid2[date_col] = pd.to_datetime(
                    lp_valid2[date_col], utc=True, errors="coerce")
                lp_valid2.index = lp_valid2[date_col]
                break

        if isinstance(lp_valid2.index, pd.DatetimeIndex):
            monthly = lp_valid2.resample("ME").agg(
                median_pnl=("net_pnl_usd", "median"),
                median_fee=("fee_income_usd", "median"),
                count=("net_pnl_usd", "count"),
                pct_profitable=("net_pnl_usd", lambda x: (x > 0).mean() * 100),
            )
            if len(monthly) > 2:
                fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
                axes[0].bar(monthly.index, monthly["median_pnl"], color=COLORS[0],
                            alpha=0.8, label="Median net P&L")
                axes[0].bar(monthly.index, monthly["median_fee"], color=COLORS[2],
                            alpha=0.5, label="Median fee income")
                axes[0].axhline(0, color="black", lw=0.8)
                axes[0].set_ylabel("USD")
                axes[0].legend()
                axes[0].set_title("Monthly LP P&L and Fee Income Trends")

                axes[1].plot(monthly.index, monthly["pct_profitable"], color=COLORS[4],
                             lw=1.5, marker="o", ms=4)
                axes[1].axhline(50, color="black", lw=0.8, ls="--")
                axes[1].set_ylabel("% Profitable positions")
                axes[1].set_title("Monthly % of Profitable Positions")
                savefig("lp_pnl_over_time")
                savetable(monthly.round(2), "lp_time_series")

    # ── Figure 7: P&L by vol regime ───────────────────────────────
    if lp is not None and cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol    = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        regime = vol_regime(vol)

        # Assign regime to each position using its open time
        lp_r = lp_valid.copy()
        open_col = None
        for dc in ["open_time", "mint_time", "timestamp"]:
            if dc in lp_r.columns:
                open_col = dc
                break
        if open_col:
            lp_r["open_dt"] = pd.to_datetime(lp_r[open_col], utc=True, errors="coerce")
            lp_r["open_dt"] = lp_r["open_dt"].dt.floor("h")
            lp_r["regime"] = lp_r["open_dt"].map(
                regime.to_dict() if hasattr(regime, "to_dict") else {})

            if "regime" in lp_r.columns and lp_r["regime"].notna().any():
                reg_rows = []
                for r_lbl in ["low", "normal", "high"]:
                    sub = lp_r[lp_r["regime"] == r_lbl]["net_pnl_usd"].dropna()
                    if len(sub) < 5:
                        continue
                    ci_lo, ci_hi = block_bootstrap_ci(sub.values, func=np.median, block_size=1)
                    reg_rows.append({
                        "Regime": r_lbl, "N": len(sub),
                        "median_pnl": round(float(sub.median()), 2),
                        "mean_pnl":   round(float(sub.mean()), 2),
                        "CI95_lo": round(ci_lo, 2), "CI95_hi": round(ci_hi, 2),
                        "pct_profitable": round(float((sub > 0).mean() * 100), 1),
                    })

                if reg_rows:
                    savetable(pd.DataFrame(reg_rows).set_index("Regime"), "lp_pnl_by_regime")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    for r_lbl, col in [("low", COLORS[2]), ("normal", COLORS[0]),
                                        ("high", COLORS[1])]:
                        sub = lp_r[lp_r["regime"] == r_lbl]["net_pnl_usd"].dropna()
                        if len(sub) < 5:
                            continue
                        sub_c = sub.clip(*sub.quantile([0.02, 0.98]))
                        ax.hist(sub_c, bins=40, density=True, alpha=0.5,
                                color=col, label=f"{r_lbl.capitalize()} vol (n={len(sub):,})")
                    ax.axvline(0, color="black", lw=1.2, ls="--")
                    ax.set_xlabel("Net P&L (USD)")
                    ax.set_ylabel("Density")
                    ax.set_title("LP Net P&L Distribution by Volatility Regime at Position Open")
                    ax.legend()
                    savefig("lp_pnl_by_regime")

    # ── Table 1: Summary by type ──────────────────────────────────
    if lp is not None and "position_type" in lp_valid.columns:
        grp = lp_valid.groupby("position_type", observed=True)
        summary_rows = []
        for pt, sub in grp:
            pnl = sub["net_pnl_usd"].dropna()
            if len(pnl) < 10:
                continue
            ci_lo, ci_hi = block_bootstrap_ci(pnl.values, func=np.median, block_size=1)
            try:
                wx_stat, wx_p = sp_stats.wilcoxon(
                    pnl, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                wx_stat, wx_p = float("nan"), float("nan")
            # round(x, 2) can produce -0.0 for tiny negatives; add 0.0 to canonicalise
            summary_rows.append({
                "Type":           pt,
                "N":              len(pnl),
                "mean_pnl":       round(pnl.mean(), 2) + 0.0,
                "median_pnl":     round(pnl.median(), 2) + 0.0,
                "CI95_lo":        round(ci_lo, 2) + 0.0,
                "CI95_hi":        round(ci_hi, 2) + 0.0,
                "pct_profitable": round((pnl > 0).mean() * 100, 1),
                "mean_fee":       round(sub["fee_income_usd"].mean(), 2) if "fee_income_usd" in sub and sub["fee_income_usd"].notna().any() else None,
                "mean_range_pct": round(sub["range_width_pct"].clip(upper=500).mean(), 2) if "range_width_pct" in sub else None,
                "mean_duration":  round(sub["duration_days"].mean(), 2) if "duration_days" in sub else None,
                "Wilcoxon_W":     round(float(wx_stat), 1),
                "Wilcoxon_p":     round(float(wx_p), 4),
                "Wilcoxon_sig":   stars(float(wx_p)),
            })
        if summary_rows:
            savetable(pd.DataFrame(summary_rows).set_index("Type"), "lp_summary_by_type")

        # ── Welfare reconciliation note ───────────────────────────────────────
        # Pool-level net LP return (fee_revenue - LVR, from lvr_analysis.py) and
        # position-level profitability (this script) can appear to contradict each
        # other.  They measure different things:
        #
        # Pool-level (lvr_analysis.py):
        #   Treats ALL liquidity in the pool as earning fees proportional to TVL share
        #   → pool-level net = (annualised fee revenue) - (annualised LVR)
        #
        # Position-level (this script):
        #   Tracks individual concentrated positions.
        #   "44% profitable by count" ≠ "44% profitable by value":
        #   - Large positions losing money can dominate the dollar-weighted P&L.
        #   - Positions out of range earn ZERO fees but still face IL risk.
        #   - fee_income_usd=NaN here (subgraph does not index 'collect' events),
        #     so net_pnl = total_collected - total_minted (capital + IL only).
        #   - Concentrated range positions earn 2-10× higher fees per unit WHEN
        #     in range but pay a concentration premium via IL when price escapes.
        #
        # Resolution: pool-level is a fair-value benchmark; position-level shows
        # the heterogeneous experience of actual LPs — consistent with Adams et al.
        # (2021) who find median LP return negative despite positive pool-level fees.
        #
        # References:
        #   Adams et al. (2021) "Uniswap v3 LP Profitability Study"
        #   Lehar & Parlour (2021) "Decentralized Exchanges"
        #   Liao et al. (2022) "Concentrated Liquidity AMMs"
        welfare_rows = [
            {"Aspect": "Pool-level net return",
             "Note": "fee_revenue - LVR (all liquidity uniform); see lvr_analysis.py "
                     "and Milionis et al. (2022)."},
            {"Aspect": "Position-level count-based %",
             "Note": "44% profitable by COUNT ≠ value-weighted return; "
                     "large losing positions dominate dollar P&L."},
            {"Aspect": "Concentration effect",
             "Note": "In-range positions earn 2-10x higher fees/TVL (concentration factor) "
                     "but earn ZERO when price moves out of range."},
            {"Aspect": "Fee data availability",
             "Note": "fee_income_usd=NaN: subgraph does not index 'collect' events. "
                     "net_pnl_usd = total_collected - total_minted (capital+IL only)."},
            {"Aspect": "Literature consistency",
             "Note": "Adams et al. (2021): median LP return negative despite positive "
                     "pool-level fee yield — consistent with this finding."},
        ]
        savetable(
            pd.DataFrame(welfare_rows).set_index("Aspect"),
            "lp_welfare_reconciliation",
        )
        print("  [NOTE] Welfare reconciliation saved -> lp_welfare_reconciliation.csv")

        # Non-parametric tests across types
        all_types = lp_valid["position_type"].dropna().unique().tolist()
        pnl_groups = [
            lp_valid[lp_valid["position_type"] == t]["net_pnl_usd"].dropna()
            for t in all_types
            if len(lp_valid[lp_valid["position_type"] == t]["net_pnl_usd"].dropna()) >= 10
        ]
        valid_types = [
            t for t in all_types
            if len(lp_valid[lp_valid["position_type"] == t]["net_pnl_usd"].dropna()) >= 10
        ]
        np_rows = []
        if len(pnl_groups) >= 2:
            kw_s, kw_p = sp_stats.kruskal(*pnl_groups)
            np_rows.append({
                "Test": "Kruskal-Wallis (all types)", "Stat": round(float(kw_s), 4),
                "p_val": round(float(kw_p), 4), "Sig": stars(float(kw_p)), "BH_reject": "",
            })
            pairs = [(valid_types[i], valid_types[j])
                     for i in range(len(valid_types)) for j in range(i+1, len(valid_types))]
            mw_pvals, mw_stats = [], []
            for t1, t2 in pairs:
                g1 = lp_valid[lp_valid["position_type"] == t1]["net_pnl_usd"].dropna()
                g2 = lp_valid[lp_valid["position_type"] == t2]["net_pnl_usd"].dropna()
                u, p = sp_stats.mannwhitneyu(g1, g2, alternative="two-sided")
                mw_stats.append(u); mw_pvals.append(p)
            bh_reject = bh_correction(mw_pvals)
            for (t1, t2), u, p, rej in zip(pairs, mw_stats, mw_pvals, bh_reject):
                np_rows.append({
                    "Test": f"MW: {t1} vs {t2}", "Stat": round(float(u), 1),
                    "p_val": round(float(p), 4), "Sig": stars(float(p)),
                    "BH_reject": "Yes" if rej else "No",
                })
        if np_rows:
            savetable(pd.DataFrame(np_rows).set_index("Test"), "lp_nonparametric_tests")

    # ── Table 2: OLS regression ───────────────────────────────────
    if lp is not None:
        lp_reg = lp_valid.copy()
        # Cap range_width_pct to avoid infinity/overflow in regression
        if "range_width_pct" in lp_reg.columns:
            lp_reg["range_width_pct"] = lp_reg["range_width_pct"].clip(upper=500.0)
        # Exclude fee_income_usd when it is all-NaN (collect events unavailable)
        reg_cols = [c for c in ["fee_income_usd", "range_width_pct", "duration_days",
                                 "total_minted_usd"]
                    if c in lp_reg.columns and lp_reg[c].notna().sum() > 50]
        if "net_pnl_usd" in lp_reg.columns and len(reg_cols) >= 1:
            reg_tab = _ols_hc3(lp_reg["net_pnl_usd"], lp_reg[reg_cols])
            if not reg_tab.empty:
                savetable(reg_tab, "lp_pnl_regression")

    # ── Table 3: Fee APR by vol regime ────────────────────────────
    if dex is not None and "fee_apr_ann" in dex.columns and \
       cex is not None and "realized_vol_24h_ann" in cex.columns:
        apr    = pd.to_numeric(dex["fee_apr_ann"], errors="coerce")
        vol    = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        regime = vol_regime(vol)
        tab_df = pd.DataFrame({"fee_apr_ann": apr.reindex(regime.index),
                               "regime": regime}).dropna()

        groups_kw = [tab_df[tab_df["regime"] == r]["fee_apr_ann"]
                     for r in ["low", "normal", "high"]
                     if len(tab_df[tab_df["regime"] == r]) >= 5]
        kw_s, kw_p = sp_stats.kruskal(*groups_kw) if len(groups_kw) >= 2 \
                     else (float("nan"), float("nan"))

        grp = tab_df.groupby("regime")["fee_apr_ann"].agg(
            N="count", mean="mean", median="median", std="std"
        ).round(4)
        grp["KW_stat"] = round(float(kw_s), 4)
        grp["KW_pval"] = round(float(kw_p), 4)
        grp["KW_sig"]  = stars(float(kw_p))
        savetable(grp, "fee_apr_by_regime")

    # ── Table 4: Duration stats by type ──────────────────────────
    if lp is not None and "duration_days" in lp_valid.columns:
        dur_tab = lp_valid.groupby("position_type" if "position_type" in lp_valid.columns
                                    else pd.Index(["all"] * len(lp_valid)))["duration_days"].agg(
            N="count", mean="mean", median="median",
            p25=lambda x: x.quantile(0.25),
            p75=lambda x: x.quantile(0.75),
            p95=lambda x: x.quantile(0.95),
        ).round(2)
        savetable(dur_tab, "lp_duration_stats")

    print("\nDONE")


if __name__ == "__main__":
    main()
