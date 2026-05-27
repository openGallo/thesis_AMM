"""
Volatility regime analysis.

Regimes defined by 24h annualized realized vol terciles:
    low    - bottom tercile
    normal - middle tercile
    high   - top tercile  (separately: stress = top decile)

Figures:
    regime_time_series          Price + vol coloured by regime
    regime_vol_distribution     Volatility distribution by regime
    regime_metrics_bar          Key metrics by regime (6-panel bar)
    stress_duration_hist        Histogram of stress episode lengths
    regime_transition_heatmap   Regime transition probability matrix
    garch_regime_overlay        GARCH conditional vol overlay on price
    regime_correlation_heatmap  Pearson correlation heatmap by regime
    vol_forecast_accuracy       HAR vs GARCH vs realized vol comparison

Tables:
    regime_stats                Key metrics by regime (mean, median, CI)
    regime_tests                KW test + pairwise Mann-Whitney + BH correction
    stress_events               Stress episodes (>90th pct vol) statistics
    regime_transition_matrix    Hour-to-hour regime transition probabilities
    regime_conditional_corr     Correlation structure by regime
    vol_forecast_comparison     RMSE + MAE of HAR, GARCH, naive forecasts
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as sp_stats

from analysis_utils import (
    COLORS, load, savefig, savetable,
    stars, block_bootstrap_ci, bh_correction, garch11_fit,
)


REGIME_COLORS = {"low": COLORS[2], "normal": COLORS[0], "high": COLORS[1]}


def assign_regimes(vol: pd.Series, stress_q: float = 0.90) -> pd.DataFrame:
    lo = vol.quantile(1 / 3)
    hi = vol.quantile(2 / 3)
    st = vol.quantile(stress_q)
    regime = pd.cut(vol, bins=[-np.inf, lo, hi, np.inf],
                    labels=["low", "normal", "high"])
    stress = vol > st
    return pd.DataFrame({"vol": vol, "regime": regime, "stress": stress})


def episode_lengths(flag: pd.Series) -> pd.Series:
    lengths, count = [], 0
    for v in flag:
        if v:
            count += 1
        elif count > 0:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return pd.Series(lengths, dtype=float)


def compute_transition_matrix(regime: pd.Series) -> pd.DataFrame:
    """Hour-to-hour empirical transition probability matrix."""
    labels = ["low", "normal", "high"]
    from_r = regime.iloc[:-1].values
    to_r   = regime.iloc[1:].values
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=float)
    for f, t in zip(from_r, to_r):
        if pd.notna(f) and pd.notna(t):
            mat.loc[f, t] += 1
    mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return mat.round(4)


def har_rv_forecast(r: pd.Series, h: int = 1) -> pd.Series:
    """
    HAR-RV one-step-ahead out-of-sample forecast using FIXED weights.

    RV proxy: rv_h = r_h^2 * 8760 (annualized hourly squared return).
    This is a noisy proxy for true realized variance.  True RV (Andersen &
    Bollerslev 1998) sums squared tick-level returns within each period; here
    we use a single hourly squared return as the tick data is unavailable.
    Hourly R^2 is therefore lower (3-10%) than daily-RV HAR (40-55%) — expected
    and consistent with Corsi (2009) and Busch, Christensen & Nielsen (2011).

    Fixed weights (0.3 / 0.4 / 0.3):
        Hourly / Daily(24h) / Weekly(120h=5d) components.
        Weights are NOT estimated from data — they are chosen to approximate
        the Corsi (2009) original cascade proportions (roughly 1/4 : 1/2 : 1/4
        for 1d/5d/22d at daily frequency, adapted here for hourly frequency).
        Consequence: these weights may not be optimal for this pool; they serve
        as a robust out-of-sample benchmark.  The estimated-weight HAR-RV
        (in price_dynamics.py) provides in-sample fit for comparison.

    All forecasts are look-ahead-free (shift(1) applied throughout).

    Refs: Corsi (2009) J. Financial Econometrics;
          Andersen & Bollerslev (1998) J. Finance;
          Busch, Christensen & Nielsen (2011) J. Econometrics.
    """
    rv  = (r ** 2) * 8760
    return (0.3 * rv.rolling(1).mean().shift(1)
            + 0.4 * rv.rolling(24).mean().shift(1)
            + 0.3 * rv.rolling(120).mean().shift(1))


def main() -> None:
    print("=" * 60)
    print("Volatility Regime Analysis")
    print("=" * 60)

    cex    = load("CEX/cex_price_hourly.csv")
    dex    = load("DEX/dex_pool_hourly.csv")
    lvr    = load("DEX/dex_lvr_hourly.csv")
    merged = load("merged/merged_hourly.csv")

    if cex is None or "realized_vol_24h_ann" not in cex.columns:
        print("  [SKIP] realized_vol_24h_ann not available")
        return

    vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce").dropna()
    reg = assign_regimes(vol)
    print(f"  Regime counts: {reg['regime'].value_counts().to_dict()}")
    print(f"  Stress hours (>90th pct): {reg['stress'].sum():,} ({reg['stress'].mean()*100:.1f}%)")

    # ── Figure 1: Price + vol coloured by regime ──────────────────
    price = pd.to_numeric(cex.get("eth_usdc_close"), errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    axes[0].plot(cex.index, price, color="black", lw=0.5, alpha=0.8)
    for regime_lbl, color in REGIME_COLORS.items():
        mask = (reg["regime"] == regime_lbl).reindex(cex.index, fill_value=False)
        axes[0].fill_between(cex.index, price.min(), price.max(),
                             where=mask, color=color, alpha=0.12)
    axes[0].set_ylabel("ETH/USDC")
    axes[0].set_title("ETH/USDC Price with Volatility Regime Shading")

    axes[1].plot(cex.index, vol.reindex(cex.index) * 100, color="black", lw=0.5, alpha=0.7)
    for regime_lbl, color in REGIME_COLORS.items():
        mask = (reg["regime"] == regime_lbl).reindex(cex.index, fill_value=False)
        axes[1].fill_between(cex.index, 0,
                             float((vol.reindex(cex.index) * 100).max()),
                             where=mask, color=color, alpha=0.25)
    axes[1].set_ylabel("Realized vol 24h (ann., %)")
    patches = [mpatches.Patch(color=c, alpha=0.6, label=r.capitalize())
               for r, c in REGIME_COLORS.items()]
    axes[1].legend(handles=patches, loc="upper right")
    savefig("regime_time_series")

    # ── Figure 2: Vol distribution by regime ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    for regime_lbl, color in REGIME_COLORS.items():
        sub = reg[reg["regime"] == regime_lbl]["vol"].dropna() * 100
        ax.hist(sub, bins=60, density=True, color=color, alpha=0.55,
                label=f"{regime_lbl.capitalize()} (n={len(sub):,})")
    ax.set_xlabel("24h realized vol (annualized, %)")
    ax.set_ylabel("Density")
    ax.set_title("Volatility Distribution by Regime")
    ax.legend()
    savefig("regime_vol_distribution")

    # ── Collect metrics per regime ─────────────────────────────────
    metrics: dict[str, pd.Series] = {"vol_24h_ann": vol}
    if dex is not None:
        for col in ["fee_apr_ann", "vol_over_tvl", "tx_count", "tvl_usd", "log_return_1h"]:
            if col in dex.columns:
                metrics[col] = pd.to_numeric(dex[col], errors="coerce")
    if lvr is not None:
        for col in ["lvr_rate_ann", "lvr_to_fee_ratio"]:
            if col in lvr.columns:
                metrics[col] = pd.to_numeric(lvr[col], errors="coerce")
    if merged is not None and "dex_cex_basis_bps" in merged.columns:
        metrics["basis_abs_bps"] = pd.to_numeric(
            merged["dex_cex_basis_bps"], errors="coerce").abs()

    aligned = pd.DataFrame(metrics)
    aligned["regime"] = reg["regime"].reindex(aligned.index)
    aligned["stress"] = reg["stress"].reindex(aligned.index)
    aligned = aligned.dropna(subset=["regime"])

    # ── Figure 3: Key metrics by regime ───────────────────────────
    bar_metrics = {k: v for k, v in {
        "fee_apr_ann":   "Fee APR (ann.)",
        "lvr_rate_ann":  "LVR rate (ann.)",
        "tx_count":      "Swap count / hour",
        "basis_abs_bps": "|Basis| (bps)",
        "vol_over_tvl":  "Volume / TVL",
        "tvl_usd":       "TVL (USD)",
    }.items() if k in aligned.columns}

    if bar_metrics:
        ncols = len(bar_metrics)
        fig, axes_bar = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4))
        if ncols == 1:
            axes_bar = [axes_bar]
        for ax, (col, label) in zip(axes_bar, bar_metrics.items()):
            grp = aligned.groupby("regime", observed=True)[col].median()
            ci_bars = []
            for r_lbl in grp.index:
                vals = aligned[aligned["regime"] == r_lbl][col].dropna().values
                lo, hi = block_bootstrap_ci(vals, func=np.median, block_size=24, n_boot=500)
                ci_bars.append(hi - lo)
            colors_bar = [REGIME_COLORS.get(r, COLORS[0]) for r in grp.index]
            ax.bar(grp.index, grp.values, color=colors_bar, alpha=0.8, yerr=ci_bars,
                   capsize=4, error_kw={"linewidth": 1})
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=15, labelsize=8)
        fig.suptitle("Median Key Metrics by Volatility Regime (with 95% CI)")
        savefig("regime_metrics_bar")

    # ── Figure 4: Stress episode duration ────────────────────────
    stress_mask = reg["stress"]
    ep_lens     = episode_lengths(stress_mask)
    if len(ep_lens) > 0:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(ep_lens, bins=min(40, len(ep_lens)), color=COLORS[1], alpha=0.75)
        ax.set_xlabel("Stress episode length (hours)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Stress Episode Duration (>90th pct vol)\n"
            f"N={len(ep_lens)} episodes, "
            f"mean={ep_lens.mean():.1f}h, max={ep_lens.max():.0f}h"
        )
        savefig("stress_duration_hist")

    # ── Figure 5: Regime transition heatmap ──────────────────────
    trans_mat = compute_transition_matrix(reg["regime"].dropna())
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(trans_mat.values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Transition prob.")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    labels_t = ["Low", "Normal", "High"]
    ax.set_xticklabels(labels_t); ax.set_yticklabels(labels_t)
    ax.set_xlabel("Next regime"); ax.set_ylabel("Current regime")
    ax.set_title("Regime Transition Probabilities (hourly)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{trans_mat.iloc[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color="white" if trans_mat.iloc[i,j] > 0.5 else "black")
    savefig("regime_transition_heatmap")
    savetable(trans_mat, "regime_transition_matrix")

    # ── Figure 6: GARCH conditional vol overlay ───────────────────
    if "log_return_1h" in cex.columns:
        r_cex = pd.to_numeric(cex["log_return_1h"], errors="coerce")
        print("  Fitting GARCH(1,1) for regime overlay...")
        garch_res = garch11_fit(r_cex)
        if garch_res and "conditional_vol_series" in garch_res:
            gvol = garch_res["conditional_vol_series"]
            idx  = r_cex.dropna().index
            if len(gvol) == len(idx):
                gvol_s = pd.Series(gvol.values * 100, index=idx)
                fig, ax = plt.subplots(figsize=(11, 4))
                ax.plot(cex.index, vol.reindex(cex.index) * 100, color=COLORS[0],
                        lw=0.4, alpha=0.6, label="Realized vol (24h, ann. %)")
                ax.plot(gvol_s.index, gvol_s.rolling(24).mean(),
                        color=COLORS[1], lw=1.0,
                        label="GARCH(1,1) cond. vol (24h MA, ann. %)")
                # Shade regime
                for regime_lbl, color in REGIME_COLORS.items():
                    mask = (reg["regime"] == regime_lbl).reindex(cex.index, fill_value=False)
                    ax.fill_between(cex.index, 0,
                                    float((vol.reindex(cex.index)*100).max()),
                                    where=mask, color=color, alpha=0.08)
                ax.set_ylabel("Annualized volatility (%)")
                ax.set_title("GARCH(1,1) Conditional Volatility vs Realized (by Regime)")
                ax.legend(fontsize=8)
                savefig("garch_regime_overlay")

    # ── Figure 7: Correlation heatmap by regime ────────────────────
    corr_cols = [c for c in metrics if c in aligned.columns and c != "vol_24h_ann"]
    if len(corr_cols) >= 3:
        fig, axes_c = plt.subplots(1, 3, figsize=(14, 5))
        for ax_c, regime_lbl in zip(axes_c, ["low", "normal", "high"]):
            sub = aligned[aligned["regime"] == regime_lbl][corr_cols].dropna()
            if len(sub) < 10:
                continue
            corr_sub = sub.corr(method="pearson")
            im2 = ax_c.imshow(corr_sub.values, cmap="RdBu_r", vmin=-1, vmax=1)
            ax_c.set_xticks(range(len(corr_cols)))
            ax_c.set_yticks(range(len(corr_cols)))
            ax_c.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=7)
            ax_c.set_yticklabels(corr_cols, fontsize=7)
            ax_c.set_title(f"{regime_lbl.capitalize()} vol regime (n={len(sub):,})")
            for i in range(len(corr_cols)):
                for j in range(len(corr_cols)):
                    ax_c.text(j, i, f"{corr_sub.iloc[i,j]:.2f}",
                              ha="center", va="center", fontsize=5.5,
                              color="white" if abs(corr_sub.iloc[i,j]) > 0.6 else "black")
        plt.colorbar(im2, ax=axes_c.ravel().tolist(), shrink=0.6, label="Pearson corr.")
        fig.suptitle("Correlation Structure by Volatility Regime")
        savefig("regime_correlation_heatmap")

        # Table: regime-conditional correlations
        corr_dfs = {}
        for regime_lbl in ["low", "normal", "high"]:
            sub = aligned[aligned["regime"] == regime_lbl][corr_cols].dropna()
            if len(sub) >= 5:
                corr_dfs[regime_lbl] = sub.corr().round(4)
                savetable(corr_dfs[regime_lbl], f"regime_corr_{regime_lbl}")

    # ── Figure 8: Vol forecast accuracy ──────────────────────────
    if "log_return_1h" in cex.columns:
        r_cex = pd.to_numeric(cex["log_return_1h"], errors="coerce")
        rv    = (r_cex ** 2) * 8760  # annualized realized variance

        har  = har_rv_forecast(r_cex)
        naive = rv.shift(1)  # naive: today's RV = yesterday's

        # GARCH forecast if available
        if "garch_res" in dir() and garch_res and "conditional_vol_series" in garch_res:
            gvol = garch_res["conditional_vol_series"]
            idx  = r_cex.dropna().index
            if len(gvol) == len(idx):
                # conditional_vol_series is already annualized vol (e.g. 0.70 = 70%).
                # Annualized variance = vol^2.  Do NOT multiply by 8760 again —
                # that would give vol^2 × 8760 ≈ 4 000, vs actual rv ≈ 0.5.
                garch_rv_hat = pd.Series(gvol.values ** 2, index=idx)
            else:
                garch_rv_hat = None
        else:
            garch_rv_hat = None

        fcst_df = pd.DataFrame({"actual": rv, "har": har, "naive": naive})
        if garch_rv_hat is not None:
            fcst_df["garch"] = garch_rv_hat.reindex(fcst_df.index)
        fcst_df = fcst_df.dropna()

        rows_f = []
        for model in ["har", "naive"] + (["garch"] if "garch" in fcst_df else []):
            e = fcst_df["actual"] - fcst_df[model]
            rows_f.append({
                "Model":   model.upper(),
                "RMSE":    round(float(np.sqrt((e**2).mean())), 6),
                "MAE":     round(float(e.abs().mean()), 6),
                "ME":      round(float(e.mean()), 6),
                "R2_OOS":  round(1 - float((e**2).sum()) / float(
                    ((fcst_df["actual"] - fcst_df["actual"].mean())**2).sum()), 4),
            })
        if rows_f:
            savetable(pd.DataFrame(rows_f).set_index("Model"), "vol_forecast_comparison")

            # Figure: actual vs HAR
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(fcst_df.index, (fcst_df["actual"]**0.5) * 100, color=COLORS[0],
                    lw=0.5, alpha=0.7, label="Realized vol (ann., %)")
            ax.plot(fcst_df.index, (fcst_df["har"]**0.5) * 100, color=COLORS[2],
                    lw=0.8, label="HAR forecast")
            if "garch" in fcst_df:
                ax.plot(fcst_df.index, (fcst_df["garch"]**0.5) * 100, color=COLORS[1],
                        lw=0.8, ls="--", label="GARCH(1,1)")
            ax.set_ylabel("Annualized vol (%)")
            ax.set_title("Volatility Forecast Comparison: HAR vs GARCH vs Realized")
            ax.legend(fontsize=8)
            savefig("vol_forecast_accuracy")

    # ── Table 1: Full regime stats ─────────────────────────────────
    rows = []
    for regime_lbl in ["low", "normal", "high"]:
        sub = aligned[aligned["regime"] == regime_lbl]
        row: dict = {"Regime": regime_lbl, "N_hours": len(sub)}
        for col in metrics:
            if col in sub.columns:
                vals = sub[col].dropna().values
                ci_lo, ci_hi = block_bootstrap_ci(vals, func=np.mean,
                                                   block_size=24, n_boot=500)
                row[f"{col}_mean"]    = round(float(sub[col].mean()), 6)
                row[f"{col}_CI95_lo"] = round(ci_lo, 6)
                row[f"{col}_CI95_hi"] = round(ci_hi, 6)
                row[f"{col}_median"]  = round(float(sub[col].median()), 6)
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("Regime")
    savetable(tab, "regime_stats")

    # ── Table 2: KW + pairwise MW + BH correction ─────────────────
    print("  Running nonparametric tests...")
    test_rows = []
    pairs = [("low", "normal"), ("low", "high"), ("normal", "high")]
    for col in [c for c in metrics if c != "vol_24h_ann"]:
        if col not in aligned.columns:
            continue
        groups = {
            r: aligned[aligned["regime"] == r][col].dropna()
            for r in ["low", "normal", "high"]
        }
        groups = {r: g for r, g in groups.items() if len(g) >= 5}
        if len(groups) < 2:
            continue
        kw_s, kw_p = sp_stats.kruskal(*groups.values())
        test_rows.append({
            "Metric": col, "Test": "Kruskal-Wallis",
            "Stat": round(float(kw_s), 4), "p_val": round(float(kw_p), 4),
            "Sig": stars(float(kw_p)), "BH_reject": "",
        })
        mw_pvals = []
        mw_pending = []
        for r1, r2 in pairs:
            if r1 not in groups or r2 not in groups:
                continue
            u, p = sp_stats.mannwhitneyu(groups[r1], groups[r2], alternative="two-sided")
            mw_pvals.append(p)
            mw_pending.append({
                "Metric": col, "Test": f"Mann-Whitney: {r1} vs {r2}",
                "Stat": round(float(u), 1), "p_val": round(float(p), 4),
                "Sig": stars(float(p)),
            })
        bh_flags = bh_correction(mw_pvals) if mw_pvals else []
        for row_r, rej in zip(mw_pending, bh_flags):
            row_r["BH_reject"] = "Yes" if rej else "No"
        test_rows.extend(mw_pending)

    if test_rows:
        savetable(pd.DataFrame(test_rows).set_index(["Metric", "Test"]), "regime_tests")

    # ── Table 3: Stress events ────────────────────────────────────
    stress_vol = vol[stress_mask.reindex(vol.index, fill_value=False)]
    stress_sub = aligned[aligned["stress"]]
    ci_lo_s = ci_hi_s = None
    if len(stress_vol) > 10:
        ci_lo_s, ci_hi_s = block_bootstrap_ci(stress_vol.values, func=np.mean,
                                               block_size=24, n_boot=500)
    st_tab = pd.DataFrame([{
        "Total stress hours":         int(stress_mask.sum()),
        "Stress fraction":            f"{stress_mask.mean()*100:.1f}%",
        "Vol threshold (90th pct)":   f"{vol.quantile(0.90)*100:.1f}%",
        "N stress episodes":          len(ep_lens),
        "Mean episode length (h)":    round(float(ep_lens.mean()), 1) if len(ep_lens) > 0 else None,
        "Median episode length (h)":  round(float(ep_lens.median()), 1) if len(ep_lens) > 0 else None,
        "Max episode length (h)":     int(ep_lens.max()) if len(ep_lens) > 0 else None,
        "Mean vol in stress":         f"{stress_vol.mean()*100:.1f}%",
        "Mean vol CI95_lo":           f"{ci_lo_s*100:.1f}%" if ci_lo_s else "--",
        "Mean vol CI95_hi":           f"{ci_hi_s*100:.1f}%" if ci_hi_s else "--",
        "Mean fee APR in stress":
            (f"{stress_sub['fee_apr_ann'].mean()*100:.2f}%"
             if "fee_apr_ann" in stress_sub.columns and len(stress_sub) > 0 else "--"),
        "Mean LVR rate in stress":
            (f"{stress_sub['lvr_rate_ann'].mean()*100:.4f}%"
             if "lvr_rate_ann" in stress_sub.columns and len(stress_sub) > 0 else "--"),
    }], index=["Value"]).T
    savetable(st_tab, "stress_events")

    print(f"\n  Stress episodes: {len(ep_lens)}")
    print(f"  Vol threshold (90th pct): {vol.quantile(0.90)*100:.1f}%")
    print("\nDONE")


if __name__ == "__main__":
    main()
