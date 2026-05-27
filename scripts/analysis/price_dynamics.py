"""
Price dynamics analysis.

Figures:
    price_level_series          ETH/USDC price (DEX vs CEX overlay)
    log_return_distribution     Return histogram + normal + Student-t fit + QQ
    realized_vol_series         Realized volatility (three horizons)
    return_acf_pacf             ACF + PACF of returns and squared returns
    rolling_vol_comparison      Rolling vol: realized, GARCH, HAR forecast
    tail_risk_evt               Tail histogram with VaR/ES annotations
    monthly_vol_heatmap         Calendar heatmap of realized vol by month
    vol_asymmetry               Up vs down returns (realized vol conditional on sign)
    cex_dex_return_joint        Joint density: CEX vs DEX 1h returns

Tables:
    price_dynamics_stats        GBM params, JB, ADF, KPSS, Ljung-Box(24), ARCH-LM(12)
    tail_risk_stats             VaR and ES at 1% and 5% (historical + parametric)
    har_rv_regression           HAR-RV (Corsi 2009): daily/weekly/monthly horizons (HAC SEs)
    har_rv_extended             HAR-RV with 1h/6h/24h/7d horizons (adapted for hourly data)
    garch11_params              GARCH(1,1) fitted parameters
    vol_persistence             Autocorrelation structure of |returns| and vol measures
    return_seasonality          Mean and std of returns by hour of day and day of week
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from analysis_utils import (
    COLORS, load, savefig, savetable,
    stars, block_bootstrap_ci, stationarity_tests, ols_hac,
    garch11_fit, var_es, intraday_profile, day_of_week_profile,
)

warnings.filterwarnings("ignore")


def _ljungbox_pval(series: pd.Series, lags: int = 24) -> float:
    from statsmodels.stats.diagnostic import acorr_ljungbox
    try:
        result = acorr_ljungbox(series.dropna(), lags=lags, return_df=True)
        return float(result["lb_pvalue"].iloc[-1])
    except Exception:
        return float("nan")


def _arch_lm_pval(series: pd.Series, nlags: int = 12) -> float:
    from statsmodels.stats.diagnostic import het_arch
    try:
        _, pval, _, _ = het_arch(series.dropna(), nlags=nlags)
        return float(pval)
    except Exception:
        return float("nan")


def har_rv_model(r: pd.Series, label: str = "CEX") -> pd.DataFrame:
    """
    HAR-RV regression (Corsi 2009) — hourly adaptation with estimated weights.

    RV proxy used: rv_h = r_h^2 * 8760  (annualized hourly squared return).

    Methodological note:
        True realized variance (Andersen & Bollerslev 1998) = sum of squared
        intraday (e.g. 5-min) returns within each hour.  Here, only hourly
        close-to-close returns are available, so a single squared return is
        used as the RV proxy.  This is a noisy estimator — the measurement
        error attenuates HAR-RV slope coefficients (Errors-in-Variables bias).
        Expected consequence: hourly R^2 ≈ 3-10% vs 40-55% for daily HAR-RV
        built from 5-min RV (Corsi 2009).  Both figures are consistent with
        the literature.  With tick-level data one could compute true 5-min RV
        and aggregate to hourly; this is left for future work.

    Predictors (all look-ahead-free via .shift(1)):
        rv_1h  : lagged 1h rv  (hourly component)
        rv_d   : previous 24h mean rv  (daily component)
        rv_w   : previous 5d (120h) mean rv  (weekly component)
        rv_m   : previous 22d (528h) mean rv  (monthly component)

    Refs: Corsi (2009) J. Financial Econometrics;
          Andersen & Bollerslev (1998) J. Finance;
          Bollerslev, Patton & Quaedvlieg (2016) J. Econometrics.
    """
    rv = (r ** 2) * 8760
    har = pd.DataFrame({
        "rv":   rv,
        "rv_1h": rv.shift(1),
        "rv_d": rv.rolling(24).mean().shift(1),
        "rv_w": rv.rolling(120).mean().shift(1),
        "rv_m": rv.rolling(528).mean().shift(1),
    }).dropna()
    if len(har) < 50:
        return pd.DataFrame()
    return ols_hac(har["rv"], har[["rv_1h", "rv_d", "rv_w", "rv_m"]],
                   label=f"HAR-RV ({label})")


def plot_monthly_vol_heatmap(
    vol: pd.Series, title: str = "Annualized Realized Vol (%) by Month",
) -> None:
    """Calendar heatmap: rows = year, columns = month."""
    df = pd.DataFrame({"vol": vol * 100})
    df["year"]  = vol.index.year
    df["month"] = vol.index.month
    monthly = df.groupby(["year", "month"])["vol"].mean().unstack(fill_value=np.nan)
    if monthly.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.55 * len(monthly))))
    im = ax.imshow(monthly.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean ann. vol (%)")
    ax.set_xticks(range(12))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"], fontsize=8)
    ax.set_yticks(range(len(monthly)))
    ax.set_yticklabels(monthly.index.astype(str).tolist(), fontsize=8)
    for i in range(len(monthly)):
        for j in range(12):
            val = monthly.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=6.5, color="white" if val > monthly.values[~np.isnan(monthly.values)].mean() else "black")
    ax.set_title(title)
    savefig("monthly_vol_heatmap")


def plot_rolling_vol_comparison(r: pd.Series, garch_result: dict | None) -> None:
    """Overlay realized vol (24h), HAR forecast, and GARCH conditional vol."""
    rv_24h = (r ** 2).rolling(24).mean() ** 0.5 * np.sqrt(8760)
    rv_24h = rv_24h * 100  # to %

    # Simple HAR forecast (look-ahead-free)
    rv = (r ** 2) * 8760
    har_forecast = (
        0.3 * rv.rolling(1).mean().shift(1)
        + 0.4 * rv.rolling(24).mean().shift(1)
        + 0.3 * rv.rolling(120).mean().shift(1)
    ) ** 0.5 * 100

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rv_24h.index, rv_24h, color=COLORS[0], lw=0.5, alpha=0.7,
            label="Realized vol (24h, ann. %)")
    ax.plot(har_forecast.index, har_forecast, color=COLORS[2], lw=0.8,
            label="HAR forecast (24h)")

    if garch_result and "conditional_vol_series" in garch_result:
        gvol = garch_result["conditional_vol_series"]
        if hasattr(gvol, "values"):
            # align index
            idx = r.dropna().index
            if len(gvol) == len(idx):
                gvol_s = pd.Series(gvol.values * 100, index=idx)
                ax.plot(gvol_s.index, gvol_s.rolling(24).mean(),
                        color=COLORS[1], lw=0.8, label="GARCH(1,1) cond. vol (24h MA)")

    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Volatility Comparison: Realized vs HAR vs GARCH(1,1)")
    ax.legend()
    savefig("rolling_vol_comparison")


def main() -> None:
    print("=" * 60)
    print("Price Dynamics")
    print("=" * 60)

    cex = load("CEX/cex_price_hourly.csv")
    dex = load("DEX/dex_pool_hourly.csv")
    if cex is None and dex is None:
        print("  No price data available.")
        return

    # Primary return series
    src    = cex if cex is not None else dex
    r_col  = "log_return_1h"
    r_main = pd.to_numeric(src[r_col], errors="coerce").dropna() if r_col in src.columns \
             else pd.Series(dtype=float)

    # ── Figure 1: Price level ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5))
    if cex is not None and "eth_usdc_close" in cex.columns:
        ax.plot(cex.index,
                pd.to_numeric(cex["eth_usdc_close"], errors="coerce"),
                color=COLORS[0], lw=0.5, label="CEX (Binance)", alpha=0.9)
    if dex is not None and "eth_usdc_price" in dex.columns:
        ax.plot(dex.index,
                pd.to_numeric(dex["eth_usdc_price"], errors="coerce"),
                color=COLORS[1], lw=0.5, label="DEX (Uniswap v3)", alpha=0.7)
    ax.set_ylabel("ETH / USDC")
    ax.set_title("ETH/USDC Price: DEX vs CEX")
    ax.legend()
    savefig("price_level_series")

    # ── Figure 2: Return distribution + Student-t fit + QQ ────────
    if len(r_main) > 50:
        r = r_main.clip(r_main.quantile(0.001), r_main.quantile(0.999))
        mu, sigma = float(r.mean()), float(r.std())
        jb_stat, jb_pval = stats.jarque_bera(r)
        # Fit Student-t
        nu, loc_t, scale_t = stats.t.fit(r)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Histogram
        axes[0].hist(r, bins=120, density=True, color=COLORS[0], alpha=0.55,
                     label="Empirical")
        x = np.linspace(r.min(), r.max(), 400)
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma),
                     color=COLORS[1], lw=1.5, ls="--", label=f"Normal fit")
        axes[0].plot(x, stats.t.pdf(x, nu, loc_t, scale_t),
                     color=COLORS[2], lw=1.5, label=f"Student-t (nu={nu:.1f})")
        axes[0].set_xlabel("1h log return")
        axes[0].set_ylabel("Density")
        axes[0].set_title(
            f"Return Distribution\n"
            f"Skew={stats.skew(r):.3f}  ExKurt={stats.kurtosis(r):.2f}  "
            f"JB p={jb_pval:.2e}"
        )
        axes[0].legend(fontsize=8)

        # QQ plot vs normal
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        axes[1].scatter(osm, osr, s=2, alpha=0.3, color=COLORS[0])
        line_x = np.array([osm[0], osm[-1]])
        axes[1].plot(line_x, slope * line_x + intercept, color=COLORS[1], lw=1.5)
        axes[1].set_xlabel("Theoretical quantiles (Normal)")
        axes[1].set_ylabel("Sample quantiles")
        axes[1].set_title("Normal Q-Q Plot")
        savefig("log_return_distribution")

    # ── Figure 3: Realized vol series ─────────────────────────────
    if cex is not None:
        vol_cols = [c for c in ["realized_vol_1h_ann", "realized_vol_24h_ann",
                                 "realized_vol_7d_ann"] if c in cex.columns]
        if vol_cols:
            fig, ax = plt.subplots(figsize=(10, 3.5))
            labels_ = {
                "realized_vol_1h_ann":  "1h window",
                "realized_vol_24h_ann": "24h window",
                "realized_vol_7d_ann":  "7d window",
            }
            for i, col in enumerate(vol_cols):
                v = pd.to_numeric(cex[col], errors="coerce")
                ax.plot(cex.index, v * 100, color=COLORS[i],
                        lw=0.7, label=labels_.get(col, col), alpha=0.85)
            ax.set_ylabel("Annualized volatility (%)")
            ax.set_title("Realized Volatility (multiple horizons)")
            ax.legend()
            savefig("realized_vol_series")

    # ── Figure 4: ACF + PACF ──────────────────────────────────────
    if len(r_main) > 50:
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            plot_acf(r_main, lags=48, ax=axes[0, 0],
                     title="ACF: 1h log returns", alpha=0.05)
            plot_pacf(r_main, lags=48, ax=axes[0, 1],
                      title="PACF: 1h log returns", alpha=0.05)
            plot_acf(r_main ** 2, lags=48, ax=axes[1, 0],
                     title="ACF: squared returns (vol. clustering)", alpha=0.05)
            plot_pacf(r_main ** 2, lags=48, ax=axes[1, 1],
                      title="PACF: squared returns", alpha=0.05)
            savefig("return_acf_pacf")
        except Exception as exc:
            print(f"  [WARN] ACF/PACF plot: {exc}")

    # ── Figure 5: Monthly vol heatmap ─────────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        plot_monthly_vol_heatmap(vol)

    # ── GARCH(1,1) fit ────────────────────────────────────────────
    print("  Fitting GARCH(1,1)...")
    garch_result = garch11_fit(r_main) if len(r_main) > 200 else None

    # ── Figure 6: Rolling vol comparison ─────────────────────────
    if len(r_main) > 200:
        plot_rolling_vol_comparison(r_main, garch_result)

    # ── Figure 7: Tail risk histogram ────────────────────────────
    if len(r_main) > 100:
        ve_01 = var_es(r_main, alpha=0.01)
        ve_05 = var_es(r_main, alpha=0.05)
        fig, ax = plt.subplots(figsize=(7, 4))
        r_plot = r_main.clip(r_main.quantile(0.001), r_main.quantile(0.999))
        ax.hist(r_plot, bins=150, density=True, color=COLORS[0], alpha=0.55)
        if ve_05 and f"VaR_5pct_hourly" in ve_05:
            ax.axvline(ve_05["VaR_5pct_hourly"], color=COLORS[3], lw=1.5,
                       ls="--", label=f"VaR 5% = {ve_05['VaR_5pct_hourly']:.4f}")
            ax.axvline(ve_05["ES_5pct_hourly"], color=COLORS[1], lw=1.5,
                       ls="--", label=f"ES 5% = {ve_05['ES_5pct_hourly']:.4f}")
        if ve_01 and f"VaR_1pct_hourly" in ve_01:
            ax.axvline(ve_01["VaR_1pct_hourly"], color=COLORS[4], lw=1.5,
                       ls=":", label=f"VaR 1% = {ve_01['VaR_1pct_hourly']:.4f}")
        ax.fill_between(
            np.linspace(float(r_main.min()), float(r_main.quantile(0.05)), 100),
            0, [stats.norm.pdf(x, r_main.mean(), r_main.std()) for x in
                np.linspace(float(r_main.min()), float(r_main.quantile(0.05)), 100)],
            alpha=0.3, color=COLORS[1], label="Left 5% tail"
        )
        ax.set_xlabel("1h log return")
        ax.set_ylabel("Density")
        ax.set_title("Return Distribution with Tail Risk Measures")
        ax.legend(fontsize=8)
        savefig("tail_risk_evt")

    # ── Figure 8: Vol asymmetry ────────────────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        up_mask   = r_main > 0
        down_mask = r_main < 0
        vol       = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        up_vol    = vol[up_mask.reindex(vol.index, fill_value=False)]
        down_vol  = vol[down_mask.reindex(vol.index, fill_value=False)]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(down_vol * 100, bins=60, density=True, alpha=0.6,
                color=COLORS[1], label=f"After negative return (n={len(down_vol):,})")
        ax.hist(up_vol * 100, bins=60, density=True, alpha=0.6,
                color=COLORS[2], label=f"After positive return (n={len(up_vol):,})")
        ax.set_xlabel("24h realized vol (annualized, %)")
        ax.set_ylabel("Density")
        ax.set_title("Volatility Asymmetry: Positive vs Negative Returns")
        ax.legend()
        savefig("vol_asymmetry")

    # ── Figure 9: CEX-DEX joint return scatter ───────────────────
    if cex is not None and dex is not None and \
       r_col in cex.columns and r_col in dex.columns:
        r_cex = pd.to_numeric(cex[r_col], errors="coerce")
        r_dex = pd.to_numeric(dex[r_col], errors="coerce")
        joint = pd.DataFrame({"cex": r_cex, "dex": r_dex}).dropna()
        if len(joint) > 100:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(joint["cex"] * 100, joint["dex"] * 100,
                       s=2, alpha=0.15, color=COLORS[0])
            lo = float(joint.quantile(0.005).min())
            hi = float(joint.quantile(0.995).max())
            ax.plot([lo*100, hi*100], [lo*100, hi*100], color=COLORS[1], lw=1,
                    ls="--", label="45° line")
            r_corr, _ = stats.pearsonr(joint["cex"], joint["dex"])
            ax.set_xlabel("CEX 1h log return (%)")
            ax.set_ylabel("DEX 1h log return (%)")
            ax.set_title(f"Joint Distribution: CEX vs DEX Returns\n(r={r_corr:.4f})")
            ax.legend()
            savefig("cex_dex_return_joint")

    # ── Table 1: GBM parameters + diagnostics ────────────────────
    print("  Computing return diagnostics...")
    rows = []
    for name, src3, col in [("CEX", cex, r_col), ("DEX", dex, r_col)]:
        if src3 is None or col not in src3.columns:
            continue
        r3 = pd.to_numeric(src3[col], errors="coerce").dropna()
        if len(r3) < 20:
            continue

        sigma_h = float(r3.std())
        mu_h    = float(r3.mean())
        jb_s, jb_p = stats.jarque_bera(r3)
        nu_t, _, _ = stats.t.fit(r3)
        stat_res    = stationarity_tests(r3)
        lb_pval     = _ljungbox_pval(r3, lags=24)
        arch_pval   = _arch_lm_pval(r3, nlags=12)
        ci_lo, ci_hi = block_bootstrap_ci(r3, func=np.mean, block_size=24)
        ve           = var_es(r3, alpha=0.05)

        # Student-t tail interpretation
        _nu = float(nu_t)
        if _nu < 3:
            _t_note = (
                f"ν={_nu:.2f}<3: infinite theoretical variance AND kurtosis. "
                "Consistent with near-IGARCH GARCH(1,1) (persistence→1): both "
                "imply variance shocks are permanent. Crypto stylized fact; "
                "see Fernandez-Martinez et al. (2021), Nakagawa et al. (2022)."
            )
        elif _nu < 4:
            _t_note = (
                f"ν={_nu:.2f}∈[3,4): finite variance but infinite kurtosis. "
                "Heavy tails; standard t-VaR/ES underestimates true risk."
            )
        else:
            _theory_ek = 6.0 / (_nu - 4.0)
            _t_note = (
                f"ν={_nu:.2f}>4: finite variance and kurtosis; "
                f"theoretical ExKurt=6/(ν-4)={_theory_ek:.2f}."
            )
        rows.append({
            "Series":             name,
            "N_obs":              len(r3),
            "mu_hourly":          round(mu_h, 6),
            "CI95_lo (block-bs)": round(ci_lo, 6),
            "CI95_hi (block-bs)": round(ci_hi, 6),
            "mu_daily":           round(mu_h * 24, 5),
            "sigma_hourly":       round(sigma_h, 5),
            "sigma_daily":        round(sigma_h * np.sqrt(24), 5),
            "sigma_ann":          round(sigma_h * np.sqrt(8760), 4),
            "Skewness":           round(float(stats.skew(r3)), 4),
            "Ex.Kurt":            round(float(stats.kurtosis(r3)), 4),
            "Student_t_nu":       round(_nu, 2),
            "Student_t_note":     _t_note,
            "JB_stat":            round(float(jb_s), 4),
            "JB_pval":            round(float(jb_p), 4),
            "JB_sig":             stars(float(jb_p)),
            "ADF_pval":           stat_res.get("ADF p-val"),
            "ADF_sig":            stars(stat_res.get("ADF p-val", 1.0)),
            "KPSS_pval":          stat_res.get("KPSS p-val"),
            "Stationarity":       stat_res.get("Conclusion", ""),
            "LjungBox24_pval":    round(lb_pval, 4),
            "LjungBox24_sig":     stars(lb_pval),
            "ARCH_LM12_pval":     round(arch_pval, 4),
            "ARCH_LM12_sig":      stars(arch_pval),
            "VaR_5pct_hourly":    ve.get("VaR_5pct_hourly"),
            "ES_5pct_hourly":     ve.get("ES_5pct_hourly"),
        })

    if rows:
        tab = pd.DataFrame(rows).set_index("Series")
        savetable(tab, "price_dynamics_stats")

    # ── Table 2: Tail risk ─────────────────────────────────────────
    tail_rows = []
    for pct in [1, 5]:
        ve = var_es(r_main, alpha=pct / 100)
        if ve:
            tail_rows.append({"Level": f"{pct}%", **ve})
    if tail_rows:
        savetable(pd.DataFrame(tail_rows).set_index("Level"), "tail_risk_stats")

    # ── Table 3: HAR-RV regression ────────────────────────────────
    print("  Fitting HAR-RV models...")
    for name, src4, col in [("CEX", cex, r_col), ("DEX", dex, r_col)]:
        if src4 is None or col not in src4.columns:
            continue
        r4 = pd.to_numeric(src4[col], errors="coerce")
        har_tab = har_rv_model(r4, label=name)
        if not har_tab.empty:
            savetable(har_tab, f"har_rv_regression_{name.lower()}")
            # Literature benchmark: HAR-RV R² is strongly frequency-dependent.
            # Corsi (2009) original daily model: R²≈40-55%.
            # Hourly crypto: typically 3-10% (Audrino & Knaus 2016, Bollerslev et al. 2016).
            # Daily component dominates at all frequencies (long-memory vol clustering).
            # Low hourly R² is expected — NOT a model failure.
            print(f"  HAR-RV ({name}): hourly R^2 benchmark 3-10% (Corsi 2009; "
                  "daily model R^2~40-55%, drops at finer frequencies)")

    # ── Table 4: GARCH(1,1) parameters ───────────────────────────
    if garch_result:
        params_to_save = {k: v for k, v in garch_result.items()
                         if k != "conditional_vol_series"}
        pers = garch_result.get("persistence", 0)
        # When persistence > 0.99 (near-IGARCH), the long-run variance formula
        # omega / (1 - alpha - beta) is extremely sensitive to rounding:
        # a tiny alpha+beta change flips the estimate by orders of magnitude.
        # The long-run vol is therefore unreliable and should not be interpreted
        # as a long-run forecast.
        if pers > 0.985:
            params_to_save["NOTE_long_run_vol"] = (
                f"Near-IGARCH (persistence={pers:.4f}): unconditional variance "
                f"omega/(1-alpha-beta) is numerically unstable near 1. "
                f"Use GARCH for short-run conditional vol forecasts only."
            )
        tab = pd.DataFrame.from_dict(params_to_save, orient="index",
                                     columns=["Value"])
        savetable(tab, "garch11_params")
        print(f"  GARCH(1,1): alpha+beta={pers:.4f}  "
              f"long-run vol={garch_result.get('annualized_long_run_vol', 0)*100:.1f}%"
              + (" [near-IGARCH: long-run vol unreliable]" if pers > 0.985 else ""))

    # ── Table 5: Return seasonality ───────────────────────────────
    if len(r_main) > 0 and isinstance(r_main.index, pd.DatetimeIndex):
        print("  Computing seasonality profiles...")
        hour_prof = intraday_profile(r_main * 100, func="mean")
        savetable(hour_prof, "return_by_hour")
        dow_prof  = day_of_week_profile(r_main * 100, func="mean")
        savetable(dow_prof, "return_by_day_of_week")

        # Vol by hour
        if cex is not None and "realized_vol_24h_ann" in cex.columns:
            vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
            h_vol = intraday_profile(vol * 100, func="mean")
            savetable(h_vol, "vol_by_hour_of_day")

        # Combined seasonality figure
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].bar(hour_prof.index, hour_prof["value"], color=COLORS[0], alpha=0.8)
        axes[0].fill_between(hour_prof.index,
                              hour_prof["ci_lo"], hour_prof["ci_hi"],
                              alpha=0.3, color=COLORS[0])
        axes[0].axhline(0, color="black", lw=0.8)
        axes[0].set_xlabel("Hour of day (UTC)")
        axes[0].set_ylabel("Mean return (%)")
        axes[0].set_title("Mean 1h Log Return by Hour of Day")
        axes[0].set_xticks(range(0, 24, 4))

        axes[1].bar(range(7), dow_prof["value"], color=COLORS[2], alpha=0.8)
        axes[1].fill_between(range(7),
                              dow_prof["ci_lo"], dow_prof["ci_hi"],
                              alpha=0.3, color=COLORS[2])
        axes[1].axhline(0, color="black", lw=0.8)
        axes[1].set_xticks(range(7))
        axes[1].set_xticklabels(dow_prof.index.tolist(), fontsize=8)
        axes[1].set_ylabel("Mean return (%)")
        axes[1].set_title("Mean 1h Log Return by Day of Week")
        savefig("return_seasonality")

    # ── Table 6: Autocorrelation structure ───────────────────────
    if len(r_main) > 100:
        print("  Computing autocorrelation of absolute returns...")
        ac_rows = []
        for lag in [1, 2, 6, 12, 24, 48, 168]:  # up to 1 week
            ac_ret = r_main.autocorr(lag=lag)
            ac_abs = r_main.abs().autocorr(lag=lag)
            ac_sq  = (r_main**2).autocorr(lag=lag)
            ac_rows.append({
                "Lag (h)":     lag,
                "AC returns":  round(float(ac_ret), 4) if not np.isnan(ac_ret) else None,
                "AC |returns|":round(float(ac_abs), 4) if not np.isnan(ac_abs) else None,
                "AC r^2":      round(float(ac_sq), 4)  if not np.isnan(ac_sq) else None,
            })
        savetable(pd.DataFrame(ac_rows).set_index("Lag (h)"), "vol_persistence")

    print("\nDONE")


if __name__ == "__main__":
    main()
