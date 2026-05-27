"""
DEX-CEX price basis analysis.

Figures:
    basis_time_series       Basis (bps) time series with arbitrage threshold
    basis_distribution      Basis histogram + fitted normal + Student-t
    basis_acf               ACF of basis and |basis| (half-life)
    basis_vs_gas            Basis magnitude vs gas price scatter
    basis_vs_vol            Basis magnitude vs realized volatility scatter
    basis_intraday          Mean |basis| by hour of day (UTC)
    basis_day_of_week       Mean |basis| by day of week
    basis_gas_quintile      |Basis| CDF by gas-price quintile
    basis_tvl_quartile      |Basis| by TVL quartile

Tables:
    basis_stats             Full summary stats + JB + ADF + KPSS + ARCH-LM + half-life
    basis_ar1               AR(1) with HAC SEs (persistence, half-life)
    basis_cointegration     Engle-Granger cointegration test (DEX vs CEX price)
    basis_vecm              VECM: speed of adjustment + long-run coefficient
    basis_determinants      OLS |basis| ~ gas + vol + hour + day_of_week + vol×gas (HAC)
    basis_by_regime         |Basis| by volatility tercile (mean, median, KW test)
    basis_by_gas_quintile   |Basis| statistics by gas-price quintile
    basis_by_tvl_quartile   |Basis| statistics by pool-TVL quartile
    basis_breakeven         Gas cost breakeven analysis: arbitrage profitability
    arbitrage_episodes      Arbitrage episode statistics
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from analysis_utils import (
    DATA_PROC, COLORS, load, load_swaps, savefig, savetable,
    stars, block_bootstrap_ci, stationarity_tests, ols_hac,
    vol_regime, intraday_profile, day_of_week_profile,
)

warnings.filterwarnings("ignore")

ARB_THRESHOLD = 5.0   # basis points - matches 0.05% pool fee


def _arch_lm_pval(series: pd.Series, nlags: int = 12) -> float:
    from statsmodels.stats.diagnostic import het_arch
    try:
        _, pval, _, _ = het_arch(series.dropna(), nlags=nlags)
        return float(pval)
    except Exception:
        return float("nan")


def _engle_granger_cointegration(p1: pd.Series, p2: pd.Series) -> dict:
    from statsmodels.tsa.stattools import coint
    aligned = pd.DataFrame({"p1": p1, "p2": p2}).dropna()
    if len(aligned) < 50:
        return {}
    try:
        eg_stat, eg_pval, eg_crit = coint(aligned["p1"], aligned["p2"])
        return {
            "N":             len(aligned),
            "EG_stat":       round(float(eg_stat), 4),
            "EG_pval":       round(float(eg_pval), 4),
            "EG_sig":        stars(float(eg_pval)),
            "Crit_1pct":     round(float(eg_crit[0]), 4),
            "Crit_5pct":     round(float(eg_crit[1]), 4),
            "Crit_10pct":    round(float(eg_crit[2]), 4),
            "Cointegrated":  "Yes" if eg_pval < 0.05 else "No",
            # Structural break caveat:
            # The Engle-Granger test (like ADF) loses power in the presence of
            # structural breaks (Gregory & Hansen 1996; Perron 1989).
            # Breaks known in this dataset: EIP-1559 (Aug 2021), The Merge
            # (Sep 2022), FTX collapse (Nov 2022) — each may shift the
            # long-run equilibrium between DEX and CEX prices.
            # If cointegration is rejected, this may reflect break-driven
            # instability in the long-run relationship rather than true
            # non-cointegration.
            # Recommendation: Gregory-Hansen (1996) cointegration test with
            # regime shift as a robustness check.
            # Refs: Gregory & Hansen (1996) J. Econometrics;
            #       Perron (1989) Econometrica.
            "Structural_break_note": (
                "EG test loses power under structural breaks "
                "(EIP-1559 Aug-2021, The Merge Sep-2022, FTX Nov-2022). "
                "Rejection may reflect break-induced instability, not true "
                "non-cointegration. Robustness: Gregory-Hansen (1996) test. "
                "Ref: Gregory & Hansen (1996) J. Econometrics."
            ),
        }
    except Exception as exc:
        return {"Error": str(exc)}


def _fit_vecm(dex_price: pd.Series, cex_price: pd.Series) -> dict:
    """
    Fit a VECM(1) by Engle-Granger two-step method.
    Step 1: OLS DEX_price = a + b×CEX_price -> get residual (EC term)
    Step 2: HAC OLS Δ(DEX) = c + α×EC_{t-1} + lags
    Returns speed-of-adjustment coefficient α and its SE.
    """
    try:
        import statsmodels.api as sm
        df = pd.DataFrame({"dex": dex_price, "cex": cex_price}).dropna()
        # Step 1: long-run relationship
        X1 = sm.add_constant(df["cex"])
        res1 = sm.OLS(df["dex"], X1).fit()
        ec_term = res1.resid.shift(1)
        # Step 2: ECM
        d_dex = df["dex"].diff()
        d_cex = df["cex"].diff()
        ecm_df = pd.DataFrame({
            "d_dex": d_dex,
            "ec":    ec_term,
            "d_cex": d_cex,
        }).dropna()
        X2 = sm.add_constant(ecm_df[["ec", "d_cex"]])
        res2 = sm.OLS(ecm_df["d_dex"], X2).fit()
        max_lags = int(np.floor(4 * (len(ecm_df) / 100) ** (2 / 9)))
        hac = res2.get_robustcov_results(cov_type="HAC", maxlags=max_lags)
        # Use positional indexing: get_robustcov_results may return params as ndarray
        param_names = list(hac.model.exog_names)
        ec_idx  = param_names.index("ec")
        cex_idx = list(res1.model.exog_names).index("cex")
        _p  = np.asarray(hac.params)
        _b  = np.asarray(hac.bse)
        _t  = np.asarray(hac.tvalues)
        _pv = np.asarray(hac.pvalues)
        alpha = float(_p[ec_idx])
        se_a  = float(_b[ec_idx])
        t_a   = float(_t[ec_idx])
        p_a   = float(_pv[ec_idx])
        lr_b  = float(np.asarray(res1.params)[cex_idx])
        half_life = -np.log(2) / np.log(1 + alpha) if -1 < alpha < 0 else float("nan")
        return {
            "N":               len(ecm_df),
            "LR_coef_beta":    round(lr_b, 6),
            "EC_alpha":        round(alpha, 6),
            "EC_alpha_se_HAC": round(se_a, 6),
            "EC_alpha_t":      round(t_a, 3),
            "EC_alpha_pval":   round(p_a, 4),
            "EC_alpha_sig":    stars(p_a),
            "half_life_hours": round(half_life, 2) if not np.isnan(half_life) else None,
        }
    except Exception as exc:
        import traceback
        print(f"  [VECM ERROR] {exc}")
        traceback.print_exc()
        return {"Error": str(exc)}


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


def main() -> None:
    print("=" * 60)
    print("DEX-CEX Basis Analysis")
    print("=" * 60)

    merged = load("merged/merged_hourly.csv")
    cex    = load("CEX/cex_price_hourly.csv")
    dex    = load("DEX/dex_pool_hourly.csv")

    if merged is None:
        print("  merged_hourly.csv not found - run process_merged_panel.py first")
        return

    basis = pd.to_numeric(merged.get("dex_cex_basis_bps"), errors="coerce").dropna()
    if len(basis) == 0:
        print("  dex_cex_basis_bps column empty.")
        return

    print(f"  Basis observations: {len(basis):,}")
    print(f"  Mean: {basis.mean():.3f} bps   Std: {basis.std():.3f} bps")
    print(f"  |basis| > {ARB_THRESHOLD} bps: {(basis.abs() > ARB_THRESHOLD).mean()*100:.1f}%")

    # ── Figure 1: Basis time series ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(basis.index, basis, color=COLORS[0], lw=0.4, alpha=0.7)
    ax.axhline(0, color="black", lw=0.8)
    for sign, label in [(1, f"+{ARB_THRESHOLD} bps"), (-1, "")]:
        ax.axhline(sign * ARB_THRESHOLD, color=COLORS[1], lw=1.0, ls="--",
                   label=label or None)
    ax.fill_between(basis.index, basis,
                    where=basis.abs() > ARB_THRESHOLD,
                    color=COLORS[1], alpha=0.15, label="Arbitrage zone")
    ax.set_ylabel("DEX-CEX basis (bps)")
    ax.set_title("DEX-CEX Price Basis (Uniswap v3 vs Binance)")
    ax.legend(fontsize=8)
    savefig("basis_time_series")

    # ── Figure 2: Basis distribution ──────────────────────────────
    basis_c = basis.clip(*basis.quantile([0.005, 0.995]))
    mu, sigma = float(basis_c.mean()), float(basis_c.std())
    nu_t, loc_t, scale_t = stats.t.fit(basis_c)
    jb_stat, jb_pval = stats.jarque_bera(basis_c)
    x = np.linspace(basis_c.min(), basis_c.max(), 400)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(basis_c, bins=100, density=True, color=COLORS[0], alpha=0.55,
            label="Empirical")
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS[1], lw=1.5,
            ls="--", label="Normal fit")
    ax.plot(x, stats.t.pdf(x, nu_t, loc_t, scale_t), color=COLORS[2], lw=1.5,
            label=f"Student-t (nu={nu_t:.1f})")
    for sign in [1, -1]:
        ax.axvline(sign * ARB_THRESHOLD, color=COLORS[3], lw=1.2, ls="--")
    ax.set_xlabel("Basis (bps)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"DEX-CEX Basis Distribution\n"
        f"Skew={basis.skew():.3f}  ExKurt={basis.kurtosis():.2f}  "
        f"JB p={jb_pval:.2e}"
    )
    ax.legend(fontsize=8)
    savefig("basis_distribution")

    # ── Figure 3: Basis ACF ────────────────────────────────────────
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        plot_acf(basis, lags=48, ax=axes[0],
                 title="ACF: DEX-CEX Basis", alpha=0.05)
        plot_acf(basis.abs(), lags=48, ax=axes[1],
                 title="ACF: |DEX-CEX Basis| (arbitrage persistence)", alpha=0.05)
        savefig("basis_acf")
    except Exception as exc:
        print(f"  [WARN] ACF plot: {exc}")

    # ── Gas price: load hourly median ────────────────────────────
    gas_hourly = pd.Series(dtype=float)
    swaps = load_swaps(cols=["timestamp", "gas_price_wei"])
    if swaps is not None:
        gas_ts = (swaps.dropna(subset=["timestamp"])
                  .set_index("timestamp")["gas_gwei"]
                  .resample("1h").median())
        gas_hourly = gas_ts

    # ── Figure 4: Basis vs gas price ──────────────────────────────
    if len(gas_hourly) > 0:
        combined = pd.DataFrame(
            {"basis_abs": basis.abs(), "gas_gwei": gas_hourly}).dropna()
        if len(combined) > 100:
            q99_g = combined["gas_gwei"].quantile(0.98)
            q99_b = combined["basis_abs"].quantile(0.98)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(combined["gas_gwei"].clip(upper=q99_g),
                       combined["basis_abs"].clip(upper=q99_b),
                       s=3, alpha=0.2, color=COLORS[0])
            # Add Lowess trend
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                sub = combined[combined["gas_gwei"] <= q99_g].copy()
                sub = sub.sort_values("gas_gwei")
                lw = lowess(sub["basis_abs"].clip(upper=q99_b), sub["gas_gwei"], frac=0.3)
                ax.plot(lw[:, 0], lw[:, 1], color=COLORS[1], lw=2, label="Lowess trend")
                ax.legend()
            except Exception:
                pass
            ax.set_xlabel("Median gas price (Gwei, hourly)")
            ax.set_ylabel("|DEX-CEX basis| (bps)")
            ax.set_title("Absolute Basis vs Gas Price")
            savefig("basis_vs_gas")

    # ── Figure 5: Basis vs vol ─────────────────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        combined = pd.DataFrame(
            {"basis_abs": basis.abs(), "vol": vol}).dropna()
        if len(combined) > 100:
            q99_v = combined["vol"].quantile(0.98)
            q99_b = combined["basis_abs"].quantile(0.98)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(combined["vol"].clip(upper=q99_v) * 100,
                       combined["basis_abs"].clip(upper=q99_b),
                       s=3, alpha=0.2, color=COLORS[2])
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                sub = combined[combined["vol"] <= q99_v].copy()
                sub = sub.sort_values("vol")
                lw = lowess(sub["basis_abs"].clip(upper=q99_b), sub["vol"] * 100, frac=0.3)
                ax.plot(lw[:, 0], lw[:, 1], color=COLORS[1], lw=2, label="Lowess trend")
                ax.legend()
            except Exception:
                pass
            ax.set_xlabel("Realized vol 24h (annualized, %)")
            ax.set_ylabel("|DEX-CEX basis| (bps)")
            ax.set_title("Absolute Basis vs Realized Volatility")
            savefig("basis_vs_vol")

    # ── Figure 6: Intraday seasonality ────────────────────────────
    basis_abs_h = basis.abs()
    hour_prof = intraday_profile(basis_abs_h, func="mean")
    dow_prof  = day_of_week_profile(basis_abs_h, func="mean")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(hour_prof.index, hour_prof["value"], color=COLORS[0], alpha=0.8)
    axes[0].fill_between(hour_prof.index,
                          hour_prof["ci_lo"], hour_prof["ci_hi"],
                          alpha=0.3, color=COLORS[0])
    axes[0].set_xlabel("Hour of day (UTC)")
    axes[0].set_ylabel("Mean |basis| (bps)")
    axes[0].set_title("Intraday Seasonality: |Basis|")
    axes[0].set_xticks(range(0, 24, 4))

    axes[1].bar(range(7), dow_prof["value"], color=COLORS[2], alpha=0.8)
    axes[1].fill_between(range(7), dow_prof["ci_lo"], dow_prof["ci_hi"],
                          alpha=0.3, color=COLORS[2])
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(dow_prof.index.tolist(), fontsize=8)
    axes[1].set_ylabel("Mean |basis| (bps)")
    axes[1].set_title("Day-of-Week Seasonality: |Basis|")
    savefig("basis_intraday")
    savetable(hour_prof, "basis_intraday")
    savetable(dow_prof, "basis_day_of_week")

    # ── Figure 7: Basis by gas-price quintile ─────────────────────
    if len(gas_hourly) > 0:
        combined = pd.DataFrame(
            {"basis_abs": basis.abs(), "gas_gwei": gas_hourly}).dropna()
        if len(combined) > 100:
            combined["gas_quintile"] = pd.qcut(
                combined["gas_gwei"], q=5,
                labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])
            fig, ax = plt.subplots(figsize=(7, 4))
            for q_lbl, sub in combined.groupby("gas_quintile", observed=True):
                s = sub["basis_abs"].clip(upper=sub["basis_abs"].quantile(0.995))
                s = s.sort_values()
                ax.plot(s.values, np.linspace(0, 1, len(s)), lw=1.5,
                        label=str(q_lbl))
            ax.axvline(ARB_THRESHOLD, color="black", lw=0.8, ls="--",
                       label=f"{ARB_THRESHOLD} bps fee")
            ax.set_xscale("log")
            ax.set_xlabel("|Basis| (bps, log scale)")
            ax.set_ylabel("CDF")
            ax.set_title("|Basis| CDF by Gas-Price Quintile")
            ax.legend(fontsize=8)
            savefig("basis_gas_quintile")

            gas_q_tab = combined.groupby("gas_quintile", observed=True)["basis_abs"].agg(
                N="count", mean="mean", median="median",
                pct_above=lambda x: (x > ARB_THRESHOLD).mean() * 100,
                p95=lambda x: x.quantile(0.95),
            ).round(3)
            savetable(gas_q_tab, "basis_by_gas_quintile")

    # ── Figure 8: Basis by TVL quartile ───────────────────────────
    if dex is not None and "tvl_usd" in dex.columns:
        tvl = pd.to_numeric(dex["tvl_usd"], errors="coerce")
        combined = pd.DataFrame(
            {"basis_abs": basis.abs(), "tvl": tvl}).dropna()
        if len(combined) > 100:
            combined["tvl_quartile"] = pd.qcut(
                combined["tvl"], q=4,
                labels=["Q1 (small)", "Q2", "Q3", "Q4 (large)"])
            tvl_tab = combined.groupby("tvl_quartile", observed=True)["basis_abs"].agg(
                N="count", mean="mean", median="median",
                pct_above=lambda x: (x > ARB_THRESHOLD).mean() * 100,
            ).round(3)
            savetable(tvl_tab, "basis_by_tvl_quartile")

    # ── Table 2 (computed first so p-value can inform Table 1) ───────
    # AR(1) with HAC — must be run before basis_stats so we can decide
    # whether the half-life is statistically meaningful.
    ar1_tab = ols_hac(
        basis.iloc[1:],
        pd.DataFrame({"basis_L1": basis.iloc[:-1].values}, index=basis.index[1:]),
    )
    ar1_pval = float("nan")
    if not ar1_tab.empty and "basis_L1" in ar1_tab.index:
        try:
            ar1_pval = float(ar1_tab.loc["basis_L1", "p-val"])
        except Exception:
            pass
    if not ar1_tab.empty:
        savetable(ar1_tab, "basis_ar1")

    # ── Table 1: Basis summary stats ──────────────────────────────
    print("  Computing basis summary stats...")
    arb_flag = basis.abs() > ARB_THRESHOLD
    ep_lens  = episode_lengths(arb_flag)
    stat_res = stationarity_tests(basis)
    arch_p   = _arch_lm_pval(basis, nlags=12)
    jb_s, jb_p = stats.jarque_bera(basis)
    ci_lo, ci_hi = block_bootstrap_ci(basis.values, func=np.mean, block_size=24)

    basis_lag = basis.shift(1)
    ar1_df = pd.DataFrame({"b": basis, "b_L1": basis_lag}).dropna()
    rho_est = float(np.corrcoef(ar1_df["b_L1"], ar1_df["b"])[0, 1])
    # Only report half-life when the AR(1) coefficient is statistically significant.
    # An insignificant rho yields an unreliable half-life estimate.
    ar1_significant = (not np.isnan(ar1_pval)) and (ar1_pval < 0.10)
    hl_ar1 = (
        -np.log(2) / np.log(abs(rho_est))
        if ar1_significant and 0 < abs(rho_est) < 1
        else float("nan")
    )

    tab = pd.DataFrame([{
        "N_obs":               len(basis),
        "mean_bps":            round(float(basis.mean()), 3),
        "CI95_lo (block-bs)":  round(ci_lo, 3),
        "CI95_hi (block-bs)":  round(ci_hi, 3),
        "std_bps":             round(float(basis.std()), 3),
        "p5_bps":              round(float(basis.quantile(0.05)), 3),
        "p95_bps":             round(float(basis.quantile(0.95)), 3),
        "Skewness":            round(float(basis.skew()), 4),
        "Ex.Kurt":             round(float(basis.kurtosis()), 4),
        "Student_t_nu":        round(float(nu_t), 2),
        # Student-t consistency check: MLE-fitted ν implies a theoretical ExKurt.
        # When empirical ExKurt >> theoretical, the t-distribution fits the central
        # bulk but underestimates the tail. EVT (Generalized Pareto Distribution)
        # should be used for extreme quantile estimation.
        # Ref: McNeil, Frey & Embrechts (2005) Quantitative Risk Management, §7.
        "Student_t_ExKurt_theory": (
            round(6.0 / (float(nu_t) - 4.0), 4)
            if float(nu_t) > 4.0 else "undefined (ν≤4; ExKurt=∞)"
        ),
        "Student_t_fit_note": (
            f"MLE fits central mass: theoretical ExKurt=6/(ν-4)="
            f"{6.0/(float(nu_t)-4.0):.2f} vs empirical ExKurt="
            f"{round(float(basis.kurtosis()), 1):.1f}. "
            "t-distribution severely underestimates tail risk. "
            "EVT/GPD recommended for tail analysis (McNeil et al. 2005)."
            if float(nu_t) > 4.0 and float(basis.kurtosis()) > 5.0 else ""
        ),
        "JB_stat":             round(float(jb_s), 4),
        "JB_pval":             round(float(jb_p), 4),
        "JB_sig":              stars(float(jb_p)),
        "ADF_pval":            stat_res.get("ADF p-val"),
        "ADF_sig":             stars(stat_res.get("ADF p-val", 1.0)),
        "KPSS_pval":           stat_res.get("KPSS p-val"),
        "Stationarity":        stat_res.get("Conclusion", ""),
        # ADF and KPSS may give conflicting results due to structural breaks.
        # Known breaks in this dataset that affect basis stationarity:
        #   EIP-1559 (Aug 2021): gas fee market overhaul → permanent basis level shift
        #   The Merge (Sep 2022): PoW→PoS → changed ETH risk premium
        #   FTX collapse (Nov 2022): systemic CEX distrust → basis volatility spike
        # Perron (1989) shows ADF loses power in the presence of structural breaks.
        # Recommendation: Zivot-Andrews or Bai-Perron tests for robust stationarity.
        # Ref: Perron (1989) Econometrica; Zivot & Andrews (1992) JBES.
        "Stationarity_note": (
            "ADF/KPSS disagreement may reflect structural breaks (EIP-1559 Aug 2021, "
            "The Merge Sep 2022, FTX Nov 2022). Perron (1989): ADF loses power with breaks. "
            "Consider Zivot-Andrews test for break-robust unit root inference."
        ),
        "ARCH_LM12_pval":      round(arch_p, 4),
        "ARCH_LM12_sig":       stars(arch_p),
        "AR1_rho":             round(rho_est, 4),
        "AR1_pval (HAC)":      round(ar1_pval, 4) if not np.isnan(ar1_pval) else None,
        "AR1_sig":             stars(ar1_pval) if not np.isnan(ar1_pval) else "",
        # half_life_hours is only reported when AR(1) is significant (p<0.10)
        "half_life_hours":     round(hl_ar1, 2) if not np.isnan(hl_ar1) else "N/A (AR1 not sig.)",
        "pct_above_threshold": round(arb_flag.mean() * 100, 2),
        "N_arb_episodes":      len(ep_lens),
        "mean_episode_h":      round(float(ep_lens.mean()), 2) if len(ep_lens) > 0 else None,
        "median_episode_h":    round(float(ep_lens.median()), 2) if len(ep_lens) > 0 else None,
        "max_episode_h":       int(ep_lens.max()) if len(ep_lens) > 0 else None,
    }], index=["Basis"])
    savetable(tab.T.rename(columns={"Basis": "Value"}), "basis_stats")

    # ── Table 3: Engle-Granger cointegration ──────────────────────
    if dex is not None and cex is not None:
        dex_p_col = next((c for c in ["eth_usdc_price", "eth_usdc_close"]
                          if c in dex.columns), None)
        cex_p_col = "eth_usdc_close" if cex is not None and "eth_usdc_close" in cex.columns else None
        if dex_p_col and cex_p_col:
            print("  Running Engle-Granger cointegration test...")
            dex_price = pd.to_numeric(dex[dex_p_col], errors="coerce")
            cex_price = pd.to_numeric(cex[cex_p_col], errors="coerce")
            eg_res = _engle_granger_cointegration(dex_price, cex_price)
            if eg_res and "EG_stat" in eg_res:
                savetable(pd.DataFrame.from_dict(eg_res, orient="index",
                                                  columns=["Value"]),
                          "basis_cointegration")

            # ── Table 4: VECM ──────────────────────────────────────
            print("  Fitting VECM...")
            vecm_res = _fit_vecm(dex_price, cex_price)
            if vecm_res and "EC_alpha" in vecm_res:
                savetable(pd.DataFrame.from_dict(vecm_res, orient="index",
                                                  columns=["Value"]),
                          "basis_vecm")
                print(f"  VECM alpha={vecm_res['EC_alpha']:.4f} "
                      f"(t={vecm_res['EC_alpha_t']:.3f}, "
                      f"half-life={vecm_res.get('half_life_hours', 'N/A')}h)")
            elif vecm_res and "Error" in vecm_res:
                print(f"  [WARN] VECM table not saved: {vecm_res['Error']}")

    # ── Table 5: OLS |basis| determinants ─────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        hour_of_day = pd.Series(basis.index.hour.astype(float), index=basis.index)
        day_of_week = pd.Series(basis.index.dayofweek.astype(float), index=basis.index)

        X_det = pd.DataFrame({
            "vol_24h_ann":  vol.reindex(basis.index),
            "hour_of_day":  hour_of_day,
            "day_of_week":  day_of_week,
        })
        if len(gas_hourly) > 0:
            X_det["gas_gwei"] = gas_hourly.reindex(basis.index)
            X_det["vol_x_gas"] = X_det["vol_24h_ann"] * X_det["gas_gwei"]

        det_tab = ols_hac(basis.abs(), X_det, label="|basis| determinants")
        if not det_tab.empty:
            savetable(det_tab, "basis_determinants")

    # ── Table 6: Basis by vol regime ──────────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol    = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        regime = vol_regime(vol)
        df_reg = pd.DataFrame({
            "basis_abs": basis.abs(),
            "regime":    regime,
        }).dropna()
        grp = df_reg.groupby("regime")["basis_abs"].agg(
            N="count", mean="mean", median="median",
            pct_above=lambda x: (x > ARB_THRESHOLD).mean() * 100,
            p95=lambda x: x.quantile(0.95),
        ).round(3)
        kw_groups = [
            df_reg[df_reg["regime"] == r]["basis_abs"].dropna()
            for r in ["low", "normal", "high"]
            if len(df_reg[df_reg["regime"] == r]) >= 5
        ]
        if len(kw_groups) >= 2:
            kw_s, kw_p = stats.kruskal(*kw_groups)
            grp["KW_stat"] = round(float(kw_s), 4)
            grp["KW_pval"] = round(float(kw_p), 4)
            grp["KW_sig"]  = stars(float(kw_p))
        savetable(grp, "basis_by_regime")

    # ── Table 7: Breakeven gas cost ───────────────────────────────
    if len(gas_hourly) > 0 and dex is not None and "tvl_usd" in dex.columns:
        # Arbitrage is profitable when fee_income > gas_cost
        # Approx: basis_bps × trade_size_usd / 10000 > gas_cost_usd
        # Here we compute: minimum trade size (USD) needed for gas break-even
        gas_aligned = gas_hourly.reindex(basis.index)
        # Median gas cost = gas_gwei × ~100000 gas units × eth_price / 1e9
        # Use median ETH price if available
        eth_price = 2000.0  # fallback
        if cex is not None and "eth_usdc_close" in cex.columns:
            eth_p = pd.to_numeric(cex["eth_usdc_close"], errors="coerce")
            eth_price = float(eth_p.median())
        gas_cost_est = gas_aligned * 150_000 * eth_price / 1e9  # 150k gas units for Uniswap swap
        breakeven_size = gas_cost_est / (basis.abs() / 10000).replace(0, np.nan)
        be_tab = pd.DataFrame({
            "mean_gas_gwei":        round(float(gas_aligned.mean()), 2),
            "median_gas_gwei":      round(float(gas_aligned.median()), 2),
            "est_gas_cost_usd":     round(float(gas_cost_est.median()), 2),
            "median_breakeven_usd": round(float(breakeven_size.median()), 0),
            "p90_breakeven_usd":    round(float(breakeven_size.quantile(0.90)), 0),
            "eth_price_used":       round(eth_price, 0),
            "pct_arb_profitable_at_100k_usd":
                round(float((breakeven_size < 100_000).mean() * 100), 1),
        }, index=["Value"]).T
        savetable(be_tab, "basis_breakeven")

    # ── Table 8: Arbitrage episode statistics ─────────────────────
    if len(ep_lens) > 0:
        ep_tab = pd.DataFrame({
            "N_episodes":          len(ep_lens),
            "Total_arb_hours":     int(ep_lens.sum()),
            "Mean_duration_h":     round(float(ep_lens.mean()), 2),
            "Median_duration_h":   round(float(ep_lens.median()), 2),
            "P90_duration_h":      round(float(ep_lens.quantile(0.90)), 1),
            "Max_duration_h":      int(ep_lens.max()),
            "Pct_1h_episodes":     round(float((ep_lens == 1).mean() * 100), 1),
            "Pct_longer_24h":      round(float((ep_lens > 24).mean() * 100), 1),
        }, index=["Value"]).T
        savetable(ep_tab, "arbitrage_episodes")

    print("\nDONE")


if __name__ == "__main__":
    main()
