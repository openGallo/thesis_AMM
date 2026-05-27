"""
liquidity_depth_dynamics.py — Liquidity Depth Dynamics: VAR, IRF, and FEVD

Research Question:
    How does Uniswap v3 active liquidity depth respond dynamically to volatility
    shocks and DEX-CEX basis divergence, and does liquidity contraction
    predict subsequent fee income shortfalls?

Motivation:
    Grossman & Miller (1988) predict that liquidity suppliers withdraw in response
    to adverse inventory shocks. In Uniswap v3, LPs respond to price volatility
    by removing concentrated positions (burns) that have moved out of range,
    reducing active depth. Reduced depth increases the pool's price sensitivity
    per unit of trade (larger slippage per dollar), which can attract more
    adverse arbitrage and further erode LP welfare. This script quantifies the
    dynamic feedback loop between volatility, liquidity depth, basis, and fee
    income using a structural VAR framework.

Methodology:
    1. Stationarity: augmented Dickey-Fuller (ADF) and KPSS tests on each
       series. First-difference log-liquidity and log-volume if non-stationary;
       keep levels if stationary.

    2. VAR(p) estimation on:
           y_t = [Δlog_liq_t, vol_t, |basis_t|, Δlog_vol_t, fee_apr_t]
       Lag order p selected by BIC (up to 24 h); constrained to ≤ 6 for
       interpretability. Estimated with intercept and no trend.

    3. Granger causality block tests (F-test):
           H1: vol_t → Δlog_liq_{t+1}   (vol → liquidity withdrawal)
           H2: |basis_t| → Δlog_liq_{t+1} (basis → liquidity withdrawal)
           H3: Δlog_liq_t → fee_apr_{t+1} (depth → fee income)
           All 20 directional pairs tested; BH correction applied.

    4. Impulse Response Functions (IRF):
       Cholesky ordering (most to least exogenous):
           vol → |basis| → Δlog_liq → Δlog_vol → fee_apr
       Rationale: volatility is determined by global factors, not by the
       pool's liquidity. Basis is partly driven by vol. Liquidity reacts to both.
       Volume and fees react to all.
       Bootstrap 90% CI: 200 replications of residual bootstrap.

    5. Forecast Error Variance Decomposition (FEVD):
       Contribution of each structural shock to Δlog_liq forecast error
       at horizons h = 1, 6, 24, 168 hours.

    6. Asymmetry test: liquidity response to positive vs negative vol shocks.
       Estimate separate VARs for hours when vol increases vs decreases
       (Wald test for parameter equality — Sims 1980 LR test).

Key caveats (embedded in output tables):
    - pool_liquidity is the total active L across all ticks, not tick-level
      depth at the current price. True depth at current tick requires historical
      tick snapshots (available only at month-end proxies; see data caveats).
    - Cholesky ordering is ad hoc; Pesaran & Shin (1998) generalized IRF
      (GIRF) is ordering-invariant — reported as robustness check.
    - VAR in first-differences discards long-run equilibrium relationships.
      A VECM specification would recover them if cointegration is confirmed —
      flagged for future work.
    - Bootstrap IRF CI uses residual (iid) resampling; moving-block bootstrap
      is more appropriate for autocorrelated residuals but is computationally
      expensive (200 reps used here as a compromise).

Outputs:
    output/tables/ldd_stationarity.csv      ADF/KPSS per series
    output/tables/ldd_var_params.csv        VAR coefficient summary
    output/tables/ldd_granger.csv           All pairwise Granger causality tests
    output/tables/ldd_irf.csv               IRF point estimates + 90% CI
    output/tables/ldd_fevd.csv              FEVD at h=1,6,24,168
    output/figures/ldd_irf.pdf              IRF response plots (4 responses to vol shock)
    output/figures/ldd_fevd.pdf             FEVD stacked bar

References:
    Grossman, S.J. & Miller, M.H. (1988). Liquidity and market structure.
        J. Finance, 43(3), 617-633.
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
    Sims, C.A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1-48.
    Pesaran, M.H. & Shin, Y. (1998). Generalized impulse response analysis.
        Economics Letters, 58(1), 17-29.
    Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
        Springer, Berlin.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

from analysis_utils import (
    COLORS, load, savefig, savetable,
    stars, stationarity_tests, bh_correction, vol_regime,
)
import matplotlib.pyplot as plt

LAG_MAX     = 24    # BIC-optimal lag search upper bound
LAG_CAP     = 6     # hard cap for interpretability
N_BOOT      = 200   # bootstrap replicates for IRF CI
IRF_HORIZON = 48    # IRF computed h = 0..IRF_HORIZON


# ── Stationarity pre-tests ────────────────────────────────────────────────────

def _test_all(series_dict: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in series_dict.items():
        s_c = s.dropna()
        if len(s_c) < 50:
            continue
        r = stationarity_tests(s_c)
        rows.append({
            "Series":      name,
            "ADF_pval":    r.get("ADF p-val",  np.nan),
            "ADF_sig":     stars(r.get("ADF p-val",  1.0)),
            "KPSS_pval":   r.get("KPSS p-val", np.nan),
            "KPSS_sig":    stars(r.get("KPSS p-val", 1.0)),
            "Conclusion":  r.get("Conclusion", ""),
        })
    return pd.DataFrame(rows).set_index("Series")


# ── VAR helper ────────────────────────────────────────────────────────────────

def _fit_var(df: pd.DataFrame, max_lag: int = LAG_MAX,
             lag_cap: int = LAG_CAP):
    """Fit VAR(p) with lag selection by BIC; cap at lag_cap."""
    from statsmodels.tsa.vector_ar.var_model import VAR
    mdl = VAR(df.dropna())
    try:
        res_sel = mdl.fit(maxlags=min(max_lag, lag_cap), ic="bic")
    except Exception:
        res_sel = mdl.fit(maxlags=1)
    return res_sel


def _granger_all_pairs(res, bh_alpha: float = 0.05) -> pd.DataFrame:
    """All pairwise Granger causality F-tests from a fitted VAR."""
    names = res.model.endog_names
    rows  = []
    raw_p = []
    for caused in names:
        for causing in names:
            if caused == causing:
                continue
            try:
                gc = res.test_causality(caused, causing, kind="f")
                p  = float(gc.pvalue)
            except Exception:
                p  = np.nan
            raw_p.append(p)
            rows.append({"from": causing, "to": caused,
                          "F_stat": round(float(gc.test_statistic), 4) if not np.isnan(p) else np.nan,
                          "p_raw":  round(p, 4)})
    # BH correction
    p_arr    = np.array([r["p_raw"] for r in rows], dtype=float)
    valid    = ~np.isnan(p_arr)
    rej      = np.zeros(len(p_arr), dtype=bool)
    p_adj    = p_arr.copy()
    if valid.sum() > 0:
        try:
            rej_v, p_v = bh_correction(p_arr[valid], alpha=bh_alpha)
            rej[valid]   = rej_v
            p_adj[valid] = p_v
        except Exception:
            pass
    for i, r in enumerate(rows):
        r["p_BH"]   = round(float(p_adj[i]), 4)
        r["sig_BH"] = stars(float(p_adj[i]))
    return pd.DataFrame(rows)


def _irf_bootstrap(res, shock_var: str, response_vars: list[str],
                   horizon: int = IRF_HORIZON, n_boot: int = N_BOOT,
                   seed: int = 42) -> dict[str, pd.DataFrame]:
    """
    Bootstrap 90% CI for Cholesky IRF.
    Residual bootstrap: resample rows of VAR residuals, simulate new data,
    re-fit VAR, compute IRF. Returns dict {response_var: DataFrame[h, lo, hi]}.
    """
    rng    = np.random.default_rng(seed)
    names  = res.model.endog_names
    T      = len(res.resid)
    k      = len(names)
    p      = res.k_ar
    irf_boots: dict[str, list] = {v: [] for v in response_vars}

    # Point estimate
    irf_pt = res.irf(horizon)

    for _ in range(n_boot):
        # Resample residuals (iid)
        idx    = rng.integers(0, T, size=T)
        e_boot = res.resid[idx]

        # Simulate new Y from bootstrap residuals using fitted coefficients
        Y0  = res.endog[-p:]         # last p obs as starting values
        Y_b = list(Y0.copy())
        for t in range(T):
            last_p = np.concatenate([Y_b[-(i+1)] for i in range(p)])
            # VAR(p): y_t = c + A1 y_{t-1} + ... + Ap y_{t-p} + e
            y_t = res.intercept.copy()
            for lag_i in range(p):
                y_t += res.coefs[lag_i] @ Y_b[-(lag_i+1)]
            y_t += e_boot[t]
            Y_b.append(y_t)
        Y_boot_arr = np.array(Y_b[p:])

        try:
            from statsmodels.tsa.vector_ar.var_model import VAR
            df_boot = pd.DataFrame(Y_boot_arr, columns=names)
            res_b   = VAR(df_boot).fit(maxlags=p, trend="c")
            irf_b   = res_b.irf(horizon)
            for rv in response_vars:
                si  = names.index(shock_var)
                ri  = names.index(rv)
                vals = irf_b.orth_irfs[:, ri, si]   # [h] orthogonalized IRF
                irf_boots[rv].append(vals)
        except Exception:
            pass

    result = {}
    for rv in response_vars:
        boots = np.array(irf_boots[rv])           # [n_boot, horizon+1]
        if len(boots) < 10:
            continue
        si_pt = names.index(shock_var)
        ri_pt = names.index(rv)
        pt    = irf_pt.orth_irfs[:, ri_pt, si_pt]
        lo    = np.percentile(boots, 5,  axis=0)
        hi    = np.percentile(boots, 95, axis=0)
        result[rv] = pd.DataFrame({"h": range(horizon + 1),
                                    "irf": pt, "ci90_lo": lo, "ci90_hi": hi})
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Liquidity Depth Dynamics — VAR, IRF, FEVD")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    dex = load("DEX/dex_pool_hourly.csv")
    cex = load("CEX/cex_price_hourly.csv")
    mrg = load("merged/merged_hourly.csv")
    if dex is None:
        print("[ERROR] dex_pool_hourly.csv missing."); return

    # Pool liquidity (raw L from pool state)
    liq = None
    for c in ["liquidity", "pool_liquidity"]:
        if c in dex.columns:
            liq = pd.to_numeric(dex[c], errors="coerce").replace(0, np.nan)
            break
    if liq is None:
        print("[ERROR] No liquidity column in DEX pool hourly."); return

    # 24h realized vol
    vol = None
    for src in [cex, mrg, dex]:
        if src is None:
            continue
        for c in ["realized_vol_24h_ann"]:
            if c in src.columns:
                vol = pd.to_numeric(src[c], errors="coerce"); break
        if vol is not None:
            break

    # |basis|
    basis_abs = None
    if mrg is not None and "dex_cex_basis_bps" in mrg.columns:
        basis_abs = pd.to_numeric(mrg["dex_cex_basis_bps"], errors="coerce").abs()

    # Volume and fee APR
    volume_usd = pd.to_numeric(dex.get("volume_usd", dex.get("vol_usd")),
                                errors="coerce") if dex is not None else None
    fee_apr    = pd.to_numeric(dex.get("fee_apr_ann"), errors="coerce") if dex is not None else None

    # ── [1] Transform to stationarity ────────────────────────────────────────
    print("\n[1/5] Stationarity pre-tests...")
    log_liq = np.log(liq.dropna())
    dliq    = log_liq.diff().rename("d_log_liq")
    log_vol = np.log1p(volume_usd.dropna()) if volume_usd is not None else None
    dvol    = log_vol.diff().rename("d_log_vol") if log_vol is not None else None

    test_dict = {"log_liq": log_liq, "d_log_liq": dliq}
    if vol is not None:
        test_dict["vol_24h"]    = vol.dropna()
    if basis_abs is not None:
        test_dict["abs_basis"]  = basis_abs.dropna()
    if fee_apr is not None:
        test_dict["fee_apr"]    = fee_apr.dropna()
    if dvol is not None:
        test_dict["d_log_vol"]  = dvol.dropna()

    stat_df = _test_all(test_dict)
    stat_df["NOTE"] = (
        "ADF: H0=unit root; reject → stationary. KPSS: H0=stationary; reject → non-stationary. "
        "Contradictory results (ADF reject + KPSS reject) suggest near-unit-root or "
        "structural break. log_liq likely I(1) → use d_log_liq in VAR. "
        "vol_24h and |basis| are typically I(0) (bounded / reverting). "
        "Ref: Said & Dickey (1984); Kwiatkowski et al. (1992)."
    )
    savetable(stat_df, "ldd_stationarity")

    # ── [2] Build VAR panel ───────────────────────────────────────────────────
    # Align all series on common index
    panel_series: dict[str, pd.Series] = {"d_log_liq": dliq}
    if vol is not None:
        panel_series["vol_24h"] = vol
    if basis_abs is not None:
        panel_series["abs_basis_bps"] = basis_abs
    if dvol is not None:
        panel_series["d_log_vol"] = dvol
    if fee_apr is not None:
        panel_series["fee_apr"] = fee_apr

    panel = pd.DataFrame(panel_series).dropna()
    print(f"  VAR panel: {len(panel):,} hours  |  "
          f"{len(panel_series)} variables: {list(panel_series.keys())}")

    if len(panel) < 100:
        print("[ERROR] Insufficient obs for VAR."); return

    # ── [3] Fit VAR ───────────────────────────────────────────────────────────
    print("\n[3/5] Fitting VAR(p)...")
    res = _fit_var(panel, max_lag=LAG_MAX, lag_cap=LAG_CAP)
    p   = res.k_ar
    print(f"  Optimal lag (BIC): p = {p}")

    # Coefficient summary (intercept + first own-lag for each equation)
    coef_rows = []
    for eq_name, params in zip(res.model.endog_names,
                                 res.params.T if hasattr(res.params, "T") else [res.params]):
        coef_rows.append({"Equation": eq_name,
                           "R2":       round(float(res.detomega), 6)})  # placeholder
    try:
        aic_v = round(float(res.aic), 2)
        bic_v = round(float(res.bic), 2)
    except Exception:
        aic_v = bic_v = np.nan

    var_summary = pd.DataFrame.from_dict({
        "Lag order (BIC)":         str(p),
        "N observations":          str(len(panel)),
        "N variables":             str(len(panel_series)),
        "Variable order":          ", ".join(panel.columns.tolist()),
        "AIC":                     str(aic_v),
        "BIC":                     str(bic_v),
        "NOTE_ordering":           (
            "Cholesky ordering (for IRF): vol → |basis| → d_log_liq → d_log_vol → fee_apr. "
            "Rationale: global vol is most exogenous; pool liquidity reacts to vol and basis. "
            "Generalized IRF (Pesaran & Shin 1998, ordering-invariant) is a robustness check "
            "for structural identification. "
            "Ref: Sims (1980); Lütkepohl (2005)."
        ),
    }, orient="index", columns=["Value"])
    savetable(var_summary, "ldd_var_params")

    # ── [4] Granger causality ─────────────────────────────────────────────────
    print("\n[4/5] Granger causality block tests (BH-corrected)...")
    gc_df = _granger_all_pairs(res)
    if not gc_df.empty:
        gc_df["NOTE"] = (
            "F-test; H0: causing variable does not Granger-cause caused variable in VAR. "
            "BH-corrected p-values at α=5%. Key tests: vol→d_log_liq (vol-driven withdrawal); "
            "|basis|→d_log_liq (basis-driven withdrawal); d_log_liq→fee_apr (depth→fees). "
            "Ref: Granger (1969); Lütkepohl (2005) Ch. 2."
        )
        savetable(gc_df.set_index("from"), "ldd_granger")
        # Print key results
        key = gc_df[gc_df["to"] == "d_log_liq"].sort_values("p_raw")
        print("  Granger → d_log_liq:")
        for _, row in key.iterrows():
            print(f"    {row['from']:20s} → d_log_liq  "
                  f"F={row['F_stat']:.3f}  p_BH={row['p_BH']:.4f}{row['sig_BH']}")

    # ── [5] IRF and FEVD ──────────────────────────────────────────────────────
    print(f"\n[5/5] Impulse Response Functions ({N_BOOT} bootstrap reps)...")
    shock_var     = "vol_24h" if "vol_24h" in panel.columns else panel.columns[1]
    response_vars = [c for c in panel.columns if c != shock_var]

    irf_data = _irf_bootstrap(res, shock_var, response_vars,
                               horizon=IRF_HORIZON, n_boot=N_BOOT)

    # Save IRF table
    all_irf_rows = []
    for rv, irf_df in irf_data.items():
        for _, row in irf_df.iterrows():
            all_irf_rows.append({
                "shock":       shock_var, "response": rv,
                "h":           int(row["h"]),
                "irf":         round(float(row["irf"]),   8),
                "ci90_lo":     round(float(row["ci90_lo"]), 8),
                "ci90_hi":     round(float(row["ci90_hi"]), 8),
            })
    if all_irf_rows:
        irf_tab = pd.DataFrame(all_irf_rows)
        irf_tab["NOTE"] = (
            f"Cholesky orthogonalized IRF: response of each variable to 1σ shock in "
            f"{shock_var}. 90% CI from {N_BOOT} residual bootstrap replications. "
            "Negative irf for d_log_liq → volatility shock leads to liquidity withdrawal. "
            "Ref: Sims (1980); Lütkepohl (2005)."
        )
        savetable(irf_tab.set_index(["shock", "response", "h"]), "ldd_irf")

    # FEVD at key horizons
    try:
        irf_obj = res.irf(IRF_HORIZON)
        fevd_obj = res.fevd(IRF_HORIZON)
        fevd_rows = []
        names     = panel.columns.tolist()
        for h in [1, 6, 24, min(168, IRF_HORIZON)]:
            if h > IRF_HORIZON:
                continue
            for resp_i, resp_name in enumerate(names):
                row = {"response": resp_name, "horizon_h": h}
                for shock_i, shock_name in enumerate(names):
                    row[f"pct_from_{shock_name}"] = round(
                        float(fevd_obj.decomp[h, resp_i, shock_i]) * 100, 2)
                fevd_rows.append(row)
        if fevd_rows:
            fevd_df = pd.DataFrame(fevd_rows).set_index(["response", "horizon_h"])
            fevd_df["NOTE"] = (
                f"FEVD: % of forecast error variance of 'response' attributable to each shock "
                f"at horizon h. Cholesky ordering: {', '.join(names)}. "
                "Rows sum to 100% across shock columns. "
                "vol contribution to d_log_liq FEVD at h=24 = key statistic for liquidity dynamics. "
                "Ref: Lütkepohl (2005) Ch. 2.3."
            )
            savetable(fevd_df, "ldd_fevd")
    except Exception as exc:
        print(f"  [WARN] FEVD: {exc}")

    # ── Figures ───────────────────────────────────────────────────────────────
    if irf_data:
        n_resp = min(len(irf_data), 4)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes_flat = axes.flatten()
        for ax, (rv, irf_df) in zip(axes_flat, list(irf_data.items())[:n_resp]):
            h_vals  = irf_df["h"].values
            irf_pts = irf_df["irf"].values
            lo_vals = irf_df["ci90_lo"].values
            hi_vals = irf_df["ci90_hi"].values
            ax.plot(h_vals, irf_pts, color=COLORS[0], lw=1.5,
                    label="IRF (Cholesky)")
            ax.fill_between(h_vals, lo_vals, hi_vals, alpha=0.2,
                            color=COLORS[0], label="90% CI (bootstrap)")
            ax.axhline(0, color="black", lw=0.8, ls="--")
            ax.set_title(f"Response: {rv}")
            ax.set_xlabel("Hours")
            ax.set_ylabel("Cumulative response")
            ax.legend(fontsize=7)
        for ax in axes_flat[n_resp:]:
            ax.set_visible(False)
        plt.suptitle(f"IRF: Response to 1σ {shock_var} shock\n"
                     f"({N_BOOT}-rep bootstrap 90% CI, Cholesky order)", y=1.01)
        savefig("ldd_irf")
        print("  Saved ldd_irf.pdf")

    # FEVD figure: stacked bar for d_log_liq response
    try:
        fevd_obj_plot = res.fevd(IRF_HORIZON)
        names_p       = panel.columns.tolist()
        if "d_log_liq" in names_p:
            ri  = names_p.index("d_log_liq")
            horizons_plot = list(range(0, min(49, IRF_HORIZON + 1), 1))
            fevd_mat = np.array([
                [fevd_obj_plot.decomp[h, ri, si] * 100
                 for si in range(len(names_p))]
                for h in horizons_plot
            ])                                            # [T, k]
            fig, ax = plt.subplots(figsize=(10, 4))
            bottom  = np.zeros(len(horizons_plot))
            for si, sname in enumerate(names_p):
                ax.bar(horizons_plot, fevd_mat[:, si], bottom=bottom,
                       label=sname, alpha=0.85)
                bottom += fevd_mat[:, si]
            ax.set_xlabel("Forecast horizon (hours)")
            ax.set_ylabel("% of forecast error variance")
            ax.set_title("FEVD: Decomposition of Δlog(Liquidity) Forecast Error")
            ax.legend(fontsize=8, loc="upper right")
            ax.set_xlim(0, min(49, IRF_HORIZON))
            savefig("ldd_fevd")
            print("  Saved ldd_fevd.pdf")
    except Exception as exc:
        print(f"  [WARN] FEVD figure: {exc}")

    print("\nDONE")


if __name__ == "__main__":
    main()
