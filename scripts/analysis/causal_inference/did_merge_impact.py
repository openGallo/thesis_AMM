"""
did_merge_impact.py — Interrupted Time Series: Ethereum Merge (Sep 15, 2022)

Research Question:
    Did the Ethereum Merge — the transition from Proof-of-Work to Proof-of-Stake
    on Sep 15, 2022 — cause a structural shift in AMM volatility dynamics, LP
    profitability, and the DEX-CEX arbitrage premium on the WETH/USDC 0.05% pool?

Motivation:
    The Merge eliminated PoW mining, making block times more predictable (12 s
    deterministic PoS vs. stochastic PoW), removing ~13 000 ETH/day of sell
    pressure from miners, and altering MEV dynamics (no uncle blocks).  Theory
    predicts: (a) lower realized volatility due to reduced miner selling;
    (b) reduced LVR from lower toxic-flow arrival (Milionis et al. 2022);
    (c) compressed basis from faster, cheaper arbitrage with predictable blocks.
    The Merge is an exogenous protocol-level change — not driven by LP or
    arbitrageur behaviour — making it a valid instrument for causal inference.

Methodology:
    1. Interrupted Time Series (ITS) — a single-unit DiD where the "control" is
       the counterfactual trend extrapolated from the pre-event window:
           y_t = α + β₁·t + β₂·Post_t + β₃·(t − T_c)·Post_t + ε_t
       t = hours since window start; Post_t = 1[t ≥ T_c]; T_c = event hour.
       β₂ = immediate level shift; β₃ = change in slope post-event.
       OLS + Newey-West HAC SEs (max_lags=168h = 1 week).
       Window: ±WINDOW_DAYS days; ±EXCL_DAYS days around event excluded.

    2. Chow (1960) structural-break F-test at the pre-specified event date:
           H₀: intercept and slope are constant across pre/post subsamples.
       Pre-specification avoids the data-mining critique of the Quandt-Andrews
       (1993) supremum-F which maximises over all possible break dates.

    3. Placebo test — apply identical ITS at the same calendar date one year
       earlier (Sep 15, 2021).  Placebo β₂, β₃ should be near-zero and
       insignificant if the parallel-trends assumption is supported.

    4. Short-run robustness — repeat ITS with a 30-day post-window to isolate
       the immediate effect, reducing FTX-collapse contamination (Nov 8, 2022
       falls 54 days post-Merge).

    5. Event-window study — weekly-average outcomes normalised by their
       pre-event mean, across ±8 weeks.

Key caveats (embedded in output tables):
    - ITS has no explicit control group; β₂/β₃ capture ALL shocks concurrent
      with the Merge.  The FTX collapse (54 days post) partly contaminates the
      full-window estimate; the 30-day robustness check addresses this.
    - Macro shocks (Fed rate hikes, Grayscale unlock) co-moved with the Merge.
      The linear trend β₁ partially absorbs slow-moving confounders.
    - ETH price level and volatility are inherently linked; lower vol may itself
      be caused by a bull-run or macro calm rather than the Merge per se.
    - The pool fee tier (0.05%) and tick spacing were unchanged before/after,
      isolating the protocol-level effect from fee-structure changes.
    - Parallel trends: untestable in strict sense; supported by (a) insignificant
      placebo, (b) Chow break localised at exact event date.

Outputs:
    output/tables/did_merge_its.csv         ITS coefs for all outcomes × specs
    output/tables/did_merge_chow.csv        Chow break F-test at The Merge
    output/tables/did_merge_placebo.csv     Placebo ITS (Sep 15, 2021)
    output/tables/did_merge_event_window.csv Weekly normalised outcomes ±8 wks
    output/figures/did_merge_its.pdf        ITS plots (actual + fitted + CF)
    output/figures/did_merge_event_window.pdf Event-window bar charts

References:
    Andrews, D.W.K. (1993). Tests for parameter instability and structural
        change. Econometrica, 61(4), 821-856.
    Bernal, J.L. et al. (2017). Interrupted time series regression for the
        evaluation of public health interventions. Int. J. Epidemiology.
    Bai, J. & Perron, P. (1998). Estimating and testing linear models with
        multiple structural changes. Econometrica, 66(1), 47-78.
    Milionis, J. et al. (2022). Automated market making and loss-versus-
        rebalancing. Working paper.
    Chow, G.C. (1960). Tests of equality between sets of coefficients.
        Econometrica, 28(3), 591-605.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from analysis_utils import (
    load, savefig, savetable, stars,
    ols_hac, block_bootstrap_ci, bh_correction,
)

# ── Event constants ───────────────────────────────────────────────────────────

THE_MERGE    = pd.Timestamp("2022-09-15 06:42:00", tz="UTC")
PLACEBO_DATE = THE_MERGE - pd.DateOffset(years=1)   # Sep 15, 2021
WINDOW_DAYS  = 180   # ±6 months
EXCL_DAYS    = 7     # exclude ±7 days around event
POST_ROB     = 30    # short-run robustness post-window

OUTCOMES = {
    "cex_realized_vol_24h_ann": "CEX realised vol (24 h ann.)",
    "lvr_rate_ann":             "LVR rate (ann.)",
    "dex_fee_apr_ann":          "Fee APR (ann.)",
    "abs_basis_bps":            "|DEX–CEX basis| (bps)",
    "dex_vol_over_tvl":         "Volume / TVL",
}

# ── Data ──────────────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame | None:
    mh  = load("merged/merged_hourly.csv")
    lvr = load("DEX/dex_lvr_hourly.csv")
    if mh is None or lvr is None:
        return None
    panel = mh.join(lvr[["lvr_rate_ann"]], how="left")
    panel["abs_basis_bps"] = panel["dex_cex_basis_bps"].abs()
    panel = panel.sort_index()
    return panel

# ── Core: ITS regression ──────────────────────────────────────────────────────

def _window_slice(series: pd.Series, event: pd.Timestamp,
                  pre_days: int = WINDOW_DAYS,
                  post_days: int = WINDOW_DAYS,
                  excl: int = EXCL_DAYS) -> pd.Series | None:
    start   = event - pd.Timedelta(days=pre_days)
    end     = event + pd.Timedelta(days=post_days)
    excl_lo = event - pd.Timedelta(days=excl)
    excl_hi = event + pd.Timedelta(days=excl)
    sub = series.loc[start:end].copy()
    sub = sub[~((sub.index >= excl_lo) & (sub.index <= excl_hi))].dropna()
    return sub if len(sub) >= 100 else None


def its_regression(series: pd.Series, event: pd.Timestamp,
                   post_days: int = WINDOW_DAYS, label: str = "") -> dict | None:
    """ITS: y = α + β₁t + β₂Post + β₃(t-Tc)Post + ε (Newey-West HAC)."""
    sub = _window_slice(series, event, post_days=post_days)
    if sub is None:
        return None

    t_vec    = np.arange(len(sub), dtype=float)
    T_c      = float(sub.index.searchsorted(event))
    if T_c <= 10 or T_c >= len(sub) - 10:
        return None

    post      = (t_vec >= T_c).astype(float)
    slope_pst = (t_vec - T_c) * post

    X = pd.DataFrame({"t": t_vec, "Post": post, "slope_post": slope_pst},
                     index=sub.index)
    tbl = ols_hac(sub.rename("y"), X, label=label)
    if tbl.empty:
        return None

    out: dict = {"N": len(sub)}
    for var in ["t", "Post", "slope_post"]:
        if var in tbl.index:
            r = tbl.loc[var]
            out[f"{var}_coef"] = r.get("Coef", np.nan)
            out[f"{var}_se"]   = r.get("SE (HAC)", np.nan)
            out[f"{var}_pval"] = r.get("p-val", np.nan)
            out[f"{var}_sig"]  = r.get("Sig", "")
    # R² from footer row
    if "-" in tbl.index:
        meta = str(tbl.loc["-", "Sig"])
        for tok in meta.split():
            if tok.startswith("R²="):
                try:
                    out["R2"] = float(tok.split("=")[1])
                except ValueError:
                    pass
    out["NOTE"] = (
        "ITS: y=α+β₁t+β₂Post+β₃(t-Tc)Post+ε. Post=β₂ is level shift; "
        "slope_post=β₃ is trend change. HAC SE (Newey-West, 168-lag). "
        "No explicit control group — assumes counterfactual trend is linear "
        "extrapolation of pre-event. Confounders co-incident with The Merge "
        "(FTX 54d later, macro) partially absorbed by β₁."
    )
    return out


# ── Chow structural-break test ────────────────────────────────────────────────

def chow_test(series: pd.Series, event: pd.Timestamp) -> dict:
    """Chow (1960) F-test for structural break at pre-specified event date."""
    sub = _window_slice(series, event)
    if sub is None:
        return {}

    t_vec = np.arange(len(sub), dtype=float)
    T_c   = float(sub.index.searchsorted(event))
    pre_mask  = t_vec < T_c
    post_mask = ~pre_mask

    def _sse(y, t):
        X_ = sm.add_constant(t.reshape(-1, 1))
        return float(sm.OLS(y, X_).fit().ssr), len(y)

    try:
        sse_r, N   = _sse(sub.values, t_vec)
        sse_1, n1  = _sse(sub.values[pre_mask],  t_vec[pre_mask])
        sse_2, n2  = _sse(sub.values[post_mask], t_vec[post_mask])
        sse_u = sse_1 + sse_2
        k = 2  # intercept + slope
        F = ((sse_r - sse_u) / k) / (sse_u / (N - 2 * k))
        pval = float(sp_stats.f.sf(F, k, N - 2 * k))
        return {
            "F_stat": round(F, 4), "df1": k, "df2": N - 2 * k,
            "p_val":  round(pval, 6), "sig": stars(pval),
            "NOTE": (
                "Pre-specified Chow (1960) break. H0: intercept+slope equal "
                "in pre/post subsamples. Pre-specification avoids Andrews (1993) "
                "data-mining critique. HAC SEs not applied (Chow uses OLS ε). "
                "Not robust to heteroskedasticity — interpret as complementary "
                "evidence alongside ITS HAC results."
            ),
        }
    except Exception:
        return {}


# ── Placebo test ──────────────────────────────────────────────────────────────

def run_placebo(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        res = its_regression(panel[col], PLACEBO_DATE, label=label)
        if res:
            res["outcome"] = col
            res["label"]   = label
            rows.append(res)
    return pd.DataFrame(rows).set_index("outcome") if rows else pd.DataFrame()


# ── Event-window study ────────────────────────────────────────────────────────

def event_window_study(panel: pd.DataFrame, event: pd.Timestamp,
                       wks: int = 8) -> pd.DataFrame:
    rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        ser = panel[col].dropna()
        pre_window = ser[(ser.index >= event - pd.Timedelta(weeks=wks))
                         & (ser.index < event)]
        pre_mean = float(pre_window.mean())
        if pre_mean == 0 or np.isnan(pre_mean):
            continue
        for w in range(-wks, wks + 1):
            lo = event + pd.Timedelta(weeks=w)
            hi = lo + pd.Timedelta(weeks=1)
            vals = ser[(ser.index >= lo) & (ser.index < hi)].dropna().values
            if len(vals) < 5:
                continue
            rows.append({
                "outcome": col, "label": label, "week": w,
                "mean_norm": float(np.mean(vals) / pre_mean - 1.0),
                "N": len(vals),
            })
    return pd.DataFrame(rows)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _its_fitted_values(series, event, post_days=WINDOW_DAYS):
    """Return (raw_weekly, fitted_series, counterfactual_series)."""
    sub = _window_slice(series, event, post_days=post_days)
    if sub is None:
        return None, None, None
    t_vec  = np.arange(len(sub), dtype=float)
    T_c    = float(sub.index.searchsorted(event))
    post   = (t_vec >= T_c).astype(float)
    slp    = (t_vec - T_c) * post
    X      = sm.add_constant(
        pd.DataFrame({"t": t_vec, "Post": post, "slope_post": slp}))
    try:
        res   = sm.OLS(sub.values, X).fit()
        fit   = pd.Series(res.fittedvalues, index=sub.index)
        Xcf   = X.copy(); Xcf["Post"] = 0.0; Xcf["slope_post"] = 0.0
        cf    = pd.Series(res.predict(Xcf), index=sub.index)
    except Exception:
        return sub, None, None
    return sub, fit, cf


def plot_its_grid(panel: pd.DataFrame, event: pd.Timestamp, tag: str) -> None:
    cols = [c for c in OUTCOMES if c in panel.columns]
    nrow = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 3.5 * nrow))
    flat = axes.flatten()

    for i, col in enumerate(cols):
        ax = flat[i]
        raw, fit, cf = _its_fitted_values(panel[col], event)
        if raw is None:
            ax.set_visible(False); continue
        wkly = raw.resample("7D").mean()
        ax.plot(wkly.index, wkly.values, color="#2563eb", lw=1.1, label="Actual (7d avg)")
        if fit is not None:
            ax.plot(fit.resample("7D").mean().index,
                    fit.resample("7D").mean().values,
                    color="#dc2626", lw=1.5, ls="--", label="ITS fit")
        if cf is not None:
            ax.plot(cf.resample("7D").mean().index,
                    cf.resample("7D").mean().values,
                    color="#16a34a", lw=1.2, ls=":", label="Counterfactual")
        ax.axvline(event, color="black", lw=1.5, label="The Merge")
        ax.set_title(OUTCOMES[col], fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)
    plt.suptitle(f"ITS — Ethereum Merge (Sep 15 2022)  ±{WINDOW_DAYS}d window",
                 fontsize=10, y=1.01)
    savefig(f"did_{tag}_its")


def plot_event_window(ew: pd.DataFrame, tag: str) -> None:
    if ew.empty:
        return
    cols = ew["outcome"].unique()
    nrow = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(13, 3 * nrow))
    flat = axes.flatten()
    colors_pos = "#dc2626"; colors_neg = "#2563eb"

    for i, col in enumerate(cols):
        ax = flat[i]
        sub = ew[ew["outcome"] == col].sort_values("week")
        colors = [colors_pos if v >= 0 else colors_neg for v in sub["mean_norm"]]
        ax.bar(sub["week"], sub["mean_norm"] * 100, color=colors, alpha=0.75)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(-0.5, color="black", lw=1.5, alpha=0.5)
        ax.set_title(sub["label"].iloc[0], fontsize=9)
        ax.set_xlabel("Weeks relative to The Merge", fontsize=8)
        ax.set_ylabel("% change from pre-event mean", fontsize=8)

    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)
    plt.suptitle("Event-window study: normalised outcomes ±8 weeks",
                 fontsize=10, y=1.01)
    savefig(f"did_{tag}_event_window")


# ── Pre-trend test ────────────────────────────────────────────────────────────

def _pre_trend_test(series: pd.Series, event: pd.Timestamp,
                    label: str = "") -> dict | None:
    """
    Pre-period linear trend: H0 β₁=0 (flat pre-event trend).
    Significant β₁ → outcome was trending before The Merge → weakens ITS.
    Window: [event - WINDOW_DAYS, event - EXCL_DAYS].
    """
    start = event - pd.Timedelta(days=WINDOW_DAYS)
    end   = event - pd.Timedelta(days=EXCL_DAYS)
    sub   = series.loc[start:end].dropna()
    if len(sub) < 30:
        return None
    t   = np.arange(len(sub), dtype=float)
    X   = pd.DataFrame({"t": t}, index=sub.index)
    tbl = ols_hac(sub.rename("y"), X, max_lags=168, label=label)
    if tbl.empty or "t" not in tbl.index:
        return None
    r = tbl.loc["t"]
    return {
        "coef_t": r.get("Coef", np.nan),
        "se_t":   r.get("SE (HAC)", np.nan),
        "pval_t": r.get("p-val", np.nan),
        "sig_t":  r.get("Sig", ""),
        "N":      len(sub),
        "NOTE": (
            "Pre-period trend test (HAC 168-lag). H0: β₁=0 (flat pre-Merge trend). "
            "Significant → outcome was already shifting before The Merge; parallel-trends "
            "assumption is weakened. Complement with placebo test at Sep 15 2021."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = load_panel()
    if panel is None:
        print("  [SKIP] Required data missing."); return

    print(f"  Panel: {panel.index[0]:%Y-%m-%d} → {panel.index[-1]:%Y-%m-%d}  "
          f"({len(panel):,} h)")
    print(f"  The Merge: {THE_MERGE}  |  Placebo: {PLACEBO_DATE.date()}")

    # 0. Pre-trend test (ITS validity: flat pre-period trend → no anticipatory dynamics)
    pt_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        res = _pre_trend_test(panel[col], THE_MERGE, label=label)
        if res:
            res["outcome"] = col; res["label"] = label
            pt_rows.append(res)
    if pt_rows:
        savetable(pd.DataFrame(pt_rows).set_index("outcome"), "did_merge_pretrend")
        sig_pt = sum(1 for r in pt_rows if r.get("pval_t", 1.0) < 0.10)
        print(f"  Pre-trend test: {sig_pt}/{len(pt_rows)} outcomes have significant pre-trend (p<0.10)")

    # 1. ITS — main and short-run robustness
    its_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            print(f"    [skip] {col}")
            continue
        for spec, post_d in [("main", WINDOW_DAYS), (f"rob_{POST_ROB}d", POST_ROB)]:
            res = its_regression(panel[col], THE_MERGE, post_days=post_d, label=label)
            if res:
                res["outcome"] = col
                res["label"]   = label
                res["spec"]    = spec
                its_rows.append(res)

    if its_rows:
        # BH correction across outcomes — add flags to dicts BEFORE building DataFrame
        main_rows = [r for r in its_rows if r.get("spec") == "main"]
        post_pvals = [r.get("Post_pval", np.nan) for r in main_rows]
        post_pvals_clean = [p for p in post_pvals if not np.isnan(p)]
        if post_pvals_clean:
            bh_flags = bh_correction(post_pvals_clean, alpha=0.10)
            for r, flag in zip(
                [r for r in main_rows if not np.isnan(r.get("Post_pval", np.nan))], bh_flags
            ):
                r["Post_BH10_reject"] = flag
            print(f"  BH(10%) rejects among main-spec Post coefficients: "
                  f"{sum(bh_flags)}/{len(bh_flags)}")
        # Build DataFrame AFTER flags are embedded in the dicts
        df_its = pd.DataFrame(its_rows).set_index(["outcome", "spec"])
        savetable(df_its, "did_merge_its")

    # 2. Chow test
    chow_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        cr = chow_test(panel[col], THE_MERGE)
        if cr:
            cr["outcome"] = col; cr["label"] = label
            chow_rows.append(cr)
    if chow_rows:
        savetable(pd.DataFrame(chow_rows).set_index("outcome"), "did_merge_chow")

    # 3. Placebo
    df_pl = run_placebo(panel)
    if not df_pl.empty:
        savetable(df_pl, "did_merge_placebo")

    # 4. Event window
    ew = event_window_study(panel, THE_MERGE)
    if not ew.empty:
        savetable(ew, "did_merge_event_window")
        plot_event_window(ew, "merge")

    # 5. ITS plots
    plot_its_grid(panel, THE_MERGE, "merge")

    print("  [DONE] did_merge_impact.py")


if __name__ == "__main__":
    main()
