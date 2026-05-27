"""
did_shapella.py — ITS: Shapella Upgrade (Apr 12, 2023) and LP Economics

Research Question:
    Did the Shapella upgrade (Apr 12, 2023) — which enabled the first-ever
    withdrawals of staked ETH from the Ethereum Beacon chain — alter AMM
    volatility dynamics, LVR, and LP economics, consistent with a dual-effect
    of (a) short-run vol spike from pent-up selling pressure and (b) long-run
    vol reduction from resolved staking-exit uncertainty?

Motivation:
    Since Ethereum's Beacon chain launch (Dec 2020), staked ETH could not be
    withdrawn — creating a one-way staking commitment.  Shapella (EIP-4895)
    unlocked ~16M ETH (~$31B at the time) in withdrawals from 500,000+
    validators. This was the single largest ETH liquidity event in history.

    Theory predicts TWO opposing AMM effects:
    (A) SHORT-RUN (0-30 days): validators who wanted to exit rush to withdraw
        → ETH selling pressure → ETH price falls → CEX vol spikes → LVR rises
        → fee APR rises (more volume) but LP P&L deteriorates.
    (B) LONG-RUN (31-120 days): the uncertainty about future ETH supply is
        permanently resolved → lower risk premium → lower baseline vol →
        tighter DEX-CEX basis → lower LVR → improved LP net economics.

    We test both effects using short and long ITS windows.  Shapella is
    exogenous: the timing was determined by Ethereum core devs months in advance
    and not correlated with Uniswap pool dynamics.

    This script is novel: the vast majority of AMM literature predates Shapella
    (Apr 2023) and no published paper has studied its impact on Uniswap v3 LPs.

Methodology:
    1. Split ITS — estimate separate post-event slopes for two sub-periods:
           SHORT = [event, event+30d];  LONG = [event+31d, event+120d]
       Full model:
         y_t = α + β₁t + β₂Post_t + β₃(t-Tc)Post_short_t + β₄(t-Tc)Post_long_t + ε
       Short/long slopes measure the speed and direction of each phase.
       HAC SE, max_lags=168h (1 week).

    2. Simple ITS — standard single-break model for comparison.

    3. Event-window study — weekly normalised outcomes ±12 weeks.

    4. Pre-trend test — regress outcome on linear time trend in pre-period
       [event - 120d, event - 7d]; test if trend is flat (ITS validity check).

    5. Placebo — same ITS at Apr 12, 2022 (1 year prior, same season).

    6. Vol regime shift test — Levene's test for variance equality of
       cex_realized_vol_24h_ann in pre vs post periods.

Key caveats:
    - Shapella was widely anticipated; the market may have partially priced in
      the event in advance. Anticipatory dynamics could bias β₂ toward zero
      (the level shift would occur BEFORE the event). The event-window study
      checks for pre-trends.
    - Validators withdrew gradually (withdrawal queue had ~80,000 validators
      waiting); the "shock" was spread over ~2-4 weeks, not a single day. The
      exclusion window of ±7 days may be insufficient — we report ±14d as a
      robustness check.
    - The LONG-RUN effect (β₄) is estimated over only 90 days of post-event
      data; given our 5-year panel, this is a WITHIN-SAMPLE short-run effect,
      not a true long-run equilibrium.
    - No explicit control group exists for a single-unit ITS. All pool-level
      shocks concurrent with Shapella are confounders (Fed rate decisions in
      April 2023; also the SVB/USDC recovery in March 2023 was ongoing).

Outputs:
    output/tables/sha_its_split.csv       Split ITS (short + long window)
    output/tables/sha_its_simple.csv      Simple ITS
    output/tables/sha_pretrend.csv        Pre-period trend test
    output/tables/sha_event_window.csv    Weekly outcomes ±12 weeks
    output/tables/sha_placebo.csv         Placebo ITS (Apr 12, 2022)
    output/tables/sha_variance_shift.csv  Levene variance test pre vs post
    output/figures/sha_its.pdf            ITS time-series plot
    output/figures/sha_event_window.pdf   Event-window bar charts

References:
    EIP-4895: Beacon chain push withdrawals (2023). Ethereum EIPs.
    Kiayias, A. & Livshits, B. (2022). Beacon chain economics. Working paper.
    Bernal, J.L. et al. (2017). ITS regression. Int. J. Epidemiology.
    Milionis, J. et al. (2022). AMM and loss-versus-rebalancing. Working paper.
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
    load, savefig, savetable, stars, ols_hac, bh_correction,
)

# ── Constants ─────────────────────────────────────────────────────────────────

SHAPELLA      = pd.Timestamp("2023-04-12 22:27:35", tz="UTC")  # block 6209536
PLACEBO       = SHAPELLA - pd.DateOffset(years=1)
PRE_DAYS      = 120
SHORT_POST    = 30
LONG_POST     = 120
EXCL_DAYS     = 7
EXCL_LONG     = 14   # robustness: wider exclusion given gradual withdrawal queue

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
    return panel.sort_index()


# ── Simple ITS ────────────────────────────────────────────────────────────────

def simple_its(series: pd.Series, event: pd.Timestamp,
               excl: int = EXCL_DAYS, label: str = "") -> dict | None:
    start = event - pd.Timedelta(days=PRE_DAYS)
    end   = event + pd.Timedelta(days=LONG_POST)
    el = event - pd.Timedelta(days=excl); eh = event + pd.Timedelta(days=excl)
    sub = series.loc[start:end].copy()
    sub = sub[~((sub.index >= el) & (sub.index <= eh))].dropna()
    if len(sub) < 80:
        return None
    t   = np.arange(len(sub), dtype=float)
    T_c = float(sub.index.searchsorted(event))
    if T_c <= 5 or T_c >= len(sub) - 5:
        return None
    post = (t >= T_c).astype(float)
    X    = pd.DataFrame({"t": t, "Post": post, "slope_post": (t - T_c) * post},
                        index=sub.index)
    tbl  = ols_hac(sub.rename("y"), X, max_lags=168, label=label)
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
    return out


# ── Split ITS (short + long) ──────────────────────────────────────────────────

def split_its(series: pd.Series, event: pd.Timestamp,
              excl: int = EXCL_DAYS, label: str = "") -> dict | None:
    """
    Split ITS: separate slopes for short (0-30d) and long (31-120d) post-windows.
    Model: y = α + β₁t + β₂Post + β₃(t-Tc)Post_short + β₄(t-Tc)Post_long + ε
    β₃: short-run slope change (expected positive for vol — selling pressure)
    β₄: long-run slope change (expected negative for vol — uncertainty resolved)
    """
    start = event - pd.Timedelta(days=PRE_DAYS)
    end   = event + pd.Timedelta(days=LONG_POST)
    el = event - pd.Timedelta(days=excl); eh = event + pd.Timedelta(days=excl)
    sub = series.loc[start:end].copy()
    sub = sub[~((sub.index >= el) & (sub.index <= eh))].dropna()
    if len(sub) < 80:
        return None
    t   = np.arange(len(sub), dtype=float)
    T_c = float(sub.index.searchsorted(event))
    T_short = float(sub.index.searchsorted(event + pd.Timedelta(days=SHORT_POST)))
    if T_c <= 5 or T_c >= len(sub) - 5:
        return None
    post      = (t >= T_c).astype(float)
    post_shrt = ((t >= T_c) & (t < T_short)).astype(float)
    post_long = (t >= T_short).astype(float)
    X = pd.DataFrame({
        "t":           t,
        "Post":        post,
        "slope_short": (t - T_c) * post_shrt,
        "slope_long":  (t - T_short) * post_long,
    }, index=sub.index)
    tbl = ols_hac(sub.rename("y"), X, max_lags=168, label=label)
    if tbl.empty:
        return None
    out: dict = {"N": len(sub)}
    for var in ["t", "Post", "slope_short", "slope_long"]:
        if var in tbl.index:
            r = tbl.loc[var]
            out[f"{var}_coef"] = r.get("Coef", np.nan)
            out[f"{var}_pval"] = r.get("p-val", np.nan)
            out[f"{var}_sig"]  = r.get("Sig", "")
    out["theory_pred"] = (
        "Shapella prediction: slope_short>0 for vol (pent-up ETH selling); "
        "slope_long<0 for vol (uncertainty resolved). "
        "Opposite signs support the dual-effect hypothesis."
    )
    return out


# ── Pre-trend test ────────────────────────────────────────────────────────────

def pre_trend_test(series: pd.Series, event: pd.Timestamp,
                   excl: int = EXCL_DAYS, label: str = "") -> dict | None:
    """
    Test H0: linear time trend = 0 in pre-period [event-PRE_DAYS, event-excl].
    A flat pre-trend supports ITS validity (no anticipatory dynamics).
    """
    start = event - pd.Timedelta(days=PRE_DAYS)
    end   = event - pd.Timedelta(days=excl)
    sub   = series.loc[start:end].dropna()
    if len(sub) < 30:
        return None
    t = np.arange(len(sub), dtype=float)
    X = pd.DataFrame({"t": t}, index=sub.index)
    tbl = ols_hac(sub.rename("y"), X, max_lags=168, label=label)
    if tbl.empty:
        return None
    if "t" in tbl.index:
        r = tbl.loc["t"]
        return {"coef_t": r.get("Coef", np.nan), "pval_t": r.get("p-val", np.nan),
                "sig_t": r.get("Sig", ""), "N": len(sub),
                "NOTE": (
                    "Pre-period trend (β₁). H0: β₁=0 (flat trend = no anticipatory dynamics). "
                    "Significant β₁ suggests Shapella was partially anticipated or the outcome "
                    "was already trending before the event — weakens ITS validity."
                )}
    return None


# ── Variance shift test ───────────────────────────────────────────────────────

def variance_shift(panel: pd.DataFrame) -> pd.DataFrame:
    """Levene (1960) test: equal variance pre vs post Shapella."""
    rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        pre  = panel.loc[SHAPELLA - pd.Timedelta(days=PRE_DAYS):
                         SHAPELLA - pd.Timedelta(days=EXCL_DAYS), col].dropna()
        post = panel.loc[SHAPELLA + pd.Timedelta(days=EXCL_DAYS):
                         SHAPELLA + pd.Timedelta(days=LONG_POST), col].dropna()
        if len(pre) < 20 or len(post) < 20:
            continue
        F, p = sp_stats.levene(pre.values, post.values)
        rows.append({
            "outcome": col, "label": label,
            "var_pre":  round(float(pre.var()), 6),
            "var_post": round(float(post.var()), 6),
            "Levene_F": round(F, 4), "Levene_p": round(p, 6),
            "sig":      stars(p),
            "direction": "post < pre" if post.var() < pre.var() else "post > pre",
            "NOTE": (
                "Levene (1960) H0: equal variance pre/post. "
                "Shapella prediction: post variance LOWER (uncertainty resolved). "
                "'post < pre' + significant p → supports long-run vol reduction hypothesis."
            ),
        })
    return pd.DataFrame(rows).set_index("outcome") if rows else pd.DataFrame()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_its(panel: pd.DataFrame) -> None:
    cols = [c for c in OUTCOMES if c in panel.columns]
    nrow = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 3.5 * nrow))
    flat = axes.flatten()
    for i, col in enumerate(cols):
        ax = flat[i]
        ser = panel[col].loc[SHAPELLA - pd.Timedelta(days=PRE_DAYS):
                              SHAPELLA + pd.Timedelta(days=LONG_POST)]
        weekly = ser.resample("7D").mean()
        ax.plot(weekly.index, weekly.values, color="#2563eb", lw=1.2)
        ax.axvline(SHAPELLA, color="black", lw=2, ls="--", label="Shapella")
        ax.axvline(SHAPELLA + pd.Timedelta(days=SHORT_POST), color="#d97706",
                   lw=1.2, ls=":", label=f"+{SHORT_POST}d (short/long split)")
        ax.set_title(OUTCOMES[col], fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)
    plt.suptitle("Shapella upgrade (Apr 12 2023) — short/long ITS split", fontsize=10, y=1.01)
    savefig("sha_its")


def plot_event_window(ew: pd.DataFrame) -> None:
    if ew.empty:
        return
    cols = ew["outcome"].unique()
    nrow = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 3 * nrow))
    flat = axes.flatten()
    for i, col in enumerate(cols):
        ax = flat[i]
        sub = ew[ew["outcome"] == col].sort_values("week")
        colors = ["#dc2626" if v >= 0 else "#2563eb" for v in sub["mean_norm"]]
        ax.bar(sub["week"], sub["mean_norm"] * 100, color=colors, alpha=0.7)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.axvline(-0.5, color="black", lw=1.5, alpha=0.5)
        ax.axvline(SHORT_POST / 7 - 0.5, color="#d97706", lw=1.2, ls=":")
        ax.set_title(OUTCOMES.get(col, col), fontsize=9)
        ax.set_xlabel("Weeks relative to Shapella", fontsize=8)
        ax.set_ylabel("% change from pre-event mean", fontsize=8)
    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)
    plt.suptitle("Event-window study ±12 wks | Orange dotted = short/long split", fontsize=10, y=1.01)
    savefig("sha_event_window")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = load_panel()
    if panel is None:
        print("  [SKIP] Required data missing."); return

    print(f"  Shapella: {SHAPELLA}  |  Placebo: {PLACEBO.date()}")

    # 1. Split ITS
    split_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        for spec, excl in [("main", EXCL_DAYS), (f"excl_{EXCL_LONG}d", EXCL_LONG)]:
            res = split_its(panel[col], SHAPELLA, excl=excl, label=label)
            if res:
                res["outcome"] = col; res["spec"] = spec
                split_rows.append(res)
    if split_rows:
        savetable(pd.DataFrame(split_rows).set_index(["outcome", "spec"]), "sha_its_split")

    # 2. Simple ITS + placebo
    simple_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        for spec, event in [("main", SHAPELLA), ("placebo", PLACEBO)]:
            res = simple_its(panel[col], event, label=label)
            if res:
                res["outcome"] = col; res["spec"] = spec
                simple_rows.append(res)
    if simple_rows:
        savetable(pd.DataFrame(simple_rows).set_index(["outcome", "spec"]), "sha_its_simple")
        # BH correction
        main_r = [r for r in simple_rows if r.get("spec") == "main"]
        pv = [r.get("Post_pval", np.nan) for r in main_r]
        pv_c = [p for p in pv if not np.isnan(p)]
        if pv_c:
            bh = bh_correction(pv_c, alpha=0.10)
            print(f"  BH(10%) rejects (Post): {sum(bh)}/{len(bh)}")

    # 3. Pre-trend test
    pt_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        res = pre_trend_test(panel[col], SHAPELLA, label=label)
        if res:
            res["outcome"] = col; res["label"] = label
            pt_rows.append(res)
    if pt_rows:
        savetable(pd.DataFrame(pt_rows).set_index("outcome"), "sha_pretrend")

    # 4. Event-window study
    ew_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        ser = panel[col].dropna()
        pm = ser[(ser.index >= SHAPELLA - pd.Timedelta(weeks=12)) & (ser.index < SHAPELLA)]
        pm_mean = float(pm.mean())
        if pm_mean == 0 or np.isnan(pm_mean):
            continue
        for w in range(-12, 13):
            lo = SHAPELLA + pd.Timedelta(weeks=w)
            hi = lo + pd.Timedelta(weeks=1)
            vals = ser[(ser.index >= lo) & (ser.index < hi)].dropna().values
            if len(vals) >= 5:
                ew_rows.append({"outcome": col, "week": w,
                                "mean_norm": float(np.mean(vals) / pm_mean - 1.0),
                                "N": len(vals)})
    if ew_rows:
        ew = pd.DataFrame(ew_rows)
        savetable(ew, "sha_event_window")
        plot_event_window(ew)

    # 5. Variance shift
    df_var = variance_shift(panel)
    if not df_var.empty:
        savetable(df_var, "sha_variance_shift")

    # 6. Plots
    plot_its(panel)

    print("  [DONE] did_shapella.py")


if __name__ == "__main__":
    main()
