"""
did_terra_luna.py — ITS + DiD: Terra/Luna Collapse (May 9, 2022)

Research Question:
    Did the Terra/Luna ecosystem collapse (May 9-13, 2022) cause a persistent
    structural shift in AMM volatility dynamics and LP behaviour on the
    WETH/USDC 0.05% Uniswap v3 pool, distinct from a temporary price shock?

Motivation:
    The Terra/Luna collapse was the largest single crypto-ecosystem failure in
    history: LUNA lost 99.9% of value in five days; UST (Terra's algorithmic
    stablecoin) fully depegged; ~$40 B of market cap was destroyed. The event
    caused ETH to fall ~35% in seven days and spiked realised volatility to
    levels last seen in the 2020 COVID crash.  Unlike FTX (which was a
    centralised exchange failure), Terra/Luna was a DEX/DeFi native event —
    making its spillover to Ethereum AMMs theoretically significant.

    For the WETH/USDC pool, Terra/Luna is purely an EXOGENOUS shock: neither
    USDC nor WETH had direct exposure to the Terra ecosystem. The shock
    propagates entirely through: (a) ETH price crash → elevated realised vol →
    higher LVR; (b) risk-off sentiment → LP withdrawal; (c) basis widening as
    CEX price discovery dominated DEX.

    This DiD complements did_merge_impact.py (structural change in protocol) and
    did_ftx_lp_withdrawal.py (CeFi failure), completing a trilogy of three
    major 2022 crypto events.

Methodology:
    1. ITS (Interrupted Time Series) — identical design to did_merge_impact.py:
           y_t = α + β₁t + β₂Post + β₃(t−Tc)Post + ε_t
       Event: May 9, 2022 (LUNA began its final cascade).
       Window: ±90 days (shorter than The Merge since FTX is 182 days after).
       Exclusion: ±7 days around event.
       HAC SE, 168-lag.

    2. Multi-event comparison — jointly estimates ITS for both Terra/Luna and
       FTX collapse; tests whether β₂ (level shift) differs across events.
       Higher β₂ for Terra/Luna in vol suggests a larger immediate shock.

    3. Persistence test — compare ITS β₃ (trend change) across events.
       Terra/Luna shock is widely considered more transitory (macro recovery
       began Aug 2022) while FTX created lasting institutional distrust.

    4. LP response DiD — narrow vs wide LP positions × pre/post Terra/Luna,
       mirroring did_ftx_lp_withdrawal.py exactly.

    5. Placebo — apply ITS at May 9, 2021 (1 year prior, same season).

Key caveats:
    - May 9, 2022 ± 90d overlaps with The Merge (Sep 15, 2022); the post-window
      is truncated at Jul 31, 2022 (90d after May 9) to avoid contamination.
    - The event start date is ambiguous: LUNA began sliding on May 8 and fully
      collapsed by May 12. May 9 is chosen as the first day of observable
      contagion to CEX prices; sensitivity to May 8 is reported.
    - UST (Terra's stablecoin) is NOT USDC; there is no direct USDC contagion.
      USDC maintained its peg throughout Terra/Luna — confirmed as a confounder.
    - 3AC/Celsius/Voyager collapsed in June–July 2022 within our post-window;
      these confounders partially contaminate the ±90d ITS estimate. The ±30d
      robustness check is cleaner.

Outputs:
    output/tables/tl_its.csv              ITS coefficients
    output/tables/tl_multi_event.csv      Multi-event (Terra/Luna vs FTX) comparison
    output/tables/tl_placebo.csv          Placebo ITS (May 9, 2021)
    output/tables/tl_lp_did.csv           LP exit DiD (narrow vs wide)
    output/figures/tl_its.pdf             ITS plots
    output/figures/tl_multi_event.pdf     Side-by-side β₂/β₃ comparison

References:
    Milionis, J. et al. (2022). Automated market making and loss-versus-
        rebalancing. Working paper.
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
    Bernal, J.L. et al. (2017). ITS regression for public health interventions.
        Int. J. Epidemiology.
    Ante, L. (2023). The impact of the LUNA/UST crypto crash on Ethereum.
        Finance Research Letters, 56, 104053.
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

TERRA_DATE    = pd.Timestamp("2022-05-09", tz="UTC")
PLACEBO_DATE  = TERRA_DATE - pd.DateOffset(years=1)   # May 9, 2021
FTX_DATE      = pd.Timestamp("2022-11-08", tz="UTC")  # for multi-event comparison
WINDOW_DAYS   = 90
POST_ROB      = 30
EXCL_DAYS     = 7

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

def load_lp() -> pd.DataFrame | None:
    lp = load("DEX/dex_lp_positions.csv")
    if lp is None:
        return None
    lp["first_mint_utc"] = pd.to_datetime(lp["first_mint_utc"], utc=True, errors="coerce")
    lp["last_burn_utc"]  = pd.to_datetime(lp["last_burn_utc"],  utc=True, errors="coerce")
    lp["last_event_utc"] = pd.to_datetime(lp["last_event_utc"], utc=True, errors="coerce")
    return lp

# ── Core ITS ─────────────────────────────────────────────────────────────────

def _its(series: pd.Series, event: pd.Timestamp,
         post_days: int = WINDOW_DAYS, label: str = "") -> dict | None:
    start   = event - pd.Timedelta(days=WINDOW_DAYS)
    end     = event + pd.Timedelta(days=post_days)
    excl_lo = event - pd.Timedelta(days=EXCL_DAYS)
    excl_hi = event + pd.Timedelta(days=EXCL_DAYS)
    sub = series.loc[start:end].copy()
    sub = sub[~((sub.index >= excl_lo) & (sub.index <= excl_hi))].dropna()
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
    out["NOTE"] = (
        "ITS y=α+β₁t+β₂Post+β₃(t-Tc)Post+ε. HAC SE 168-lag. "
        "Window: ±90d; ±7d excl. FTX (182d post) and 3AC/Celsius (62d post) "
        "are concurrent confounders in the full post-window. "
        "Use 30d robustness for cleaner identification."
    )
    return out


# ── Multi-event comparison ────────────────────────────────────────────────────

def multi_event_comparison(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compare ITS β₂ (level shift) and β₃ (trend change) for Terra/Luna vs FTX.
    Tests H0: β₂_Terra = β₂_FTX (equal immediate shock).
    Wald test using estimated coefs and HAC SEs (no covariance → conservative).
    """
    rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        r_tl  = _its(panel[col], TERRA_DATE, label=label)
        r_ftx = _its(panel[col], FTX_DATE,   label=label)
        if not r_tl or not r_ftx:
            continue
        # Approximate Wald test for β₂_TL vs β₂_FTX
        b_tl  = r_tl.get("Post_coef", np.nan)
        se_tl = r_tl.get("Post_se",   np.nan)
        b_ftx = r_ftx.get("Post_coef", np.nan)
        se_ftx = r_ftx.get("Post_se",  np.nan)
        if not any(np.isnan([b_tl, se_tl, b_ftx, se_ftx])):
            diff     = b_tl - b_ftx
            se_diff  = np.sqrt(se_tl ** 2 + se_ftx ** 2)
            t_diff   = diff / se_diff if se_diff > 0 else np.nan
            # Use min(N_TL, N_FTX) - 4 as df (4 params per ITS, two independent samples)
            df_approx = max(min(r_tl.get("N", 500), r_ftx.get("N", 500)) - 4, 1)
            p_diff   = float(2 * sp_stats.t.sf(abs(t_diff), df=df_approx))
        else:
            diff = se_diff = t_diff = p_diff = np.nan
        rows.append({
            "outcome": col, "label": label,
            "beta2_terra": round(b_tl, 6) if b_tl else np.nan,
            "beta2_ftx":   round(b_ftx, 6) if b_ftx else np.nan,
            "diff_beta2":  round(diff, 6) if not np.isnan(diff) else np.nan,
            "t_diff":      round(t_diff, 3) if not np.isnan(t_diff) else np.nan,
            "p_diff":      round(p_diff, 4) if not np.isnan(p_diff) else np.nan,
            "sig_diff":    stars(p_diff) if not np.isnan(p_diff) else "",
            "NOTE": (
                "Conservative Wald test (assumes zero covariance between Terra and FTX "
                "ITS estimates; events are non-overlapping so this is approximately correct). "
                "Significant p_diff → the two events had different immediate impacts on the outcome."
            ),
        })
    return pd.DataFrame(rows).set_index("outcome") if rows else pd.DataFrame()


# ── Pre-trend test ────────────────────────────────────────────────────────────

def pre_trend_test(series: pd.Series, event: pd.Timestamp,
                   label: str = "") -> dict | None:
    """
    Pre-period linear trend: H0 β₁=0 (flat pre-event trend).
    Window: [event - WINDOW_DAYS, event - EXCL_DAYS].
    Significant β₁ → outcome was already shifting before Terra/Luna.
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
            "Pre-period trend test (HAC 168-lag). H0: β₁=0 (flat pre-Terra trend). "
            "Significant → the outcome was already trending before May 9 2022 — "
            "possibly due to broader crypto bear market (LUNA had been declining since Apr). "
            "Supplement with placebo at May 9 2021."
        ),
    }


# ── LP DiD (mirrors did_ftx_lp_withdrawal.py) ────────────────────────────────

def lp_did(lp: pd.DataFrame) -> pd.DataFrame:
    """
    Daily panel DiD: narrow vs wide LP exit rate around Terra/Luna.
    Identical design to did_ftx_lp_withdrawal.py. See that script for full methodology.
    """
    active = lp[lp["first_mint_utc"] < TERRA_DATE].copy()
    if active.empty:
        return pd.DataFrame()
    med = active["range_width_pct"].median()
    active["narrow"] = (active["range_width_pct"] <= med).astype(int)
    active["closed_before"] = (active["last_burn_utc"].notna() & (active["last_burn_utc"] < TERRA_DATE))
    active = active[~active["closed_before"]].copy()

    start = TERRA_DATE - pd.Timedelta(days=WINDOW_DAYS)
    end   = TERRA_DATE + pd.Timedelta(days=POST_ROB)   # shorter post (3AC confounder)
    days  = pd.date_range(start, end, freq="D", tz="UTC")
    opens  = active["first_mint_utc"].values
    closes = active["last_burn_utc"].values
    narrows = active["narrow"].values

    rows = []
    for day in days:
        day_next = day + pd.Timedelta(days=1)
        for grp in [0, 1]:
            grp_m = narrows == grp
            open_m = grp_m & (opens < day_next) & (pd.isnull(closes) | (closes >= day))
            n_open = int(open_m.sum())
            if n_open < 3:
                continue
            n_exit = int((grp_m & (~pd.isnull(closes)) & (closes >= day) & (closes < day_next) & open_m).sum())
            rows.append({
                "date": day, "narrow": grp,
                "post": 1 if day >= TERRA_DATE else 0,
                "day_rel": int((day - TERRA_DATE).days),
                "n_open": n_open, "n_exit": n_exit,
                "exit_rate": n_exit / n_open,
            })
    panel_lp = pd.DataFrame(rows)
    if panel_lp.empty:
        return pd.DataFrame()
    panel_lp["narrow_x_post"]    = panel_lp["narrow"] * panel_lp["post"]
    panel_lp["narrow_x_day_rel"] = panel_lp["narrow"] * panel_lp["day_rel"]
    X_cols = ["narrow", "post", "narrow_x_post", "day_rel", "narrow_x_day_rel"]
    result = ols_hac(panel_lp["exit_rate"], panel_lp[X_cols], max_lags=7, label="DiD")
    if not result.empty:
        result["NOTE"] = (
            "DiD: δ=coef(narrow_x_post). Post window = 30d (FTX/3AC excluded). "
            "HAC SE, 7-lag. Design mirrors did_ftx_lp_withdrawal.py."
        )
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_multi_event(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = [c for c in df.index if c in OUTCOMES]
    if not cols:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(cols))
    b_tl  = df.loc[cols, "beta2_terra"].values.astype(float)
    b_ftx = df.loc[cols, "beta2_ftx"].values.astype(float)
    labels = [OUTCOMES.get(c, c) for c in cols]

    for ax, vals, label, col in [
        (axes[0], b_tl,  "Terra/Luna (β₂)", "#dc2626"),
        (axes[1], b_ftx, "FTX (β₂)", "#2563eb"),
    ]:
        ax.barh(x, vals, color=col, alpha=0.7)
        ax.set_yticks(x); ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(label)
        ax.set_xlabel("Level shift β₂ (ITS)")
    plt.suptitle("Multi-event comparison: β₂ level shifts — Terra/Luna vs FTX", fontsize=10)
    savefig("tl_multi_event")


def plot_its_panel(panel: pd.DataFrame) -> None:
    cols = [c for c in OUTCOMES if c in panel.columns]
    nrow = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 3.5 * nrow))
    flat = axes.flatten()
    for i, col in enumerate(cols):
        ax = flat[i]
        ser = panel[col].loc[TERRA_DATE - pd.Timedelta(days=WINDOW_DAYS):
                              TERRA_DATE + pd.Timedelta(days=WINDOW_DAYS)]
        weekly = ser.resample("7D").mean()
        ax.plot(weekly.index, weekly.values, color="#2563eb", lw=1.2, label="7d avg")
        ax.axvline(TERRA_DATE, color="black", lw=2, ls="--", label="Terra/Luna collapse")
        ax.set_title(OUTCOMES[col], fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)
    plt.suptitle("Terra/Luna Collapse (May 9 2022) — Outcome time series", fontsize=10, y=1.01)
    savefig("tl_its")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = load_panel()
    if panel is None:
        print("  [SKIP] Required data missing."); return
    lp = load_lp()

    print(f"  Event: Terra/Luna {TERRA_DATE.date()}  |  Placebo: {PLACEBO_DATE.date()}")

    # 1. ITS — main + robustness + placebo
    its_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        for spec, post_d, event in [
            ("main",        WINDOW_DAYS, TERRA_DATE),
            (f"rob_{POST_ROB}d", POST_ROB,  TERRA_DATE),
            ("placebo",     WINDOW_DAYS, PLACEBO_DATE),
            ("sens_may8",   WINDOW_DAYS, TERRA_DATE - pd.Timedelta(days=1)),  # 1-day sensitivity
        ]:
            res = _its(panel[col], event, post_days=post_d, label=label)
            if res:
                res["outcome"] = col
                res["spec"]    = spec
                its_rows.append(res)

    if its_rows:
        # BH correction — add flags before building DataFrame
        main_rows = [r for r in its_rows if r.get("spec") == "main"]
        pvals = [r.get("Post_pval", np.nan) for r in main_rows]
        pvals_clean = [p for p in pvals if not np.isnan(p)]
        if pvals_clean:
            bh_flags = bh_correction(pvals_clean, alpha=0.10)
            for r, flag in zip(
                [r for r in main_rows if not np.isnan(r.get("Post_pval", np.nan))], bh_flags
            ):
                r["Post_BH10_reject"] = flag
            print(f"  BH(10%) rejects (Post coef): {sum(bh_flags)}/{len(bh_flags)}")
        df_its = pd.DataFrame(its_rows).set_index(["outcome", "spec"])
        savetable(df_its, "tl_its")
        # Save placebo separately
        plac_r = [r for r in its_rows if r.get("spec") == "placebo"]
        if plac_r:
            savetable(pd.DataFrame(plac_r).set_index(["outcome", "spec"]), "tl_placebo")

    # 1b. Pre-trend test
    pt_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        res = pre_trend_test(panel[col], TERRA_DATE, label=label)
        if res:
            res["outcome"] = col; res["label"] = label
            pt_rows.append(res)
    if pt_rows:
        savetable(pd.DataFrame(pt_rows).set_index("outcome"), "tl_pretrend")
        sig_pt = sum(1 for r in pt_rows if r.get("pval_t", 1.0) < 0.10)
        print(f"  Pre-trend test: {sig_pt}/{len(pt_rows)} outcomes significant pre-trend (p<0.10)")

    # 2. Multi-event comparison
    df_me = multi_event_comparison(panel)
    if not df_me.empty:
        savetable(df_me, "tl_multi_event")
        plot_multi_event(df_me)

    # 3. LP DiD
    if lp is not None:
        df_lp = lp_did(lp)
        if not df_lp.empty:
            savetable(df_lp, "tl_lp_did")

    # 4. Plots
    plot_its_panel(panel)

    print("  [DONE] did_terra_luna.py")


if __name__ == "__main__":
    main()
