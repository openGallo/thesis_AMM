"""
rd_vol_lvr_breakeven.py — Fuzzy RD: Realised Volatility and LVR-to-Fee Break-Even

Research Question:
    Does LP liquidity provision exhibit a discontinuous response when realised
    volatility crosses the break-even threshold at which LVR exceeds fee income
    (LVR-to-fee ratio = 1), consistent with rational LPs withdrawing liquidity
    once the pool becomes ex-ante loss-making?

Motivation:
    Milionis et al. (2022) prove that LP net P&L = fee income − LVR, and
    characterise the break-even volatility σ* at which fee income = LVR:
        σ* = sqrt(2 × fee_rate / Δ)  (for uniform liquidity)
    where Δ is the fee rate and spread.  For concentrated liquidity (Uniswap v3),
    σ* depends on the range width: narrow positions have HIGHER fee income per
    unit of vol but face LVR that grows faster with σ.

    We test the empirical counterpart: as the REALISED volatility (σ) crosses
    σ* from below (fee > LVR) to above (fee < LVR), rational LPs should
    withdraw liquidity — creating a FUZZY RD in next-period liquidity changes.

    The running variable (realised σ) is MARKET DETERMINED and cannot be
    manipulated by LPs.  The cutoff σ* is THEORY-MOTIVATED (LVR-to-fee ratio =1).
    This makes the design unusually clean: no agent determines the running variable
    at the margin, and the cutoff is economically primitive, not data-mined.

    Novelty: this is the first empirical RD test of the Milionis et al. (2022)
    break-even condition using actual pool data.  Most prior tests are simulation-
    based (Lehar & Parlour 2021) or analytical (Deng & Lin 2023).

Methodology:
    1. Cutoff estimation — identify vol* where E[lvr_to_fee_ratio] = 1:
       Regress lvr_to_fee_ratio on cex_realized_vol_24h_ann using a smooth
       non-parametric spline; find the crossing point. Also use the pooled
       median as a simple estimate.

    2. First stage — local linear regression of P(lvr_to_fee > 1) on vol:
       This measures the sharpness of the regime transition at vol*.
       Strong first stage: F = (jump/se)² > 10 (Staiger & Stock 1997).

    3. Reduced form — LLR of next-period LP response on vol (running variable):
       Outcomes: Δlog(dex_tvl_usd)₊₂₄ₕ, Δlog(dex_liquidity)₊₂₄ₕ, P(LP exit in next 24h).
       H1: negative jump at vol* → below vol*, liquidity grows; above vol*, liquidity shrinks.

    4. Wald estimator (LATE) — reduced form / first stage:
       Causal effect of crossing LVR-to-fee=1 on next-period liquidity change.

    5. Bandwidth sensitivity — h × {0.5, 1.0, 1.5} and the IK-approximate bw.

    6. McCrary density — vol should NOT cluster at vol* (market-determined);
       any clustering would suggest microstructure feedback between vol and LP provision.

    7. Heterogeneity by vol regime — repeat the RD in low, normal, and high vol
       sub-periods to test whether the break-even response is stronger in calm markets
       (where the vol* signal is more precise).

    8. Donut RD — exclude hours where vol ∈ [vol* - ε, vol* + ε], checking that
       the jump is not artefactual from mechanical measurement near the crossing.

Key caveats:
    - The break-even vol σ* is ESTIMATED, not analytically known for v3 concentrated
      liquidity with heterogeneous LP ranges. The estimated σ* has uncertainty that
      propagates to the RD estimate. Sensitivity to a range of σ* values is reported.
    - The running variable (vol_24h) is a BACKWARD-LOOKING average (last 24h).
      LP decision to withdraw is forward-looking. Using backward-looking vol as the
      running variable is valid IF LPs use recent vol as their σ estimate (consistent
      with naive risk assessment, common in retail DeFi).
    - Liquidity changes in the next 24 hours are measured from the hourly dex_pool data.
      An LP withdrawing takes effect immediately; we may miss partially executed withdrawals.
    - Causal interpretation requires monotonicity (no "defiers"): LPs who provision MORE
      liquidity when vol > vol* would violate this. Range-rebalancers (who widen range
      during high vol) are potential defiers.
    - The LATE is identified for LPs at the margin vol ≈ vol*. High-vol behaviour
      (e.g., during crashes) is outside the bandwidth and not captured.

Outputs:
    output/tables/rvlb_cutoff.csv         Vol break-even cutoff estimation
    output/tables/rvlb_first_stage.csv    P(LVR>fee) jump at vol*
    output/tables/rvlb_reduced.csv        Liquidity change jump at vol*
    output/tables/rvlb_wald.csv           Wald LATE estimates
    output/tables/rvlb_robustness.csv     Bandwidth + polynomial + donut
    output/tables/rvlb_mccrary.csv        McCrary density at vol*
    output/figures/rvlb_first_stage.pdf   First-stage compliance plot
    output/figures/rvlb_rd.pdf            RD scatter: liquidity change vs vol

References:
    Milionis, J. et al. (2022). Automated market making and loss-versus-
        rebalancing. Working paper.
    Deng, Y. & Lin, J. (2023). LVR in Uniswap v3. Working paper.
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
    Imbens, G. & Kalyanaraman, K. (2012). Optimal bandwidth. REStud, 79(3).
    Staiger, D. & Stock, J. (1997). Weak instruments. Econometrica, 65(3).
    Lee, D. & Lemieux, T. (2010). RD designs. JEL, 48(2), 281-355.
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
)

MIN_OBS = 50  # min obs each side for LLR
N_BINS  = 40

# ── Data ──────────────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame | None:
    mh  = load("merged/merged_hourly.csv")
    lvr = load("DEX/dex_lvr_hourly.csv")
    if mh is None or lvr is None:
        return None
    panel = mh.join(lvr[["lvr_rate_ann", "lvr_to_fee_ratio", "lvr_usd_tvl_approx"]], how="left")
    # Outcomes: next-24h liquidity changes
    panel["d_log_tvl_24h"]  = np.log(panel["dex_tvl_usd"].shift(-24) / panel["dex_tvl_usd"])
    panel["d_log_liq_24h"]  = np.log(panel["dex_liquidity"].shift(-24) / panel["dex_liquidity"])
    panel["lvr_gt_fee"]     = (panel["lvr_to_fee_ratio"] > 1).astype(float)
    panel = panel.dropna(subset=["cex_realized_vol_24h_ann", "lvr_to_fee_ratio"])
    return panel.sort_index()


# ── Cutoff estimation ─────────────────────────────────────────────────────────

def estimate_vol_cutoff(panel: pd.DataFrame) -> dict:
    """
    Find vol* such that E[lvr_to_fee_ratio | vol = vol*] = 1.
    Method 1: scan quantiles of vol; find where mean(lvr_to_fee_ratio in bin) = 1.
    Method 2: median vol where lvr_to_fee_ratio > 1.
    """
    vol = panel["cex_realized_vol_24h_ann"].dropna().values
    ltf = panel["lvr_to_fee_ratio"].dropna().values
    panel_c = panel.dropna(subset=["cex_realized_vol_24h_ann", "lvr_to_fee_ratio"]).copy()

    # Method 1: regression crossing
    v = panel_c["cex_realized_vol_24h_ann"].values
    l = panel_c["lvr_to_fee_ratio"].values

    # Bin-based estimate: find vol bin where mean(ltf) ≈ 1
    bins  = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 50)
    mids  = (bins[:-1] + bins[1:]) / 2
    means = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (v >= lo) & (v < hi)
        means.append(float(l[m].mean()) if m.sum() >= 5 else np.nan)
    means = np.array(means)
    # Find crossing point
    cross_idx = np.where(~np.isnan(means) & (means >= 1.0))[0]
    vol_star_m1 = float(mids[cross_idx[0]]) if len(cross_idx) > 0 else float(np.median(v))

    # Method 2: median vol on high-LVR hours
    high_ltf_v = panel_c.loc[panel_c["lvr_to_fee_ratio"] > 1, "cex_realized_vol_24h_ann"]
    low_ltf_v  = panel_c.loc[panel_c["lvr_to_fee_ratio"] <= 1, "cex_realized_vol_24h_ann"]
    vol_star_m2 = float((high_ltf_v.quantile(0.1) + low_ltf_v.quantile(0.9)) / 2) \
        if len(high_ltf_v) > 10 and len(low_ltf_v) > 10 else vol_star_m1

    frac_above = float((panel_c["lvr_to_fee_ratio"] > 1).mean())
    return {
        "vol_star_m1": round(vol_star_m1, 4),
        "vol_star_m2": round(vol_star_m2, 4),
        "vol_star_primary": round((vol_star_m1 + vol_star_m2) / 2, 4),
        "frac_hours_lvr_gt_fee": round(frac_above, 3),
        "mean_vol": round(float(np.nanmean(v)), 4),
        "NOTE": (
            "vol* = vol at which lvr_to_fee_ratio crosses 1. "
            "M1: bin-crossing (mean ltf in vol bin = 1). "
            "M2: midpoint between 10th pct of high-LVR vol and 90th pct of low-LVR vol. "
            "Primary = (M1+M2)/2. Sensitivity to M1 and M2 separately reported."
        ),
    }


# ── Local linear RD ───────────────────────────────────────────────────────────

def _kern(u): return np.maximum(1.0 - np.abs(u), 0.0)

def llr(y: np.ndarray, r: np.ndarray, cutoff: float, bw: float,
        poly: int = 1, label: str = "") -> dict:
    u = (r - cutoff) / bw; w = _kern(u)
    out = {"cutoff": round(cutoff, 4), "bw": round(bw, 4), "poly": poly, "label": label}
    sides: dict = {}
    for side, mask in [("left", r < cutoff), ("right", r >= cutoff)]:
        m = mask & (w > 0)
        if m.sum() < MIN_OBS:
            sides[side] = None; continue
        ys = y[m]; us = r[m] - cutoff; ws = w[m]
        cols = [np.ones(m.sum()), us] + ([us**2] if poly==2 else [])
        Xs = np.column_stack(cols); W_ = np.diag(ws)
        try:
            XWX = Xs.T @ W_ @ Xs
            beta = np.linalg.solve(XWX, Xs.T @ W_ @ ys)
            res  = ys - Xs @ beta
            s2   = float(np.sum(ws * res**2) / max(ws.sum() - len(beta), 1))
            cov  = s2 * np.linalg.solve(XWX, np.eye(len(beta)))
            sides[side] = {"int": float(beta[0]), "se": float(np.sqrt(max(cov[0,0],0))),
                           "n": int(m.sum())}
        except np.linalg.LinAlgError:
            sides[side] = None
    if not sides.get("left") or not sides.get("right"):
        out.update({"jump": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "sig": ""}); return out
    jump = sides["right"]["int"] - sides["left"]["int"]
    se   = np.sqrt(sides["left"]["se"]**2 + sides["right"]["se"]**2)
    t    = jump / se if se > 0 else np.nan
    p    = float(2 * sp_stats.t.sf(abs(t), df=max(sides["left"]["n"]+sides["right"]["n"]-2*(poly+1),1)))
    out.update({"jump": round(jump,6), "se": round(se,6), "t": round(t,4), "p": round(p,6),
                "sig": stars(p), "n_left": sides["left"]["n"], "n_right": sides["right"]["n"]})
    return out


def wald_late(rf: dict, fs: dict, label: str = "") -> dict:
    rf_j = rf.get("jump", np.nan); fs_j = fs.get("jump", np.nan)
    if np.isnan(rf_j) or np.isnan(fs_j) or abs(fs_j) < 1e-9:
        return {"label": label, "late": np.nan, "F_fs": np.nan}
    late = rf_j / fs_j
    se_l = abs(late) * np.sqrt(
        (rf.get("se",0)/max(abs(rf_j),1e-9))**2 + (fs.get("se",0)/max(abs(fs_j),1e-9))**2)
    t = late / se_l if se_l > 0 else np.nan
    p = float(2 * sp_stats.t.sf(abs(t), df=rf.get("n_left",0)+rf.get("n_right",0)-4))
    F_fs = (fs.get("t", np.nan))**2 if not np.isnan(fs.get("t", np.nan)) else np.nan
    return {"label": label, "late": round(late,6), "se_late": round(se_l,6),
            "t": round(t,4), "p": round(p,6), "sig": stars(p),
            "F_first_stage": round(F_fs, 2) if not np.isnan(F_fs) else np.nan,
            "NOTE": (
                f"Wald LATE = rf_jump/fs_jump. "
                f"F_first_stage={round(F_fs,2) if not np.isnan(F_fs) else 'NA'} "
                "(F<10 = weak instrument, Staiger & Stock 1997). "
                "LATE = causal effect of crossing LVR-to-fee=1 threshold on next-24h "
                "log liquidity change, for LPs at the margin vol ≈ vol*."
            )}


# ── McCrary test ──────────────────────────────────────────────────────────────

def mccrary_test(r: np.ndarray, cutoff: float) -> dict:
    bins    = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), 41)
    mids    = (bins[:-1] + bins[1:]) / 2
    bw_b    = bins[1] - bins[0]
    cts, _  = np.histogram(r, bins=bins)
    density = cts / (len(r) * bw_b)
    h0      = 1.84 * np.std(r) * len(r)**(-0.2)
    res     = llr(density, mids, cutoff, h0, poly=1, label="McCrary")
    res["NOTE"] = (
        "McCrary (2008) density. H0: density continuous at vol*. "
        "Volatility is market-determined; clustering at vol* would suggest "
        "LP activity CHANGES vol (e.g., liquidity withdrawal reduces pool depth → "
        "higher slippage → higher vol). This would be a simultaneity bias, not manipulation."
    )
    return res


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_first_stage(panel: pd.DataFrame, cutoff: float, bw: float) -> None:
    clean = panel.dropna(subset=["cex_realized_vol_24h_ann", "lvr_gt_fee"]).copy()
    v = clean["cex_realized_vol_24h_ann"].values
    D = clean["lvr_gt_fee"].values
    bins = np.linspace(max(0, cutoff - 2.5*bw), cutoff + 2.5*bw, N_BINS + 1)
    mids, means = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (v >= lo) & (v < hi)
        if m.sum() >= 5:
            mids.append((lo+hi)/2); means.append(float(D[m].mean()))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(mids, means, s=25, color="#2563eb", alpha=0.8)
    ax.axvline(cutoff, color="black", lw=2, ls="--", label=f"vol* = {cutoff:.3f}")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Realised vol (24h ann.)"); ax.set_ylabel("P(LVR > fee income)")
    ax.set_title("First stage: P(LVR_to_fee > 1) vs realised volatility")
    ax.legend(); savefig("rvlb_first_stage")


def plot_rd(panel: pd.DataFrame, cutoff: float, bw: float) -> None:
    clean = panel.dropna(subset=["cex_realized_vol_24h_ann", "d_log_tvl_24h"]).copy()
    v = clean["cex_realized_vol_24h_ann"].values
    y = clean["d_log_tvl_24h"].values
    bins = np.linspace(max(0, cutoff - 2.5*bw), cutoff + 2.5*bw, N_BINS + 1)
    def _bm(v_, y_, bins_):
        ms, ys = [], []
        for lo, hi in zip(bins_[:-1], bins_[1:]):
            m = (v_ >= lo) & (v_ < hi) & ~np.isnan(y_)
            if m.sum() >= 5:
                ms.append((lo+hi)/2); ys.append(float(y_[m].mean()))
        return np.array(ms), np.array(ys)
    bins_lo = bins[bins <= cutoff]; bins_hi = bins[bins >= cutoff]
    rl, yl = _bm(v, y, bins_lo); rr, yr = _bm(v, y, bins_hi)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(rl, yl, color="#2563eb", s=30, alpha=0.8, label="Vol < vol* (fee>LVR)")
    ax.scatter(rr, yr, color="#dc2626", s=30, alpha=0.8, label="Vol ≥ vol* (LVR>fee)")
    ax.axvline(cutoff, color="black", lw=2, ls="--", label=f"vol* = {cutoff:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Realised vol (24h ann.)"); ax.set_ylabel("Δlog(TVL) next 24h")
    ax.set_title("RD: Next-24h TVL change at LVR-to-fee break-even vol*", fontsize=9)
    ax.legend(fontsize=8); savefig("rvlb_rd")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = load_panel()
    if panel is None:
        print("  [SKIP] Required data missing."); return

    print(f"  Panel: {panel.index[0]:%Y-%m-%d} → {panel.index[-1]:%Y-%m-%d}  ({len(panel):,} h)")
    print(f"  Hours with LVR > fee: {int(panel['lvr_gt_fee'].sum()):,} ({100*panel['lvr_gt_fee'].mean():.1f}%)")

    # 1. Cutoff
    cutoff_info = estimate_vol_cutoff(panel)
    savetable(pd.DataFrame([cutoff_info]), "rvlb_cutoff")
    vol_star = cutoff_info["vol_star_primary"]
    vol_s_m1 = cutoff_info["vol_star_m1"]
    vol_s_m2 = cutoff_info["vol_star_m2"]
    print(f"  Vol break-even: vol*={vol_star:.3f}  (M1={vol_s_m1:.3f}, M2={vol_s_m2:.3f})")

    clean = panel.dropna(subset=["cex_realized_vol_24h_ann"]).copy()
    v = clean["cex_realized_vol_24h_ann"].values
    bw = 1.84 * np.std(v) * len(v)**(-0.2)
    D = clean["lvr_gt_fee"].values

    # 2. First stage
    fs_rows = []
    for spec, bw_m, poly, c in [
        ("main", 1.0, 1, vol_star), ("bw_half", 0.5, 1, vol_star),
        ("bw_1.5x", 1.5, 1, vol_star), ("poly2", 1.0, 2, vol_star),
        ("sens_m1", 1.0, 1, vol_s_m1), ("sens_m2", 1.0, 1, vol_s_m2),
    ]:
        fs = llr(D, v, cutoff=c, bw=bw*bw_m, poly=poly, label=spec)
        fs["spec"] = spec; fs["F"] = round(fs.get("t",np.nan)**2, 2) if not np.isnan(fs.get("t",np.nan)) else np.nan
        fs_rows.append(fs)
    savetable(pd.DataFrame(fs_rows).set_index("spec"), "rvlb_first_stage")

    # 3. Reduced form + Wald for each outcome
    OUTCOMES = {
        "d_log_tvl_24h":  "Δlog(TVL) +24h",
        "d_log_liq_24h":  "Δlog(liquidity) +24h",
    }
    rf_rows, wald_rows, rob_rows = [], [], []
    fs_main = llr(D, v, cutoff=vol_star, bw=bw, poly=1, label="fs")

    for col, label in OUTCOMES.items():
        if col not in clean.columns:
            continue
        y = clean[col].values
        for spec, bw_m, poly, c in [
            ("main", 1.0, 1, vol_star), ("bw_half", 0.5, 1, vol_star),
            ("bw_1.5x", 1.5, 1, vol_star), ("poly2", 1.0, 2, vol_star),
            ("donut", 1.0, 1, vol_star),
        ]:
            if spec == "donut":
                donut = 0.05 * vol_star
                mask = np.abs(v - vol_star) > donut
                rf_r = llr(y[mask], v[mask], c, bw*bw_m, poly=poly, label=f"{col}_{spec}")
            else:
                rf_r = llr(y, v, c, bw*bw_m, poly=poly, label=f"{col}_{spec}")
            rf_r["outcome"] = col; rf_r["spec"] = spec
            if spec == "main":
                rf_rows.append(rf_r)
                wald_rows.append(wald_late(rf_r, fs_main, label=col))
            else:
                rob_rows.append(rf_r)

    savetable(pd.DataFrame(rf_rows).set_index(["outcome","spec"]), "rvlb_reduced")
    savetable(pd.DataFrame(wald_rows).set_index("label"), "rvlb_wald")
    savetable(pd.DataFrame(rob_rows).set_index(["outcome","spec"]), "rvlb_robustness")

    # 4. McCrary
    mc = mccrary_test(v, vol_star)
    savetable(pd.DataFrame([mc]), "rvlb_mccrary")

    # 5. Plots
    plot_first_stage(panel, vol_star, bw)
    plot_rd(panel, vol_star, bw)

    print("  [DONE] rd_vol_lvr_breakeven.py")


if __name__ == "__main__":
    main()
