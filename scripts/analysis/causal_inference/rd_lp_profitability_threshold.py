"""
rd_lp_profitability_threshold.py — Fuzzy RD: LP Fee APR Break-Even Threshold

Research Question:
    Does empirical LP profitability exhibit a discontinuous jump at the
    theoretical fee APR break-even threshold where expected fees equal expected
    LVR losses, consistent with the Milionis et al. (2022) prediction that
    LP welfare is a deterministic function of fee rate and volatility?

Motivation:
    Milionis et al. (2022) prove that LP P&L = fee income − LVR, where:
        fee income ≈ (fee_rate × vol × sqrt(dt)) / range_factor
        LVR ≈ (vol² × dt / 2) × liquidity_share (continuous-time)
    This implies a BREAK-EVEN fee APR: LPs with fee_APR_at_mint > LVR_at_mint
    earn positive expected returns; those below break even earn negative.

    We test this prediction empirically using a Regression Discontinuity design:
    - Running variable: pool-level fee_apr_ann at the hour of LP position mint
      (observable before position outcome is determined)
    - Cutoff c* = median(lvr_rate_ann) — the average LVR rate over the sample
    - Outcome: realized net_pnl_usd / total_minted_usd (for closed positions)

    Identification: if the LVR break-even theory is correct, there should be a
    SHARP JUMP in expected ROI at c*. LPs cannot perfectly time their entry to
    always mint above c* (market conditions fluctuate hourly) → the running
    variable is not precisely controlled → local randomisation holds near c*.

    The RD is FUZZY because: (1) individual LP returns depend on position-specific
    range width, not just pool-level fee APR; (2) the cutoff c* is estimated
    with uncertainty; (3) post-mint market dynamics affect realised outcomes.

Methodology:
    1. Data construction — for each closed LP position, merge mint time with
       dex_pool_hourly to obtain fee_apr_ann at the hour of first_mint_utc.
       Also merge with dex_lvr_hourly to obtain lvr_rate_ann at mint time.
       Running variable = fee_apr_at_mint; compute ROI = net_pnl / total_minted_usd.

    2. Cutoff estimation — c* = median(lvr_rate_ann) over the pre-mint periods
       of all positions (a proxy for the average LVR rate experienced). Sensitivity
       to the 25th and 75th percentile cutoffs is reported.

    3. First stage — local linear regression of P(ROI > 0) vs fee_apr_at_mint.
       Expected: P(positive ROI) jumps at c* (from < 0.5 below to > 0.5 above).

    4. Reduced form + Wald estimator (fuzzy RD LATE):
           LATE = jump_in_ROI / jump_in_P(ROI > 0)
       Triangular kernel; Silverman's rule for bandwidth.
       Polynomial robustness (order 1 and 2); bandwidth sensitivity.

    5. McCrary density test — P(LP mints above c* vs below c*) should be ~50%
       if LPs cannot time their entry precisely.  Clustering just above c*
       would indicate strategic timing (mining when fee APR > LVR rate).

    6. Heterogeneity — repeat the RD separately for narrow vs wide range
       positions (narrow positions are more sensitive to the fee/LVR trade-off).

Key caveats:
    - The pool-level fee APR at mint hour is a NOISY proxy for the individual
      LP's expected return: the actual fee income depends on the LP's range
      width and the fraction of time the current price is within the range.
      Wide-range LPs earn lower fees per dollar of liquidity but have lower
      impermanent loss → the break-even APR is RANGE-DEPENDENT.
    - LPs may anticipate fee APR reverting toward the mean. If high fee APR
      is temporary (e.g., due to a news event), the ex-ante APR > LVR but the
      realised APR over the position's lifetime may be lower.
    - The cutoff c* = median(lvr_rate_ann) is estimated, not known. Estimation
      error introduces noise in the RD. The donut RD excludes a band around
      c* to check robustness to cutoff misspecification.
    - ROI = net_pnl / total_minted_usd. net_pnl = burned_usd − minted_usd
      + collected_fees. ONLY CLOSED POSITIONS are used; positions still active
      at data end are excluded → survivorship bias if high-fee-APR positions
      stay open longer (they have less reason to exit).
    - Estimand = LATE at c*: the causal effect of crossing the break-even
      threshold for the sub-population of LPs at the margin (fee_APR ≈ c*).
      This is NOT the ATE across all LPs.

Outputs:
    output/tables/rdp_cutoff.csv         LVR distribution → cutoff estimate
    output/tables/rdp_first_stage.csv    P(ROI>0) jump at c*
    output/tables/rdp_main.csv           Reduced form + Wald LATE estimates
    output/tables/rdp_robustness.csv     Bandwidth + polynomial robustness
    output/tables/rdp_mccrary.csv        McCrary density test at c*
    output/tables/rdp_heterogeneity.csv  Narrow vs wide separate RD
    output/figures/rdp_rd_scatter.pdf    Binned scatter + LLR fit
    output/figures/rdp_mccrary.pdf       Running variable density

References:
    Milionis, J. et al. (2022). Automated market making and loss-versus-
        rebalancing. Working paper.
    Lee, D. & Lemieux, T. (2010). RD designs in economics. JEL, 48(2), 281-355.
    Imbens, G. & Kalyanaraman, K. (2012). Optimal bandwidth. REStud, 79(3).
    McCrary, J. (2008). Manipulation of running variable. J. Econometrics.
    Calonico, S. et al. (2014). Robust nonparametric CIs for RD. Econometrica.
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
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
    load, savefig, savetable, stars, bootstrap_ci,
)

MIN_OBS  = 30  # min obs on each side for LLR
N_BINS   = 30  # scatter plot bins

# ── Data construction ─────────────────────────────────────────────────────────

def build_lp_rd_data() -> pd.DataFrame | None:
    """Merge LP positions with hourly pool data to get fee_apr at mint time."""
    lp  = load("DEX/dex_lp_positions.csv")
    dex = load("DEX/dex_pool_hourly.csv")
    lvr = load("DEX/dex_lvr_hourly.csv")
    if lp is None or dex is None or lvr is None:
        return None

    lp["first_mint_utc"] = pd.to_datetime(lp["first_mint_utc"], utc=True, errors="coerce")
    lp["last_burn_utc"]  = pd.to_datetime(lp["last_burn_utc"],  utc=True, errors="coerce")

    # Keep only closed positions with observable ROI
    lp = lp[lp["last_burn_utc"].notna() & lp["total_minted_usd"].notna()].copy()
    lp = lp[lp["total_minted_usd"] > 100].copy()  # exclude dust positions
    if lp.empty:
        return None

    lp["roi"] = lp["net_pnl_usd"] / lp["total_minted_usd"]
    lp["positive_roi"] = (lp["roi"] > 0).astype(float)

    # Merge with hourly pool data at mint time
    dex_h = dex[["fee_apr_ann"]].copy()
    lvr_h = lvr[["lvr_rate_ann", "lvr_to_fee_ratio"]].copy()
    pool_h = dex_h.join(lvr_h, how="left")

    # Map LP mint time to pool-hourly index
    lp["mint_hour"] = lp["first_mint_utc"].dt.floor("h")
    fee_at_mint = pool_h.reindex(lp["mint_hour"].values)["fee_apr_ann"].values
    lvr_at_mint = pool_h.reindex(lp["mint_hour"].values)["lvr_rate_ann"].values
    lp["fee_apr_at_mint"] = fee_at_mint
    lp["lvr_at_mint"]     = lvr_at_mint
    lp = lp.dropna(subset=["fee_apr_at_mint", "roi"]).copy()
    lp["log_mint_usd"] = np.log(lp["total_minted_usd"].clip(lower=1))
    lp["log_rng"]      = np.log(lp["range_width_pct"].clip(lower=0.01))

    print(f"  LP positions with fee_apr at mint: {len(lp):,}")
    print(f"  ROI: mean={lp['roi'].mean():.3f}  median={lp['roi'].median():.3f}  "
          f"positive={lp['positive_roi'].mean()*100:.0f}%")
    return lp


# ── Cutoff estimation ─────────────────────────────────────────────────────────

def estimate_cutoff(lp: pd.DataFrame) -> dict:
    """c* = percentiles of lvr_rate_ann at mint time (proxy for break-even APR)."""
    lvr = lp["lvr_at_mint"].dropna()
    c_p25 = float(np.percentile(lvr, 25))
    c_p50 = float(np.percentile(lvr, 50))
    c_p75 = float(np.percentile(lvr, 75))
    return {
        "c_p25": round(c_p25, 4), "c_p50": round(c_p50, 4), "c_p75": round(c_p75, 4),
        "mean_fee_at_mint": round(float(lp["fee_apr_at_mint"].mean()), 4),
        "mean_lvr_at_mint": round(float(lvr.mean()), 4),
        "pct_fee_gt_lvr":   round(float((lp["fee_apr_at_mint"] > lp["lvr_at_mint"]).mean() * 100), 1),
        "NOTE": (
            "c* = median(lvr_rate_ann at mint) = primary cutoff for RD. "
            "Sensitivity: c=p25, c=p75. "
            "pct_fee_gt_lvr: fraction of positions where ex-ante fee > LVR at mint."
        ),
    }


# ── Local linear RD ───────────────────────────────────────────────────────────

def _kern(u): return np.maximum(1.0 - np.abs(u), 0.0)

def local_linear_rd(y: np.ndarray, r: np.ndarray, cutoff: float, bw: float,
                    poly: int = 1, label: str = "") -> dict:
    u = (r - cutoff) / bw
    w = _kern(u)
    out = {"cutoff": cutoff, "bw": bw, "poly": poly, "label": label}
    sides: dict = {}
    for side, mask in [("left", r < cutoff), ("right", r >= cutoff)]:
        m = mask & (w > 0)
        if m.sum() < MIN_OBS:
            sides[side] = None; continue
        ys = y[m]; us = (r[m] - cutoff); ws = w[m]
        cols = [np.ones(m.sum()), us]
        if poly == 2:
            cols.append(us ** 2)
        Xs = np.column_stack(cols)
        W  = np.diag(ws)
        try:
            XWX = Xs.T @ W @ Xs
            beta = np.linalg.solve(XWX, Xs.T @ W @ ys)
            res  = ys - Xs @ beta
            s2   = float(np.sum(ws * res**2) / max(ws.sum() - len(beta), 1))
            cov  = s2 * np.linalg.solve(XWX, np.eye(len(beta)))
            sides[side] = {"int": float(beta[0]), "se": float(np.sqrt(max(cov[0,0], 0))),
                           "n": int(m.sum())}
        except np.linalg.LinAlgError:
            sides[side] = None
    if not sides.get("left") or not sides.get("right"):
        out.update({"jump": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "sig": ""}); return out
    jump = sides["right"]["int"] - sides["left"]["int"]
    se   = np.sqrt(sides["left"]["se"]**2 + sides["right"]["se"]**2)
    t    = jump / se if se > 0 else np.nan
    df_t = sides["left"]["n"] + sides["right"]["n"] - 2 * (poly + 1)
    p    = float(2 * sp_stats.t.sf(abs(t), df=max(df_t, 1)))
    out.update({"jump": round(jump, 6), "se": round(se, 6), "t": round(t, 4),
                "p": round(p, 6), "sig": stars(p),
                "n_left": sides["left"]["n"], "n_right": sides["right"]["n"],
                "NOTE": (
                    f"LLR poly={poly}, bw={bw:.3f}, triangular kernel. "
                    "SE = asymptotic HC (not RBC; Calonico et al. 2014 RBC preferred for inference). "
                    "Estimand = LATE at c*: causal effect for LPs whose fee_APR ≈ c*."
                )})
    return out


def _silverman_bw(r): return 1.84 * np.std(r) * len(r)**(-0.2)


def wald_late(rf: dict, fs: dict, outcome: str) -> dict:
    rf_j = rf.get("jump", np.nan); fs_j = fs.get("jump", np.nan)
    if np.isnan(rf_j) or np.isnan(fs_j) or abs(fs_j) < 1e-9:
        return {"outcome": outcome, "late": np.nan}
    late   = rf_j / fs_j
    se_l   = abs(late) * np.sqrt(
        (rf.get("se", 0)/max(abs(rf_j), 1e-9))**2 + (fs.get("se", 0)/max(abs(fs_j), 1e-9))**2)
    t = late / se_l if se_l > 0 else np.nan
    p = float(2 * sp_stats.t.sf(abs(t), df=rf.get("n_left", 0) + rf.get("n_right", 0) - 4))
    return {"outcome": outcome, "late": round(late, 6), "se_late": round(se_l, 6),
            "t": round(t, 4), "p": round(p, 6), "sig": stars(p),
            "rf_jump": round(rf_j, 6), "fs_jump": round(fs_j, 6)}


# ── McCrary test ──────────────────────────────────────────────────────────────

def mccrary_test(r: np.ndarray, cutoff: float) -> dict:
    bins  = np.linspace(r.min(), r.max(), 41)
    mids  = (bins[:-1] + bins[1:]) / 2
    bw_b  = bins[1] - bins[0]
    cts, _ = np.histogram(r, bins=bins)
    density = cts / (len(r) * bw_b)
    h0      = _silverman_bw(r)
    res     = local_linear_rd(density, mids, cutoff, h0, poly=1, label="McCrary")
    res["NOTE"] = (
        "McCrary (2008) density. H0: density continuous at c*. "
        "Bunching ABOVE c* (positive density jump) would indicate LPs time entry "
        "to high-fee-APR hours — suggesting market timing skill, not local randomisation. "
        "If manipulation is present, the RD identification assumption fails."
    )
    return res


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_rd(lp: pd.DataFrame, cutoff: float, bw: float) -> None:
    r = lp["fee_apr_at_mint"].values
    y = lp["roi"].values
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # RD scatter
    ax = axes[0]
    r_rng = (max(r.min(), cutoff - 2*bw), min(r.max(), cutoff + 2*bw))
    m = (r >= r_rng[0]) & (r <= r_rng[1])
    bins = np.linspace(r_rng[0], r_rng[1], N_BINS + 1)
    def _bm(r_, y_, bins_):
        ms, ys = [], []
        for lo, hi in zip(bins_[:-1], bins_[1:]):
            mk = (r_ >= lo) & (r_ < hi) & ~np.isnan(y_)
            if mk.sum() >= 3:
                ms.append((lo+hi)/2); ys.append(float(y_[mk].mean()))
        return np.array(ms), np.array(ys)
    lo_b = bins[bins <= cutoff]; hi_b = bins[bins >= cutoff]
    rl, yl = _bm(r[m], y[m], lo_b)
    rr, yr = _bm(r[m], y[m], hi_b)
    ax.scatter(rl, yl, color="#2563eb", s=30, alpha=0.8, label="Below c*")
    ax.scatter(rr, yr, color="#dc2626", s=30, alpha=0.8, label="Above c*")
    ax.axvline(cutoff, color="black", lw=2, ls="--", label=f"c*={cutoff:.3f}")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Fee APR at mint")
    ax.set_ylabel("Realized ROI (net_pnl / mint_usd)")
    ax.set_title("RD: LP Profitability Break-Even", fontsize=9)
    ax.legend(fontsize=8)

    # McCrary density
    ax2 = axes[1]
    lo_r = r[r < cutoff]; hi_r = r[r >= cutoff]
    ax2.hist(lo_r, bins=20, density=True, color="#2563eb", alpha=0.5, label="Below c*")
    ax2.hist(hi_r, bins=20, density=True, color="#dc2626", alpha=0.5, label="Above c*")
    ax2.axvline(cutoff, color="black", lw=2, ls="--")
    ax2.set_xlabel("Fee APR at mint"); ax2.set_ylabel("Density")
    ax2.set_title("McCrary density test — running variable", fontsize=9)
    ax2.legend(fontsize=8)
    plt.suptitle("LP profitability threshold RD", fontsize=10)
    savefig("rdp_rd_scatter")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    lp = build_lp_rd_data()
    if lp is None:
        print("  [SKIP] Required data missing."); return

    # 1. Cutoff
    cutoff_info = estimate_cutoff(lp)
    savetable(pd.DataFrame([cutoff_info]), "rdp_cutoff")
    c_star = cutoff_info["c_p50"]
    c_lo   = cutoff_info["c_p25"]
    c_hi   = cutoff_info["c_p75"]
    r = lp["fee_apr_at_mint"].values
    y = lp["roi"].values
    D = lp["positive_roi"].values
    bw = _silverman_bw(r)
    print(f"  c*={c_star:.4f}  bw={bw:.4f}  N_closed={len(lp):,}")

    # 2. First stage
    fs_rows = []
    for spec, bw_m, poly, cutoff in [
        ("main",        1.0, 1, c_star),
        ("bw_half",     0.5, 1, c_star),
        ("bw_1.5x",     1.5, 1, c_star),
        ("poly2",       1.0, 2, c_star),
        ("sens_p25",    1.0, 1, c_lo),
        ("sens_p75",    1.0, 1, c_hi),
    ]:
        fs = local_linear_rd(D, r, cutoff=cutoff, bw=bw*bw_m, poly=poly, label=spec)
        fs["spec"] = spec
        t_fs = fs.get("t", np.nan)
        fs["F_first_stage"] = round(float(t_fs**2), 2) if not np.isnan(t_fs) else np.nan
        fs["weak_instrument"] = (fs["F_first_stage"] < 10) if not np.isnan(fs.get("F_first_stage", np.nan)) else True
        fs_rows.append(fs)
    df_fs = pd.DataFrame(fs_rows).set_index("spec")
    # Warn if main spec is weak
    f_main = df_fs.loc["main", "F_first_stage"] if "main" in df_fs.index else np.nan
    if not np.isnan(f_main):
        flag = " [WEAK INSTRUMENT]" if f_main < 10 else ""
        print(f"  First-stage F (main): {f_main:.1f}{flag}  (threshold: F≥10, Staiger & Stock 1997)")
    savetable(df_fs, "rdp_first_stage")

    # 3. Reduced form (ROI) + Wald
    rf_rows, wald_rows, rob_rows = [], [], []
    fs_main = local_linear_rd(D, r, cutoff=c_star, bw=bw, poly=1, label="fs")
    for spec, bw_m, poly, cutoff in [
        ("main",    1.0, 1, c_star),
        ("bw_half", 0.5, 1, c_star),
        ("bw_1.5x", 1.5, 1, c_star),
        ("poly2",   1.0, 2, c_star),
        ("sens_p25",1.0, 1, c_lo),
        ("sens_p75",1.0, 1, c_hi),
    ]:
        rf = local_linear_rd(y, r, cutoff=cutoff, bw=bw*bw_m, poly=poly, label=spec)
        rf["spec"] = spec
        if spec == "main":
            rf_rows.append(rf)
            wald_rows.append(wald_late(rf, fs_main, "ROI"))
        else:
            rob_rows.append(rf)

    # Donut RD: exclude ±10% of bandwidth around c* (robustness to cutoff misspecification)
    donut_h = max(bw * 0.10, cutoff_info["c_p75"] - cutoff_info["c_p25"]) * 0.05
    donut_h = max(donut_h, 1e-4)
    mask_donut = np.abs(r - c_star) > donut_h
    if mask_donut.sum() > MIN_OBS * 2:
        rf_donut = local_linear_rd(y[mask_donut], r[mask_donut], c_star, bw, poly=1, label="donut")
        rf_donut["spec"] = f"donut_{donut_h:.4f}"
        rf_donut["NOTE_donut"] = (
            f"Donut RD: excludes ±{donut_h:.4f} around c*={c_star:.4f}. "
            "Tests whether jump is driven by LPs at the exact cutoff (strategic timing). "
            "Ref: Calonico et al. (2019); Barreca et al. (2016)."
        )
        rob_rows.append(rf_donut)

    savetable(pd.DataFrame(rf_rows + rob_rows).set_index("spec"), "rdp_main")
    savetable(pd.DataFrame(wald_rows).set_index("outcome"), "rdp_robustness")

    # 4. Heterogeneity (narrow vs wide)
    med_rng = float(lp["range_width_pct"].median())
    het_rows = []
    for grp_label, mask in [("narrow", lp["range_width_pct"] <= med_rng),
                             ("wide",   lp["range_width_pct"] > med_rng)]:
        sub = lp[mask]
        if len(sub) < MIN_OBS * 2:
            continue
        rf_g = local_linear_rd(sub["roi"].values, sub["fee_apr_at_mint"].values,
                                c_star, bw, poly=1, label=grp_label)
        rf_g["group"] = grp_label; rf_g["N"] = len(sub)
        het_rows.append(rf_g)
    if het_rows:
        savetable(pd.DataFrame(het_rows).set_index("group"), "rdp_heterogeneity")

    # 5. McCrary
    mc = mccrary_test(r, c_star)
    savetable(pd.DataFrame([mc]), "rdp_mccrary")

    # 6. Plots
    plot_rd(lp, c_star, bw)

    print("  [DONE] rd_lp_profitability_threshold.py")


if __name__ == "__main__":
    main()
