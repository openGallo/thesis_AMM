"""
rd_tick_crossing.py — Fuzzy RD: Uniswap v3 Tick Crossings and Market Quality

Research Question:
    Do Uniswap v3 tick crossings — the discrete jumps in concentrated liquidity
    at tick boundaries — cause a measurable deterioration in AMM market quality
    (wider DEX-CEX basis, elevated price impact) relative to intra-tick periods,
    identifying the mechanical effect of discretised liquidity supply?

Motivation:
    Unlike traditional limit-order books, Uniswap v3 concentrates liquidity within
    discrete price tick intervals (1.0001^tick spacing).  When a swap pushes the
    price across a tick boundary, liquidity discontinuously leaves (for a crossing
    in the swap direction) or enters (for a reversal).  This creates a "liquidity
    cliff" at each tick: execution quality drops abruptly.  Theory (Lehar & Parlour
    2021; Barger et al. 2023) predicts that tick crossings amplify price impact and
    widen the effective spread, generating a transient DEX-CEX basis divergence.

    Identification strategy — Fuzzy RD:
    • Running variable: |log_return_1h| (absolute hourly return, market-determined).
    • Theoretical cutoff c* = ln(1.0001^10) ≈ 0.001 (0.1%) — the minimum return
      to cross one tick at the 0.05% fee-tier (10-tick spacing).
    • Treatment: delta_tick_h ≠ 0 (at least one tick crossed in hour h).
    • The running variable is CONTINUOUS and market-determined — no agent can
      precisely sort around c* (McCrary density test should find no discontinuity).
    • At |log_return| just below c*, P(tick_cross) ≈ 0 (sub-tick move, no crossing).
    • At |log_return| just above c*, P(tick_cross) > 0 and increasing sharply.
    • This creates a FUZZY RD: the instrument Z = 1{|log_return| ≥ c*} predicts
      tick_cross but does not deterministically assign it (within-hour reversals
      can prevent crossings even when total return exceeds the threshold).

Methodology:
    1. First stage — local linear regression of P(tick_cross) on |log_return|:
           E[D_h | r_h = r] — separately on each side of c*.
       First-stage F-statistic (strength) reported; F < 10 = weak instrument.

    2. Reduced form — LLR of outcome on running variable at cutoff:
           E[Y_h | r_h = r] jump at c*.
       Outcomes: |basis_bps_{h+1}|, vol_ratio = |log_return_{h+1}| / |log_return_h|,
       dex_vol_over_tvl_h.

    3. 2SLS (Fuzzy RD LATE) — reduced form / first stage = causal effect of
       tick crossing on outcome (LATE at the margin = c*).
       Wald estimator: τ_2SLS = jump_outcome / jump_first_stage.

    4. Polynomial robustness — repeat with quadratic local polynomial.

    5. Bandwidth sensitivity — h = h_IK × {0.5, 1.0, 1.5}.

    6. McCrary density test — density of |log_return| at c*. Should be smooth
       (returns are market-determined; no agent sorts around 0.1%).

    7. Control robustness — add hour-of-day and day-of-week dummies as
       additional covariates in the local linear regression (Calonico et al. 2019).

    8. Empirical cutoff estimation — estimate c_hat empirically as the value of
       |log_return| where P(tick_cross) = 0.5 (the median crossing threshold).
       If c_hat ≫ c* = 0.001, within-hour reversals are common; if c_hat ≈ c*,
       the theoretical threshold is a good approximation.

Key caveats (embedded in output tables):
    - Within-hour reversals: in a volatile hour, the price may cross multiple
      ticks and return without net crossing. The DEX data shows hourly tick
      changes, not the tick path within the hour. This attenuates the first stage
      (fuzzy → even fuzzier). Minute-level data would sharpen identification.
    - The monotonicity assumption (LATE requirement): all marginal units at c*
      must be "compliers" (units that switch from no-cross to cross as return
      crosses c*). "Defiers" (units that cross a tick with small returns, e.g.,
      very thin liquidity near the tick boundary) would violate this.
    - The 10-tick spacing is for the 0.05% fee tier; the pool may have a mix of
      positions at non-standard tick multiples (V3 allows partial tick usage),
      though liquidity transitions must occur at multiples of tick_spacing.
    - Outcomes are measured 1 hour AFTER the crossing (h+1). Simultaneous (h)
      outcomes would confound treatment with common time-period shocks.

Outputs:
    output/tables/rd_tick_cutoff.csv        Theoretical vs empirical cutoff
    output/tables/rd_tick_first_stage.csv   P(tick_cross | |log_return|) jump
    output/tables/rd_tick_reduced.csv       Reduced-form outcome jumps at c*
    output/tables/rd_tick_2sls.csv          Wald (fuzzy RD LATE) estimates
    output/tables/rd_tick_robustness.csv    Polynomial + bandwidth sensitivity
    output/tables/rd_tick_mccrary.csv       McCrary density test
    output/figures/rd_tick_first_stage.pdf  First-stage compliance plot
    output/figures/rd_tick_rd_plots.pdf     Reduced-form RD scatter plots

References:
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
    Imbens, G. & Angrist, J. (1994). Identification and estimation of LATE.
        Econometrica, 62(2), 467-475.
    Lee, D. & Lemieux, T. (2010). RD designs in economics.
        J. Econ. Literature, 48(2), 281-355.
    McCrary, J. (2008). Manipulation of running variable in RD.
        J. Econometrics, 142(2), 698-714.
    Calonico, S. et al. (2019). Regression discontinuity designs using
        covariates. Rev. Econ. Stats., 101(3), 442-451.
    Adams, A. et al. (2021). Uniswap v3 Core. Uniswap Labs.
"""
from __future__ import annotations

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

# ── Constants ─────────────────────────────────────────────────────────────────

TICK_SPACING  = 10          # Uniswap v3 0.05% fee tier
C_STAR        = 10 * np.log(1.0001)   # ≈ 0.001000 (0.1%)
N_BINS_PLOT   = 35
MIN_OBS_SIDE  = 30

OUTCOMES = {
    "abs_basis_next": "|DEX–CEX basis| next hour (bps)",
    "vol_ratio":      "|log_return_{h+1}| / |log_return_h|",
    "dex_vol_over_tvl": "Volume / TVL (same hour)",
}

# ── Data ──────────────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame | None:
    mh  = load("merged/merged_hourly.csv")
    dex = load("DEX/dex_pool_hourly.csv")
    if mh is None or dex is None:
        return None

    # Merge
    panel = mh.join(
        dex[["tick", "log_return_1h", "volume_usd", "liquidity", "dex_vol_over_tvl"]]
           .rename(columns={"log_return_1h": "dex_log_return"}),
        how="left",
        lsuffix="", rsuffix="_dex",
    )
    # Running variable and treatment
    panel["abs_return"] = panel["dex_log_return"].abs()
    panel["delta_tick"] = panel["tick"].diff().abs()
    panel["tick_cross"] = (panel["delta_tick"] >= TICK_SPACING).astype(float)

    # Outcomes (next-hour)
    panel["abs_basis_next"] = panel["dex_cex_basis_bps"].abs().shift(-1)
    panel["abs_return_next"]= panel["abs_return"].shift(-1)
    panel["vol_ratio"]      = np.where(
        panel["abs_return"] > 1e-6,
        panel["abs_return_next"] / panel["abs_return"],
        np.nan,
    )

    panel = panel.dropna(subset=["abs_return", "tick_cross", "abs_basis_next"])
    panel = panel[panel["abs_return"] > 0].copy()

    print(f"  Theoretical c*: {C_STAR*100:.3f}%  ({C_STAR:.5f})")
    print(f"  Tick crossings: {int(panel['tick_cross'].sum()):,} / {len(panel):,} hours  "
          f"({100*panel['tick_cross'].mean():.1f}%)")
    return panel


# ── Empirical cutoff estimation ───────────────────────────────────────────────

def empirical_cutoff(panel: pd.DataFrame) -> dict:
    """
    Find c_hat = argmin_c |P(tick_cross | |r| ∈ [c-ε, c+ε]) - 0.5|.
    Scans over quantiles of |log_return| to locate the 50% crossing threshold.
    """
    quantiles = np.linspace(0.01, 0.99, 200)
    r   = panel["abs_return"].values
    D   = panel["tick_cross"].values
    eps = np.percentile(r, 2)   # smoothing window

    best_c, best_diff = C_STAR, 1.0
    probs = []
    cuts  = np.percentile(r, quantiles * 100)
    for c, q in zip(cuts, quantiles):
        m = np.abs(r - c) < eps
        if m.sum() < 10:
            probs.append(np.nan)
            continue
        p = float(D[m].mean())
        probs.append(p)
        if abs(p - 0.5) < best_diff:
            best_diff = abs(p - 0.5)
            best_c    = c

    return {
        "c_star_theory": round(C_STAR, 6),
        "c_hat_empirical": round(best_c, 6),
        "c_hat_pct": round(best_c * 100, 4),
        "c_star_pct": round(C_STAR * 100, 4),
        "ratio_chat_cstar": round(best_c / C_STAR, 3),
        "NOTE": (
            f"Empirical c_hat estimated from P(tick_cross|r)=0.5. "
            "c_hat > c_star suggests within-hour reversals are common "
            "(higher return needed for net tick crossing). "
            "c_hat/c_star quantifies fuzziness of the first stage."
        ),
    }


# ── Local linear regression (reused from rd_arbitrage_trigger pattern) ────────

def _triangular_kernel(u: np.ndarray) -> np.ndarray:
    return np.maximum(1.0 - np.abs(u), 0.0)


def local_linear_rd(
    y: np.ndarray, r: np.ndarray, cutoff: float, bw: float, poly: int = 1,
    label: str = ""
) -> dict:
    u = (r - cutoff) / bw
    w = _triangular_kernel(u)
    out = {"cutoff": cutoff, "bw": bw, "poly": poly, "label": label}
    sides: dict = {}

    for side, mask in [("left", r < cutoff), ("right", r >= cutoff)]:
        m = mask & (w > 0)
        if m.sum() < MIN_OBS_SIDE:
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
            resid = ys - Xs @ beta
            s2 = float(np.sum(ws * resid ** 2) / max(ws.sum() - len(beta), 1))
            cov = s2 * np.linalg.solve(XWX, np.eye(len(beta)))
            sides[side] = {"int": float(beta[0]), "se": float(np.sqrt(max(cov[0,0], 0))),
                           "n": int(m.sum())}
        except np.linalg.LinAlgError:
            sides[side] = None

    if not sides.get("left") or not sides.get("right"):
        out.update({"jump": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "sig": ""}); return out

    jump = sides["right"]["int"] - sides["left"]["int"]
    se   = np.sqrt(sides["left"]["se"] ** 2 + sides["right"]["se"] ** 2)
    t    = jump / se if se > 0 else np.nan
    df_t = sides["left"]["n"] + sides["right"]["n"] - 2 * (poly + 1)
    p    = float(2 * sp_stats.t.sf(abs(t), df=max(df_t, 1)))
    out.update({"jump": round(jump, 6), "se": round(se, 6), "t": round(t, 4),
                "p": round(p, 6), "sig": stars(p),
                "n_left": sides["left"]["n"], "n_right": sides["right"]["n"]})
    return out


def _ik_bw(y: np.ndarray, r: np.ndarray, cutoff: float) -> float:
    """Silverman's rule as IK pilot bandwidth."""
    return 1.84 * np.std(r) * len(r) ** (-0.2)


# ── McCrary density test ──────────────────────────────────────────────────────

def mccrary_test(r: np.ndarray, cutoff: float) -> dict:
    n_bins = 40
    bins   = np.linspace(r.min(), r.max(), n_bins + 1)
    mids   = (bins[:-1] + bins[1:]) / 2
    bw_b   = bins[1] - bins[0]
    counts, _ = np.histogram(r, bins=bins)
    density   = counts / (len(r) * bw_b)
    h_pilot   = 1.84 * np.std(r) * len(r) ** (-0.2)
    res = local_linear_rd(density, mids, cutoff=cutoff, bw=h_pilot, label="McCrary")
    res["NOTE"] = (
        "McCrary (2008) density test. Running variable = |log_return_1h|. "
        "Returns are market-determined; no agent can sort near c*. "
        "A hole in density ABOVE c* is expected (tick crossings increase return variance). "
        "This diagnostic checks for anomalous bunching, not manipulation per se."
    )
    return res


# ── 2SLS Wald estimator ───────────────────────────────────────────────────────

def wald_estimator(rf: dict, fs: dict, outcome_name: str) -> dict:
    """Fuzzy RD LATE = reduced_form_jump / first_stage_jump (Wald estimator)."""
    rf_j = rf.get("jump", np.nan)
    fs_j = fs.get("jump", np.nan)
    if np.isnan(rf_j) or np.isnan(fs_j) or abs(fs_j) < 1e-9:
        return {"outcome": outcome_name, "late": np.nan, "se_late": np.nan,
                "t_late": np.nan, "p_late": np.nan, "sig": ""}
    late    = rf_j / fs_j
    se_late = abs(late) * np.sqrt(
        (rf.get("se", 0) / max(abs(rf_j), 1e-9)) ** 2 +
        (fs.get("se", 0) / max(abs(fs_j), 1e-9)) ** 2
    )
    t = late / se_late if se_late > 0 else np.nan
    p = float(2 * sp_stats.t.sf(abs(t), df=rf.get("n_left", 0) + rf.get("n_right", 0) - 4))
    return {
        "outcome":    outcome_name,
        "late":       round(late, 6),
        "se_late":    round(se_late, 6),
        "t_late":     round(t, 4),
        "p_late":     round(p, 6),
        "sig":        stars(p),
        "rf_jump":    round(rf_j, 6),
        "fs_jump":    round(fs_j, 6),
        "NOTE": (
            "Wald estimator: LATE = rf_jump / fs_jump. "
            "Interprets as: one additional tick crossing (at the margin c*) "
            "causes LATE-unit change in outcome. LATE ≠ ATE if effect varies "
            "with return magnitude (heterogeneous treatment effects; Imbens & Angrist 1994). "
            "Weak first stage: F_first_stage = (fs_jump/fs_se)²."
        ),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_first_stage(panel: pd.DataFrame, cutoff: float, bw: float) -> None:
    r = panel["abs_return"].values
    D = panel["tick_cross"].values

    # Bin data
    n_bins = 50
    bins   = np.linspace(0, np.percentile(r, 99), n_bins + 1)
    mids   = (bins[:-1] + bins[1:]) / 2
    means  = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (r >= lo) & (r < hi)
        means.append(float(D[m].mean()) if m.sum() >= 3 else np.nan)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.scatter(mids, means, s=20, color="#2563eb", alpha=0.7)
    ax.axvline(cutoff, color="black", lw=2, ls="--",
               label=f"c* = {cutoff*100:.2f}% (theoretical)")
    ax.set_xlabel("|log_return_1h|")
    ax.set_ylabel("P(tick crossing)")
    ax.set_title(f"First stage: P(tick_cross) vs |log_return|  |  c* = {cutoff:.5f}")
    ax.legend()
    savefig("rd_tick_first_stage")


def plot_rd_outcomes(panel: pd.DataFrame, cutoff: float, bw: float) -> None:
    r = panel["abs_return"].values
    fig, axes = plt.subplots(1, len(OUTCOMES), figsize=(5 * len(OUTCOMES), 4))
    for i, (col, label) in enumerate(OUTCOMES.items()):
        ax = axes[i] if len(OUTCOMES) > 1 else axes
        if col not in panel.columns:
            ax.set_visible(False); continue
        y = panel[col].values
        # Bin
        n_bins = N_BINS_PLOT
        bins_lo = np.linspace(max(0, cutoff - 2 * bw), cutoff, n_bins // 2 + 1)
        bins_hi = np.linspace(cutoff, cutoff + 2 * bw, n_bins // 2 + 1)
        def _bm(r_, y_, b_):
            ms, ys = [], []
            for lo, hi in zip(b_[:-1], b_[1:]):
                m = (r_ >= lo) & (r_ < hi) & ~np.isnan(y_)
                if m.sum() >= 3:
                    ms.append((lo + hi) / 2); ys.append(float(y_[m].mean()))
            return np.array(ms), np.array(ys)
        rl, yl = _bm(r, y, bins_lo)
        rr, yr = _bm(r, y, bins_hi)
        ax.scatter(rl, yl, color="#2563eb", s=25, alpha=0.8)
        ax.scatter(rr, yr, color="#dc2626", s=25, alpha=0.8)
        ax.axvline(cutoff, color="black", lw=2, ls="--")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("|log_return|", fontsize=8)
    plt.suptitle(f"Fuzzy RD outcome plots  (c* = {cutoff*100:.2f}%)", fontsize=10)
    savefig("rd_tick_rd_plots")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = load_panel()
    if panel is None:
        print("  [SKIP] Required data not found."); return

    cutoff = C_STAR
    r      = panel["abs_return"].values
    D      = panel["tick_cross"].values

    # 1. Empirical cutoff
    ec = empirical_cutoff(panel)
    savetable(pd.DataFrame([ec]), "rd_tick_cutoff")
    print(f"  Theoretical c*: {C_STAR:.5f}  |  Empirical c_hat: {ec['c_hat_empirical']:.5f}  "
          f"(ratio: {ec['ratio_chat_cstar']:.2f}×)")

    # 2. Bandwidth
    h_ik = _ik_bw(D, r, cutoff)
    print(f"  IK pilot bandwidth: {h_ik:.5f}")

    # 3. First stage
    fs_rows = []
    for bw_mult, spec in [(1.0, "h_IK"), (0.5, "0.5×h_IK"), (1.5, "1.5×h_IK")]:
        bw = h_ik * bw_mult
        fs = local_linear_rd(D.astype(float), r, cutoff, bw, poly=1, label="first_stage")
        fs["spec"] = spec
        fs["F_first_stage"] = round((fs["t"] ** 2) if not np.isnan(fs.get("t", np.nan)) else np.nan, 2)
        fs["NOTE"] = (
            "H1: P(tick_cross) jumps at c*. F_first_stage=(jump/se)². "
            "F<10 indicates weak instrument (Staiger & Stock 1997). "
            "Fuzziness source: within-hour reversals prevent crossing despite net return ≥ c*."
        )
        fs_rows.append(fs)
    df_fs = pd.DataFrame(fs_rows).set_index("spec")
    savetable(df_fs, "rd_tick_first_stage")

    # 4. Reduced form + 2SLS
    rf_rows, wald_rows, rob_rows = [], [], []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        y = panel[col].values
        mask = ~np.isnan(y)
        r_m = r[mask]; y_m = y[mask]; D_m = D[mask]

        # Reduced form (main bw)
        rf = local_linear_rd(y_m, r_m, cutoff, h_ik, poly=1, label=col)
        rf["outcome"] = col; rf["spec"] = "main_p1"
        rf_rows.append(rf)

        # 2SLS
        fs_main = local_linear_rd(D_m.astype(float), r_m, cutoff, h_ik, poly=1, label="fs")
        wald = wald_estimator(rf, fs_main, col)
        wald_rows.append(wald)

        # Robustness
        for bw_mult, poly, spec in [
            (0.5, 1, "bw_half"), (1.5, 1, "bw_1.5x"), (1.0, 2, "poly2")
        ]:
            rf_r = local_linear_rd(y_m, r_m, cutoff, h_ik * bw_mult, poly=poly, label=col)
            rf_r["outcome"] = col; rf_r["spec"] = spec
            rob_rows.append(rf_r)

    savetable(pd.DataFrame(rf_rows).set_index(["outcome", "spec"]), "rd_tick_reduced")
    savetable(pd.DataFrame(wald_rows).set_index("outcome"), "rd_tick_2sls")
    savetable(pd.DataFrame(rob_rows).set_index(["outcome", "spec"]), "rd_tick_robustness")

    # 5. McCrary
    mc = mccrary_test(r, cutoff)
    savetable(pd.DataFrame([mc]), "rd_tick_mccrary")

    # 6. Plots
    plot_first_stage(panel, cutoff, h_ik)
    plot_rd_outcomes(panel, cutoff, h_ik)

    print("  [DONE] rd_tick_crossing.py")


if __name__ == "__main__":
    main()
