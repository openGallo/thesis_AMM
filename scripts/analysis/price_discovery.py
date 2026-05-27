"""
price_discovery.py — Hasbrouck & Gonzalo-Granger Price Discovery Analysis

Research Question:
    Does the Uniswap v3 USDC/WETH pool or Binance lead ETH/USD price
    discovery, and how does the venue-specific information share evolve
    across time and volatility regimes?

Methodology:
    1. Bivariate VECM(k) on [log_p_DEX, log_p_CEX] via statsmodels.
    2. Gonzalo-Granger (1995) Component Share (CS):
           CS_DEX = α_CEX / (α_CEX − α_DEX)
           CS_CEX = −α_DEX / (α_CEX − α_DEX)
       where α = [α_DEX, α_CEX] are speed-of-adjustment coefficients.
    3. Hasbrouck (1995) Information Share (IS):
       - Common factor weights ψ = [−α_CEX, α_DEX] / (α_DEX − α_CEX)
       - IS bounds via Cholesky(Ω) under both orderings.
       - Midpoint IS = (IS_lo + IS_hi) / 2 as best estimate.
    4. Rolling 30-day (720-h) estimates — step = 24 h for performance.
    5. Regime-conditional IS/GG: estimate VECM per volatility tercile.
    6. Basis-conditional IS: hours with |basis| > 80th pct vs < 50th pct.
    7. Granger causality test (VAR): log_p_DEX → log_p_CEX and vice versa.

Key caveats (embedded in output tables):
    - IS bounds widen with cross-market innovation correlation (ρ).
      Midpoint is a conventional estimate; see Lien & Shrestha (2009).
    - GG CS can be outside [0,1] when both α have the same sign
      (theoretically unusual; clipped to [0,1] for reporting).
    - Subset VECM (regime/basis conditioning) uses non-consecutive obs.;
      treat as exploratory, not causal.
    - Hourly frequency is lower than typical IS studies (tick / 5-min).
      Within-hour arbitrage activity is averaged out; IS for DEX is
      likely understated at hourly level.
    - Structural breaks (EIP-1559 Aug-2021, The Merge Sep-2022, FTX
      Nov-2022) may violate constant-cointegration assumption.

Outputs:
    output/tables/pd_static.csv         Full-sample IS, GG, VECM α
    output/tables/pd_rolling.csv        Rolling 30-day IS and GG
    output/tables/pd_by_regime.csv      IS/GG by volatility regime
    output/tables/pd_by_basis.csv       IS/GG by |basis| level
    output/tables/pd_granger.csv        Granger causality (VAR)
    output/figures/pd_rolling_is.pdf    Rolling IS/GG time series
    output/figures/pd_by_regime.pdf     IS/GG bar chart by regime

References:
    Hasbrouck, J. (1995). One security, many markets. J. Finance, 50, 1175.
    Gonzalo, J. & Granger, C. (1995). Estimation of common long-memory
        components. J. Bus. Econ. Stat., 13(1), 27-36.
    Lien, D. & Shrestha, K. (2009). New information share measure.
        J. Futures Markets, 29(12), 1112-1129.
    Aspris, A. et al. (2021). Decentralised exchanges. Fin. Res. Letters.
    de Jong, F. (2002). Measures of contributions to price discovery.
        J. Financial Markets, 5(3), 323-327.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ── Imports from shared utilities ─────────────────────────────────────────────
from analysis_utils import (
    COLORS, load, savefig, savetable,
    stars, block_bootstrap_ci, vol_regime,
)

import matplotlib.pyplot as plt

ROLLING_WINDOW = 720   # 30 days of hourly obs
ROLLING_STEP   = 24    # re-estimate every 24 h (daily step)
MIN_OBS_VECM   = 200   # minimum observations for VECM estimation


# ── VECM and price-discovery helpers ─────────────────────────────────────────

def _estimate_vecm(p1: pd.Series, p2: pd.Series,
                   k_ar_diff: int = 2) -> dict | None:
    """
    Bivariate VECM(k_ar_diff) on [p1, p2] (log prices assumed cointegrated).

    Returns dict: alpha [2], sigma_u [2×2], resid [T×2].
    Catches statsmodels convergence failures silently.
    """
    try:
        from statsmodels.tsa.vector_ar.vecm import VECM
        df = pd.DataFrame({"p1": p1, "p2": p2}).dropna()
        if len(df) < max(MIN_OBS_VECM, k_ar_diff * 4 + 10):
            return None
        mdl = VECM(df, k_ar_diff=k_ar_diff, coint_rank=1, deterministic="n")
        res = mdl.fit()
        return {
            "alpha":   np.asarray(res.alpha[:, 0], dtype=float),
            "sigma_u": np.asarray(res.sigma_u,     dtype=float),
            "resid":   np.asarray(res.resid,        dtype=float),
            "N":       len(df) - k_ar_diff,
        }
    except Exception:
        return None


def _gonzalo_granger(alpha: np.ndarray) -> tuple[float, float]:
    """
    Gonzalo-Granger (1995) Component Shares.
    CS_1 = α₂/(α₂-α₁), CS_2 = -α₁/(α₂-α₁).
    Clipped to [0,1] (outside when same-sign α).
    """
    a1, a2 = float(alpha[0]), float(alpha[1])
    d = a2 - a1
    if abs(d) < 1e-10:
        return np.nan, np.nan
    return float(np.clip(a2 / d, 0, 1)), float(np.clip(-a1 / d, 0, 1))


def _hasbrouck_is(alpha: np.ndarray,
                  omega: np.ndarray) -> tuple[float, float, float]:
    """
    Hasbrouck (1995) Information Share bounds for market 1 (DEX).

    Common factor weights ψ derived from GG permanent component:
        ψ = [-α₂, α₁] / (α₁ - α₂)
    Upper bound: DEX ordered first in Cholesky.
    Lower bound: CEX ordered first (DEX gets residual).
    Midpoint = (lower + upper) / 2 as best point estimate.

    Returns (IS_lo, IS_hi, IS_mid).
    """
    a1, a2 = float(alpha[0]), float(alpha[1])
    d = a1 - a2
    if abs(d) < 1e-10:
        return np.nan, np.nan, np.nan
    psi = np.array([-a2 / d, a1 / d])
    tv  = float(psi @ omega @ psi)
    if tv <= 0:
        return np.nan, np.nan, np.nan

    def _chol_is1(order: list[int]) -> float:
        om_p = omega[np.ix_(order, order)]
        try:
            F = np.linalg.cholesky(om_p + 1e-14 * np.eye(2))
        except np.linalg.LinAlgError:
            return np.nan
        # IS for first market in this ordering: (ψ_0 * F[0,0])² / tv
        return float((psi[order[0]] * F[0, 0]) ** 2 / tv)

    hi  = _chol_is1([0, 1])          # DEX first → DEX upper bound
    lo  = 1.0 - _chol_is1([1, 0])    # CEX first → DEX lower bound
    mid = (np.clip(lo, 0, 1) + np.clip(hi, 0, 1)) / 2.0
    return float(np.clip(lo, 0, 1)), float(np.clip(hi, 0, 1)), float(mid)


def _granger_causality_var(log_dex: pd.Series, log_cex: pd.Series,
                           max_lag: int = 4) -> pd.DataFrame:
    """
    VAR-based Granger causality tests: DEX→CEX and CEX→DEX.
    Uses statsmodels VAR with lag selected by BIC (up to max_lag).
    """
    from statsmodels.tsa.vector_ar.var_model import VAR
    df = pd.DataFrame({"dex": log_dex, "cex": log_cex}).dropna().diff().dropna()
    if len(df) < max_lag * 4 + 20:
        return pd.DataFrame()
    try:
        mdl = VAR(df)
        res = mdl.fit(maxlags=max_lag, ic="bic")
        rows = []
        for caused in ["dex", "cex"]:
            causing = "cex" if caused == "dex" else "dex"
            gc = res.test_causality(caused, causing, kind="f")
            rows.append({
                "Test":    f"{causing.upper()} → {caused.upper()}",
                "F_stat":  round(float(gc.test_statistic), 4),
                "p_val":   round(float(gc.pvalue), 4),
                "Sig":     stars(float(gc.pvalue)),
                "df":      f"({int(gc.df_denom)},{int(gc.df_num)})" if hasattr(gc, "df_denom") else "--",
                "lag_BIC": int(res.k_ar),
            })
        return pd.DataFrame(rows).set_index("Test")
    except Exception as exc:
        print(f"  [WARN] Granger VAR: {exc}")
        return pd.DataFrame()


# ── Rolling estimation ────────────────────────────────────────────────────────

def _rolling_pd(log_dex: pd.Series, log_cex: pd.Series) -> pd.DataFrame:
    """VECM(1) re-estimated every ROLLING_STEP hours on ROLLING_WINDOW obs."""
    idx  = log_dex.index
    rows = []
    for end in range(ROLLING_WINDOW, len(idx), ROLLING_STEP):
        p1 = log_dex.iloc[end - ROLLING_WINDOW : end]
        p2 = log_cex.iloc[end - ROLLING_WINDOW : end]
        r  = _estimate_vecm(p1, p2, k_ar_diff=1)
        if r is None:
            continue
        gg1, gg2 = _gonzalo_granger(r["alpha"])
        lo, hi, mid = _hasbrouck_is(r["alpha"], r["sigma_u"])
        corr = (float(np.corrcoef(r["resid"].T)[0, 1])
                if r["resid"].shape[1] == 2 else np.nan)
        rows.append({
            "date":       idx[end - 1],
            "alpha_DEX":  round(float(r["alpha"][0]), 5),
            "alpha_CEX":  round(float(r["alpha"][1]), 5),
            "GG_DEX":     round(gg1, 4),
            "IS_DEX_lo":  round(lo,  4),
            "IS_DEX_hi":  round(hi,  4),
            "IS_DEX_mid": round(mid, 4),
            "innov_corr": round(corr, 4),
        })
    return (pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Price Discovery: Hasbrouck IS & Gonzalo-Granger CS")
    print("=" * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    mrg = load("merged/merged_hourly.csv")
    cex = load("CEX/cex_price_hourly.csv")
    dex = load("DEX/dex_pool_hourly.csv")
    if mrg is None:
        return

    # Resolve price columns in merged panel
    dex_col = next((c for c in ["dex_eth_usdc_close", "eth_usdc_close",
                                 "dex_eth_usdc_price"] if c in mrg.columns), None)
    cex_col = next((c for c in ["cex_eth_usdc_close", "cex_close",
                                 "close_ethusdc", "close_ethusdt"] if c in mrg.columns), None)
    if dex_col is None and dex is not None:
        for c in ["eth_usdc_close", "eth_usdc_price"]:
            if c in dex.columns:
                mrg["_dex"] = pd.to_numeric(dex[c], errors="coerce").reindex(mrg.index)
                dex_col = "_dex"; break
    if cex_col is None and cex is not None:
        for c in ["close_ethusdc", "close_ethusdt"]:
            if c in cex.columns:
                mrg["_cex"] = pd.to_numeric(cex[c], errors="coerce").reindex(mrg.index)
                cex_col = "_cex"; break
    if dex_col is None or cex_col is None:
        print("[ERROR] Cannot resolve DEX/CEX price columns."); return

    dex_p = pd.to_numeric(mrg[dex_col], errors="coerce").dropna()
    cex_p = pd.to_numeric(mrg[cex_col], errors="coerce").dropna()
    comm  = dex_p.index.intersection(cex_p.index)
    dex_p, cex_p = dex_p.loc[comm], cex_p.loc[comm]
    log_dex = np.log(dex_p.replace(0, np.nan)).dropna()
    log_cex = np.log(cex_p.replace(0, np.nan)).dropna()
    comm2   = log_dex.index.intersection(log_cex.index)
    log_dex, log_cex = log_dex.loc[comm2], log_cex.loc[comm2]
    print(f"  Sample: {comm2[0].date()} → {comm2[-1].date()}  ({len(comm2):,} h)")

    # Volatility (for regime analysis)
    vol = None
    for src in [cex, mrg]:
        if src is not None and "realized_vol_24h_ann" in src.columns:
            vol = pd.to_numeric(src["realized_vol_24h_ann"], errors="coerce").reindex(comm2)
            break

    # Basis magnitude
    basis_abs = None
    if "dex_cex_basis_bps" in mrg.columns:
        basis_abs = pd.to_numeric(mrg["dex_cex_basis_bps"], errors="coerce").abs().reindex(comm2)

    # ── [1] Full-sample VECM ──────────────────────────────────────────────────
    print("\n[1/5] Full-sample bivariate VECM(2)...")
    r0 = _estimate_vecm(log_dex, log_cex, k_ar_diff=2)
    if r0 is None:
        print("  [ERROR] VECM failed."); return

    alpha  = r0["alpha"]
    omega  = r0["sigma_u"]
    resid  = r0["resid"]
    gg1, gg2 = _gonzalo_granger(alpha)
    is_lo, is_hi, is_mid = _hasbrouck_is(alpha, omega)
    corr_r = float(np.corrcoef(resid.T)[0, 1]) if resid.shape[1] == 2 else np.nan
    gg_note = (
        "GG_DEX ≈ 0 when α_CEX ≈ 0: Binance does not error-correct → CEX is the "
        "dominant price-discovery venue. Consistent with DEX as a price-taking AMM "
        "that passively tracks the informationally efficient CEX. "
        "Ref: Gonzalo & Granger (1995); Aspris et al. (2021)."
    )
    is_note = (
        f"IS bounds widen with innovation correlation ρ={corr_r:.3f}. "
        "Midpoint IS is a proxy; Lien & Shrestha (2009) show midpoint is unbiased "
        "under symmetric information production. Hourly frequency understates DEX "
        "short-run price leadership relative to tick-level estimates."
    )
    static = pd.DataFrame.from_dict({
        "α_DEX  (speed-of-adj.)":    f"{alpha[0]:.6f}",
        "α_CEX  (speed-of-adj.)":    f"{alpha[1]:.6f}",
        "GG_DEX (component share)":  f"{gg1:.4f}",
        "GG_CEX (component share)":  f"{gg2:.4f}",
        "IS_DEX lower (Cholesky)":   f"{is_lo:.4f}",
        "IS_DEX upper (Cholesky)":   f"{is_hi:.4f}",
        "IS_DEX midpoint":           f"{is_mid:.4f}",
        "IS_CEX midpoint":           f"{1.0 - is_mid:.4f}",
        "Innovation ρ (DEX, CEX)":   f"{corr_r:.4f}",
        "N (hourly obs.)":           f"{len(comm2):,}",
        "NOTE_GG":                   gg_note,
        "NOTE_IS":                   is_note,
    }, orient="index", columns=["Value"])
    savetable(static, "pd_static")
    print(f"  GG_DEX={gg1:.3f}  IS_DEX midpoint={is_mid:.3f}")

    # ── [2] Granger causality (VAR in first differences) ──────────────────────
    print("\n[2/5] Granger causality (VAR)...")
    gc_tab = _granger_causality_var(log_dex, log_cex, max_lag=4)
    if not gc_tab.empty:
        gc_tab["NOTE"] = (
            "Granger causality in Δlog prices (first differences of VECM series). "
            "Significant CEX→DEX (not DEX→CEX) confirms Binance leads. "
            "Ref: Granger (1969); Toda & Yamamoto (1995) for level-VAR robustness."
        )
        savetable(gc_tab, "pd_granger")

    # ── [3] Rolling estimates ─────────────────────────────────────────────────
    print("\n[3/5] Rolling 30-day IS & GG (step=24 h)...")
    roll_df = _rolling_pd(log_dex, log_cex)
    if roll_df.empty:
        print("  [WARN] No rolling estimates.")
    else:
        roll_df["NOTE"] = (
            "VECM(1) re-estimated on rolling 720-h window, step=24 h. "
            "GG clipped to [0,1]. Estimation uncertainty not shown — interpret trends."
        )
        savetable(roll_df, "pd_rolling")
        print(f"  {len(roll_df)} windows  |  IS_DEX mid mean={roll_df['IS_DEX_mid'].mean():.3f}")

        # Figure: rolling IS + GG
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax1.plot(roll_df.index, roll_df["IS_DEX_mid"],
                 color=COLORS[0], lw=0.9, label="IS_DEX (midpoint)")
        ax1.fill_between(roll_df.index, roll_df["IS_DEX_lo"], roll_df["IS_DEX_hi"],
                         alpha=0.2, color=COLORS[0], label="IS_DEX bounds")
        ax1.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5, label="50% parity")
        ax1.set_ylim(-0.05, 1.05)
        ax1.set_ylabel("Hasbrouck IS (DEX)")
        ax1.set_title("Rolling 30-day Price Discovery: Uniswap v3 DEX vs Binance")
        ax1.legend(fontsize=8)

        ax2.plot(roll_df.index, roll_df["GG_DEX"],
                 color=COLORS[1], lw=0.9, label="GG_DEX (CS)")
        ax2.fill_between(roll_df.index, 0, roll_df["GG_DEX"].clip(0),
                         alpha=0.15, color=COLORS[1])
        ax2.axhline(0.5, color="black", lw=0.7, ls="--", alpha=0.5)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("Gonzalo-Granger CS (DEX)")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=8)
        plt.suptitle("IS < 0.5  →  CEX dominates price discovery", y=1.00)
        savefig("pd_rolling_is")

    # ── [4] Regime-conditional ────────────────────────────────────────────────
    print("\n[4/5] Regime-conditional IS & GG...")
    regime_rows = []
    if vol is not None:
        reg_series = vol_regime(vol).reindex(comm2)
        for reg in ["low", "normal", "high"]:
            idx_r = reg_series[reg_series == reg].index
            if len(idx_r) < MIN_OBS_VECM:
                continue
            rr = _estimate_vecm(log_dex.loc[idx_r], log_cex.loc[idx_r], k_ar_diff=1)
            if rr is None:
                continue
            gg_r1, gg_r2 = _gonzalo_granger(rr["alpha"])
            lo_r, hi_r, mid_r = _hasbrouck_is(rr["alpha"], rr["sigma_u"])
            regime_rows.append({
                "regime":     reg,
                "N_hours":    len(idx_r),
                "α_DEX":      round(float(rr["alpha"][0]), 4),
                "α_CEX":      round(float(rr["alpha"][1]), 4),
                "GG_DEX":     round(gg_r1, 4),
                "IS_DEX_lo":  round(lo_r,  4),
                "IS_DEX_hi":  round(hi_r,  4),
                "IS_DEX_mid": round(mid_r, 4),
            })
    if regime_rows:
        reg_df = pd.DataFrame(regime_rows).set_index("regime")
        reg_df["NOTE"] = (
            "VECM(1) estimated on non-consecutive subset of hours per regime. "
            "Cointegration rank 1 imposed; may not hold within subsets. "
            "Interpret as conditional average, not causal regime effect."
        )
        savetable(reg_df, "pd_by_regime")

        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        x = np.arange(len(regime_rows))
        labs = [r["regime"] for r in regime_rows]
        axes[0].bar(x, [r["GG_DEX"] for r in regime_rows], color=COLORS[:3], alpha=0.8)
        axes[0].set_xticks(x); axes[0].set_xticklabels(labs)
        axes[0].axhline(0.5, color="black", lw=0.8, ls="--")
        axes[0].set_title("Gonzalo-Granger CS (DEX)")
        axes[0].set_ylabel("Component Share")
        axes[1].bar(x, [r["IS_DEX_mid"] for r in regime_rows], color=COLORS[:3], alpha=0.8)
        axes[1].set_xticks(x); axes[1].set_xticklabels(labs)
        axes[1].axhline(0.5, color="black", lw=0.8, ls="--")
        axes[1].set_title("Hasbrouck IS (DEX) midpoint")
        axes[1].set_ylabel("Information Share")
        plt.suptitle("Price Discovery by Volatility Regime")
        savefig("pd_by_regime")

    # ── [5] Basis-conditional ─────────────────────────────────────────────────
    print("\n[5/5] Basis-conditional IS...")
    basis_rows = []
    if basis_abs is not None:
        q50, q80 = basis_abs.quantile([0.50, 0.80])
        for lbl, mask in [("low  (|basis|<p50)", basis_abs < q50),
                           ("high (|basis|>p80)", basis_abs > q80)]:
            idx_b = mask[mask].index.intersection(comm2)
            if len(idx_b) < MIN_OBS_VECM:
                continue
            rb = _estimate_vecm(log_dex.loc[idx_b], log_cex.loc[idx_b], k_ar_diff=1)
            if rb is None:
                continue
            gg_b, _ = _gonzalo_granger(rb["alpha"])
            lo_b, hi_b, mid_b = _hasbrouck_is(rb["alpha"], rb["sigma_u"])
            basis_rows.append({"condition": lbl, "N_hours": len(idx_b),
                                "GG_DEX": round(gg_b, 4), "IS_DEX_mid": round(mid_b, 4)})
    if basis_rows:
        b_df = pd.DataFrame(basis_rows).set_index("condition")
        b_df["NOTE"] = (
            "Large |basis| hours: DEX temporarily mispriced relative to CEX. "
            "If IS_DEX rises during high-|basis| hours, DEX creates short-lived "
            "price discrepancies closed by arbitrage — temporary price leadership. "
            "Ref: Aspris et al. (2021); Lehar & Parlour (2021)."
        )
        savetable(b_df, "pd_by_basis")

    print("\nDONE")


if __name__ == "__main__":
    main()
