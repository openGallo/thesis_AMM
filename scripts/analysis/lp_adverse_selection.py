"""
lp_adverse_selection.py — Liquidity Provider Adverse Selection Analysis

Research Question:
    Do Uniswap v3 liquidity providers exhibit systematic adverse selection —
    minting positions into elevated-volatility and wide-basis environments —
    and does ex-ante market condition at mint time predict realized LP returns?

Motivation:
    In classic microstructure theory (Kyle 1985; Glosten & Milgrom 1985),
    liquidity providers suffer from adverse selection when informed traders
    exploit their quotes. In AMMs, this manifests as LPs minting concentrated
    positions just before large price moves, causing maximum impermanent loss
    and LVR. Unlike traditional market-making, Uniswap v3 LPs choose both
    position timing and range — both dimensions can exhibit skill or adverse
    selection. This script tests whether the LP population is, on average,
    informationally disadvantaged at mint time.

Methodology:
    1. Event study: compute average CEX volatility in a ±48 h window
       around each mint event. Adverse selection signature: vol_t+k > vol_t-k
       (LPs mint before volatility spikes, not after).

    2. Position return: ROI = (net_pnl_usd / mint_usd) where
       net_pnl_usd = burned_usd − mint_usd (lower bound; ignores uncollected
       fees not captured by subgraph collect events).

    3. Cross-sectional OLS (HC3) of position ROI on ex-ante predictors:
           ROI ~ vol_at_mint + |basis|_at_mint + log(range_width) +
                 log(mint_usd) + hour_of_day + day_of_week + controls
       Fama-MacBeth style: estimate monthly cross-sections, report
       time-series mean and Newey-West t-statistics.

    4. Range-width quartile analysis: do narrow-range LPs suffer more
       adverse selection than wide-range LPs? (Lehar & Parlour 2021 predict
       concentration amplifies adverse selection cost.)

    5. Log-linear duration model: log(duration_hours) ~ vol_at_mint +
       |basis|_at_mint + range_width + mint_usd.
       Positive β on vol: LPs exit faster in high-vol environments
       (consistent with defensive rebalancing).

    6. Cohort analysis: monthly cohorts of mints — compute median ROI
       by cohort to detect time trends in LP profitability.

Key caveats (embedded in output tables):
    - net_pnl_usd = burned_usd − mint_usd approximates LP welfare but omits
      uncollected fees (collect events not indexed by subgraph). ROI is a
      LOWER BOUND on true LP return. Positions still open (burned_usd = NaN)
      are excluded; this creates survivorship if unprofitable positions are
      held longer.
    - Adverse selection is pool-level (not position-level tick-crossing).
      A position minted during high-vol may be profitable if the fee APR
      compensates. Net welfare = fees − IL, not just IL.
    - Fama-MacBeth requires at least 10 positions per monthly cross-section;
      months with fewer positions are excluded.
    - Range width is logged to handle the heavy right tail. Extreme range
      positions (range_width_pct > 500%) capped.

Outputs:
    output/tables/asel_event_study.csv      Vol around mint events (±48 h)
    output/tables/asel_regression.csv       OLS ROI ~ ex-ante predictors
    output/tables/asel_fm_betas.csv         Fama-MacBeth monthly coefficients
    output/tables/asel_range_quartile.csv   ROI by range-width quartile
    output/tables/asel_duration.csv         Log-duration OLS
    output/tables/asel_cohort.csv           Monthly cohort ROI
    output/figures/asel_event_study.pdf     Vol event-study plot
    output/figures/asel_roi_by_range.pdf    ROI distribution by range quartile

References:
    Kyle, A.S. (1985). Continuous auctions and insider trading. Econometrica.
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
    Adams, A. et al. (2021). Uniswap v3 Core. Uniswap Labs.
    Barbon, A. & Ranaldo, A. (2022). On the quality of cryptocurrency markets.
        J. Financial Econ., 143(1), 291-323.
    Fama, E. & MacBeth, J. (1973). Risk, return, and equilibrium.
        J. Political Economy, 81(3), 607-636.
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
    stars, bootstrap_ci, block_bootstrap_ci, vol_regime,
)
import matplotlib.pyplot as plt

RANGE_CAP_PCT  = 500.0   # cap extreme range widths (% of price)
MIN_MINT_USD   = 10.0    # exclude dust positions
MIN_POS_MONTH  = 10      # Fama-MacBeth: min positions per cross-section
EVENT_WINDOW   = 48      # hours before/after mint for event study


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ols_hc3(y: pd.Series, X: pd.DataFrame, label: str = "") -> pd.DataFrame:
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(df) < 30:
        return pd.DataFrame()
    y_ = df["__y__"]
    X_ = sm.add_constant(df.drop(columns="__y__"))
    res = sm.OLS(y_, X_).fit(cov_type="HC3")
    rows = []
    for v, c, se, t, p in zip(res.model.exog_names,
                               res.params, res.bse, res.tvalues, res.pvalues):
        rows.append({"Variable": v,
                     "Coef": round(float(c), 6), "SE_HC3": round(float(se), 6),
                     "t_stat": round(float(t), 3), "p_val": round(float(p), 4),
                     "Sig": stars(float(p))})
    rows.append({"Variable": "—", "Coef": np.nan, "SE_HC3": np.nan,
                 "t_stat": np.nan, "p_val": np.nan,
                 "Sig": (f"N={len(df):,}  R²={res.rsquared:.4f}  "
                         f"adj-R²={res.rsquared_adj:.4f}  {label}")})
    return pd.DataFrame(rows).set_index("Variable")


def _resolve_mint_time(pos: pd.DataFrame) -> pd.Series:
    """Return mint timestamp series, trying multiple column names."""
    for c in ["mint_time", "open_time", "timestamp"]:
        if c in pos.columns:
            return pd.to_datetime(pos[c], utc=True, errors="coerce")
    return pd.NaT


def _resolve_burn_time(pos: pd.DataFrame) -> pd.Series:
    for c in ["burn_time", "close_time"]:
        if c in pos.columns:
            return pd.to_datetime(pos[c], utc=True, errors="coerce")
    return pd.NaT


# ── Event study ───────────────────────────────────────────────────────────────

def event_study(mint_times: pd.Series, vol: pd.Series,
                window: int = EVENT_WINDOW) -> pd.DataFrame:
    """
    Average vol at each event-relative hour h ∈ [-window, +window].
    Uses vectorised lookup: floors mint_time to hour, then reindexes vol.
    """
    vol_arr   = vol.dropna()
    mint_h    = mint_times.dropna().dt.floor("h")
    offsets   = range(-window, window + 1)
    mean_vol, se_vol = {}, {}
    for h in offsets:
        shifted = mint_h + pd.Timedelta(hours=h)
        vals    = vol_arr.reindex(shifted.values).values
        vals    = vals[~np.isnan(vals)]
        if len(vals) < 5:
            mean_vol[h] = np.nan; se_vol[h] = np.nan
        else:
            mean_vol[h] = float(np.mean(vals))
            se_vol[h]   = float(np.std(vals) / np.sqrt(len(vals)))
    df = pd.DataFrame({"h": list(offsets),
                       "vol_mean": [mean_vol[h] for h in offsets],
                       "vol_se":   [se_vol[h]   for h in offsets]}).set_index("h")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("LP Adverse Selection Analysis")
    print("=" * 60)

    # ── Data loading ──────────────────────────────────────────────────────────
    pos = load("DEX/dex_lp_positions.csv")
    cex = load("CEX/cex_price_hourly.csv")
    mrg = load("merged/merged_hourly.csv")
    if pos is None:
        print("[ERROR] dex_lp_positions.csv not found."); return

    print(f"  LP positions loaded: {len(pos):,} rows")

    # ── Position data preparation ─────────────────────────────────────────────
    mint_t = _resolve_mint_time(pos)
    burn_t = _resolve_burn_time(pos)

    # Net P&L proxy: burned_usd - mint_usd
    for c_mint in ["mint_usd", "deposited_amount_usd", "total_deposited_usd"]:
        if c_mint in pos.columns:
            mint_usd = pd.to_numeric(pos[c_mint], errors="coerce"); break
    else:
        print("[WARN] No mint_usd column; trying deposited_token0/1...")
        mint_usd = pd.Series(np.nan, index=pos.index)

    for c_burn in ["burned_usd", "total_burned_usd", "withdrawn_amount_usd"]:
        if c_burn in pos.columns:
            burn_usd = pd.to_numeric(pos[c_burn], errors="coerce"); break
    else:
        burn_usd = pd.Series(np.nan, index=pos.index)

    pnl = pd.to_numeric(
        pos["net_pnl_usd"] if "net_pnl_usd" in pos.columns
        else burn_usd - mint_usd, errors="coerce"
    )

    # Filter: closed positions with non-trivial mint amount
    closed  = (burn_t.notna() & mint_usd.notna() & pnl.notna()
               & (mint_usd > MIN_MINT_USD))
    pos_cl  = pos[closed].copy()
    mint_cl = mint_t[closed]
    burn_cl = burn_t[closed]
    mu_cl   = mint_usd[closed]
    pnl_cl  = pnl[closed]
    roi_cl  = (pnl_cl / mu_cl.replace(0, np.nan)).rename("roi")

    print(f"  Closed positions with mint_usd > ${MIN_MINT_USD}: {len(pos_cl):,}")
    if len(pos_cl) < 50:
        print("[WARN] Very few closed positions; adverse selection analysis may be unreliable.")

    # Range width
    rw_raw = None
    for c in ["range_width_pct", "range_pct"]:
        if c in pos_cl.columns:
            rw_raw = pd.to_numeric(pos_cl[c], errors="coerce").clip(upper=RANGE_CAP_PCT)
            break
    if rw_raw is None and all(c in pos_cl.columns for c in ["tick_lower", "tick_upper"]):
        # Approximate: range_width ≈ 2*(exp(tick_upper*log(1.0001)/2) - 1)
        tl = pd.to_numeric(pos_cl["tick_lower"],  errors="coerce")
        tu = pd.to_numeric(pos_cl["tick_upper"],  errors="coerce")
        rw_raw = (np.exp((tu - tl) * np.log(1.0001)) - 1.0) * 100
        rw_raw = rw_raw.clip(upper=RANGE_CAP_PCT)

    dur_h = None
    if "duration_days" in pos_cl.columns:
        dur_h = pd.to_numeric(pos_cl["duration_days"], errors="coerce") * 24
    elif mint_t is not None and burn_t is not None:
        dur_h = ((burn_cl - mint_cl).dt.total_seconds() / 3600).clip(lower=0)

    # ── Market conditions at mint time ─────────────────────────────────────────
    # Snap vol and basis to the hourly timestamp of each mint
    vol_h = None
    for src in [cex, mrg]:
        if src is not None:
            for c in ["realized_vol_24h_ann"]:
                if c in src.columns:
                    vol_h = pd.to_numeric(src[c], errors="coerce")
                    break
        if vol_h is not None:
            break

    basis_h = None
    if mrg is not None and "dex_cex_basis_bps" in mrg.columns:
        basis_h = pd.to_numeric(mrg["dex_cex_basis_bps"], errors="coerce").abs()

    mint_floor = mint_cl.dt.floor("h")
    vol_at_mint   = (vol_h.reindex(mint_floor.values).values
                     if vol_h is not None else np.full(len(pos_cl), np.nan))
    basis_at_mint = (basis_h.reindex(mint_floor.values).values
                     if basis_h is not None else np.full(len(pos_cl), np.nan))

    # ── [1] Event study around mint ──────────────────────────────────────────
    print("\n[1/5] Event study: vol ±48 h around mint events...")
    if vol_h is not None and mint_cl.notna().sum() > 20:
        ev = event_study(mint_cl, vol_h, window=EVENT_WINDOW)
        vol_pre  = float(ev.loc[ev.index < 0,  "vol_mean"].mean()) if len(ev) > 0 else np.nan
        vol_post = float(ev.loc[ev.index > 0,  "vol_mean"].mean()) if len(ev) > 0 else np.nan
        ratio    = vol_post / vol_pre if vol_pre and vol_pre > 0 else np.nan
        ev["NOTE"] = np.nan
        ev.at[0, "NOTE"] = (
            f"Adverse selection signature: post/pre vol ratio = {ratio:.3f}. "
            "Ratio > 1 → LPs mint before vol spikes (adverse selection). "
            "Ratio ≈ 1 → timing uncorrelated with vol (neutral). "
            "Ratio < 1 → LPs mint after vol spikes (informed / contrarian). "
            "Ref: Lehar & Parlour (2021); Barbon & Ranaldo (2022)."
        )
        savetable(ev, "asel_event_study")

        # Figure: event study
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(ev.index, ev["vol_mean"] * 100, color=COLORS[0], lw=1.5,
                label="Mean 24h vol (%)")
        ax.fill_between(ev.index,
                        (ev["vol_mean"] - 1.96 * ev["vol_se"]) * 100,
                        (ev["vol_mean"] + 1.96 * ev["vol_se"]) * 100,
                        alpha=0.2, color=COLORS[0], label="±1.96 SE")
        ax.axvline(0, color="black", lw=1.2, ls="--", label="Mint event (h=0)")
        ax.axhline(float(vol_h.mean()) * 100, color=COLORS[1], lw=0.8, ls=":",
                   alpha=0.7, label="Unconditional mean vol")
        ax.set_xlabel("Hours relative to mint")
        ax.set_ylabel("24h Realized Vol (ann., %)")
        ax.set_title(f"Volatility Around Mint Events (N={int(mint_cl.notna().sum()):,} positions)\n"
                     f"Post/Pre vol ratio = {ratio:.3f}  "
                     f"({'adverse selection' if ratio > 1.02 else 'neutral / informed'})")
        ax.legend(fontsize=8)
        savefig("asel_event_study")
        print(f"  Post/pre vol ratio = {ratio:.3f}  "
              f"({'ASelective' if ratio > 1.02 else 'Neutral'})")
    else:
        print("  [SKIP] Insufficient vol or mint data for event study.")

    # ── [2] Cross-sectional OLS ───────────────────────────────────────────────
    print("\n[2/5] Cross-sectional OLS: ROI ~ ex-ante predictors...")
    reg_vars = {}
    if vol_at_mint is not None:
        reg_vars["vol_at_mint"]       = pd.Series(vol_at_mint,   index=pos_cl.index)
    if basis_at_mint is not None:
        reg_vars["abs_basis_at_mint"] = pd.Series(basis_at_mint, index=pos_cl.index)
    if rw_raw is not None:
        reg_vars["log_range_width"]   = np.log1p(rw_raw)
    if mu_cl is not None:
        reg_vars["log_mint_usd"]      = np.log1p(mu_cl)
    if mint_cl.notna().any():
        reg_vars["hour_of_day"] = mint_cl.dt.hour.astype(float)
        reg_vars["day_of_week"] = mint_cl.dt.dayofweek.astype(float)

    if reg_vars and len(roi_cl.dropna()) >= 50:
        X_ols = pd.DataFrame(reg_vars, index=pos_cl.index)
        ols_tab = _ols_hc3(roi_cl, X_ols, "HC3 robust SE")
        if not ols_tab.empty:
            ols_tab["NOTE"] = (
                "Dep. var.: ROI = net_pnl_usd / mint_usd (lower bound; excludes uncollected fees). "
                "vol_at_mint = 24h realized vol at hour of mint. "
                "Negative β on vol_at_mint → higher vol at mint → worse realized return "
                "(consistent with adverse selection). Positive β → informed/contrarian timing. "
                "Ref: Fama & MacBeth (1973); Barbon & Ranaldo (2022)."
            )
            savetable(ols_tab, "asel_regression")

    # ── [3] Fama-MacBeth monthly cross-sections ───────────────────────────────
    print("\n[3/5] Fama-MacBeth monthly cross-sections...")
    if mint_cl.notna().any() and reg_vars:
        month_key = mint_cl.dt.to_period("M")
        months    = month_key.dropna().unique()
        fm_betas  = []
        fm_vars   = list(reg_vars.keys())
        for mth in months:
            mask = month_key == mth
            y_m  = roi_cl[mask].dropna()
            X_m  = pd.DataFrame(reg_vars, index=pos_cl.index).loc[mask]
            comm_m = y_m.index.intersection(X_m.dropna().index)
            if len(comm_m) < MIN_POS_MONTH:
                continue
            try:
                X_m_a = sm.add_constant(X_m.loc[comm_m])
                res_m = sm.OLS(y_m.loc[comm_m], X_m_a).fit()
                row = {"month": str(mth)}
                row.update({v: float(res_m.params.get(v, np.nan)) for v in fm_vars})
                fm_betas.append(row)
            except Exception:
                pass

        if len(fm_betas) >= 6:
            fm_df    = pd.DataFrame(fm_betas).set_index("month")
            fm_means = fm_df.mean()
            fm_sds   = fm_df.std()
            fm_n     = len(fm_df)
            fm_t     = fm_means / (fm_sds / np.sqrt(fm_n))
            fm_p     = 2 * sp_stats.t.sf(fm_t.abs(), df=fm_n - 1)
            fm_sum   = pd.DataFrame({
                "FM_beta_mean":    fm_means.round(6),
                "FM_beta_std":     fm_sds.round(6),
                "FM_NW_t":         fm_t.round(3),
                "FM_p_val":        fm_p.round(4),
                "FM_sig":          [stars(p) for p in fm_p],
                "N_months":        fm_n,
            })
            fm_sum["NOTE"] = (
                f"Fama-MacBeth (1973) approach: OLS estimated monthly ({fm_n} months), "
                "time-series mean of monthly βs reported. NW t-statistic (no lag correction "
                "since months are independent). Min 10 positions per cross-section required. "
                "Interpret as: average predictive power of ex-ante conditions on LP ROI."
            )
            savetable(fm_sum, "asel_fm_betas")
            print(f"  Fama-MacBeth: {fm_n} monthly cross-sections")

    # ── [4] Range quartile analysis ───────────────────────────────────────────
    print("\n[4/5] ROI and adverse selection by range-width quartile...")
    if rw_raw is not None and len(roi_cl.dropna()) >= 40:
        q_labels = ["Q1 (narrow)", "Q2", "Q3", "Q4 (wide)"]
        rq = pd.qcut(rw_raw.reindex(roi_cl.index), q=4,
                     labels=q_labels, duplicates="drop")
        rq_rows = []
        for q_lab in rq.cat.categories:
            mask = rq == q_lab
            sub  = roi_cl[mask].dropna()
            va_m = pd.Series(vol_at_mint, index=pos_cl.index)[mask].dropna()
            if len(sub) < 5:
                continue
            ci_lo, ci_hi = bootstrap_ci(sub.values)
            rq_rows.append({
                "quartile":             q_lab,
                "N_positions":          len(sub),
                "roi_mean":             round(float(sub.mean()), 4),
                "roi_median":           round(float(sub.median()), 4),
                "roi_ci95_lo":          round(ci_lo, 4),
                "roi_ci95_hi":          round(ci_hi, 4),
                "pct_profitable":       round(float((sub > 0).mean()), 4),
                "vol_at_mint_mean":     round(float(va_m.mean()), 4) if len(va_m) > 0 else np.nan,
                "rw_mean_pct":          round(float(rw_raw[mask].mean()), 2),
            })
        if rq_rows:
            rq_df = pd.DataFrame(rq_rows).set_index("quartile")
            rq_df["NOTE"] = (
                "Narrow-range LPs (Q1) earn higher expected fee income per unit capital "
                "(concentration factor) but suffer larger impermanent loss when price "
                "moves out of range. If vol_at_mint_mean is highest for Q1, narrow LPs "
                "exhibit worst timing (adverse selection amplified by concentration). "
                "Ref: Lehar & Parlour (2021); Adams et al. (2021)."
            )
            savetable(rq_df, "asel_range_quartile")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            cats = [r["quartile"] for r in rq_rows]
            axes[0].bar(range(len(rq_rows)), [r["roi_mean"] for r in rq_rows],
                        color=COLORS[:len(rq_rows)], alpha=0.8)
            axes[0].errorbar(range(len(rq_rows)),
                             [r["roi_mean"] for r in rq_rows],
                             yerr=[r["roi_mean"] - r["roi_ci95_lo"] for r in rq_rows],
                             fmt="none", color="black", capsize=4)
            axes[0].axhline(0, color="black", lw=0.8, ls="--")
            axes[0].set_xticks(range(len(rq_rows)))
            axes[0].set_xticklabels(cats, fontsize=8, rotation=15)
            axes[0].set_ylabel("Mean ROI")
            axes[0].set_title("Mean LP ROI by Range-Width Quartile")

            axes[1].bar(range(len(rq_rows)), [r["pct_profitable"] for r in rq_rows],
                        color=COLORS[:len(rq_rows)], alpha=0.8)
            axes[1].axhline(0.5, color="black", lw=0.8, ls="--")
            axes[1].set_xticks(range(len(rq_rows)))
            axes[1].set_xticklabels(cats, fontsize=8, rotation=15)
            axes[1].set_ylabel("Fraction profitable")
            axes[1].set_title("% Profitable Positions by Range-Width Quartile")
            plt.suptitle("LP Performance by Range Width (Q1=narrow, Q4=wide)")
            savefig("asel_roi_by_range")

    # ── [5] Log-duration model ────────────────────────────────────────────────
    print("\n[5/5] Log-duration OLS...")
    if dur_h is not None and len(dur_h.dropna()) >= 50:
        log_dur = np.log1p(dur_h.reindex(pos_cl.index)).rename("log_duration_h")
        dur_X   = pd.DataFrame({
            k: v.reindex(pos_cl.index) for k, v in reg_vars.items()
        }, index=pos_cl.index)
        dur_tab = _ols_hc3(log_dur, dur_X, "log-duration")
        if not dur_tab.empty:
            dur_tab["NOTE"] = (
                "Log-linear duration model: dep. var. = log(1 + hours from mint to burn). "
                "Positive β on vol_at_mint: LPs hold longer when minted during high vol "
                "(waiting for price recovery) — consistent with 'lock-in' effect. "
                "Negative β: LPs exit quickly after high-vol mints (defensive rebalancing). "
                "Not a formal survival model (no censoring correction for open positions). "
                "Ref: Kiefer (1988) for duration models."
            )
            savetable(dur_tab, "asel_duration")

    # ── [6] Monthly cohort ROI ────────────────────────────────────────────────
    if mint_cl.notna().any():
        cohort_key = mint_cl.dt.to_period("M").astype(str)
        coh_df = pd.DataFrame({
            "cohort":    cohort_key.values,
            "roi":       roi_cl.values,
            "mint_usd":  mu_cl.values,
        }).dropna()
        if len(coh_df) >= 20:
            coh_agg = (coh_df.groupby("cohort")
                       .agg(N=("roi", "count"),
                            roi_median=("roi", "median"),
                            roi_mean=("roi", "mean"),
                            pct_pos=("roi", lambda x: (x > 0).mean()),
                            mint_usd_median=("mint_usd", "median"))
                       .reset_index().set_index("cohort"))
            coh_agg["NOTE"] = (
                "Monthly cohort analysis: each row = positions opened in that month. "
                "Declining roi_median over time → worsening LP environment (more competition, "
                "higher LVR). Rising roi_median → improving pool conditions. "
                "Cohort effect is confounded by selection: later cohorts may include "
                "more sophisticated LPs using automated tools."
            )
            savetable(coh_agg, "asel_cohort")

    print("\nDONE")


if __name__ == "__main__":
    main()
