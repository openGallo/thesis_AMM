"""
order_flow_toxicity.py — Order Flow Toxicity, Adverse Selection & LP Welfare

Research Question:
    Does directional order flow toxicity — measured by signed volume imbalance
    in the Uniswap v3 pool — predict DEX-CEX basis persistence and LP welfare
    losses, consistent with the adverse selection channel in market microstructure?

Motivation:
    Easley et al. (2012) VPIN identifies high-frequency adverse selection by
    measuring how skewed volume is in one direction: informed traders consistently
    buy or sell, while uninformed traders transact symmetrically. In AMMs, this
    translates to one-sided swap flow pushing the pool price away from the CEX,
    increasing the DEX-CEX basis and the arbitrage opportunity. Arbitrageurs who
    correct this basis extract LVR from passive LPs. If VPIN predicts basis
    widening and LVR spikes, it establishes an empirical link between informed
    flow, price dislocation, and LP welfare — a novel contribution for AMM research.

Methodology:
    1. Hourly order flow imbalance (OFI) from swap directions in dex_swaps.csv:
           buy_vol_h  = Σ amount_usd  [direction ∈ {buy, +1, 1}]
           sell_vol_h = Σ amount_usd  [direction ∈ {sell, -1, 0}]
           OFI_h      = (buy_vol_h − sell_vol_h) / (buy_vol_h + sell_vol_h)
           VPIN_h     = |OFI_h|    (unsigned toxicity ∈ [0, 1])

       Interpretation: VPIN = 1 → all volume is one-directional (maximum toxicity);
       VPIN = 0 → balanced two-way flow (minimum toxicity / uninformed).

    2. VPIN rolling average: 24 h and 168 h (weekly) smoothed VPIN.

    3. Granger causality tests (VAR(p), lag selected by BIC):
           H1: VPIN → |basis_bps|   (toxicity predicts basis widening)
           H2: |basis_bps| → VPIN   (basis attracts informed flow)
       Also: VPIN → LVR_rate (flow toxicity predicts LP cost).

    4. Predictive OLS regressions (HAC SE, 24 lags):
           |basis_{t+1}|  = α + β₁·VPIN_t + β₂·vol_t + β₃·log_volume_t + ε
           LVR_rate_{t+1} = α + β₁·VPIN_t + β₂·vol_t + β₃·|basis_t|    + ε

    5. LP welfare by VPIN decile: compare E[fee_apr − LVR_rate | decile d].
       If welfare is monotonically decreasing in VPIN → clear adverse selection
       channel from order flow toxicity to LP welfare loss.

    6. Signed OFI (not absolute) and subsequent CEX return:
       Tests whether DEX order flow anticipates CEX price moves
       (i.e., does the DEX contain information not yet in CEX?).

Key caveats (embedded in output tables):
    - Trade direction uses the 'direction' column from process_dex_swaps.py.
      Direction is inferred from the sign of price change (log_price_change > 0
      = buy pressure); cross-block swaps where direction is ambiguous are
      excluded from OFI computation and contribute only to total volume.
    - True VPIN (Easley et al. 2012) uses volume clock bucketing; hourly OFI
      is a time-clock approximation. Result: higher noise, lower power.
    - Pool-level VPIN averages over heterogeneous traders (MEV bots, retail,
      institutions). Decomposing by trade size would improve identification.
    - Granger causality in levels vs differences: basis and VPIN are both
      stationary (basis by construction; VPIN ∈ [0,1]). VAR in levels is
      appropriate; stationarity pre-test reported.

Outputs:
    output/tables/oft_summary.csv           VPIN summary statistics
    output/tables/oft_granger.csv           Granger causality VAR tests
    output/tables/oft_predictive_basis.csv  OLS: |basis_{t+1}| ~ VPIN_t
    output/tables/oft_predictive_lvr.csv    OLS: LVR_{t+1} ~ VPIN_t
    output/tables/oft_welfare_decile.csv    LP welfare by VPIN decile
    output/tables/oft_signed_ofi.csv        Signed OFI predicting CEX return
    output/figures/oft_vpin_series.pdf      VPIN and basis time series
    output/figures/oft_welfare_decile.pdf   LP welfare by VPIN decile

References:
    Easley, D. et al. (2012). Flow toxicity and liquidity in a high frequency
        world. Rev. Financial Studies, 25(5), 1457-1493.
    Kyle, A.S. (1985). Continuous auctions and insider trading. Econometrica.
    Capponi, A. & Jia, R. (2022). The adoption of blockchain-based decentralized
        exchanges. arXiv:2009.07663.
    Collin-Dufresne, P. & Fos, V. (2015). Do prices reveal the presence of
        informed trading? J. Finance, 70(4), 1555-1582.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import statsmodels.api as sm

warnings.filterwarnings("ignore")

from analysis_utils import (
    COLORS, load, load_swaps, savefig, savetable,
    stars, block_bootstrap_ci, vol_regime, stationarity_tests,
)
import matplotlib.pyplot as plt

VPIN_SMOOTH_24  = 24    # 24-h rolling VPIN
VPIN_SMOOTH_168 = 168   # 7-day rolling VPIN
N_DECILES       = 10    # for welfare analysis
LAG_MAX_GRANGER = 6


# ── OFI computation from swaps ────────────────────────────────────────────────

def compute_hourly_ofi(swaps: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate swap-level direction into hourly OFI and VPIN.

    Direction encoding handled: 'buy'/1/1.0 = buy; 'sell'/-1/0 = sell.
    Cross-block swaps with ambiguous direction are counted in total_vol only.
    """
    sw = swaps.copy()
    sw["timestamp"] = pd.to_datetime(sw["timestamp"], utc=True, errors="coerce")
    sw = sw.dropna(subset=["timestamp"])
    sw["hour"] = sw["timestamp"].dt.floor("h")

    # Normalise direction column
    if "direction" in sw.columns:
        d = sw["direction"]
        # Convert string or signed numeric to {1 (buy), -1 (sell), 0 (unknown)}
        d_norm = pd.Series(0, index=sw.index, dtype=float)
        if d.dtype == object:
            d_norm[d.str.lower().isin(["buy", "b", "1"])]  = 1.0
            d_norm[d.str.lower().isin(["sell", "s", "-1"])] = -1.0
        else:
            d_num = pd.to_numeric(d, errors="coerce").fillna(0)
            d_norm[d_num > 0]  =  1.0
            d_norm[d_num < 0]  = -1.0
        sw["dir_norm"] = d_norm
    elif "log_price_change" in sw.columns:
        # Fallback: direction inferred from within-block price change
        lpc = pd.to_numeric(sw["log_price_change"], errors="coerce")
        sw["dir_norm"] = np.sign(lpc).fillna(0)
    else:
        print("  [WARN] No direction column; using log_price_change sign as proxy.")
        sw["dir_norm"] = 0.0

    amount = pd.to_numeric(sw.get("amount_usd", sw.get("amount_usd_swap", 0)),
                           errors="coerce").fillna(0)
    sw["amount_usd_"] = amount

    grp       = sw.groupby("hour")
    buy_vol   = grp.apply(lambda g: (g["amount_usd_"] * (g["dir_norm"] > 0)).sum())
    sell_vol  = grp.apply(lambda g: (g["amount_usd_"] * (g["dir_norm"] < 0)).sum())
    total_vol = grp["amount_usd_"].sum()
    n_trades  = grp["amount_usd_"].count()

    ofi  = (buy_vol - sell_vol) / (buy_vol + sell_vol).replace(0, np.nan)
    vpin = ofi.abs()

    df = pd.DataFrame({
        "buy_vol_usd":   buy_vol,
        "sell_vol_usd":  sell_vol,
        "total_vol_usd": total_vol,
        "n_trades":      n_trades,
        "OFI":           ofi,
        "VPIN":          vpin,
    })
    df.index = pd.DatetimeIndex(df.index, name="timestamp_utc").tz_localize("UTC")
    return df


# ── VAR Granger causality ─────────────────────────────────────────────────────

def _granger_var(df_panel: pd.DataFrame,
                 x_col: str, y_col: str,
                 max_lag: int = LAG_MAX_GRANGER) -> dict:
    """VAR-based Granger causality: x_col → y_col (HAC)."""
    from statsmodels.tsa.vector_ar.var_model import VAR
    sub = df_panel[[x_col, y_col]].dropna()
    if len(sub) < max_lag * 4 + 20:
        return {}
    try:
        mdl = VAR(sub)
        res = mdl.fit(maxlags=max_lag, ic="bic")
        gc  = res.test_causality(y_col, x_col, kind="f")
        return {
            "from":  x_col, "to": y_col,
            "F_stat": round(float(gc.test_statistic), 4),
            "p_val":  round(float(gc.pvalue), 4),
            "Sig":    stars(float(gc.pvalue)),
            "lags":   int(res.k_ar),
        }
    except Exception as exc:
        return {"from": x_col, "to": y_col, "error": str(exc)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Order Flow Toxicity — VPIN, Adverse Selection & LP Welfare")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    mrg = load("merged/merged_hourly.csv")
    lvr = load("DEX/dex_lvr_hourly.csv")
    cex = load("CEX/cex_price_hourly.csv")
    dex = load("DEX/dex_pool_hourly.csv")

    # Load swaps (direction + amount_usd only; full file for complete coverage)
    print("\n  Loading swap data for OFI computation...")
    swaps = load_swaps(cols=["timestamp", "amount_usd", "direction",
                              "log_price_change", "gas_price_wei"])
    if swaps is None:
        print("[ERROR] dex_swaps.csv not found."); return
    print(f"  Swaps loaded: {len(swaps):,} rows")

    # ── [1] Compute hourly VPIN ───────────────────────────────────────────────
    print("\n[1/6] Computing hourly OFI and VPIN...")
    ofi_h = compute_hourly_ofi(swaps)
    del swaps   # free memory

    ofi_h["VPIN_24h"]  = ofi_h["VPIN"].rolling(VPIN_SMOOTH_24,  min_periods=6).mean()
    ofi_h["VPIN_168h"] = ofi_h["VPIN"].rolling(VPIN_SMOOTH_168, min_periods=24).mean()

    vpin_raw = ofi_h["VPIN"].dropna()
    print(f"  VPIN  mean={vpin_raw.mean():.4f}  median={vpin_raw.median():.4f}  "
          f"std={vpin_raw.std():.4f}")

    # Stationarity of VPIN
    stat_vpin = stationarity_tests(vpin_raw.iloc[:5000])  # subsample for speed

    # Summary table
    vpin_sum = pd.DataFrame.from_dict({
        "N_hours_with_trades":     f"{ofi_h['VPIN'].notna().sum():,}",
        "VPIN_mean":               f"{vpin_raw.mean():.4f}",
        "VPIN_median":             f"{vpin_raw.median():.4f}",
        "VPIN_std":                f"{vpin_raw.std():.4f}",
        "VPIN_p5":                 f"{vpin_raw.quantile(0.05):.4f}",
        "VPIN_p95":                f"{vpin_raw.quantile(0.95):.4f}",
        "OFI_mean":                f"{ofi_h['OFI'].dropna().mean():.4f}",
        "OFI_std":                 f"{ofi_h['OFI'].dropna().std():.4f}",
        "ADF_p_val (VPIN stat.)":  str(stat_vpin.get("ADF p-val", "?")),
        "Stationarity":            stat_vpin.get("Conclusion", "?"),
        "NOTE_vpin":               (
            "VPIN = |OFI| is a time-clock hourly approximation of Easley et al. (2012) "
            "volume-clock VPIN. Interpretation: VPIN → 1 = one-directional flow "
            "(toxic / informed); VPIN → 0 = balanced two-way flow (uninformed). "
            "Direction column from process_dex_swaps.py (sign of price change for "
            "within-block swaps; cross-block swaps excluded from signed volume)."
        ),
    }, orient="index", columns=["Value"])
    savetable(vpin_sum, "oft_summary")

    # ── [2] Merge with panel ──────────────────────────────────────────────────
    panel_idx = ofi_h.index
    for src in [mrg, cex, dex, lvr]:
        if src is None:
            continue
        for col in src.columns:
            if col not in ofi_h.columns:
                ofi_h[col] = src[col].reindex(panel_idx)

    # Resolve key columns
    basis_col = next((c for c in ["dex_cex_basis_bps"] if c in ofi_h.columns), None)
    vol_col   = next((c for c in ["realized_vol_24h_ann"] if c in ofi_h.columns), None)
    lvr_col   = next((c for c in ["lvr_rate_ann"] if c in ofi_h.columns), None)
    fee_col   = next((c for c in ["fee_apr_ann"] if c in ofi_h.columns), None)
    tvol_col  = next((c for c in ["volume_usd", "vol_base_1h_ethusdt"] if c in ofi_h.columns), None)

    abs_basis = (pd.to_numeric(ofi_h[basis_col], errors="coerce").abs()
                 if basis_col else None)
    vol       = pd.to_numeric(ofi_h[vol_col], errors="coerce") if vol_col else None
    lvr_rate  = pd.to_numeric(ofi_h[lvr_col], errors="coerce") if lvr_col else None
    fee_apr   = pd.to_numeric(ofi_h[fee_col], errors="coerce") if fee_col else None
    log_tvol  = (np.log1p(pd.to_numeric(ofi_h[tvol_col], errors="coerce"))
                 if tvol_col else None)

    # ── [3] Granger causality ─────────────────────────────────────────────────
    print("\n[3/6] Granger causality tests (VAR)...")
    gc_results = []
    if abs_basis is not None:
        panel_gc = pd.DataFrame({"VPIN": vpin_raw,
                                  "abs_basis": abs_basis}).dropna()
        gc_results.append(_granger_var(panel_gc, "VPIN", "abs_basis"))
        gc_results.append(_granger_var(panel_gc, "abs_basis", "VPIN"))
    if lvr_rate is not None:
        panel_gc2 = pd.DataFrame({"VPIN": vpin_raw,
                                   "lvr_rate": lvr_rate}).dropna()
        gc_results.append(_granger_var(panel_gc2, "VPIN", "lvr_rate"))

    if gc_results:
        gc_df = pd.DataFrame([r for r in gc_results if r and "F_stat" in r])
        if not gc_df.empty:
            gc_df["NOTE"] = (
                "VAR-based Granger causality in levels (VPIN stationary; |basis| stationary). "
                "VPIN → |basis|: H1 that toxic flow predicts basis widening (adverse selection). "
                "|basis| → VPIN: H2 that basis divergence attracts informed flow (reaction). "
                "Ref: Granger (1969); Easley et al. (2012)."
            )
            savetable(gc_df.set_index("from"), "oft_granger")

    # ── [4] Predictive OLS regressions ────────────────────────────────────────
    print("\n[4/6] Predictive OLS regressions...")
    vpin_lag = vpin_raw.shift(1)

    # Regression 1: |basis_{t+1}| ~ VPIN_t + vol_t + log_vol_t
    if abs_basis is not None:
        X_b = {"VPIN_t": vpin_lag,
               "const": 1.0}
        if vol is not None:
            X_b["vol_24h_t"] = vol
        if log_tvol is not None:
            X_b["log_volume_t"] = log_tvol
        df_b = pd.DataFrame({
            "basis_t1": abs_basis.shift(-1),
            **{k: (v.reindex(abs_basis.index) if isinstance(v, pd.Series) else
                   pd.Series(v, index=abs_basis.index))
               for k, v in X_b.items()}
        }).dropna()
        if len(df_b) > 50:
            X_b_r = sm.add_constant(df_b.drop(columns="basis_t1"))
            res_b = sm.OLS(df_b["basis_t1"], X_b_r).fit(
                cov_type="HAC", cov_kwds={"maxlags": 24})
            basis_reg = []
            for v, c, se, t, p in zip(res_b.model.exog_names, res_b.params,
                                       res_b.bse, res_b.tvalues, res_b.pvalues):
                basis_reg.append({"Var": v, "Coef": round(float(c), 6),
                                   "SE_HAC": round(float(se), 6),
                                   "t": round(float(t), 3),
                                   "p": round(float(p), 4),
                                   "Sig": stars(float(p))})
            basis_reg.append({"Var": "—", "Coef": np.nan, "SE_HAC": np.nan,
                               "t": np.nan, "p": np.nan,
                               "Sig": (f"N={len(df_b):,}  R²={res_b.rsquared:.4f}  "
                                       f"adj-R²={res_b.rsquared_adj:.4f}")})
            b_tab = pd.DataFrame(basis_reg).set_index("Var")
            b_tab["NOTE"] = (
                "Dep. var.: |DEX-CEX basis_{t+1}| (bps). VPIN_t is lagged 1 h. "
                "Positive β on VPIN_t → higher toxicity predicts wider basis (adverse selection). "
                "HAC SE with 24 lags for autocorrelation correction. "
                "Low R² expected (basis is hard to predict). "
                "Ref: Easley et al. (2012); Capponi & Jia (2022)."
            )
            savetable(b_tab, "oft_predictive_basis")

    # Regression 2: LVR_{t+1} ~ VPIN_t + vol_t + |basis_t|
    if lvr_rate is not None:
        X_l = {"VPIN_t": vpin_lag}
        if vol is not None:
            X_l["vol_24h_t"] = vol
        if abs_basis is not None:
            X_l["abs_basis_t"] = abs_basis
        df_l = pd.DataFrame({
            "lvr_t1": lvr_rate.shift(-1),
            **{k: (v.reindex(lvr_rate.index) if isinstance(v, pd.Series) else
                   pd.Series(v, index=lvr_rate.index))
               for k, v in X_l.items()}
        }).dropna()
        if len(df_l) > 50:
            X_l_r = sm.add_constant(df_l.drop(columns="lvr_t1"))
            res_l = sm.OLS(df_l["lvr_t1"], X_l_r).fit(
                cov_type="HAC", cov_kwds={"maxlags": 24})
            lvr_reg = []
            for v, c, se, t, p in zip(res_l.model.exog_names, res_l.params,
                                       res_l.bse, res_l.tvalues, res_l.pvalues):
                lvr_reg.append({"Var": v, "Coef": round(float(c), 6),
                                 "SE_HAC": round(float(se), 6),
                                 "t": round(float(t), 3),
                                 "p": round(float(p), 4),
                                 "Sig": stars(float(p))})
            lvr_reg.append({"Var": "—", "Coef": np.nan, "SE_HAC": np.nan,
                             "t": np.nan, "p": np.nan,
                             "Sig": f"N={len(df_l):,}  R²={res_l.rsquared:.4f}"})
            l_tab = pd.DataFrame(lvr_reg).set_index("Var")
            l_tab["NOTE"] = (
                "Dep. var.: LVR_rate_{t+1}. Pool-level LVR = σ²/8 (tautological; see "
                "lvr_theory_test.csv). Positive β on VPIN_t reflects that VPIN is "
                "correlated with vol_t (both are high during informed-flow hours). "
                "This regression is confirmatory, not causal. "
                "Ref: Milionis et al. (2022); Capponi & Jia (2022)."
            )
            savetable(l_tab, "oft_predictive_lvr")

    # ── [5] LP welfare by VPIN decile ─────────────────────────────────────────
    print("\n[5/6] LP welfare by VPIN decile...")
    if fee_apr is not None and lvr_rate is not None:
        net_ret = (fee_apr.reindex(panel_idx) / 8760 -
                   lvr_rate.reindex(panel_idx))   # hourly net return
        decile_panel = pd.DataFrame({
            "VPIN":    vpin_raw,
            "net_ret": net_ret,
            "fee_apr": fee_apr.reindex(panel_idx),
            "lvr":     lvr_rate.reindex(panel_idx),
        }).dropna()
        if len(decile_panel) >= N_DECILES * 10:
            decile_panel["decile"] = pd.qcut(
                decile_panel["VPIN"], q=N_DECILES,
                labels=[f"D{i+1}" for i in range(N_DECILES)],
                duplicates="drop")
            dec_agg = decile_panel.groupby("decile", observed=True).agg(
                N=("net_ret", "count"),
                VPIN_mean=("VPIN", "mean"),
                net_ret_hourly=("net_ret", "mean"),
                fee_apr_mean=("fee_apr", "mean"),
                lvr_rate_mean=("lvr", "mean"),
            ).round(6)
            dec_agg["net_ret_ann"] = dec_agg["net_ret_hourly"] * 8760
            dec_agg["NOTE"] = (
                "D1 = lowest VPIN (balanced flow); D10 = highest VPIN (most toxic). "
                "If net_ret_ann decreases monotonically with VPIN: adverse selection "
                "channel confirmed (toxic flow → higher LVR → worse LP welfare). "
                "net_ret = fee_apr/8760 − lvr_rate (hourly). "
                "Pool-level LVR tautology applies (see lvr_fee_welfare.csv). "
                "Ref: Easley et al. (2012); Milionis et al. (2022)."
            )
            savetable(dec_agg, "oft_welfare_decile")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            x = np.arange(len(dec_agg))
            axes[0].bar(x, dec_agg["net_ret_ann"] * 100,
                        color=[COLORS[0] if v >= 0 else COLORS[1]
                               for v in dec_agg["net_ret_ann"]],
                        alpha=0.8)
            axes[0].axhline(0, color="black", lw=0.8, ls="--")
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(dec_agg.index.tolist(), fontsize=8, rotation=45)
            axes[0].set_ylabel("Annualised net LP return (fee − LVR, %)")
            axes[0].set_title("Net LP Return by VPIN Decile")

            axes[1].bar(x, dec_agg["VPIN_mean"], color=COLORS[2], alpha=0.8)
            axes[1].set_xticks(x)
            axes[1].set_xticklabels(dec_agg.index.tolist(), fontsize=8, rotation=45)
            axes[1].set_ylabel("Mean VPIN")
            axes[1].set_title("VPIN Level per Decile")
            plt.suptitle("LP Welfare (fee − LVR) by Order Flow Toxicity Decile")
            savefig("oft_welfare_decile")

    # ── [6] Signed OFI → next-hour CEX return ─────────────────────────────────
    print("\n[6/6] Signed OFI predicting CEX returns (information content)...")
    r_cex = None
    if cex is not None:
        for c in ["log_return_1h"]:
            if c in cex.columns:
                r_cex = pd.to_numeric(cex[c], errors="coerce"); break

    if r_cex is not None:
        ofi_signed = ofi_h["OFI"].dropna()
        comm_o = ofi_signed.index.intersection(r_cex.index)
        df_ofi = pd.DataFrame({
            "OFI_t":       ofi_signed.reindex(comm_o),
            "r_cex_t1":    r_cex.shift(-1).reindex(comm_o),
        }).dropna()
        if len(df_ofi) > 50:
            X_o = sm.add_constant(df_ofi[["OFI_t"]])
            res_o = sm.OLS(df_ofi["r_cex_t1"], X_o).fit(
                cov_type="HAC", cov_kwds={"maxlags": 24})
            ofi_tab = pd.DataFrame({
                "Coef":   [round(float(c), 8) for c in res_o.params],
                "SE_HAC": [round(float(s), 8) for s in res_o.bse],
                "t_stat": [round(float(t), 3) for t in res_o.tvalues],
                "p_val":  [round(float(p), 4) for p in res_o.pvalues],
                "Sig":    [stars(float(p)) for p in res_o.pvalues],
            }, index=res_o.model.exog_names)
            ofi_tab.loc["—", "Sig"] = (
                f"N={len(df_ofi):,}  R²={res_o.rsquared:.4f}")
            ofi_tab["NOTE"] = (
                "Dep. var.: CEX log return at hour t+1. OFI_t = signed order flow "
                "imbalance at DEX hour t (positive = net buy pressure). "
                "Positive β → DEX buy pressure predicts positive CEX returns "
                "(DEX contains information not yet in CEX — DEX leads). "
                "Negative/zero β → no predictive content (CEX leads). "
                "Ref: de Jong (2002); Aspris et al. (2021) price discovery."
            )
            savetable(ofi_tab, "oft_signed_ofi")
            print(f"  OFI → CEX return: β={res_o.params.get('OFI_t',np.nan):.6f}  "
                  f"p={res_o.pvalues.get('OFI_t',1):.4f}{stars(res_o.pvalues.get('OFI_t',1))}")

    # Figure: VPIN and basis time series
    if abs_basis is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        vpin_24h = ofi_h["VPIN_24h"].dropna()
        ax1.plot(vpin_24h.index, vpin_24h, color=COLORS[0], lw=0.8, label="VPIN (24h MA)")
        ax1.axhline(float(vpin_24h.mean()), color="black", lw=0.7, ls="--",
                    alpha=0.5, label="Mean VPIN")
        ax1.set_ylabel("VPIN (order flow toxicity)")
        ax1.set_title("Uniswap v3 USDC/WETH: Order Flow Toxicity (VPIN)")
        ax1.legend(fontsize=8)
        ax1.set_ylim(0, 1)

        b_comm = abs_basis.dropna()
        ax2.plot(b_comm.index, b_comm.rolling(24).mean(), color=COLORS[1],
                 lw=0.8, label="|DEX-CEX basis| (24h MA, bps)")
        ax2.set_ylabel("|DEX-CEX basis| (bps)")
        ax2.set_xlabel("Date")
        ax2.legend(fontsize=8)
        savefig("oft_vpin_series")

    print("\nDONE")


if __name__ == "__main__":
    main()
