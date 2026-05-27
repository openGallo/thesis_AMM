"""
did_usdc_depeg.py — ITS + DiD: USDC Silicon Valley Bank Depeg (Mar 10-13, 2023)

Research Question:
    Did the USDC depeg event (Mar 10-13, 2023) — when Circle disclosed $3.3B in
    reserves at the failed Silicon Valley Bank — cause a measurable disruption to
    the WETH/USDC 0.05% Uniswap v3 pool's market quality, LP economics, and
    liquidity provision, distinct from a generic ETH price shock?

Motivation:
    This is uniquely relevant to our specific pool: USDC is the QUOTE CURRENCY
    of the WETH/USDC pair.  When USDC temporarily lost its $1 peg (trading at
    $0.87-0.93 on March 11, 2023), the pool's "ETH price" (in USDC) was
    artificially elevated by the USDC discount.  This creates a pool-specific
    natural experiment unavailable to any other Ethereum AMM pair not containing
    USDC.

    Key channels of causal impact:
    1. DEX-CEX basis distortion: CEX prices ETH in USDT (stably pegged),
       while DEX prices ETH in USDC (at a discount) → artificial basis spike.
    2. LP impermanent loss: positions calibrated for USDC = $1 experience
       unexpected range violations if the USDC-denominated price diverges.
    3. Panic withdrawals: LPs may exit regardless of the actual ETH/USD price
       change, responding to stablecoin counterparty risk.
    4. Fee APR spike: the enormous trading volume during the depeg (basis
       traders, panic sellers) generates anomalously high fees for remaining LPs.

    The depeg is exogenous to LP behaviour: it originated from SVB's failure,
    which was caused by the Federal Reserve's rate hikes — not by any action
    of Uniswap LPs or arbitrageurs.

Methodology:
    1. ITS on pool metrics:
           y_t = α + β₁t + β₂Post + β₃(t−Tc)Post + ε_t
       Event: Mar 10, 2023 (Friday evening, USDC price began declining).
       Window: ±60 days; exclusion: ±1 day (depeg lasted only 3 days).
       Outcomes: |basis_bps|, LVR rate, fee APR, TVL (dex_tvl_usd), Vol/TVL.
       Note: |basis_bps| is contaminated during the depeg itself
       (USDC discount ≠ genuine ETH price discrepancy). Post-period estimates
       (after Mar 14) are cleaner as USDC re-pegged.

    2. Depeg episode study — compare pool metrics DURING the depeg (Mar 10-13)
       vs matched pre-period (-3 to 0 days) and post-period (+3 to +6 days).
       One-sample Wilcoxon signed-rank test for each metric.

    3. LP withdrawal DiD — narrow vs wide LP positions:
           exit_rate_{g,d} = α + β_narrow·Narrow + β_post·Post + δ·Narrow×Post + ε
       Window: ±30 days.  Prediction: narrow positions exit faster (δ > 0).

    4. Placebo — apply ITS at Mar 10, 2022 (same calendar date, 1 year prior).

Key caveats:
    - |basis_bps| DURING the depeg is NOT a genuine DEX-CEX price discrepancy —
      it reflects USDC vs USDT pricing. Outcomes measured after re-peg (Mar 14+)
      are cleaner. All ITS estimates should be interpreted with this caveat.
    - The depeg lasted 3-4 days; with hourly data, N in the exclusion window is
      72-96 obs. ITS estimates rely on the LEVEL SHIFT (β₂) as measured from
      the post-re-peg period, which may confound depeg effects with the relief
      rally in USDC after Fed/FDIC backstop (announced Mar 12-13).
    - USDC re-pegged before US markets opened on March 13 (Monday). The LP
      response captures both panic and relief dynamics.
    - Our dataset may show artificially high CEX vol because the CEX data is
      ETH/USDT from Binance — USDT was unaffected by SVB. The USDC-USDT basis
      was not ~0 during the depeg, so the CEX/DEX basis metric is contaminated.

Outputs:
    output/tables/usdc_its.csv             ITS coefficients
    output/tables/usdc_depeg_episode.csv   Depeg episode vs matched periods
    output/tables/usdc_lp_did.csv          LP exit DiD (narrow vs wide)
    output/tables/usdc_placebo.csv         Placebo ITS (Mar 10, 2022)
    output/figures/usdc_its.pdf            ITS time series plots
    output/figures/usdc_depeg_episode.pdf  Episode comparison bar charts

References:
    Circle (2023). USDC and Silicon Valley Bank. Press release, Mar 11, 2023.
    FDIC (2023). Silicon Valley Bank receivership. Press release, Mar 10, 2023.
    Lee, D. & Lemieux, T. (2010). RD designs in economics. JEL.
    Bernal, J.L. et al. (2017). ITS regression. Int. J. Epidemiology.
    Adams, A. et al. (2021). Uniswap v3 Core. Uniswap Labs.
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

DEPEG_START  = pd.Timestamp("2023-03-10", tz="UTC")  # Friday evening
DEPEG_END    = pd.Timestamp("2023-03-14", tz="UTC")  # re-peg complete
PLACEBO      = DEPEG_START - pd.DateOffset(years=1)
WINDOW_DAYS  = 60
EXCL_DAYS    = 1    # ±1 day (depeg lasted 3d → 7d excl. would remove the whole event)
WINDOW_LP_D  = 30

OUTCOMES = {
    "abs_basis_bps":  "|DEX–CEX basis| bps [CONTAMINATED during depeg]",
    "lvr_rate_ann":   "LVR rate (ann.)",
    "dex_fee_apr_ann":"Fee APR (ann.)",
    "dex_tvl_usd":    "Pool TVL (USD)",
    "dex_vol_over_tvl":"Volume / TVL",
}

# ── Data ──────────────────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame | None:
    mh  = load("merged/merged_hourly.csv")
    lvr = load("DEX/dex_lvr_hourly.csv")
    if mh is None:
        return None
    panel = mh.copy()
    if lvr is not None:
        panel = panel.join(lvr[["lvr_rate_ann"]], how="left")
    panel["abs_basis_bps"] = panel["dex_cex_basis_bps"].abs()
    # Flag contaminated hours (during depeg)
    panel["depeg_period"] = (
        (panel.index >= DEPEG_START) & (panel.index < DEPEG_END)
    ).astype(int)
    return panel.sort_index()


# ── ITS ───────────────────────────────────────────────────────────────────────

def its(series: pd.Series, event: pd.Timestamp,
        excl: int = EXCL_DAYS, label: str = "") -> dict | None:
    start = event - pd.Timedelta(days=WINDOW_DAYS)
    end   = event + pd.Timedelta(days=WINDOW_DAYS)
    el    = event - pd.Timedelta(days=excl)
    eh    = event + pd.Timedelta(days=excl)
    sub   = series.loc[start:end].copy()
    sub   = sub[~((sub.index >= el) & (sub.index <= eh))].dropna()
    if len(sub) < 60:
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
    out: dict = {"N": len(sub), "label": label}
    for var in ["t", "Post", "slope_post"]:
        if var in tbl.index:
            r = tbl.loc[var]
            out[f"{var}_coef"] = r.get("Coef", np.nan)
            out[f"{var}_se"]   = r.get("SE (HAC)", np.nan)
            out[f"{var}_pval"] = r.get("p-val", np.nan)
            out[f"{var}_sig"]  = r.get("Sig", "")
    return out


# ── Depeg episode study ───────────────────────────────────────────────────────

def depeg_episode_study(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Compare pool metrics DURING depeg (Mar 10-13) vs matched windows.
    PRE:  [Mar 7, Mar 10)  — 3 days before
    DURING: [Mar 10, Mar 14) — 4 days of depeg
    POST: [Mar 14, Mar 17) — 3 days after re-peg

    Wilcoxon signed-rank test (non-parametric; appropriate for small N=72-96h).
    Note: results for abs_basis_bps DURING the depeg are distorted by USDC pricing.
    """
    pre_lo  = DEPEG_START - pd.Timedelta(days=3)
    post_hi = DEPEG_END   + pd.Timedelta(days=3)

    rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        pre_v    = panel.loc[pre_lo:DEPEG_START, col].dropna().values
        during_v = panel.loc[DEPEG_START:DEPEG_END, col].dropna().values
        post_v   = panel.loc[DEPEG_END:post_hi, col].dropna().values
        if len(pre_v) < 3 or len(during_v) < 3:
            continue
        # Wilcoxon rank-sum (Mann-Whitney) during vs pre
        if len(during_v) > 0 and len(pre_v) > 0:
            stat, p = sp_stats.ranksums(during_v, pre_v)
        else:
            stat, p = np.nan, np.nan
        rows.append({
            "outcome": col, "label": label[:40],
            "mean_pre":    round(float(np.mean(pre_v)),    4),
            "mean_during": round(float(np.mean(during_v)), 4),
            "mean_post":   round(float(np.mean(post_v)),   4) if len(post_v) > 0 else np.nan,
            "ratio_during_pre": round(float(np.mean(during_v) / max(abs(np.mean(pre_v)), 1e-9)), 3),
            "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else np.nan,
            "wilcoxon_p":    round(float(p),    4) if not np.isnan(p) else np.nan,
            "sig":           stars(float(p)) if not np.isnan(p) else "",
            "NOTE": (
                f"Wilcoxon rank-sum during vs pre. N_pre={len(pre_v)}, N_during={len(during_v)}. "
                "abs_basis_bps: contaminated by USDC discount (not genuine ETH price gap). "
                "lvr/fee_apr: cleaner measures of pool economics during depeg."
            ),
        })
    return pd.DataFrame(rows).set_index("outcome") if rows else pd.DataFrame()


# ── LP withdrawal DiD ─────────────────────────────────────────────────────────

def lp_did(lp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily panel DiD: narrow vs wide LP exit rate around USDC depeg.
    Design mirrors did_ftx_lp_withdrawal.py.
    Window: ±WINDOW_LP_D days around depeg start.
    """
    lp = lp_df[lp_df["first_mint_utc"] < DEPEG_START].copy()
    med = lp["range_width_pct"].median()
    lp["narrow"] = (lp["range_width_pct"] <= med).astype(int)
    lp["already_closed"] = lp["last_burn_utc"].notna() & (lp["last_burn_utc"] < DEPEG_START)
    lp = lp[~lp["already_closed"]].copy()
    if lp.empty:
        return pd.DataFrame()

    start = DEPEG_START - pd.Timedelta(days=WINDOW_LP_D)
    end   = DEPEG_START + pd.Timedelta(days=WINDOW_LP_D)
    days  = pd.date_range(start, end, freq="D", tz="UTC")
    opens  = lp["first_mint_utc"].values
    closes = lp["last_burn_utc"].values
    narrows = lp["narrow"].values

    rows = []
    for day in days:
        dn = day + pd.Timedelta(days=1)
        for grp in [0, 1]:
            m = narrows == grp
            op = m & (opens < dn) & (pd.isnull(closes) | (closes >= day))
            n_op = int(op.sum())
            if n_op < 3:
                continue
            n_ex = int((m & ~pd.isnull(closes) & (closes >= day) & (closes < dn) & op).sum())
            rows.append({"date": day, "narrow": grp,
                         "post": 1 if day >= DEPEG_START else 0,
                         "day_rel": int((day - DEPEG_START).days),
                         "n_open": n_op, "n_exit": n_ex,
                         "exit_rate": n_ex / n_op})
    if not rows:
        return pd.DataFrame()
    pl = pd.DataFrame(rows)
    pl["narrow_x_post"]    = pl["narrow"] * pl["post"]
    pl["narrow_x_day_rel"] = pl["narrow"] * pl["day_rel"]
    result = ols_hac(pl["exit_rate"], pl[["narrow", "post", "narrow_x_post",
                                          "day_rel", "narrow_x_day_rel"]],
                     max_lags=7, label="DiD")
    if not result.empty:
        result["NOTE"] = (
            "DiD: δ=coef(narrow_x_post). Window ±30d. HAC SE, 7-lag. "
            "USDC depeg prediction: narrow positions exit faster (δ>0) because "
            "tight ranges are more sensitive to USDC-denominated price oscillations."
        )
    return result


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_its(panel: pd.DataFrame) -> None:
    cols = [c for c in ["lvr_rate_ann", "dex_fee_apr_ann", "dex_tvl_usd", "dex_vol_over_tvl"]
            if c in panel.columns]
    nrow = (len(cols) + 1) // 2
    fig, axes = plt.subplots(nrow, 2, figsize=(12, 3.5 * nrow))
    flat = axes.flatten()
    for i, col in enumerate(cols):
        ax = flat[i]
        ser = panel[col].loc[DEPEG_START - pd.Timedelta(days=WINDOW_DAYS):
                              DEPEG_START + pd.Timedelta(days=WINDOW_DAYS)]
        daily = ser.resample("D").mean()
        ax.plot(daily.index, daily.values, color="#2563eb", lw=1.2)
        ax.axvspan(DEPEG_START, DEPEG_END, color="#fca5a5", alpha=0.4, label="USDC depeg")
        ax.axvline(DEPEG_START, color="black", lw=2, ls="--")
        ax.set_title(OUTCOMES.get(col, col)[:50], fontsize=9)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cols), len(flat)):
        flat[j].set_visible(False)
    plt.suptitle("USDC SVB Depeg (Mar 10-13 2023) — Pool metrics", fontsize=10, y=1.01)
    savefig("usdc_its")


def plot_depeg_episode(df: pd.DataFrame) -> None:
    if df.empty:
        return
    cols = [c for c in df.index if "abs_basis" not in c]  # skip contaminated series
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    flat = [axes] if len(cols) == 1 else axes
    for ax, col in zip(flat, cols):
        row = df.loc[col]
        vals = [row.get("mean_pre", 0), row.get("mean_during", 0), row.get("mean_post", 0)]
        bars = ax.bar(["Pre", "During", "Post"], vals,
                      color=["#2563eb", "#dc2626", "#16a34a"], alpha=0.75)
        ax.set_title(row.get("label", col)[:30], fontsize=9)
        sig = row.get("sig", "")
        ax.set_xlabel(f"Wilcoxon p={row.get('wilcoxon_p', np.nan):.3f}{sig}", fontsize=8)
    plt.suptitle("USDC Depeg Episode: Pool metric comparison (Pre / During / Post)", fontsize=10)
    savefig("usdc_depeg_episode")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    panel = load_panel()
    if panel is None:
        print("  [SKIP] Required data missing."); return

    lp_raw = load("DEX/dex_lp_positions.csv")
    if lp_raw is not None:
        lp_raw["first_mint_utc"] = pd.to_datetime(lp_raw["first_mint_utc"], utc=True, errors="coerce")
        lp_raw["last_burn_utc"]  = pd.to_datetime(lp_raw["last_burn_utc"],  utc=True, errors="coerce")

    print(f"  USDC depeg: {DEPEG_START.date()} → {DEPEG_END.date()}")
    print(f"  Placebo: {PLACEBO.date()}")

    # 1. ITS
    its_rows = []
    for col, label in OUTCOMES.items():
        if col not in panel.columns:
            continue
        for spec, event in [("main", DEPEG_START), ("placebo", PLACEBO)]:
            res = its(panel[col], event, label=label)
            if res:
                res["outcome"] = col; res["spec"] = spec
                its_rows.append(res)
    if its_rows:
        # BH on non-contaminated main outcomes (add flags before building DataFrame)
        main_r = [r for r in its_rows if r.get("spec") == "main"
                  and "abs_basis" not in r.get("outcome", "")]
        pv = [r.get("Post_pval", np.nan) for r in main_r]
        pv_c = [p for p in pv if not np.isnan(p)]
        if pv_c:
            bh = bh_correction(pv_c, alpha=0.10)
            for r, flag in zip(
                [r for r in main_r if not np.isnan(r.get("Post_pval", np.nan))], bh
            ):
                r["Post_BH10_reject"] = flag
            print(f"  BH(10%) rejects (non-contaminated Post): {sum(bh)}/{len(bh)}")
        # Save main ITS; save placebo separately
        df_its = pd.DataFrame(its_rows).set_index(["outcome", "spec"])
        savetable(df_its, "usdc_its")
        plac_r = [r for r in its_rows if r.get("spec") == "placebo"]
        if plac_r:
            savetable(pd.DataFrame(plac_r).set_index(["outcome", "spec"]), "usdc_placebo")

    # 2. Depeg episode
    df_ep = depeg_episode_study(panel)
    if not df_ep.empty:
        savetable(df_ep, "usdc_depeg_episode")
        plot_depeg_episode(df_ep)

    # 3. LP DiD
    if lp_raw is not None:
        df_did = lp_did(lp_raw)
        if not df_did.empty:
            savetable(df_did, "usdc_lp_did")

    # 4. Plots
    plot_its(panel)

    print("  [DONE] did_usdc_depeg.py")


if __name__ == "__main__":
    main()
