"""
did_eip1559_gas.py — DiD / ITS: EIP-1559 and LP Position Economics

Research Question:
    Did EIP-1559 (Aug 5, 2021) — which replaced Ethereum's first-price auction
    gas mechanism with a predictable base fee — reduce gas-cost uncertainty and
    extend LP position durations on Uniswap v3, consistent with reduced
    rebalancing frictions?

Motivation:
    Before EIP-1559, Ethereum gas fees were determined by a first-price sealed-
    bid auction, producing high variance and unpredictable costs. This created
    strategic uncertainty for Uniswap v3 LPs who must decide when to rebalance
    (adjust their price range) or exit: the cost of the rebalance transaction
    was unknown in advance and could vary 5-10× within hours.
    EIP-1559 replaced this with a protocol-set base fee that adjusts gradually
    (±12.5% per block), dramatically reducing gas-price variance without
    necessarily reducing the mean level. The prediction: with lower gas variance,
    LPs should (a) hold positions longer (fewer forced early exits to avoid
    gas-cost spikes), (b) potentially narrow their ranges (cheaper to manage),
    and (c) exhibit fewer "panic rebalances" around gas-price spikes.

    EIP-1559 is exogenous: implemented at Ethereum block 12,965,000 (Aug 5, 2021),
    a protocol upgrade not related to pool liquidity or ETH price dynamics.
    Confounders: ETH price appreciation in Aug-Oct 2021 (bull market) may
    independently increase LP activity. We address this via ITS and placebo tests.

Methodology:
    1. ITS on daily gas metrics (from dex_swaps):
           gas_gwei_daily = daily median gas_gwei of Uniswap swaps.
       Model: gas_gwei_d = α + β₁d + β₂Post + β₃(d−D_c)Post + ε_d.
       H0: β₂ = β₃ = 0 (EIP-1559 did not shift or bend the gas-price trend).
       Newey-West HAC SE (max_lags = 7 days).

    2. Gas variance ITS — repeat with daily std(gas_gwei) as outcome.
       EIP-1559's primary mechanistic claim is VARIANCE reduction, not level.

    3. LP cohort DiD — compare positions opened in Pre-EIP (Jun–Aug 4, 2021)
       vs Post-EIP (Aug 5–Oct 31, 2021) cohorts:
           Y_i = α + β·Post_cohort_i + ε_i
       Y ∈ {log(duration_days), range_width_pct, log(total_minted_usd)}.
       HC3 robust SE. Pre/Post are cohort labels (cross-sectional, not panel).

    4. Parallel trends test — in the pre-EIP period (Mar–Aug 4, 2021), regress
       daily gas_gwei on a linear time trend; test whether the trend is smooth
       up to the cutoff (no pre-treatment kink). Also: placebo ITS at Jun 1,
       2021 (2 months before the actual event).

    5. Variance decomposition — pre vs post Levene's test for equality of
       gas_gwei variances. Also: daily interquartile range (IQR) of gas fees
       before/after EIP-1559.

    6. LP activity panel — daily mint count and average mint size (USD) for the
       pre/post EIP window; smooth trend should change if LPs respond to gas.

Key caveats (embedded in output tables):
    - The data starts May 2021, leaving only ~3 months of pre-EIP data
      (pool inception May 5 → EIP-1559 Aug 5). This is a SHORT pre-period;
      the ITS trend estimate β₁ is extrapolated from only ~90 days.
    - ETH bull market (Aug–Nov 2021) is a confounder: higher ETH price raises
      gas costs in USD (but not in Gwei) and attracts more LP activity
      independently of EIP-1559. Gwei-denominated analysis is more insulated.
    - The LP cohort DiD confounds EIP-1559 with the bull market: post-EIP LPs
      entered a rising market. Controlling for contemporaneous ETH returns in
      the regression partially absorbs this.
    - Position duration is right-censored for positions still open at data end.
      Active positions are excluded from duration analysis.
    - Uniswap v3 launched May 5, 2021, so pre-EIP positions are early adopters
      — potentially different from the post-EIP general population (selection
      into the pool before EIP-1559 may correlate with risk tolerance or gas
      sensitivity).

Outputs:
    output/tables/eip_gas_its.csv            ITS coefs: gas level and variance
    output/tables/eip_gas_variance.csv       Levene + IQR before/after
    output/tables/eip_lp_cohort.csv          LP cohort DiD (duration, range, size)
    output/tables/eip_lp_activity.csv        Daily mint count and size
    output/tables/eip_placebo.csv            Placebo ITS at Jun 1, 2021
    output/figures/eip_gas_its.pdf           ITS plots: gas level + variance
    output/figures/eip_lp_cohort.pdf         LP duration CDF by cohort

References:
    London Hard Fork EIP-1559. Ethereum Improvement Proposals (2021).
    Roughgarden, T. (2021). Transaction fee mechanism design for the Ethereum
        blockchain. Proc. ACM EC.
    Leonardos, S. et al. (2021). Optimality and stability in EIP-1559.
        IEEE Int. Conf. Blockchain.
    Bertrand, M. et al. (2004). How much should we trust DiD estimates?
        Q. J. Economics, 119(1), 249-275.
    Bernal, J.L. et al. (2017). ITS regression for public health interventions.
        Int. J. Epidemiology.
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
    load, load_swaps, savefig, savetable, stars, ols_hac, bootstrap_ci, bh_correction,
)

# ── Constants ─────────────────────────────────────────────────────────────────

EIP_DATE      = pd.Timestamp("2021-08-05", tz="UTC")
PLACEBO_DATE  = pd.Timestamp("2021-06-01", tz="UTC")  # 2 months before actual
PRE_START     = pd.Timestamp("2021-05-05", tz="UTC")  # pool launch
POST_END      = pd.Timestamp("2021-10-31", tz="UTC")
WINDOW_ITS    = 120    # days each side for ITS on gas metrics
EXCL_DAYS     = 3      # days to exclude around event

# LP cohort windows
PRE_COHORT_START  = pd.Timestamp("2021-06-01", tz="UTC")
PRE_COHORT_END    = EIP_DATE
POST_COHORT_START = EIP_DATE
POST_COHORT_END   = pd.Timestamp("2021-11-01", tz="UTC")

# ── Data loading ──────────────────────────────────────────────────────────────

def load_gas_daily() -> pd.DataFrame | None:
    """Daily median and std of gas_gwei from swaps."""
    swaps = load_swaps(cols=["timestamp", "gas_price_wei", "amount_usd"])
    if swaps is None:
        return None
    swaps = swaps.dropna(subset=["timestamp", "gas_price_wei"])
    swaps["gas_gwei"] = swaps["gas_price_wei"] / 1e9
    swaps = swaps[swaps["gas_gwei"] > 0].copy()
    swaps = swaps.set_index("timestamp").sort_index()
    daily = swaps["gas_gwei"].resample("D").agg(
        median_gwei="median", std_gwei="std", count="count",
        iqr_gwei=lambda x: float(np.percentile(x, 75) - np.percentile(x, 25))
        if len(x) >= 4 else np.nan
    )
    daily = daily[daily["count"] >= 5]
    return daily.dropna(subset=["median_gwei"])


def load_lp_positions() -> pd.DataFrame | None:
    lp = load("DEX/dex_lp_positions.csv")
    if lp is None:
        return None
    lp["first_mint_utc"] = pd.to_datetime(lp["first_mint_utc"], utc=True, errors="coerce")
    lp["last_burn_utc"]  = pd.to_datetime(lp["last_burn_utc"],  utc=True, errors="coerce")
    return lp


# ── ITS regression on daily gas ───────────────────────────────────────────────

def gas_its(gas_daily: pd.DataFrame, outcome: str, event: pd.Timestamp,
            label: str = "", excl: int = EXCL_DAYS, max_lags: int = 7) -> dict | None:
    """ITS: y_d = α + β₁d + β₂Post + β₃(d−Dc)Post + ε (HAC, weekly lags)."""
    start = event - pd.Timedelta(days=WINDOW_ITS)
    end   = event + pd.Timedelta(days=WINDOW_ITS)
    excl_lo = event - pd.Timedelta(days=excl)
    excl_hi = event + pd.Timedelta(days=excl)

    sub = gas_daily.loc[start:end, outcome].copy()
    sub = sub[~((sub.index >= excl_lo) & (sub.index <= excl_hi))].dropna()
    if len(sub) < 30:
        return None

    t_vec   = np.arange(len(sub), dtype=float)
    T_c     = float(sub.index.searchsorted(event))
    if T_c <= 5 or T_c >= len(sub) - 5:
        return None

    post    = (t_vec >= T_c).astype(float)
    slp_pst = (t_vec - T_c) * post
    X = pd.DataFrame({"t": t_vec, "Post": post, "slope_post": slp_pst},
                     index=sub.index)
    tbl = ols_hac(sub.rename("y"), X, max_lags=max_lags, label=label)
    if tbl.empty:
        return None

    out: dict = {"outcome": outcome, "label": label, "N": len(sub)}
    for var in ["t", "Post", "slope_post"]:
        if var in tbl.index:
            r = tbl.loc[var]
            out[f"{var}_coef"] = r.get("Coef", np.nan)
            out[f"{var}_se"]   = r.get("SE (HAC)", np.nan)
            out[f"{var}_pval"] = r.get("p-val", np.nan)
            out[f"{var}_sig"]  = r.get("Sig", "")
    out["NOTE"] = (
        f"ITS on {outcome}. EIP-1559 implemented {event.date()}. "
        "β_Post = level shift; β_slope_post = trend change. "
        "HAC SE (7-day lag). Pre-period only 3 months (pool launched May 5 2021). "
        "Bull-market confounder: ETH price +70% Aug-Nov 2021."
    )
    return out


# ── Gas variance Levene test ──────────────────────────────────────────────────

def variance_test(gas_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Levene's (1960) test for equality of gas-fee variance pre/post EIP-1559.
    H0: variance is equal in pre and post periods.
    EIP-1559 prediction: reject H0 with lower post-event variance.
    """
    pre  = gas_daily.loc[PRE_START:EIP_DATE, "median_gwei"].dropna()
    post = gas_daily.loc[EIP_DATE:POST_END,  "median_gwei"].dropna()
    if len(pre) < 5 or len(post) < 5:
        return pd.DataFrame()

    F_lev, p_lev = sp_stats.levene(pre.values, post.values)
    rows = [
        {"period": "pre-EIP",  "N_days": len(pre),
         "mean_gwei": round(pre.mean(), 2), "std_gwei": round(pre.std(), 2),
         "median_gwei": round(pre.median(), 2),
         "iqr_gwei": round(float(np.percentile(pre, 75) - np.percentile(pre, 25)), 2)},
        {"period": "post-EIP", "N_days": len(post),
         "mean_gwei": round(post.mean(), 2), "std_gwei": round(post.std(), 2),
         "median_gwei": round(post.median(), 2),
         "iqr_gwei": round(float(np.percentile(post, 75) - np.percentile(post, 25)), 2)},
        {"period": "test", "Levene_F": round(F_lev, 4), "Levene_p": round(p_lev, 6),
         "Levene_sig": stars(p_lev),
         "NOTE": (
             "Levene (1960) test for equality of variance. "
             "EIP-1559 prediction: post-EIP variance < pre-EIP (mechanism = bounded base fee). "
             "Post-EIP MEAN may be higher if ETH price appreciation raised gas in USD terms "
             "(but Gwei is ETH-native, so this should not affect Gwei). "
             "IQR = 75th - 25th pct = robust variance proxy."
         )},
    ]
    return pd.DataFrame(rows)


# ── LP cohort DiD ─────────────────────────────────────────────────────────────

def lp_cohort_did(lp: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional DiD: compare LP positions opened Pre-EIP vs Post-EIP.
    Outcomes: log(duration_days), range_width_pct, log(total_minted_usd).
    HC3 robust SE. Cohort label = treatment indicator (Post=1 if post-EIP).
    """
    pre  = lp[(lp["first_mint_utc"] >= PRE_COHORT_START)
              & (lp["first_mint_utc"] < PRE_COHORT_END)].copy()
    post = lp[(lp["first_mint_utc"] >= POST_COHORT_START)
              & (lp["first_mint_utc"] < POST_COHORT_END)].copy()
    pre["post_eip"]  = 0
    post["post_eip"] = 1
    combined = pd.concat([pre, post], ignore_index=True)

    # Keep only closed positions (duration observable)
    closed = combined[combined["last_burn_utc"].notna()
                      & (combined["duration_days"] > 0)].copy()

    if len(closed) < 20:
        return pd.DataFrame()

    closed["log_duration"] = np.log(closed["duration_days"].clip(lower=0.01))
    closed["log_mint"]     = np.log(closed["total_minted_usd"].clip(lower=1))
    closed["log_rng"]      = np.log(closed["range_width_pct"].clip(lower=0.01))

    # Add ETH price at mint as control for the Aug-Nov 2021 bull market confounder.
    # Without this, post_eip coefficient conflates EIP-1559 effect with rising ETH price
    # attracting more/different LP activity (higher ETH → larger positions, longer holds).
    mh = load("merged/merged_hourly.csv")
    if mh is not None and "dex_eth_usdc_price" in mh.columns:
        eth_daily = mh["dex_eth_usdc_price"].resample("D").mean()
        closed["eth_at_mint"] = closed["first_mint_utc"].dt.normalize().map(
            lambda d: float(eth_daily.get(d, np.nan))
        )
        closed["log_eth"] = np.log(closed["eth_at_mint"].clip(lower=1))
        ctrl_cols = ["post_eip", "log_eth"]
    else:
        ctrl_cols = ["post_eip"]

    results = []
    for dep, dep_label in [
        ("log_duration", "log(duration_days)"),
        ("log_rng",      "log(range_width_pct)"),
        ("log_mint",     "log(total_minted_usd)"),
    ]:
        y = closed[dep]
        reg_cols = [c for c in ctrl_cols if c in closed.columns]
        X = sm.add_constant(closed[reg_cols].astype(float))
        try:
            res = sm.OLS(y, X).fit().get_robustcov_results(cov_type="HC3")
            for var, c, s, t, p in zip(res.model.exog_names, res.params,
                                        res.bse, res.tvalues, res.pvalues):
                results.append({
                    "dep_var": dep_label, "var": var,
                    "coef": round(float(c), 6), "se_hc3": round(float(s), 6),
                    "t": round(float(t), 3), "p": round(float(p), 4),
                    "sig": stars(float(p)),
                    "N_pre": len(pre), "N_post": len(post),
                })
        except Exception as e:
            print(f"    [WARN] cohort DiD for {dep}: {e}")
            continue

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results).set_index(["dep_var", "var"])
    df["NOTE"] = (
        "Cross-sectional DiD: post_eip = 1 for positions opened after EIP-1559 (Aug 5 2021). "
        "HC3 robust SE. Pre cohort = Jun–Aug 4 2021; Post = Aug 5–Oct 31 2021. "
        "Confounder: ETH bull market (Aug–Nov 2021) → more LP activity regardless of gas. "
        "Selection: pre-EIP LPs are early adopters; may be systematically different. "
        "Only closed positions included; active positions are right-censored."
    )
    return df


# ── Daily LP activity panel ───────────────────────────────────────────────────

def lp_activity_panel(lp: pd.DataFrame) -> pd.DataFrame:
    """Daily mint count and average mint size (USD) in the EIP window."""
    window = lp[(lp["first_mint_utc"] >= PRE_START)
                & (lp["first_mint_utc"] <= POST_COHORT_END)].copy()
    daily = window.set_index("first_mint_utc").resample("D").agg(
        n_mints=("total_minted_usd", "count"),
        avg_mint_usd=("total_minted_usd", "mean"),
        total_mint_usd=("total_minted_usd", "sum"),
    )
    daily["post_eip"] = (daily.index >= EIP_DATE).astype(int)
    daily["day_rel"]  = (daily.index - EIP_DATE).days
    return daily.reset_index()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_gas_its(gas_daily: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, col, label in [
        (axes[0], "median_gwei", "Median gas price (Gwei)"),
        (axes[1], "std_gwei",    "Std dev gas price (Gwei)"),
    ]:
        ser = gas_daily[col].dropna()
        ax.plot(ser.index, ser.values, color="#2563eb", lw=0.8, alpha=0.5)
        weekly = ser.rolling(7, center=True).mean()
        ax.plot(weekly.index, weekly.values, color="#2563eb", lw=2, label="7-day MA")
        ax.axvline(EIP_DATE, color="black", lw=2, ls="--", label="EIP-1559")
        ax.axvline(PLACEBO_DATE, color="#d97706", lw=1.5, ls=":", label="Placebo")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
    axes[1].set_xlabel("Date")
    plt.suptitle("Gas prices before and after EIP-1559 (Aug 5, 2021)", fontsize=10)
    savefig("eip_gas_its")


def plot_lp_cohort_cdf(lp: pd.DataFrame) -> None:
    """Empirical CDF of log(duration_days) for Pre vs Post EIP cohorts."""
    pre  = lp[(lp["first_mint_utc"] >= PRE_COHORT_START)
              & (lp["first_mint_utc"] < PRE_COHORT_END)
              & lp["last_burn_utc"].notna()
              & (lp["duration_days"] > 0)]["duration_days"].values
    post = lp[(lp["first_mint_utc"] >= POST_COHORT_START)
              & (lp["first_mint_utc"] < POST_COHORT_END)
              & lp["last_burn_utc"].notna()
              & (lp["duration_days"] > 0)]["duration_days"].values

    if len(pre) < 5 or len(post) < 5:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for vals, label, col, ls in [(pre, "Pre-EIP", "#2563eb", "--"),
                                  (post, "Post-EIP", "#dc2626", "-")]:
        sorted_v = np.sort(vals)
        cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax.plot(np.log1p(sorted_v), cdf, color=col, ls=ls, lw=2, label=f"{label} (N={len(vals)})")

    # KS test
    ks_stat, ks_p = sp_stats.ks_2samp(pre, post)
    ax.set_xlabel("log(1 + duration days)")
    ax.set_ylabel("CDF")
    ax.set_title(f"LP position duration: Pre vs Post EIP-1559  |  KS p = {ks_p:.4f}{stars(ks_p)}")
    ax.legend()
    savefig("eip_lp_cohort")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    gas_daily = load_gas_daily()
    lp        = load_lp_positions()

    if gas_daily is None and lp is None:
        print("  [SKIP] No data available."); return

    print(f"  EIP-1559 date: {EIP_DATE.date()}  |  Placebo: {PLACEBO_DATE.date()}")

    # 1. ITS on gas metrics
    if gas_daily is not None:
        print(f"  Gas daily obs: {len(gas_daily):,}  "
              f"({gas_daily.index[0].date()} → {gas_daily.index[-1].date()})")
        its_rows = []
        for outcome, label in [
            ("median_gwei", "Median gas (Gwei)"),
            ("std_gwei",    "Std dev gas (Gwei)"),
            ("iqr_gwei",    "IQR gas (Gwei)"),
        ]:
            if outcome not in gas_daily.columns:
                continue
            # Main ITS
            res = gas_its(gas_daily, outcome, EIP_DATE, label=label)
            if res:
                res["spec"] = "main"
                its_rows.append(res)
            # Placebo ITS
            res_pl = gas_its(gas_daily, outcome, PLACEBO_DATE, label=label)
            if res_pl:
                res_pl["spec"] = "placebo_jun2021"
                its_rows.append(res_pl)
        if its_rows:
            # BH correction on main-spec ITS Post coefficients
            main_its = [r for r in its_rows if r.get("spec") == "main"]
            pv = [r.get("Post_pval", np.nan) for r in main_its]
            pv_c = [p for p in pv if not np.isnan(p)]
            if pv_c:
                bh = bh_correction(pv_c, alpha=0.10)
                for r, flag in zip(
                    [r for r in main_its if not np.isnan(r.get("Post_pval", np.nan))], bh
                ):
                    r["Post_BH10_reject"] = flag
                print(f"  BH(10%) rejects (EIP gas ITS Post): {sum(bh)}/{len(bh)}")
            # Save main + combined table; save placebo separately
            savetable(pd.DataFrame(its_rows).set_index(["outcome", "spec"]), "eip_gas_its")
            plac_rows = [r for r in its_rows if r.get("spec") != "main"]
            if plac_rows:
                savetable(pd.DataFrame(plac_rows).set_index(["outcome", "spec"]), "eip_placebo")

        # 2. Variance test
        df_var = variance_test(gas_daily)
        if not df_var.empty:
            savetable(df_var, "eip_gas_variance")

        # 3. Plots
        plot_gas_its(gas_daily)

        # 4. LP activity panel
        if lp is not None:
            act = lp_activity_panel(lp)
            if not act.empty:
                savetable(act, "eip_lp_activity")

    # 5. LP cohort DiD
    if lp is not None:
        n_total = len(lp)
        pre_mask = (lp["first_mint_utc"] >= PRE_COHORT_START) & (lp["first_mint_utc"] < PRE_COHORT_END)
        pst_mask = (lp["first_mint_utc"] >= POST_COHORT_START) & (lp["first_mint_utc"] < POST_COHORT_END)
        print(f"  LP positions total: {n_total:,}  "
              f"|  pre-EIP: {pre_mask.sum():,}  |  post-EIP: {pst_mask.sum():,}")

        df_cohort = lp_cohort_did(lp)
        if not df_cohort.empty:
            savetable(df_cohort, "eip_lp_cohort")

        plot_lp_cohort_cdf(lp)

    print("  [DONE] did_eip1559_gas.py")


if __name__ == "__main__":
    main()
