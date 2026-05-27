"""
did_ftx_lp_withdrawal.py — DiD: FTX Collapse and LP Withdrawal Behaviour

Research Question:
    Did the FTX collapse (Nov 8, 2022) trigger a differential liquidity
    withdrawal response by narrow-range versus wide-range LPs in the
    WETH/USDC 0.05% Uniswap v3 pool, and did concentrated (narrow) positions
    exit disproportionately faster?

Motivation:
    Narrow-range (concentrated) LPs in Uniswap v3 face amplified impermanent
    loss during large price dislocations because their entire liquidity is
    active within a tighter price band (Adams et al. 2021; Lehar & Parlour 2021).
    The FTX collapse (Alameda Research/FTX insolvency announced Nov 8, 2022) was
    a sudden, unanticipated shock that caused ETH to fall ~25% in five days and
    spike realized volatility.  It is therefore a near-ideal natural experiment:
    (a) exogenous to LP positioning choices, (b) large enough to force out-of-
    range concentrated positions, (c) sharply dated (pre/post is unambiguous).

    The DiD design exploits the within-pool heterogeneity in range width to
    identify a DIFFERENTIAL effect across LP types, controlling for pool-level
    time trends that affect all LPs equally.

Methodology:
    1. Daily panel DiD — for each day d in [−WINDOW_D, +WINDOW_D] relative to
       the FTX date, compute:
           exit_rate_{g,d} = (# positions of group g closed on day d) /
                             (# positions of group g open at start of day d)
       Groups: narrow = range_width_pct ≤ median; wide = range_width_pct > median.
       Estimate:
           exit_rate_{g,d} = α + β_narrow·Narrow_g + β_post·Post_d
                             + δ·(Narrow_g × Post_d) + γ_g·d + ε_{g,d}
       δ is the DiD estimator. γ_g·d allows group-specific linear time trends.
       OLS + HAC SEs (Newey-West, 7-lag = 1 week).

    2. Parallel trends test — in the pre-period [−WINDOW_D, 0], test H0 that
       narrow and wide groups share the same linear exit-rate trend (β_narrow·d
       interaction).  Rejection would invalidate parallel trends.

    3. LP flow event study — daily net flow = (daily mints USD − daily burns USD)
       for each group; normalised by pre-event mean. Plots ±WINDOW_D days.

    4. Survival analysis (non-parametric) — for positions active at FTX date,
       compare time-to-exit (days from FTX to burn) between narrow and wide
       using Kaplan-Meier estimates and a log-rank test.  Censoring: positions
       still active at end of data = right-censored.

    5. Cohort regression — cross-sectional OLS of log(days_to_exit) on
       range_width_pct, log(mint_usd), position_type dummies, and Post_ftx
       interaction.  HC3 robust SEs.

Key caveats (embedded in output tables):
    - Parallel trends is an assumption, not testable in the post-period.
      The pre-period test partially validates it; concurrent macro shocks
      (e.g., Binance FUD) cannot be fully distinguished from the FTX effect.
    - Wide positions may have had their price range out-of-active-range too,
      but face lower impermanent loss per percentage move: the group comparison
      is a clean margin even if not perfectly binary.
    - LP positions dataset records only final mint/burn events.  Intra-day
      rebalancing by adding to the same position appears as same-position
      re-mint and is not split out; this biases the panel slightly.
    - Survivorship: positions opened AFTER the FTX shock are excluded from the
      panel by construction (they couldn't survive before FTX if they didn't exist).
    - The pool is USDC/WETH; USDC was widely considered "safe" during FTX
      (no USDC exposure to FTX), so the shock is pure volatility/panic, not
      a stablecoin depeg. This simplifies interpretation.

Outputs:
    output/tables/ftx_did_panel.csv          DiD daily panel regression
    output/tables/ftx_parallel_trends.csv    Pre-period parallel trends test
    output/tables/ftx_lp_flow.csv            Daily net LP flows ±WINDOW_D days
    output/tables/ftx_survival.csv           KM survival + log-rank test
    output/tables/ftx_cohort_regression.csv  Cross-sectional duration OLS
    output/figures/ftx_did_exit_rate.pdf     Exit-rate DiD plot
    output/figures/ftx_lp_flow.pdf           Net flow event-study plot
    output/figures/ftx_survival.pdf          Kaplan-Meier survival curves

References:
    Adams, A. et al. (2021). Uniswap v3 Core. Uniswap Labs.
    Lehar, A. & Parlour, C. (2021). Decentralized exchanges. SSRN 3905316.
    Bertrand, M. et al. (2004). How much should we trust DiD estimates?
        Q. J. Economics, 119(1), 249-275.
    Callaway, B. & Sant'Anna, P. (2021). Difference-in-differences with
        multiple time periods. J. Econometrics, 225(2), 200-230.
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
    load, savefig, savetable, stars, ols_hac, bootstrap_ci,
)

# ── Constants ─────────────────────────────────────────────────────────────────

FTX_DATE   = pd.Timestamp("2022-11-08", tz="UTC")
WINDOW_D   = 90       # days each side of FTX
MIN_POS    = 5        # min positions per group-day for inclusion
HC_TYPE    = "HC3"    # heteroskedasticity-consistent SE type

# ── Data loading ──────────────────────────────────────────────────────────────

def _parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def load_positions() -> pd.DataFrame | None:
    lp = load("DEX/dex_lp_positions.csv")
    if lp is None:
        return None
    lp = _parse_dates(lp, ["first_mint_utc", "last_mint_utc", "last_burn_utc",
                            "last_event_utc"])
    # Keep only positions opened before FTX
    lp = lp[lp["first_mint_utc"] < FTX_DATE].copy()
    # Median-split into narrow / wide
    med = lp["range_width_pct"].median()
    lp["narrow"] = (lp["range_width_pct"] <= med).astype(int)
    lp["group"]  = lp["narrow"].map({1: "narrow", 0: "wide"})
    # Classify closed vs active at FTX
    lp["closed_before_ftx"] = (
        lp["last_burn_utc"].notna() & (lp["last_burn_utc"] < FTX_DATE)
    )
    lp["active_at_ftx"] = ~lp["closed_before_ftx"]
    return lp


# ── Daily panel construction ──────────────────────────────────────────────────

def build_daily_panel(lp: pd.DataFrame) -> pd.DataFrame | None:
    """
    For each calendar day in [FTX - WINDOW_D, FTX + WINDOW_D], compute
    exit_rate = exits_on_day / open_at_start_of_day  for narrow and wide groups.
    """
    active = lp[lp["active_at_ftx"]].copy()
    if active.empty:
        return None

    start = FTX_DATE - pd.Timedelta(days=WINDOW_D)
    end   = FTX_DATE + pd.Timedelta(days=WINDOW_D)
    days  = pd.date_range(start, end, freq="D", tz="UTC")

    rows = []
    for grp, sub in active.groupby("group"):
        opens  = sub["first_mint_utc"].values
        closes = sub["last_burn_utc"].values   # NaT if still active
        for day in days:
            day_next = day + pd.Timedelta(days=1)
            # Open at start of day: opened before day AND not yet closed before day
            open_mask = (opens < day_next) & (
                pd.isnull(closes) | (closes >= day)
            )
            n_open = int(open_mask.sum())
            if n_open < MIN_POS:
                continue
            # Closed on this specific day
            exit_mask = (
                ~pd.isnull(closes) &
                (closes >= day) &
                (closes < day_next) &
                open_mask
            )
            n_exit = int(exit_mask.sum())
            rows.append({
                "date":      day,
                "group":     grp,
                "narrow":    1 if grp == "narrow" else 0,
                "post":      1 if day >= FTX_DATE else 0,
                "day_rel":   int((day - FTX_DATE).days),
                "n_open":    n_open,
                "n_exit":    n_exit,
                "exit_rate": n_exit / n_open,
            })

    panel = pd.DataFrame(rows)
    if panel.empty:
        return None
    panel = panel.sort_values(["date", "group"]).reset_index(drop=True)
    return panel


# ── DiD regression ────────────────────────────────────────────────────────────

def did_regression(panel: pd.DataFrame) -> pd.DataFrame:
    """
    DiD: exit_rate = α + β_narrow·Narrow + β_post·Post
                       + δ·(Narrow×Post) + γ₁·day_rel + γ₂·Narrow·day_rel + ε
    HAC SE; δ = DiD estimator.
    """
    df = panel.copy()
    df["narrow_x_post"]    = df["narrow"] * df["post"]
    df["narrow_x_day_rel"] = df["narrow"] * df["day_rel"]
    X_cols = ["narrow", "post", "narrow_x_post", "day_rel", "narrow_x_day_rel"]
    y = df["exit_rate"]
    X = df[X_cols]
    result = ols_hac(y, X, max_lags=7, label="DiD")
    if not result.empty:
        result["NOTE"] = (
            "DiD: δ = coef(narrow_x_post). H1: δ>0 → narrow LPs exit faster "
            "post-FTX. Includes group-specific linear time trend (narrow×day_rel). "
            "HAC SE, 7-lag. N = group × day observations."
        )
    return result


def parallel_trends_test(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-period only: test H0 that the linear trend in exit_rate is the same
    across narrow and wide groups. Rejection → parallel trends violated.
    """
    pre = panel[panel["post"] == 0].copy()
    pre["narrow_x_day"] = pre["narrow"] * pre["day_rel"]
    X_cols = ["narrow", "day_rel", "narrow_x_day"]
    y = pre["exit_rate"]
    X = pre[X_cols]
    result = ols_hac(y, X, max_lags=7, label="ParallelTrends")
    if not result.empty:
        result["NOTE"] = (
            "Pre-period only (post=0). H0: coef(narrow×day_rel)=0 → parallel trends. "
            "Significant coef suggests narrow/wide diverged pre-FTX (violates assumption)."
        )
    return result


# ── LP flow event study ───────────────────────────────────────────────────────

def lp_flow_study(lp: pd.DataFrame, mh: pd.DataFrame | None) -> pd.DataFrame:
    """Daily net LP flow (burns-per-day volume) for narrow and wide groups."""
    rows = []
    start = FTX_DATE - pd.Timedelta(days=WINDOW_D)
    end   = FTX_DATE + pd.Timedelta(days=WINDOW_D)
    days  = pd.date_range(start, end, freq="D", tz="UTC")

    for grp, sub in lp.groupby("group"):
        # Daily mints
        mints = sub.set_index("first_mint_utc")["total_minted_usd"].resample("D").sum()
        # Daily burns
        burned = sub.dropna(subset=["last_burn_utc"]).set_index("last_burn_utc")
        burns_d = burned["total_burned_usd"].resample("D").sum()
        for day in days:
            day_str = day.normalize()
            m = float(mints.get(day_str, 0.0) or 0.0)
            b = float(burns_d.get(day_str, 0.0) or 0.0)
            rows.append({
                "date":    day,
                "group":   grp,
                "day_rel": int((day - FTX_DATE).days),
                "mints_usd": m,
                "burns_usd": b,
                "net_flow_usd": m - b,
            })

    df = pd.DataFrame(rows)
    # Normalise by group pre-event mean
    for grp in df["group"].unique():
        mask_pre = (df["group"] == grp) & (df["day_rel"] < 0)
        pre_mean = df.loc[mask_pre, "net_flow_usd"].mean()
        if pre_mean and not np.isnan(pre_mean):
            df.loc[df["group"] == grp, "net_flow_norm"] = (
                df.loc[df["group"] == grp, "net_flow_usd"] / abs(pre_mean)
            )
        else:
            df.loc[df["group"] == grp, "net_flow_norm"] = np.nan

    return df


# ── Kaplan-Meier survival ─────────────────────────────────────────────────────

def kaplan_meier(lp: pd.DataFrame) -> pd.DataFrame:
    """
    Non-parametric Kaplan-Meier survival estimate.
    Event = position exit; time = days from FTX to exit.
    Censoring = positions still active at end of data.

    Log-rank test between narrow and wide groups.
    Ref: Kaplan & Meier (1958); Mantel (1966) for log-rank.
    """
    active = lp[lp["active_at_ftx"]].copy()
    active["time_days"] = np.where(
        active["last_burn_utc"].notna(),
        (active["last_burn_utc"] - FTX_DATE).dt.days,
        (active["last_event_utc"] - FTX_DATE).dt.days,
    )
    active["event"] = active["last_burn_utc"].notna().astype(int)
    active = active[active["time_days"] >= 0].copy()

    def _km(sub: pd.DataFrame):
        if sub.empty:
            return pd.DataFrame()
        sub = sub.sort_values("time_days").copy()
        times = sub["time_days"].values
        events = sub["event"].values
        unique_t = np.unique(times[events == 1])
        S = 1.0
        rows = [{"t": 0, "S": 1.0, "n_risk": len(sub), "n_event": 0}]
        for t in unique_t:
            n_risk  = int((times >= t).sum())
            n_event = int(((times == t) & (events == 1)).sum())
            if n_risk > 0:
                S *= 1.0 - n_event / n_risk
            rows.append({"t": int(t), "S": round(S, 6),
                         "n_risk": n_risk, "n_event": n_event})
        return pd.DataFrame(rows)

    rows_out = []
    km_data = {}
    for grp, sub in active.groupby("group"):
        km = _km(sub)
        km["group"] = grp
        rows_out.append(km)
        km_data[grp] = sub

    # Log-rank test (Mantel 1966) if both groups present
    logrank_p = np.nan
    if len(km_data) == 2:
        from scipy.stats import chi2
        grps = list(km_data.keys())
        def _logrank(a, b):
            t_a = a["time_days"].values; e_a = a["event"].values
            t_b = b["time_days"].values; e_b = b["event"].values
            all_t = np.unique(np.concatenate([t_a[e_a == 1], t_b[e_b == 1]]))
            O_a = E_a = 0.0
            for t in all_t:
                n_a = int((t_a >= t).sum()); n_b = int((t_b >= t).sum())
                d_a = int(((t_a == t) & (e_a == 1)).sum())
                d_b = int(((t_b == t) & (e_b == 1)).sum())
                d = d_a + d_b; n = n_a + n_b
                if n <= 0:
                    continue
                O_a += d_a
                E_a += d * n_a / n
            stat = (O_a - E_a) ** 2 / max(E_a, 1e-9)
            return float(chi2.sf(stat, df=1))
        logrank_p = _logrank(km_data[grps[0]], km_data[grps[1]])

    df_km = pd.concat(rows_out, ignore_index=True) if rows_out else pd.DataFrame()
    df_km["logrank_p_narrow_vs_wide"] = logrank_p
    df_km["logrank_sig"] = stars(logrank_p)
    df_km["NOTE"] = (
        "KM survival: time = days from FTX to position exit. "
        "Censored = still active at data end. Log-rank test (Mantel 1966) "
        "for H0: narrow/wide survival curves are equal. "
        "Does not control for position size, range width continuous, or LP identity."
    )
    return df_km


# ── Cross-sectional duration regression ──────────────────────────────────────

def cohort_regression(lp: pd.DataFrame) -> pd.DataFrame:
    """HC3 OLS of log(days_to_exit) on position characteristics."""
    active = lp[lp["active_at_ftx"]].copy()
    active = active[active["last_burn_utc"].notna()].copy()
    active["time_days"] = (active["last_burn_utc"] - FTX_DATE).dt.days
    active = active[active["time_days"] > 0].copy()
    if len(active) < 20:
        return pd.DataFrame()

    active["log_t"]   = np.log(active["time_days"])
    active["log_mint"]= np.log(active["total_minted_usd"].clip(lower=1))
    active["log_rng"] = np.log(active["range_width_pct"].clip(lower=0.01))

    y = active["log_t"]
    X = active[["narrow", "log_mint", "log_rng"]].copy()
    X = sm.add_constant(X)
    try:
        res = sm.OLS(y, X).fit().get_robustcov_results(cov_type=HC_TYPE)
        rows = []
        for v, c, s, t, p in zip(res.model.exog_names, res.params,
                                  res.bse, res.tvalues, res.pvalues):
            rows.append({"var": v, "coef": round(float(c), 6),
                         "se_hc3": round(float(s), 6),
                         "t": round(float(t), 3), "p": round(float(p), 4),
                         "sig": stars(float(p))})
        df = pd.DataFrame(rows).set_index("var")
        df["NOTE"] = (
            "Dep var = log(days from FTX to exit). HC3 robust SE. "
            "Narrow<0 → narrow positions exit faster post-FTX (shorter survival). "
            "Does not control for LP identity (repeated-LP bias possible)."
        )
        return df
    except Exception as exc:
        print(f"    [WARN] cohort regression failed: {exc}")
        return pd.DataFrame()


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_exit_rate(panel: pd.DataFrame) -> None:
    if panel is None or panel.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    for grp, col in [("narrow", "#dc2626"), ("wide", "#2563eb")]:
        sub = panel[panel["group"] == grp].set_index("day_rel")["exit_rate"]
        weekly = sub.rolling(7, center=True).mean()
        ax.plot(sub.index, sub.values, alpha=0.25, color=col, lw=0.8)
        ax.plot(weekly.index, weekly.values, color=col, lw=2, label=f"{grp} (7d MA)")
    ax.axvline(0, color="black", lw=2, ls="--", label="FTX collapse (Nov 8)")
    ax.set_xlabel("Days relative to FTX collapse")
    ax.set_ylabel("Daily exit rate (exits / open positions)")
    ax.set_title("LP exit rate: narrow vs wide range — DiD design")
    ax.legend()
    savefig("ftx_did_exit_rate")


def plot_lp_flow(flow: pd.DataFrame) -> None:
    if flow.empty:
        return
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for i, (grp, col) in enumerate([("narrow", "#dc2626"), ("wide", "#2563eb")]):
        ax = axes[i]
        sub = flow[flow["group"] == grp].sort_values("day_rel")
        ax.bar(sub["day_rel"], sub["net_flow_usd"] / 1e6, color=col, alpha=0.7)
        ax.axvline(0, color="black", lw=1.5, ls="--")
        ax.set_title(f"Net LP flow — {grp} positions (M USD)")
        ax.set_ylabel("Net flow (M USD)")
    axes[1].set_xlabel("Days relative to FTX collapse")
    plt.suptitle("Daily net LP flow (mints − burns) around FTX collapse", fontsize=10)
    savefig("ftx_lp_flow")


def plot_survival(df_km: pd.DataFrame) -> None:
    if df_km.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    for grp, col, ls in [("narrow", "#dc2626", "--"), ("wide", "#2563eb", "-")]:
        sub = df_km[df_km["group"] == grp].sort_values("t")
        if sub.empty:
            continue
        ax.step(sub["t"], sub["S"], color=col, lw=2, ls=ls, where="post",
                label=f"{grp} positions")
    ax.axhline(0.5, color="gray", lw=0.8, ls=":")
    ax.set_xlabel("Days from FTX collapse")
    ax.set_ylabel("Survival probability (position not yet exited)")
    logrank_p = df_km["logrank_p_narrow_vs_wide"].iloc[0]
    sig = stars(logrank_p) if not np.isnan(logrank_p) else ""
    ax.set_title(f"Kaplan-Meier survival  |  Log-rank p = {logrank_p:.3f}{sig}")
    ax.legend()
    savefig("ftx_survival")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    lp = load_positions()
    if lp is None:
        print("  [SKIP] dex_lp_positions.csv not found."); return

    mh = load("merged/merged_hourly.csv")

    n_active = lp["active_at_ftx"].sum()
    print(f"  FTX date: {FTX_DATE.date()}")
    print(f"  LP positions before FTX: {len(lp):,}  |  active at FTX: {n_active:,}")
    med_rng = lp["range_width_pct"].median()
    print(f"  Median range width: {med_rng:.1f}%  →  narrow ≤ {med_rng:.1f}%")

    # 1. Daily panel DiD
    panel = build_daily_panel(lp)
    if panel is not None and not panel.empty:
        print(f"  Daily panel: {len(panel):,} group-day observations")
        df_did = did_regression(panel)
        if not df_did.empty:
            savetable(df_did, "ftx_did_panel")
        df_pt = parallel_trends_test(panel)
        if not df_pt.empty:
            savetable(df_pt, "ftx_parallel_trends")
        plot_exit_rate(panel)
    else:
        print("  [WARN] Could not build daily panel (insufficient data)")

    # 2. LP flow event study
    flow = lp_flow_study(lp, mh)
    if not flow.empty:
        savetable(flow, "ftx_lp_flow")
        plot_lp_flow(flow)

    # 3. Kaplan-Meier survival
    df_km = kaplan_meier(lp)
    if not df_km.empty:
        savetable(df_km, "ftx_survival")
        plot_survival(df_km)

    # 4. Cross-sectional duration regression
    df_cohort = cohort_regression(lp)
    if not df_cohort.empty:
        savetable(df_cohort, "ftx_cohort_regression")

    print("  [DONE] did_ftx_lp_withdrawal.py")


if __name__ == "__main__":
    main()
