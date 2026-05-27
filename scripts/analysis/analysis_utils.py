"""
Shared utilities for all analysis scripts.

Provides:
  - I/O helpers      : load(), savefig(), savetable(), load_swaps()
  - Statistical tests: stationarity_tests(), ols_hac(), bh_correction()
  - Bootstrap CIs    : bootstrap_ci() (i.i.d.), block_bootstrap_ci() (time-series)
  - GARCH utilities  : garch11_fit()
  - Misc helpers     : stars(), vol_regime(), amihud_illiquidity()
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC    = PROJECT_ROOT / "data_processed"
FIG_DIR      = PROJECT_ROOT / "output" / "figures"
TAB_DIR      = PROJECT_ROOT / "output" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.constrained_layout.use": True,
})

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
          "#0891b2", "#db2777", "#65a30d", "#ea580c", "#0d9488"]

# ── I/O helpers ───────────────────────────────────────────────────────────────

def savefig(name: str, png: bool = True) -> None:
    """Save current figure as PDF (and optionally PNG for quick preview)."""
    pdf_path = FIG_DIR / f"{name}.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    if png:
        png_path = FIG_DIR / f"{name}.png"
        plt.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved {pdf_path.relative_to(PROJECT_ROOT)}")


def savetable(df: pd.DataFrame, name: str) -> None:
    csv_path = TAB_DIR / f"{name}.csv"
    # Always write UTF-8 so Unicode characters in note fields (e.g. ν, ≈, →)
    # survive on Windows where the default encoding is cp1252.
    df.to_csv(csv_path, encoding="utf-8")
    try:
        tex = df.to_latex(float_format="%.4f", na_rep="--", escape=False)
        (TAB_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
    except Exception:
        pass
    print(f"  Saved {csv_path.relative_to(PROJECT_ROOT)}")


def load(rel: str, **kwargs) -> pd.DataFrame | None:
    """Load a processed CSV with DatetimeIndex (UTC)."""
    path = DATA_PROC / rel
    if not path.exists():
        print(f"  [SKIP] {rel} not found - run processing pipeline first")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False, **kwargs)
    try:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    except Exception:
        pass
    return df


def load_swaps(
    cols: list[str] | None = None,
    nrows: int | None = None,
    chunksize: int | None = None,
) -> pd.DataFrame | None:
    """
    Load dex_swaps.csv with optional column selection and row limit.

    If chunksize is given, reads in chunks and concatenates (saves memory for
    very large files).
    """
    path = DATA_PROC / "DEX" / "dex_swaps.csv"
    if not path.exists():
        print("  [SKIP] dex_swaps.csv not found")
        return None

    default_cols = [
        "timestamp", "amount_usd", "gas_price_wei", "gas_cost_eth",
        "gas_cost_usd", "direction", "log_price_change",
        "trade_size_bucket", "eth_price_at_trade",
    ]
    use_cols = cols if cols is not None else default_cols

    read_kwargs: dict = dict(low_memory=False)
    read_kwargs["usecols"] = lambda c: c in use_cols
    if nrows:
        read_kwargs["nrows"] = nrows

    if chunksize:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):
            chunks.append(chunk)
        swaps = pd.concat(chunks, ignore_index=True)
    else:
        swaps = pd.read_csv(path, **read_kwargs)

    if "timestamp" in swaps.columns:
        swaps["timestamp"] = pd.to_datetime(swaps["timestamp"], utc=True, errors="coerce")
    for col in ["amount_usd", "gas_price_wei", "gas_cost_usd", "gas_cost_eth",
                "log_price_change", "eth_price_at_trade"]:
        if col in swaps.columns:
            swaps[col] = pd.to_numeric(swaps[col], errors="coerce")
    if "gas_price_wei" in swaps.columns:
        swaps["gas_gwei"] = swaps["gas_price_wei"] / 1e9
    return swaps


# ── Statistical helpers ───────────────────────────────────────────────────────

def stars(pval: float) -> str:
    """Return significance stars for a p-value."""
    if pval is None or np.isnan(pval):
        return ""
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def bootstrap_ci(
    series: np.ndarray | pd.Series,
    func: Callable = np.mean,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    max_n: int = 50_000,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval (i.i.d. resampling).
    Use block_bootstrap_ci() for autocorrelated time series.

    max_n: if len > max_n, subsample without replacement first.
           50K is more than enough — CI barely changes past ~10K obs (CLT).
           Without this cap, 10M-row inputs take hours.
    """
    arr = np.asarray(series)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 5:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if len(arr) > max_n:
        arr = rng.choice(arr, size=max_n, replace=False)
    boot = [func(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def block_bootstrap_ci(
    series: np.ndarray | pd.Series,
    func: Callable = np.mean,
    block_size: int = 24,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    max_n: int = 100_000,
) -> tuple[float, float]:
    """
    Moving-block bootstrap CI - corrects for autocorrelation in hourly time series.
    Default block_size=24 (one calendar day) preserves intraday dependence.

    max_n: if len > max_n, subsample contiguous blocks to this size.
           Hourly panels rarely exceed 50K; the cap exists for swap-level data.
    """
    arr = np.asarray(series)
    arr = arr[~np.isnan(arr)]
    n = len(arr)
    if n < block_size * 2:
        return bootstrap_ci(arr, func, n_boot=n_boot, alpha=alpha, seed=seed)

    # If array is huge, take a contiguous random window to preserve autocorrelation
    if n > max_n:
        rng_sub = np.random.default_rng(seed)
        start = int(rng_sub.integers(0, n - max_n))
        arr = arr[start : start + max_n]
        n = max_n

    rng = np.random.default_rng(seed)

    if block_size == 1:
        # Degenerate case: plain i.i.d. bootstrap — use vectorised fancy-indexing
        # instead of a list comprehension of 100 K 1-element slices (O(n_boot*n) list ops).
        idx = rng.integers(0, n, size=(n_boot, n))
        boot_stats = [func(arr[idx[i]]) for i in range(n_boot)]
    else:
        n_blocks = (n + block_size - 1) // block_size
        starts = np.arange(n - block_size + 1)
        boot_stats = []
        for _ in range(n_boot):
            idx_starts = rng.choice(starts, size=n_blocks, replace=True)
            sample = np.concatenate([arr[s : s + block_size] for s in idx_starts])[:n]
            boot_stats.append(func(sample))

    lo, hi = np.percentile(boot_stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def stationarity_tests(series: pd.Series, name: str = "") -> dict:
    """ADF (H0: unit root) + KPSS (H0: stationary). Both rejecting => inconclusive."""
    from statsmodels.tsa.stattools import adfuller, kpss
    clean = series.dropna()
    if len(clean) < 20:
        return {}
    adf_s, adf_p, *_, adf_cv, _ = adfuller(clean, autolag="AIC")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_s, kpss_p, _, kpss_cv = kpss(clean, regression="c", nlags="auto")
    return {
        "N":                len(clean),
        "ADF stat":         round(float(adf_s), 4),
        "ADF p-val":        round(float(adf_p), 4),
        "ADF 5% crit":      round(float(adf_cv["5%"]), 4),
        "ADF reject H0":    "Yes" if adf_p < 0.05 else "No",
        "KPSS stat":        round(float(kpss_s), 4),
        "KPSS p-val":       round(float(kpss_p), 4),
        "KPSS 5% crit":     round(float(kpss_cv["5%"]), 4),
        "KPSS reject H0":   "Yes" if kpss_p < 0.05 else "No",
        "Conclusion":       _stationarity_conclusion(adf_p, kpss_p),
    }


def _stationarity_conclusion(adf_p: float, kpss_p: float) -> str:
    adf_rej  = adf_p  < 0.05
    kpss_rej = kpss_p < 0.05
    if adf_rej and not kpss_rej:
        return "Stationary"
    if not adf_rej and kpss_rej:
        return "Non-stationary"
    if adf_rej and kpss_rej:
        return "Inconclusive"
    return "Weak evidence"


def ols_hac(
    y: pd.Series,
    X: pd.DataFrame,
    max_lags: int | None = None,
    label: str = "OLS-HAC",
) -> pd.DataFrame:
    """
    OLS with Newey-West HAC standard errors.
    Returns tidy DataFrame: variable | coef | se_hac | t_stat | p_val | sig
    """
    import statsmodels.api as sm
    df = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if len(df) < 20:
        return pd.DataFrame()
    y_  = df["__y__"]
    X_  = sm.add_constant(df.drop(columns="__y__"))
    res = sm.OLS(y_, X_).fit()
    if max_lags is None:
        max_lags = int(np.floor(4 * (len(df) / 100) ** (2 / 9)))
    hac = res.get_robustcov_results(cov_type="HAC", maxlags=max_lags, use_correction=True)
    rows = []
    for var, coef, se, tstat, pval in zip(
        hac.model.exog_names, hac.params, hac.bse, hac.tvalues, hac.pvalues
    ):
        rows.append({
            "Variable": var,
            "Coef":     round(float(coef), 6),
            "SE (HAC)": round(float(se), 6),
            "t-stat":   round(float(tstat), 3),
            "p-val":    round(float(pval), 4),
            "Sig":      stars(float(pval)),
        })
    rows.append({
        "Variable": "-",
        "Coef":     None, "SE (HAC)": None, "t-stat": None, "p-val": None,
        "Sig":      f"N={len(df)}  R²={hac.rsquared:.4f}  adj-R²={hac.rsquared_adj:.4f}"
                    f"  F-p={hac.f_pvalue:.4f}",
    })
    return pd.DataFrame(rows).set_index("Variable")


def bh_correction(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Benjamini-Hochberg FDR correction. Returns list of rejection booleans."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    sorted_p = np.array(pvals)[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    reject_sorted = sorted_p <= thresholds
    if reject_sorted.any():
        last = np.where(reject_sorted)[0][-1]
        reject_sorted[:last + 1] = True
    result = np.empty(n, dtype=bool)
    result[order] = reject_sorted
    return result.tolist()


def vol_regime(series: pd.Series, low_q: float = 1/3, high_q: float = 2/3) -> pd.Series:
    """Label each observation as low / normal / high volatility by tercile."""
    lo = series.quantile(low_q)
    hi = series.quantile(high_q)
    return pd.cut(series, bins=[-np.inf, lo, hi, np.inf],
                  labels=["low", "normal", "high"])


# ── GARCH(1,1) utility ────────────────────────────────────────────────────────

def garch11_fit(returns: pd.Series) -> dict | None:
    """
    Fit GARCH(1,1) model to a return series using arch library.
    Returns dict with fitted params or None if arch not installed.

    Keys: omega, alpha, beta, log_likelihood, aic, bic,
          annualized_long_run_vol, persistence
    """
    try:
        from arch import arch_model
    except ImportError:
        print("  [WARN] arch library not installed - skipping GARCH fit")
        return None

    # Accept both pandas Series and numpy arrays
    r = pd.Series(returns).dropna() * 100  # scale for numerical stability
    if len(r) < 200:
        return None
    try:
        model = arch_model(r, vol="Garch", p=1, q=1, dist="normal", rescale=False)
        res   = model.fit(disp="off", show_warning=False)
        omega = float(res.params["omega"])
        alpha = float(res.params["alpha[1]"])
        beta  = float(res.params["beta[1]"])
        # Long-run variance (undoes the ×100 scaling above)
        long_run_var = omega / max(1 - alpha - beta, 1e-9) / 1e4
        return {
            "omega":                  round(omega, 8),
            "alpha":                  round(alpha, 6),
            "beta":                   round(beta, 6),
            "persistence":            round(alpha + beta, 6),
            "half_life_days":         round(-np.log(2) / np.log(alpha + beta) / 24, 2)
                                      if 0 < alpha + beta < 1 else None,
            "log_likelihood":         round(float(res.loglikelihood), 4),
            "aic":                    round(float(res.aic), 4),
            "bic":                    round(float(res.bic), 4),
            "annualized_long_run_vol": round(float(np.sqrt(long_run_var * 8760)), 4),
            "conditional_vol_series": res.conditional_volatility / 100 * np.sqrt(8760),
            # Distributional assumption note:
            # Fitted with Normal distribution -> quasi-maximum likelihood (QML).
            # QML estimates are consistent under non-normality (Bollerslev &
            # Wooldridge 1992) but less efficient than MLE with correct distribution.
            # For heavy-tailed crypto returns (Student-t nu≈2-3) a t-GARCH or
            # GJR-GARCH would better capture tail risk and give more efficient
            # parameter estimates. Near-IGARCH persistence (alpha+beta≈1) means
            # the unconditional variance omega/(1-alpha-beta) is numerically
            # unstable; use the model for conditional vol forecasts only.
            # Refs: Bollerslev & Wooldridge (1992) Econometric Reviews;
            #       Nelson (1991) Econometrica; Engle & Bollerslev (1986) ER.
            "NOTE_dist": (
                "dist='normal' -> QML estimates: consistent but inefficient under "
                "fat tails (Bollerslev & Wooldridge 1992). t-GARCH preferred for "
                "crypto returns (nu~2-3). Near-IGARCH: use for short-run conditional "
                "vol forecasts only, not unconditional variance."
            ),
        }
    except Exception as exc:
        print(f"  [WARN] GARCH fit failed: {exc}")
        return None


# ── Amihud (2002) illiquidity ─────────────────────────────────────────────────

def amihud_illiquidity(
    price_change: pd.Series,
    volume_usd: pd.Series,
    window: int = 24,
) -> pd.Series:
    """
    Amihud (2002) illiquidity ratio adapted for hourly DEX data.

    Original formulation (Amihud 2002, J. Financial Markets):
        ILLIQ_d = |r_d| / DailyVolume_USD      (daily, averaged over a year)

    Adaptation used here:
        ILLIQ_h = |r_h| / HourlyVolume_USD      (hourly)
        Rolling mean over `window` hours (default 24h = 1 calendar day).

    Rationale: Uniswap v3 provides only hourly OHLCV; daily aggregation loses
    intraday resolution essential for this study. The hourly ratio is consistent
    with Goyenko, Holden & Trzcinka (2009) JFE who use the Amihud ratio on
    daily and higher-frequency data.

    Scale note: for a pool with $500M+ daily volume, ILLIQ_h ≈ 1e-9 — near-zero
    but non-trivial relative to other pools.  Compare across pools using
    log(ILLIQ) or rank-normalized values.

    Refs: Amihud (2002) JFM; Goyenko, Holden & Trzcinka (2009) JFE.
    """
    illiq = price_change.abs() / volume_usd.replace(0, np.nan)
    return illiq.rolling(window).mean()


# ── Intraday seasonality helper ───────────────────────────────────────────────

def intraday_profile(
    series: pd.Series,
    func: str = "mean",
) -> pd.DataFrame:
    """
    Compute mean (or median) of a series by hour of day (0-23, UTC).

    Returns DataFrame indexed 0-23 with columns: value, ci_lo, ci_hi (block bootstrap).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have DatetimeIndex")
    df = pd.DataFrame({"val": series, "hour": series.index.hour})
    fn = np.mean if func == "mean" else np.median
    rows = []
    for h in range(24):
        vals = df[df["hour"] == h]["val"].dropna().values
        if len(vals) < 5:
            rows.append({"hour": h, "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        est = fn(vals)
        ci_lo, ci_hi = bootstrap_ci(vals, func=fn, n_boot=500)
        rows.append({"hour": h, "value": round(float(est), 6),
                     "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6)})
    return pd.DataFrame(rows).set_index("hour")


def day_of_week_profile(
    series: pd.Series,
    func: str = "mean",
) -> pd.DataFrame:
    """
    Mean (or median) by day of week (0=Monday … 6=Sunday, UTC).
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("series must have DatetimeIndex")
    df = pd.DataFrame({"val": series, "dow": series.index.dayofweek})
    fn = np.mean if func == "mean" else np.median
    DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    rows = []
    for d in range(7):
        vals = df[df["dow"] == d]["val"].dropna().values
        if len(vals) < 5:
            rows.append({"dow": d, "label": DOW[d],
                         "value": np.nan, "ci_lo": np.nan, "ci_hi": np.nan})
            continue
        est = fn(vals)
        ci_lo, ci_hi = bootstrap_ci(vals, func=fn, n_boot=500)
        rows.append({"dow": d, "label": DOW[d],
                     "value": round(float(est), 6),
                     "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6)})
    return pd.DataFrame(rows).set_index("label")


# ── Tail-risk statistics ──────────────────────────────────────────────────────

def var_es(
    returns: pd.Series,
    alpha: float = 0.05,
    ann_factor: float = np.sqrt(8760),
) -> dict:
    """
    Historical VaR and Expected Shortfall (CVaR) at level alpha.

    Returns annualized vol, VaR_{alpha} (1-sided lower tail), ES_{alpha}.
    All expressed as fractions (not %).

    Methodology note — square-root-of-time scaling:
        VaR_1d = VaR_1h × sqrt(24) uses the square-root-of-time rule, which
        assumes i.i.d. returns (no serial correlation, constant variance).
        This is invalid when returns exhibit GARCH-type volatility clustering:
        - During high-vol regimes: actual 1-day VaR > i.i.d. estimate.
        - During low-vol regimes: actual 1-day VaR < i.i.d. estimate.
        For GARCH-adjusted multi-period VaR see Diebold, Schuermann & Stroughair
        (1998) and McNeil & Frey (2000). Historical simulation over 24h windows
        is an alternative that requires no distributional assumption.
        Refs: McNeil & Frey (2000) J. Empirical Finance;
              Diebold et al. (1998) J. Derivatives.
    """
    r = returns.dropna()
    if len(r) < 50:
        return {}
    var_h = float(np.percentile(r, alpha * 100))        # negative number
    es_h  = float(r[r <= var_h].mean())
    return {
        "sigma_ann":   round(float(r.std()) * ann_factor, 4),
        f"VaR_{int(alpha*100)}pct_hourly":  round(var_h, 6),
        f"ES_{int(alpha*100)}pct_hourly":   round(es_h, 6),
        f"VaR_{int(alpha*100)}pct_1d":      round(var_h * np.sqrt(24), 6),
        f"ES_{int(alpha*100)}pct_1d":       round(es_h  * np.sqrt(24), 6),
        "NOTE_sqrt_time": (
            "VaR_1d = VaR_1h x sqrt(24) assumes i.i.d. returns. "
            "Invalid under GARCH clustering: actual 1-day VaR is higher in "
            "high-vol regimes. Ref: McNeil & Frey (2000); Diebold et al. (1998)."
        ),
    }


# ── Rolling window helper ─────────────────────────────────────────────────────

def rolling_sharpe(
    returns: pd.Series,
    window: int = 24 * 30,
    ann_factor: float = 8760,
) -> pd.Series:
    """
    Rolling annualized Sharpe ratio (zero risk-free rate, consistent with DeFi framing).
    window in hours, default = 30 calendar days.
    """
    mu  = returns.rolling(window).mean()
    sig = returns.rolling(window).std()
    return (mu / sig.replace(0, np.nan)) * np.sqrt(ann_factor)
