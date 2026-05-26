"""
DEX-CEX price basis analysis.

Figures:
    basis_time_series       Basis (bps) time series with arbitrage threshold
    basis_distribution      Basis distribution histogram + fitted normal
    basis_acf               Autocorrelation of basis (half-life estimation)
    basis_vs_gas            Basis magnitude vs gas price scatter
    basis_vs_vol            Basis magnitude vs realized volatility scatter

Tables:
    basis_stats             Summary statistics and half-life
    basis_by_regime         Basis statistics by volatility regime
    arbitrage_persistence   Mean duration of arbitrage episodes
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from analysis_utils import COLORS, load, savefig, savetable, vol_regime

warnings.filterwarnings("ignore")

ARB_THRESHOLD = 5.0   # basis points — pool fee is 5 bps


def half_life(series: pd.Series) -> float:
    """Estimate AR(1) half-life of mean reversion in periods."""
    clean = series.dropna()
    if len(clean) < 20:
        return float("nan")
    y = clean.values[1:]
    x = clean.values[:-1]
    rho = float(np.corrcoef(x, y)[0, 1])
    if rho >= 1 or rho <= 0:
        return float("nan")
    return float(-np.log(2) / np.log(abs(rho)))


def episode_lengths(flag: pd.Series) -> pd.Series:
    """Return lengths (in periods) of consecutive True runs."""
    lengths = []
    count = 0
    for v in flag:
        if v:
            count += 1
        elif count > 0:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return pd.Series(lengths)


def main() -> None:
    print("=" * 60)
    print("DEX-CEX Basis Analysis")
    print("=" * 60)

    merged = load("merged/merged_hourly.csv")
    cex    = load("CEX/cex_price_hourly.csv")

    if merged is None:
        print("  merged_hourly.csv not found — run process_merged_panel.py first")
        return

    basis = pd.to_numeric(merged.get("dex_cex_basis_bps"), errors="coerce").dropna()
    if len(basis) == 0:
        print("  dex_cex_basis_bps column empty.")
        return

    print(f"  Basis observations: {len(basis):,}")
    print(f"  Mean: {basis.mean():.2f} bps   Std: {basis.std():.2f} bps")
    print(f"  |basis| > {ARB_THRESHOLD} bps: {(basis.abs() > ARB_THRESHOLD).mean()*100:.1f}% of hours")

    # ── Figure 1: Basis time series ───────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(basis.index, basis, color=COLORS[0], lw=0.4, alpha=0.7)
    ax.axhline(0,                color="black",    lw=0.8)
    ax.axhline( ARB_THRESHOLD,   color=COLORS[1], lw=1.0, ls="--",
                label=f"+{ARB_THRESHOLD} bps (fee threshold)")
    ax.axhline(-ARB_THRESHOLD,   color=COLORS[1], lw=1.0, ls="--")
    ax.fill_between(basis.index, basis,
                    where=basis.abs() > ARB_THRESHOLD,
                    color=COLORS[1], alpha=0.15, label="Arbitrage zone")
    ax.set_ylabel("DEX-CEX basis (bps)")
    ax.set_title("DEX-CEX Price Basis (Uniswap v3 vs Binance)")
    ax.legend()
    savefig("basis_time_series")

    # ── Figure 2: Basis distribution ─────────────────────────────
    basis_clipped = basis.clip(*basis.quantile([0.005, 0.995]))
    mu, sigma = basis_clipped.mean(), basis_clipped.std()
    x = np.linspace(basis_clipped.min(), basis_clipped.max(), 300)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(basis_clipped, bins=100, density=True, color=COLORS[0],
            alpha=0.6, label="Empirical")
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS[1], lw=1.5,
            label=f"Normal fit ($\\mu$={mu:.2f}, $\\sigma$={sigma:.2f})")
    ax.axvline( ARB_THRESHOLD, color=COLORS[2], lw=1.2, ls="--", label="+5 bps fee")
    ax.axvline(-ARB_THRESHOLD, color=COLORS[2], lw=1.2, ls="--")
    ax.set_xlabel("Basis (bps)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of DEX-CEX Price Basis")
    ax.legend()
    savefig("basis_distribution")

    # ── Figure 3: Basis ACF ───────────────────────────────────────
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, ax = plt.subplots(figsize=(7, 3.5))
        plot_acf(basis, lags=48, ax=ax,
                 title="Autocorrelation of DEX-CEX Basis (lags in hours)", alpha=0.05)
        savefig("basis_acf")
    except Exception as exc:
        print(f"  [WARN] ACF plot: {exc}")

    # ── Figure 4: Basis vs gas price ─────────────────────────────
    swaps_path = load.__module__  # trick to get DATA_PROC
    from analysis_utils import DATA_PROC
    swaps_path = DATA_PROC / "DEX" / "dex_swaps.csv"
    if swaps_path.exists():
        try:
            gas_df = pd.read_csv(swaps_path, usecols=["timestamp", "gas_price_wei"],
                                 low_memory=False, nrows=500_000)
            gas_df["timestamp"] = pd.to_datetime(gas_df["timestamp"], utc=True,
                                                  errors="coerce")
            gas_df["gas_gwei"] = pd.to_numeric(gas_df["gas_price_wei"],
                                               errors="coerce") / 1e9
            gas_hourly = (gas_df.set_index("timestamp")["gas_gwei"]
                          .resample("1h").median().rename("gas_gwei"))
            combined = pd.DataFrame({"basis_abs": basis.abs(), "gas_gwei": gas_hourly}).dropna()
            if len(combined) > 100:
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.scatter(combined["gas_gwei"].clip(upper=combined["gas_gwei"].quantile(0.98)),
                           combined["basis_abs"].clip(upper=combined["basis_abs"].quantile(0.98)),
                           s=3, alpha=0.2, color=COLORS[0])
                ax.set_xlabel("Median gas price (Gwei, hourly)")
                ax.set_ylabel("|DEX-CEX basis| (bps)")
                ax.set_title("Absolute Basis vs Gas Price")
                savefig("basis_vs_gas")
        except Exception as exc:
            print(f"  [WARN] basis_vs_gas: {exc}")

    # ── Figure 5: Basis vs realized volatility ────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        combined = pd.DataFrame({"basis_abs": basis.abs(), "vol": vol}).dropna()
        if len(combined) > 100:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            ax.scatter(combined["vol"].clip(upper=combined["vol"].quantile(0.98)) * 100,
                       combined["basis_abs"].clip(upper=combined["basis_abs"].quantile(0.98)),
                       s=3, alpha=0.2, color=COLORS[2])
            ax.set_xlabel("Realized vol 24h (annualized, %)")
            ax.set_ylabel("|DEX-CEX basis| (bps)")
            ax.set_title("Absolute Basis vs Realized Volatility")
            savefig("basis_vs_vol")

    # ── Table: basis stats + half-life ────────────────────────────
    hl = half_life(basis)
    arb_flag = basis.abs() > ARB_THRESHOLD
    ep_lens  = episode_lengths(arb_flag)

    tab = pd.DataFrame([{
        "N_obs":                  len(basis),
        "mean_bps":               round(basis.mean(), 3),
        "std_bps":                round(basis.std(), 3),
        "p5_bps":                 round(basis.quantile(0.05), 3),
        "p95_bps":                round(basis.quantile(0.95), 3),
        "skewness":               round(float(basis.skew()), 4),
        "pct_above_threshold":    round(arb_flag.mean() * 100, 2),
        "half_life_hours":        round(hl, 2) if not np.isnan(hl) else None,
        "mean_episode_hours":     round(ep_lens.mean(), 2) if len(ep_lens) > 0 else None,
        "median_episode_hours":   round(ep_lens.median(), 2) if len(ep_lens) > 0 else None,
    }], index=["Basis"])
    savetable(tab.T.rename(columns={"Basis": "Value"}), "basis_stats")

    # ── Table: basis by vol regime ────────────────────────────────
    if cex is not None and "realized_vol_24h_ann" in cex.columns:
        vol    = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce")
        regime = vol_regime(vol)
        df     = pd.DataFrame({"basis_abs": basis.abs(), "regime": regime}).dropna()
        grp    = df.groupby("regime")["basis_abs"].agg(
            N="count", mean="mean", median="median",
            pct_above_threshold=lambda x: (x > ARB_THRESHOLD).mean() * 100
        ).round(3)
        savetable(grp, "basis_by_regime")

    print("\nDONE")


if __name__ == "__main__":
    main()
