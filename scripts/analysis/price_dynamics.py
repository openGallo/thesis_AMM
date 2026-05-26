"""
Price dynamics analysis.

Figures:
    price_level_series      ETH/USDC price (DEX vs CEX overlay)
    log_return_distribution Log return histogram + normal overlay
    realized_vol_series     Realized volatility time series (three horizons)
    return_acf              Autocorrelation of returns and squared returns

Tables:
    price_dynamics_stats    GBM parameter estimates + stationarity tests
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from analysis_utils import COLORS, load, savefig, savetable

warnings.filterwarnings("ignore")


def _adf_pvalue(series: pd.Series) -> float:
    from statsmodels.tsa.stattools import adfuller
    clean = series.dropna()
    if len(clean) < 20:
        return float("nan")
    try:
        return float(adfuller(clean, autolag="AIC")[1])
    except Exception:
        return float("nan")


def main() -> None:
    print("=" * 60)
    print("Price Dynamics")
    print("=" * 60)

    cex = load("CEX/cex_price_hourly.csv")
    dex = load("DEX/dex_pool_hourly.csv")
    if cex is None and dex is None:
        print("  No price data available.")
        return

    # ── Figure 1: Price level time series ────────────────────────
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if cex is not None and "eth_usdc_close" in cex.columns:
        ax.plot(cex.index, pd.to_numeric(cex["eth_usdc_close"], errors="coerce"),
                color=COLORS[0], lw=0.6, label="CEX (Binance)", alpha=0.9)
    if dex is not None and "eth_usdc_close" in dex.columns:
        ax.plot(dex.index, pd.to_numeric(dex["eth_usdc_close"], errors="coerce"),
                color=COLORS[1], lw=0.6, label="DEX (Uniswap v3)", alpha=0.7)
    ax.set_ylabel("ETH / USDC")
    ax.set_title("ETH/USDC Price: DEX vs CEX")
    ax.legend()
    savefig("price_level_series")

    # ── Figure 2: Log return distribution ────────────────────────
    src = cex if cex is not None else dex
    col = "log_return_1h"
    if src is not None and col in src.columns:
        r = pd.to_numeric(src[col], errors="coerce").dropna()
        r = r[r.between(r.quantile(0.001), r.quantile(0.999))]

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        # Histogram + normal fit
        mu, sigma = r.mean(), r.std()
        axes[0].hist(r, bins=120, density=True, color=COLORS[0], alpha=0.6, label="Empirical")
        x = np.linspace(r.min(), r.max(), 300)
        axes[0].plot(x, stats.norm.pdf(x, mu, sigma), color=COLORS[1], lw=1.5, label="Normal fit")
        axes[0].set_xlabel("1h log return")
        axes[0].set_ylabel("Density")
        axes[0].set_title("Return distribution")
        axes[0].legend()

        # QQ plot
        (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
        axes[1].scatter(osm, osr, s=2, alpha=0.3, color=COLORS[0])
        line_x = np.array([osm[0], osm[-1]])
        axes[1].plot(line_x, slope * line_x + intercept, color=COLORS[1], lw=1.5)
        axes[1].set_xlabel("Theoretical quantiles")
        axes[1].set_ylabel("Sample quantiles")
        axes[1].set_title("Normal Q-Q plot")
        savefig("log_return_distribution")

    # ── Figure 3: Realized volatility series ─────────────────────
    if cex is not None:
        vol_cols = [c for c in ["realized_vol_1h_ann", "realized_vol_24h_ann",
                                 "realized_vol_7d_ann"] if c in cex.columns]
        if vol_cols:
            fig, ax = plt.subplots(figsize=(9, 3.5))
            labels = {"realized_vol_1h_ann":  "1h window",
                      "realized_vol_24h_ann": "24h window",
                      "realized_vol_7d_ann":  "7d window"}
            for i, col in enumerate(vol_cols):
                v = pd.to_numeric(cex[col], errors="coerce")
                ax.plot(cex.index, v * 100, color=COLORS[i],
                        lw=0.7, label=labels.get(col, col), alpha=0.85)
            ax.set_ylabel("Annualized volatility (%)")
            ax.set_title("Realized Volatility (multiple horizons)")
            ax.legend()
            savefig("realized_vol_series")

    # ── Figure 4: ACF of returns and squared returns ──────────────
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        src = cex if cex is not None else dex
        if src is not None and "log_return_1h" in src.columns:
            r = pd.to_numeric(src["log_return_1h"], errors="coerce").dropna()
            fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
            plot_acf(r,  lags=48, ax=axes[0], title="ACF: 1h log returns",    alpha=0.05)
            plot_acf(r**2, lags=48, ax=axes[1], title="ACF: squared returns (vol clustering)", alpha=0.05)
            savefig("return_acf")
    except Exception as exc:
        print(f"  [WARN] ACF plot failed: {exc}")

    # ── Table: GBM parameters + stationarity ─────────────────────
    rows = []
    for name, src, col in [
        ("CEX", cex, "log_return_1h"),
        ("DEX", dex, "log_return_1h"),
    ]:
        if src is None or col not in src.columns:
            continue
        r = pd.to_numeric(src[col], errors="coerce").dropna()
        sigma_h = float(r.std())
        mu_h    = float(r.mean())
        rows.append({
            "Series":        name,
            "mu_hourly":     round(mu_h, 6),
            "mu_daily":      round(mu_h * 24, 5),
            "sigma_hourly":  round(sigma_h, 5),
            "sigma_daily":   round(sigma_h * np.sqrt(24), 5),
            "sigma_ann":     round(sigma_h * np.sqrt(8760), 4),
            "skewness":      round(float(stats.skew(r)), 4),
            "excess_kurt":   round(float(stats.kurtosis(r)), 4),
            "ADF_pval":      round(_adf_pvalue(r), 4),
            "N_obs":         len(r),
        })

    if rows:
        tab = pd.DataFrame(rows).set_index("Series")
        savetable(tab, "price_dynamics_stats")

    print("\nDONE")


if __name__ == "__main__":
    main()
