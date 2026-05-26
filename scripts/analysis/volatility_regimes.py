"""
Volatility regime analysis.

Regimes defined by 24h annualized realized volatility:
    low    — bottom tercile
    normal — middle tercile
    high   — top tercile (separately: stress = top decile)

Figures:
    regime_time_series      Price + vol coloured by regime
    regime_vol_distribution Volatility distribution by regime
    regime_metrics_bar      Key metrics (fee APR, LVR, tx count, basis) by regime

Tables:
    regime_stats            All key metrics by regime
    stress_events           Summary of stress episodes (>90th pct vol)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from analysis_utils import COLORS, load, savefig, savetable


REGIME_COLORS = {"low": COLORS[2], "normal": COLORS[0], "high": COLORS[1]}


def assign_regimes(vol: pd.Series, stress_q: float = 0.90) -> pd.DataFrame:
    lo = vol.quantile(1/3)
    hi = vol.quantile(2/3)
    st = vol.quantile(stress_q)
    regime = pd.cut(vol, bins=[-np.inf, lo, hi, np.inf],
                    labels=["low", "normal", "high"])
    stress = vol > st
    return pd.DataFrame({"vol": vol, "regime": regime, "stress": stress})


def main() -> None:
    print("=" * 60)
    print("Volatility Regime Analysis")
    print("=" * 60)

    cex    = load("CEX/cex_price_hourly.csv")
    dex    = load("DEX/dex_pool_hourly.csv")
    lvr    = load("DEX/dex_lvr_hourly.csv")
    merged = load("merged/merged_hourly.csv")

    if cex is None or "realized_vol_24h_ann" not in cex.columns:
        print("  [SKIP] realized_vol_24h_ann not available")
        return

    vol = pd.to_numeric(cex["realized_vol_24h_ann"], errors="coerce").dropna()
    reg = assign_regimes(vol)
    print(f"  Regime counts: {reg['regime'].value_counts().to_dict()}")
    print(f"  Stress hours (>90th pct): {reg['stress'].sum():,} ({reg['stress'].mean()*100:.1f}%)")

    # ── Figure 1: Price + vol coloured by regime ──────────────────
    price = pd.to_numeric(cex.get("eth_usdc_close"), errors="coerce")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(cex.index, price, color="black", lw=0.5, alpha=0.8)
    for regime, color in REGIME_COLORS.items():
        mask = reg["regime"] == regime
        axes[0].fill_between(cex.index, price.min(), price.max(),
                             where=mask.reindex(cex.index, fill_value=False),
                             color=color, alpha=0.12)
    axes[0].set_ylabel("ETH/USDC")
    axes[0].set_title("ETH/USDC Price with Volatility Regime Shading")

    axes[1].plot(cex.index, vol * 100, color="black", lw=0.5, alpha=0.7)
    for regime, color in REGIME_COLORS.items():
        mask = reg["regime"] == regime
        axes[1].fill_between(cex.index, 0, (vol * 100).max(),
                             where=mask.reindex(cex.index, fill_value=False),
                             color=color, alpha=0.20,
                             label=f"{regime.capitalize()} vol")
    axes[1].set_ylabel("Realized vol 24h (ann., %)")
    patches = [mpatches.Patch(color=c, alpha=0.5, label=r.capitalize())
               for r, c in REGIME_COLORS.items()]
    axes[1].legend(handles=patches, loc="upper right")
    savefig("regime_time_series")

    # ── Figure 2: Vol distribution by regime ─────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    for regime, color in REGIME_COLORS.items():
        sub = reg[reg["regime"] == regime]["vol"].dropna() * 100
        ax.hist(sub, bins=60, density=True, color=color, alpha=0.55,
                label=f"{regime.capitalize()} (n={len(sub):,})")
    ax.set_xlabel("24h realized vol (annualized, %)")
    ax.set_ylabel("Density")
    ax.set_title("Volatility Distribution by Regime")
    ax.legend()
    savefig("regime_vol_distribution")

    # ── Collect metrics per regime ────────────────────────────────
    metrics: dict[str, pd.Series] = {"vol_24h_ann": vol}

    if dex is not None:
        for col in ["fee_apr_ann", "vol_over_tvl", "tx_count", "tvl_usd"]:
            if col in dex.columns:
                metrics[col] = pd.to_numeric(dex[col], errors="coerce")

    if lvr is not None:
        for col in ["lvr_rate_ann", "lvr_to_fee_ratio"]:
            if col in lvr.columns:
                metrics[col] = pd.to_numeric(lvr[col], errors="coerce")

    if merged is not None and "dex_cex_basis_bps" in merged.columns:
        metrics["basis_abs_bps"] = pd.to_numeric(
            merged["dex_cex_basis_bps"], errors="coerce").abs()

    aligned = pd.DataFrame(metrics)
    aligned["regime"] = reg["regime"].reindex(aligned.index)
    aligned["stress"] = reg["stress"].reindex(aligned.index)
    aligned = aligned.dropna(subset=["regime"])

    # ── Figure 3: Key metrics by regime (bar chart) ───────────────
    bar_metrics = {k: v for k, v in {
        "fee_apr_ann":     "Fee APR (ann.)",
        "lvr_rate_ann":    "LVR rate (ann.)",
        "tx_count":        "Swap count / hour",
        "basis_abs_bps":   "|Basis| (bps)",
    }.items() if k in aligned.columns}

    if bar_metrics:
        fig, axes = plt.subplots(1, len(bar_metrics), figsize=(3.5 * len(bar_metrics), 4))
        if len(bar_metrics) == 1:
            axes = [axes]
        for ax, (col, label) in zip(axes, bar_metrics.items()):
            grp = aligned.groupby("regime", observed=True)[col].median()
            ax.bar(grp.index, grp.values,
                   color=[REGIME_COLORS[r] for r in grp.index], alpha=0.8)
            ax.set_title(label)
            ax.set_xlabel("Regime")
            ax.tick_params(axis="x", rotation=15)
        fig.suptitle("Median Key Metrics by Volatility Regime")
        savefig("regime_metrics_bar")

    # ── Table: full regime stats ──────────────────────────────────
    rows = []
    for regime in ["low", "normal", "high"]:
        sub = aligned[aligned["regime"] == regime]
        row = {"Regime": regime, "N_hours": len(sub)}
        for col in metrics:
            if col in sub.columns:
                row[f"{col}_mean"]   = round(sub[col].mean(), 6)
                row[f"{col}_median"] = round(sub[col].median(), 6)
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("Regime")
    savetable(tab, "regime_stats")

    # ── Table: stress episodes ────────────────────────────────────
    stress_mask = reg["stress"]
    # Count consecutive stress blocks
    blocks, n, total_hrs = 0, 0, int(stress_mask.sum())
    prev = False
    for v in stress_mask:
        if v and not prev:
            blocks += 1
        prev = v

    stress_vol   = vol[stress_mask]
    stress_sub   = aligned[aligned["stress"]]
    stress_tab = pd.DataFrame([{
        "Total stress hours":       total_hrs,
        "Stress fraction":          f"{stress_mask.mean()*100:.1f}%",
        "Vol threshold (90th pct)": f"{vol.quantile(0.90)*100:.1f}%",
        "N stress episodes":        blocks,
        "Mean episode length (h)":  round(total_hrs / blocks, 1) if blocks > 0 else None,
        "Mean vol in stress":       f"{stress_vol.mean()*100:.1f}%",
        "Mean fee APR in stress":   f"{stress_sub['fee_apr_ann'].mean()*100:.2f}%" if "fee_apr_ann" in stress_sub else "--",
        "Mean LVR rate in stress":  f"{stress_sub['lvr_rate_ann'].mean()*100:.4f}%" if "lvr_rate_ann" in stress_sub else "--",
    }], index=["Value"]).T
    savetable(stress_tab, "stress_events")

    print(f"\n  Stress episodes: {blocks}")
    print(f"  Vol threshold (90th pct): {vol.quantile(0.90)*100:.1f}%")
    print("\nDONE")


if __name__ == "__main__":
    main()
