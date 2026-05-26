"""
Trade microstructure analysis.

Figures:
    swap_size_distribution      Swap size histogram + lognormal fit
    swap_size_by_direction      Swap size CDF: buy-ETH vs sell-ETH
    trade_arrival_by_hour       Average swap count by hour of day (UTC)
    gas_price_series            Gas price time series (Gwei)
    gas_cost_vs_size            Gas cost USD vs swap size scatter

Tables:
    microstructure_summary      Trade size lognormal fit + KS test
    arrival_by_hour             Mean swap count by hour of day
    gas_summary                 Gas price and cost statistics
    direction_balance           Buy vs sell counts and volumes
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from analysis_utils import DATA_PROC, COLORS, savefig, savetable

warnings.filterwarnings("ignore")


def main() -> None:
    print("=" * 60)
    print("Trade Microstructure Analysis")
    print("=" * 60)

    swaps_path = DATA_PROC / "DEX" / "dex_swaps.csv"
    if not swaps_path.exists():
        print("  [SKIP] dex_swaps.csv not found")
        return

    print("  Loading swaps...")
    swaps = pd.read_csv(swaps_path, low_memory=False, usecols=lambda c: c in [
        "timestamp", "amount_usd", "gas_price_wei", "gas_cost_usd",
        "direction", "log_price_change", "trade_size_bucket",
    ])
    swaps["timestamp"] = pd.to_datetime(swaps["timestamp"], utc=True, errors="coerce")
    swaps["amount_usd"]    = pd.to_numeric(swaps.get("amount_usd"),    errors="coerce").abs()
    swaps["gas_price_wei"] = pd.to_numeric(swaps.get("gas_price_wei"), errors="coerce")
    swaps["gas_cost_usd"]  = pd.to_numeric(swaps.get("gas_cost_usd"),  errors="coerce")
    swaps["gas_gwei"]      = swaps["gas_price_wei"] / 1e9

    q = swaps["amount_usd"].dropna()
    q = q[q > 0]
    print(f"  Swaps loaded: {len(swaps):,}  |  valid sizes: {len(q):,}")

    # ── Figure 1: Swap size distribution + lognormal fit ─────────
    log_q = np.log(q[q > 1])
    mu_ln, sigma_ln = float(log_q.mean()), float(log_q.std())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Linear scale
    clip_usd = q.quantile(0.99)
    axes[0].hist(q.clip(upper=clip_usd), bins=100, density=True,
                 color=COLORS[0], alpha=0.65, label="Empirical")
    x = np.linspace(1, clip_usd, 400)
    axes[0].plot(x, stats.lognorm.pdf(x, s=sigma_ln, scale=np.exp(mu_ln)),
                 color=COLORS[1], lw=1.8, label=f"Lognormal fit\n$\\mu$={mu_ln:.2f}, $\\sigma$={sigma_ln:.2f}")
    axes[0].set_xlabel("Swap size (USD)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Swap Size Distribution")
    axes[0].legend(fontsize=8)

    # Log scale
    axes[1].hist(log_q, bins=80, density=True, color=COLORS[0], alpha=0.65, label="Empirical")
    xl = np.linspace(log_q.min(), log_q.max(), 300)
    axes[1].plot(xl, stats.norm.pdf(xl, mu_ln, sigma_ln),
                 color=COLORS[1], lw=1.8, label="Normal (log scale)")
    axes[1].set_xlabel("log(Swap size USD)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Log Swap Size Distribution")
    axes[1].legend(fontsize=8)
    savefig("swap_size_distribution")

    # ── Figure 2: CDF by direction ────────────────────────────────
    if "direction" in swaps.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        for direction, color, label in [
            ("buy_eth",  COLORS[0], "Buy ETH (sell USDC)"),
            ("sell_eth", COLORS[1], "Sell ETH (buy USDC)"),
        ]:
            sub = swaps[swaps["direction"] == direction]["amount_usd"].dropna()
            sub = sub[sub > 0].sort_values()
            if len(sub) == 0:
                continue
            ax.plot(sub.clip(upper=sub.quantile(0.995)),
                    np.linspace(0, 1, len(sub)),
                    color=color, lw=1.5, label=f"{label} (n={len(sub):,})")
        ax.set_xscale("log")
        ax.set_xlabel("Swap size (USD, log scale)")
        ax.set_ylabel("CDF")
        ax.set_title("Swap Size CDF by Direction")
        ax.legend()
        savefig("swap_size_by_direction")

    # ── Figure 3: Swap count by hour of day ───────────────────────
    dex_pool_path = DATA_PROC / "DEX" / "dex_pool_hourly.csv"
    if dex_pool_path.exists():
        dex = pd.read_csv(dex_pool_path, index_col=0, parse_dates=True,
                          low_memory=False, usecols=["timestamp_utc", "tx_count"] if
                          "timestamp_utc" in pd.read_csv(dex_pool_path, nrows=0).columns
                          else [0, "tx_count"])
        if "tx_count" in dex.columns:
            tx = pd.to_numeric(dex["tx_count"], errors="coerce")
            by_hour = tx.groupby(dex.index.hour).mean()
            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.bar(by_hour.index, by_hour.values, color=COLORS[0], alpha=0.8)
            ax.set_xlabel("Hour of day (UTC)")
            ax.set_ylabel("Mean swap count")
            ax.set_title("Average Swap Arrival Rate by Hour of Day")
            ax.set_xticks(range(0, 24, 2))
            savefig("trade_arrival_by_hour")
            by_hour_tab = by_hour.rename("mean_tx_count").to_frame()
            savetable(by_hour_tab, "arrival_by_hour")

    # ── Figure 4: Gas price time series ──────────────────────────
    gas_ts = swaps.dropna(subset=["timestamp", "gas_gwei"]).set_index("timestamp")["gas_gwei"]
    gas_hourly = gas_ts.resample("1h").median()
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(gas_hourly.index, gas_hourly, color=COLORS[3], lw=0.5, alpha=0.7)
    ax.plot(gas_hourly.index, gas_hourly.rolling(24 * 7).median(),
            color=COLORS[1], lw=1.2, label="7-day median")
    ax.set_ylabel("Gas price (Gwei)")
    ax.set_title("Gas Price Over Time (hourly median)")
    ax.legend()
    savefig("gas_price_series")

    # ── Figure 5: Gas cost vs swap size ──────────────────────────
    if "gas_cost_usd" in swaps.columns:
        sub = swaps[["amount_usd", "gas_cost_usd"]].dropna()
        sub = sub[(sub["amount_usd"] > 0) & (sub["gas_cost_usd"] > 0)]
        sub = sub[sub["amount_usd"] < sub["amount_usd"].quantile(0.99)]
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(sub["amount_usd"], sub["gas_cost_usd"],
                   s=2, alpha=0.1, color=COLORS[0])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Swap size (USD, log)")
        ax.set_ylabel("Gas cost (USD, log)")
        ax.set_title("Gas Cost vs Swap Size")
        # Break-even line: gas = 0.05% of swap size (pool fee)
        x = np.logspace(0, 7, 200)
        ax.plot(x, x * 0.0005, color=COLORS[1], lw=1.5, ls="--",
                label="0.05% of swap (pool fee)")
        ax.legend()
        savefig("gas_cost_vs_size")

    # ── Table: lognormal fit + KS test ────────────────────────────
    ks_stat, ks_pval = stats.kstest(
        log_q.sample(min(5000, len(log_q)), random_state=42),
        "norm", args=(mu_ln, sigma_ln)
    )
    tab = pd.DataFrame([{
        "N_swaps":         len(q),
        "mean_usd":        round(q.mean(), 2),
        "median_usd":      round(q.median(), 2),
        "p95_usd":         round(q.quantile(0.95), 2),
        "p99_usd":         round(q.quantile(0.99), 2),
        "mu_lognormal":    round(mu_ln, 4),
        "sigma_lognormal": round(sigma_ln, 4),
        "ks_stat":         round(ks_stat, 4),
        "ks_pval":         round(ks_pval, 4),
    }], index=["Swaps"]).T.rename(columns={"Swaps": "Value"})
    savetable(tab, "microstructure_summary")

    # ── Table: gas summary ────────────────────────────────────────
    gas = swaps["gas_gwei"].dropna()
    gas = gas[gas > 0]
    gas_usd = swaps["gas_cost_usd"].dropna() if "gas_cost_usd" in swaps.columns else pd.Series(dtype=float)
    gas_tab = pd.DataFrame([{
        "mean_gas_gwei":         round(gas.mean(), 2),
        "median_gas_gwei":       round(gas.median(), 2),
        "std_gas_gwei":          round(gas.std(), 2),
        "p95_gas_gwei":          round(gas.quantile(0.95), 2),
        "mean_gas_usd_per_swap": round(gas_usd.mean(), 4) if len(gas_usd) > 0 else None,
        "median_gas_usd_per_swap": round(gas_usd.median(), 4) if len(gas_usd) > 0 else None,
    }], index=["Gas"]).T.rename(columns={"Gas": "Value"})
    savetable(gas_tab, "gas_summary")

    # ── Table: direction balance ──────────────────────────────────
    if "direction" in swaps.columns:
        dir_tab = swaps.groupby("direction").agg(
            count=("amount_usd", "count"),
            total_volume_usd=("amount_usd", "sum"),
            mean_size_usd=("amount_usd", "mean"),
        ).round(2)
        savetable(dir_tab, "direction_balance")

    print("\nDONE")


if __name__ == "__main__":
    main()
