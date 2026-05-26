"""
Extract simulation calibration parameters from all processed datasets.

This script reads the processed data outputs and fits the reduced-form
parameter values needed to initialize and run the AMM simulation described
in the thesis theoretical model.

Parameters extracted:

  Price dynamics:
    mu_daily, sigma_daily      — GBM drift and volatility (daily)
    sigma_hourly               — hourly realized vol (for simulation step calibration)

  Trade arrival (DEX):
    lambda_day, lambda_hour    — Poisson arrival rate (trades per unit time)
    mu_q_lognormal             — log-mean of trade size in USD (lognormal fit)
    sigma_q_lognormal          — log-std of trade size in USD
    p25/p50/p75/p95_trade_usd  — trade size percentiles

  Gas dynamics:
    mean_gas_gwei              — mean gas price during sample (Gwei)
    std_gas_gwei               — std of gas price
    ar1_rho_gas                — AR(1) persistence of log gas price
    mean_gas_usd_per_swap      — average gas cost per swap in USD

  Pool state:
    fee_tier                   — always 0.0005
    median_tvl_usd             — median TVL
    median_liquidity           — median active liquidity (raw units)
    vol_over_tvl_mean          — mean hourly volume / TVL

  Stress regime:
    vol_stress_threshold_ann   — 90th percentile vol (label as "stress")
    stress_fraction            — fraction of hours above the threshold

  LVR:
    lvr_rate_ann_mean          — mean annualized LVR rate (as fraction of TVL)
    lvr_to_fee_ratio_mean      — mean LVR / fee income ratio

Inputs:
    data_processed/DEX/dex_pool_hourly.csv
    data_processed/DEX/dex_swaps.csv
    data_processed/DEX/dex_lvr_hourly.csv         (optional)
    data_processed/CEX/cex_price_hourly.csv

Outputs:
    data_processed/calibration/calibration_params.json
    data_processed/calibration/calibration_summary.csv
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_PROC    = PROJECT_ROOT / "data_processed"
DATA_OUT     = DATA_PROC / "calibration"


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        return None if np.isnan(v) or np.isinf(v) else round(v, 8)
    except Exception:
        return None


def fit_ar1(series: pd.Series) -> tuple[float, float]:
    """Return (rho, sigma_eps) for a log-price AR(1) model."""
    log_s = np.log(series.dropna().clip(lower=1e-10))
    if len(log_s) < 10:
        return float("nan"), float("nan")
    y = log_s.iloc[1:].values
    x = log_s.iloc[:-1].values
    rho = float(np.corrcoef(x, y)[0, 1])
    resid = y - rho * x
    return rho, float(np.std(resid))


def fit_lognormal(series: pd.Series) -> tuple[float, float]:
    """Return (mu, sigma) of the lognormal fit to positive values."""
    s = series.dropna()
    s = s[s > 0]
    if len(s) < 10:
        return float("nan"), float("nan")
    log_s = np.log(s)
    return float(log_s.mean()), float(log_s.std())


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Simulation Calibration Parameter Extraction")
    print("=" * 60)

    params: dict = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pool_address":     "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640",
        "fee_tier":         0.0005,
    }
    missing: list[str] = []

    # ── CEX price dynamics ────────────────────────────────────────────────────
    cex_path = DATA_PROC / "CEX" / "cex_price_hourly.csv"
    if cex_path.exists():
        cex = pd.read_csv(cex_path, index_col=0, parse_dates=True)
        cex.index = pd.to_datetime(cex.index, utc=True)

        lr = cex["log_return_1h"].dropna()
        sigma_hourly = float(lr.std())
        sigma_daily  = sigma_hourly * np.sqrt(24)
        mu_daily     = float(lr.mean()) * 24

        params["price_dynamics"] = {
            "period_start":   str(cex.index.min()),
            "period_end":     str(cex.index.max()),
            "mu_daily":       _safe_float(mu_daily),
            "sigma_daily":    _safe_float(sigma_daily),
            "sigma_hourly":   _safe_float(sigma_hourly),
            "vol_24h_ann_mean": _safe_float(
                cex["realized_vol_24h_ann"].mean() if "realized_vol_24h_ann" in cex.columns else None
            ),
        }
        print(f"  Price: σ_daily={sigma_daily:.4f}  μ_daily={mu_daily:.6f}")

        # Stress regime: hours where 24h realized vol > 90th percentile
        if "realized_vol_24h_ann" in cex.columns:
            vol = cex["realized_vol_24h_ann"].dropna()
            threshold = float(vol.quantile(0.90))
            stress_frac = float((vol > threshold).mean())
            params["stress_regime"] = {
                "vol_stress_threshold_ann": _safe_float(threshold),
                "stress_fraction":          _safe_float(stress_frac),
            }
            print(f"  Stress: vol_threshold={threshold:.4f}  stress_frac={stress_frac:.3f}")
    else:
        missing.append("cex_price_hourly.csv — run process_cex_price.py")
        print("[WARN] CEX price data missing")

    # ── DEX pool metrics ──────────────────────────────────────────────────────
    dex_path = DATA_PROC / "DEX" / "dex_pool_hourly.csv"
    if dex_path.exists():
        dex = pd.read_csv(dex_path, index_col=0, parse_dates=True)
        dex.index = pd.to_datetime(dex.index, utc=True)

        params["pool_state"] = {
            "period_start":       str(dex.index.min()),
            "period_end":         str(dex.index.max()),
            "median_tvl_usd":     _safe_float(dex["tvl_usd"].median()),
            "mean_tvl_usd":       _safe_float(dex["tvl_usd"].mean()),
            "median_liquidity":   _safe_float(
                pd.to_numeric(dex["liquidity"], errors="coerce").median()
            ),
            "vol_over_tvl_mean":  _safe_float(dex["vol_over_tvl"].mean()),
            "fee_apr_ann_mean":   _safe_float(dex["fee_apr_ann"].mean()),
        }

        # Trade arrival intensity: tx_count per hour → per day
        lam_hour = float(dex["tx_count"].mean())
        lam_day  = lam_hour * 24
        params["trade_arrival"] = {
            "lambda_hour": _safe_float(lam_hour),
            "lambda_day":  _safe_float(lam_day),
        }
        print(f"  Trade arrival: λ_hour={lam_hour:.1f}  λ_day={lam_day:.0f}")
    else:
        missing.append("dex_pool_hourly.csv — run process_dex_pool_hourly.py")
        print("[WARN] DEX pool hourly data missing")

    # ── Swap-level trade size and gas ─────────────────────────────────────────
    swaps_path = DATA_PROC / "DEX" / "dex_swaps.csv"
    if swaps_path.exists():
        print("Loading swap data for trade-size and gas calibration...")
        swaps = pd.read_csv(swaps_path, usecols=[
            "amount_usd", "gas_price_wei", "gas_cost_eth", "gas_cost_usd",
        ], low_memory=False)

        # Trade size (positive USD notional)
        q = pd.to_numeric(swaps["amount_usd"], errors="coerce").abs().dropna()
        q = q[q > 0]
        mu_q, sig_q = fit_lognormal(q)
        params["trade_size"] = {
            "mu_lognormal":  _safe_float(mu_q),
            "sigma_lognormal": _safe_float(sig_q),
            "mean_usd":      _safe_float(float(q.mean())),
            "median_usd":    _safe_float(float(q.median())),
            "p25_usd":       _safe_float(float(q.quantile(0.25))),
            "p75_usd":       _safe_float(float(q.quantile(0.75))),
            "p95_usd":       _safe_float(float(q.quantile(0.95))),
            "p99_usd":       _safe_float(float(q.quantile(0.99))),
        }
        print(f"  Trade size: median=${q.median():,.0f}  μ_log={mu_q:.3f}  σ_log={sig_q:.3f}")

        # Gas
        gwei = pd.to_numeric(swaps["gas_price_wei"], errors="coerce").dropna() / 1e9
        gwei = gwei[gwei > 0]
        rho_g, sig_g = fit_ar1(gwei)
        gas_usd = pd.to_numeric(swaps["gas_cost_usd"], errors="coerce").dropna()
        gas_usd = gas_usd[gas_usd > 0]
        params["gas_dynamics"] = {
            "mean_gas_gwei":       _safe_float(float(gwei.mean())),
            "median_gas_gwei":     _safe_float(float(gwei.median())),
            "std_gas_gwei":        _safe_float(float(gwei.std())),
            "ar1_rho_log_gas":     _safe_float(rho_g),
            "ar1_sigma_log_gas":   _safe_float(sig_g),
            "mean_gas_usd_per_swap":   _safe_float(float(gas_usd.mean())),
            "median_gas_usd_per_swap": _safe_float(float(gas_usd.median())),
        }
        print(f"  Gas: mean={gwei.mean():.1f} Gwei  ρ={rho_g:.3f}  "
              f"mean_cost=${gas_usd.mean():.2f}")
    else:
        missing.append("dex_swaps.csv — run process_dex_swaps.py")
        print("[WARN] Swap data missing")

    # ── LVR ───────────────────────────────────────────────────────────────────
    lvr_path = DATA_PROC / "DEX" / "dex_lvr_hourly.csv"
    if lvr_path.exists():
        lvr = pd.read_csv(lvr_path, index_col=0, parse_dates=True)
        params["lvr"] = {
            "lvr_rate_ann_mean":    _safe_float(lvr["lvr_rate_ann"].mean()),
            "lvr_rate_ann_median":  _safe_float(lvr["lvr_rate_ann"].median()),
            "lvr_usd_hourly_mean":  _safe_float(lvr["lvr_usd_tvl_approx"].mean()),
            "lvr_to_fee_ratio_mean": _safe_float(
                lvr["lvr_to_fee_ratio"].mean() if "lvr_to_fee_ratio" in lvr.columns else None
            ),
        }
        print(f"  LVR: rate_ann_mean={lvr['lvr_rate_ann'].mean():.4%}")
    else:
        missing.append("dex_lvr_hourly.csv — run process_dex_lvr.py")

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_json = DATA_OUT / "calibration_params.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2, default=str)
    print(f"\nSaved {out_json}")

    # Flat CSV version for quick inspection
    flat: list[dict] = []
    for section, values in params.items():
        if isinstance(values, dict):
            for k, v in values.items():
                flat.append({"section": section, "parameter": k, "value": v})
        else:
            flat.append({"section": "meta", "parameter": section, "value": values})
    out_csv = DATA_OUT / "calibration_summary.csv"
    pd.DataFrame(flat).to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    if missing:
        print(f"\n[WARN] {len(missing)} data source(s) not available yet:")
        for m in missing:
            print(f"  • {m}")

    print("\nDONE")


if __name__ == "__main__":
    main()
