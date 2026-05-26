"""
Build 1-minute and hourly CEX price panels from raw Binance klines.

Inputs (auto-discovered):
    data_raw/CEX/binance/spot/daily/klines/{SYMBOL}/1m/*.csv
    data_raw/CEX/binance/spot/monthly/klines/{SYMBOL}/1m/*.csv
    Raw Binance CSVs have no header row.

Key variables produced:
    eth_usdc_price      — direct ETHUSDC close, or synthetic = ETHUSDT / USDCUSDT
    eth_usdc_synthetic  — always-available synthetic reference
    log_return_1m       — per-bar log return on eth_usdc_price
    realized_vol_*_ann  — rolling annualized realized vol (1h / 24h / 7d window)

Outputs:
    data_processed/CEX/cex_price_1m.csv
    data_processed/CEX/cex_price_hourly.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data_raw"
DATA_OUT     = PROJECT_ROOT / "data_processed" / "CEX"

KLINE_COLUMNS = [
    "open_time_ms", "open", "high", "low", "close", "volume_base",
    "close_time_ms", "quote_asset_volume", "number_of_trades",
    "taker_buy_base_volume", "taker_buy_quote_volume", "ignore",
]

_MINY = 365.25 * 24 * 60  # minutes per year


def load_klines(symbol: str, interval: str = "1m") -> pd.DataFrame:
    """Combine daily + monthly kline CSVs, deduplicate, sort by open_time_ms."""
    frames: list[pd.DataFrame] = []
    for gran in ("daily", "monthly"):
        folder = DATA_RAW / "CEX" / "binance" / "spot" / gran / "klines" / symbol / interval
        if not folder.exists():
            continue
        for f in sorted(folder.glob("*.csv")):
            try:
                frames.append(pd.read_csv(f, header=None, names=KLINE_COLUMNS, dtype=str))
            except Exception as exc:
                print(f"  [WARN] {f.name}: {exc}")

    if not frames:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.concat(frames, ignore_index=True)
    df["open_time_ms"] = pd.to_numeric(df["open_time_ms"], errors="coerce")
    df = df.dropna(subset=["open_time_ms"])
    df["open_time_ms"] = df["open_time_ms"].astype("int64")
    for col in ("open", "high", "low", "close", "volume_base",
                "quote_asset_volume", "number_of_trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.drop_duplicates("open_time_ms").sort_values("open_time_ms").reset_index(drop=True)


def to_series(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty or col not in df.columns:
        return pd.Series(dtype=float, name=col)
    idx = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    return pd.Series(df[col].values, index=idx, name=col)


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CEX Price Processing — Binance 1m klines")
    print("=" * 60)

    print("Loading ETHUSDC 1m klines...")
    df_ethusdc  = load_klines("ETHUSDC")
    print(f"  {len(df_ethusdc):,} bars")

    print("Loading ETHUSDT 1m klines...")
    df_ethusdt  = load_klines("ETHUSDT")
    print(f"  {len(df_ethusdt):,} bars")

    print("Loading USDCUSDT 1m klines...")
    df_usdcusdt = load_klines("USDCUSDT")
    print(f"  {len(df_usdcusdt):,} bars")

    s_ethusdc  = to_series(df_ethusdc,  "close")
    s_ethusdt  = to_series(df_ethusdt,  "close")
    s_usdcusdt = to_series(df_usdcusdt, "close")

    all_idx = s_ethusdc.index.union(s_ethusdt.index).union(s_usdcusdt.index)
    if all_idx.empty:
        raise RuntimeError("No kline data found — run main_import_cex.py first.")

    print(f"\nBuilding 1m panel: {all_idx.min()} → {all_idx.max()}")

    p = pd.DataFrame(index=all_idx)
    p.index.name = "timestamp_utc"
    p["ethusdc_close"]      = s_ethusdc.reindex(p.index)
    p["ethusdt_close"]      = s_ethusdt.reindex(p.index)
    p["usdcusdt_close"]     = s_usdcusdt.reindex(p.index)

    # ETHUSDC is unavailable on Binance before ~2022-08; fill with synthetic
    p["eth_usdc_synthetic"] = p["ethusdt_close"] / p["usdcusdt_close"]
    p["eth_usdc_price"]     = p["ethusdc_close"].fillna(p["eth_usdc_synthetic"])

    # Log returns and rolling realized volatility (annualized)
    lr = np.log(p["eth_usdc_price"] / p["eth_usdc_price"].shift(1))
    p["log_return_1m"] = lr
    f = np.sqrt(_MINY)
    p["realized_vol_1h_ann"]  = lr.rolling(60,    min_periods=30).std()   * f
    p["realized_vol_24h_ann"] = lr.rolling(1440,  min_periods=720).std()  * f
    p["realized_vol_7d_ann"]  = lr.rolling(10080, min_periods=3600).std() * f

    # Volume from ETHUSDT (the most liquid pair)
    if not df_ethusdt.empty:
        p["volume_base_ethusdt"]    = to_series(df_ethusdt, "volume_base").reindex(p.index)
        p["n_trades_ethusdt"]       = to_series(df_ethusdt, "number_of_trades").reindex(p.index)

    out_1m = DATA_OUT / "cex_price_1m.csv"
    p.to_csv(out_1m)
    print(f"Saved {out_1m}  ({len(p):,} rows, {out_1m.stat().st_size / 1e6:.0f} MB)")

    # ── Hourly resample ───────────────────────────────────────────────────────
    price = p["eth_usdc_price"].dropna()
    h = price.resample("1h").ohlc()
    h.columns = pd.Index(["eth_usdc_open", "eth_usdc_high", "eth_usdc_low", "eth_usdc_close"])
    h.index.name = "timestamp_utc"

    h["log_return_1h"]        = np.log(h["eth_usdc_close"] / h["eth_usdc_close"].shift(1))
    h["realized_vol_1h_ann"]  = p["realized_vol_1h_ann"].resample("1h").last()
    h["realized_vol_24h_ann"] = p["realized_vol_24h_ann"].resample("1h").last()
    h["realized_vol_7d_ann"]  = p["realized_vol_7d_ann"].resample("1h").last()

    if "volume_base_ethusdt" in p.columns:
        h["vol_base_1h_ethusdt"] = p["volume_base_ethusdt"].resample("1h").sum()
        h["n_trades_1h_ethusdt"] = p["n_trades_ethusdt"].resample("1h").sum()

    out_hourly = DATA_OUT / "cex_price_hourly.csv"
    h.to_csv(out_hourly)
    print(f"Saved {out_hourly}  ({len(h):,} rows)")

    print("\nDONE")


if __name__ == "__main__":
    main()
