"""
Aggregate live Binance order-book snapshots into daily spread and depth summaries.

Inputs (auto-discovered):
    data_raw/CEX/binance/live_orderbook/{SYMBOL}/depth_summary_*.csv
    Columns written by collect_binance_orderbook_rest.py:
        timestamp_utc, exchange, symbol, last_update_id,
        best_bid, best_ask, mid_price, spread_abs, spread_bps,
        bid_levels_returned, ask_levels_returned,
        bid_depth_quote_{1,5,10,50}bps, ask_depth_quote_{1,5,10,50}bps

Key variables produced:
    spread_bps_mean / median / p95  - intraday spread distribution
    bid/ask_depth_{N}bps_mean       - average notional depth within N bps of mid

Output:
    data_processed/CEX/cex_orderbook_daily.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data_raw"
DATA_OUT     = PROJECT_ROOT / "data_processed" / "CEX"

DEPTH_BPS = [1, 5, 10, 50]


def load_depth_summaries(symbol_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(symbol_dir.glob("depth_summary_*.csv")):
        try:
            frames.append(pd.read_csv(f))
        except Exception as exc:
            print(f"  [WARN] {f.name}: {exc}")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    return df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc").reset_index(drop=True)


def daily_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = df["timestamp_utc"].dt.date

    grp = df.groupby(["date", "symbol"], sort=True)

    result = (
        grp["timestamp_utc"]
        .count()
        .rename("n_snapshots")
        .reset_index()
    )

    for stat_col, src, func in [
        ("spread_bps_mean",   "spread_bps", "mean"),
        ("spread_bps_median", "spread_bps", "median"),
    ]:
        result[stat_col] = grp[src].agg(func).values

    # p95 requires separate call; quantile not universally supported in named agg
    p95 = (
        grp["spread_bps"]
        .quantile(0.95)
        .rename("spread_bps_p95")
        .reset_index()
    )
    result = result.merge(p95, on=["date", "symbol"], how="left")

    for bps in DEPTH_BPS:
        for side in ("bid", "ask"):
            src = f"{side}_depth_quote_{bps}bps"
            if src in df.columns:
                result[f"{side}_depth_{bps}bps_mean"] = grp[src].mean().values

    return result


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CEX Order-Book Processing")
    print("=" * 60)

    ob_root = DATA_RAW / "CEX" / "binance" / "live_orderbook"
    if not ob_root.exists():
        print("[INFO] No live_orderbook data found.")
        print("       Run collect_binance_orderbook_rest.py first to collect prospective snapshots.")
        return

    symbol_dirs = [d for d in sorted(ob_root.iterdir()) if d.is_dir()]
    if not symbol_dirs:
        print("[INFO] No symbol directories found in live_orderbook/.")
        return

    frames = []
    for sd in symbol_dirs:
        print(f"  Loading {sd.name}...")
        df = load_depth_summaries(sd)
        if df.empty:
            print(f"  [WARN] No snapshots for {sd.name}")
        else:
            print(f"  {len(df):,} snapshots")
            frames.append(df)

    if not frames:
        print("No order-book data to process.")
        return

    combined = pd.concat(frames, ignore_index=True)
    daily    = daily_stats(combined)

    out = DATA_OUT / "cex_orderbook_daily.csv"
    daily.to_csv(out, index=False)
    n_sym = daily["symbol"].nunique() if "symbol" in daily.columns else "?"
    print(f"\nSaved {out}  ({len(daily):,} daily rows, {n_sym} symbol(s))")
    print("\nDONE")


if __name__ == "__main__":
    main()
