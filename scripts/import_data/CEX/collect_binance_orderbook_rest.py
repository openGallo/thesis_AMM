r"""
Collect live Binance order-book snapshots through the public REST endpoint.

Purpose:
    - prospectively collect bid-ask spreads
    - prospectively collect CEX depth within 1, 5, 10, and 50 bps
    - save both full snapshots and compact summaries

Default output:
    C:\Courses\thesis_AMM\data_raw\CEX\binance\live_orderbook\<SYMBOL>\

Examples:
    python collect_binance_orderbook_rest.py --symbols ETHUSDC --seconds 60 --interval-sec 10
    python collect_binance_orderbook_rest.py --symbols ETHUSDC,ETHUSDT --seconds 0 --interval-sec 10
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from cex_config import BINANCE_OUTPUT_ROOT, BINANCE_SPOT_API_BASE_URL, REQUEST_TIMEOUT_SECONDS


DEFAULT_DEPTH_BPS = [1, 5, 10, 50]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().replace(microsecond=0).isoformat()


def fetch_depth(symbol: str, limit: int = 1000) -> dict[str, Any]:
    url = f"{BINANCE_SPOT_API_BASE_URL}/api/v3/depth"
    params = {"symbol": symbol.upper().strip(), "limit": limit}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def notional_depth_within_bps(
    levels: list[list[str]],
    mid_price: float,
    side: str,
    bps: float,
) -> float:
    """
    Compute quote notional depth within a basis-point distance from mid.

    For bids, include price >= mid * (1 - bps/10000).
    For asks, include price <= mid * (1 + bps/10000).
    """
    if side not in {"bid", "ask"}:
        raise ValueError("side must be 'bid' or 'ask'")

    threshold = mid_price * (1 - bps / 10000.0) if side == "bid" else mid_price * (1 + bps / 10000.0)

    total = 0.0
    for price_str, qty_str in levels:
        price = float(price_str)
        qty = float(qty_str)
        include = price >= threshold if side == "bid" else price <= threshold
        if include:
            total += price * qty
    return total


def summarize_depth(symbol: str, raw_depth: dict[str, Any], depth_bps: list[int]) -> dict[str, Any]:
    bids = raw_depth.get("bids", [])
    asks = raw_depth.get("asks", [])

    if not bids or not asks:
        raise ValueError(f"No bids or asks returned for {symbol}")

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = 0.5 * (best_bid + best_ask)
    spread_abs = best_ask - best_bid
    spread_bps = 10000.0 * spread_abs / mid if mid > 0 else None

    row: dict[str, Any] = {
        "timestamp_utc": utc_now_iso(),
        "exchange": "binance",
        "symbol": symbol.upper().strip(),
        "last_update_id": raw_depth.get("lastUpdateId"),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread_abs": spread_abs,
        "spread_bps": spread_bps,
        "bid_levels_returned": len(bids),
        "ask_levels_returned": len(asks),
    }

    for bps in depth_bps:
        row[f"bid_depth_quote_{bps}bps"] = notional_depth_within_bps(bids, mid, "bid", bps)
        row[f"ask_depth_quote_{bps}bps"] = notional_depth_within_bps(asks, mid, "ask", bps)

    return row


def append_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_json_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")


def collect_orderbook_snapshots(
    symbols: list[str],
    output_root: Path = BINANCE_OUTPUT_ROOT,
    seconds: int = 60,
    interval_sec: float = 10.0,
    limit: int = 1000,
    depth_bps: list[int] | None = None,
    save_full_snapshots: bool = True,
) -> None:
    """
    Collect snapshots for a fixed number of seconds.

    If seconds = 0, run until Ctrl+C.
    """
    if depth_bps is None:
        depth_bps = DEFAULT_DEPTH_BPS

    symbols = [s.upper().strip() for s in symbols if s.strip()]
    started = time.time()
    run_forever = seconds == 0

    print("Starting Binance order-book collector.")
    print(f"Symbols: {symbols}")
    print(f"Output root: {output_root}")
    print("Press Ctrl+C to stop.")

    try:
        while run_forever or (time.time() - started < seconds):
            date_key = utc_now().strftime("%Y%m%d")

            for symbol in symbols:
                try:
                    raw = fetch_depth(symbol=symbol, limit=limit)
                    row = summarize_depth(symbol=symbol, raw_depth=raw, depth_bps=depth_bps)

                    symbol_dir = output_root / "live_orderbook" / symbol
                    summary_path = symbol_dir / f"depth_summary_{date_key}.csv"
                    append_csv_row(summary_path, row)

                    if save_full_snapshots:
                        payload = {
                            "timestamp_utc": row["timestamp_utc"],
                            "exchange": "binance",
                            "symbol": symbol,
                            "depth_limit": limit,
                            "depth": raw,
                        }
                        snapshot_path = symbol_dir / f"depth_snapshots_{date_key}.ndjson"
                        append_json_line(snapshot_path, payload)

                    print(
                        f"[{row['timestamp_utc']}] {symbol} "
                        f"mid={row['mid_price']:.6f} spread_bps={row['spread_bps']:.4f}"
                    )

                except Exception as exc:
                    print(f"[ERROR] {symbol}: {type(exc).__name__}: {exc}")

            time.sleep(interval_sec)

    except KeyboardInterrupt:
        print("Stopped by user.")


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_bps_arg(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ETHUSDC")
    parser.add_argument("--seconds", type=int, default=60, help="0 = run until Ctrl+C")
    parser.add_argument("--interval-sec", type=float, default=10.0)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--depth-bps", default="1,5,10,50")
    parser.add_argument("--output-root", default=str(BINANCE_OUTPUT_ROOT))
    parser.add_argument("--no-full-snapshots", action="store_true")
    args = parser.parse_args()

    collect_orderbook_snapshots(
        symbols=parse_csv_arg(args.symbols),
        output_root=Path(args.output_root),
        seconds=args.seconds,
        interval_sec=args.interval_sec,
        limit=args.limit,
        depth_bps=parse_bps_arg(args.depth_bps),
        save_full_snapshots=not args.no_full_snapshots,
    )


if __name__ == "__main__":
    main()
