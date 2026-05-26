"""
Download Binance historical spot kline data from data.binance.vision.

This script can be run directly, but the recommended entry point is:
    python main_import_cex.py

Direct example:
    python fetch_binance_klines.py --symbols ETHUSDC --intervals 1m,5m --start 2024-01-01 --end 2024-01-02 --granularity daily
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from cex_config import BINANCE_OUTPUT_ROOT, DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS
from binance_download_utils import DownloadResult, download_one_binance_public_file, periods_for_granularity


def download_binance_klines(
    symbols: Iterable[str],
    intervals: Iterable[str],
    start_date: str,
    end_date: str,
    granularity: str,
    output_root: Path = BINANCE_OUTPUT_ROOT,
    overwrite: bool = False,
    extract: bool = True,
    keep_zip: bool = True,
    sleep_seconds: float = DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS,
) -> list[DownloadResult]:
    periods = periods_for_granularity(start_date, end_date, granularity)
    results = []

    for symbol in symbols:
        symbol = symbol.upper().strip()
        for interval in intervals:
            interval = interval.strip()
            for period in periods:
                print(f"[klines] {symbol} {interval} {granularity} {period}")
                result = download_one_binance_public_file(
                    output_root=output_root,
                    data_type="klines",
                    symbol=symbol,
                    interval=interval,
                    period=period,
                    granularity=granularity,
                    overwrite=overwrite,
                    extract=extract,
                    keep_zip=keep_zip,
                    sleep_seconds=sleep_seconds,
                )
                print(f"  -> {result.status}")
                results.append(result)

    return results


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="ETHUSDC")
    parser.add_argument("--intervals", default="1m,5m")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-01-02")
    parser.add_argument("--granularity", choices=["daily", "monthly"], default="daily")
    parser.add_argument("--output-root", default=str(BINANCE_OUTPUT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")
    args = parser.parse_args()

    download_binance_klines(
        symbols=parse_csv_arg(args.symbols),
        intervals=parse_csv_arg(args.intervals),
        start_date=args.start,
        end_date=args.end,
        granularity=args.granularity,
        output_root=Path(args.output_root),
        overwrite=args.overwrite,
        extract=not args.no_extract,
        keep_zip=not args.delete_zip_after_extract,
    )


if __name__ == "__main__":
    main()
