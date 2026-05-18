r"""
MAIN CEX IMPORT SCRIPT

Put this file in:
    C:\Courses\thesis_AMM\scripts\import_data\CEX

Default behavior:
    TEST IMPORT ONLY.

The parameters below are deliberately small:
    - one symbol: ETHUSDC
    - two days: 2024-01-01 to 2024-01-02
    - daily Binance files
    - 1m and 5m klines
    - aggTrades for the same short period

Raw output goes to:
    C:\Courses\thesis_AMM\data_raw\CEX\binance

After the test works, change the USER PARAMETERS section or run with CLI
arguments. For a full import, use monthly files, not daily files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cex_config import BINANCE_OUTPUT_ROOT, DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS
from fetch_binance_agg_trades import download_binance_agg_trades
from fetch_binance_klines import download_binance_klines
from collect_binance_orderbook_rest import collect_orderbook_snapshots


# =====================================================================
# USER PARAMETERS — EDIT THESE FIRST
# =====================================================================

# Output folder. This should remain aligned with your thesis folder structure.
OUTPUT_ROOT = BINANCE_OUTPUT_ROOT

# Test import: small, safe, and fast.
# END_DATE is inclusive for daily imports.
START_DATE = "2024-01-01"
END_DATE = "2024-01-02"

# Use "daily" for tests. Use "monthly" for full historical imports.
ARCHIVE_GRANULARITY = "daily"

# For a first test, use ETHUSDC only.
# For the full thesis import, use:
#     SYMBOLS = ["ETHUSDC", "ETHUSDT", "USDCUSDT"]
SYMBOLS = ["ETHUSDC"]

# Kline intervals needed for reference price and realized volatility.
INTERVALS = ["1m", "5m"]

# Data blocks to download.
DOWNLOAD_KLINES = True
DOWNLOAD_AGG_TRADES = True

# Keep original ZIPs for reproducibility and extract raw CSVs for easier later processing.
EXTRACT_ZIP = True
KEEP_ZIP = True

# Do not overwrite files unless you intentionally want to refresh them.
OVERWRITE = False

# Public-data pause between files.
SLEEP_BETWEEN_DOWNLOADS_SECONDS = DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS

# Live order-book collection is OFF by default.
# Turn it on only when you want to collect spread/depth prospectively.
COLLECT_LIVE_ORDERBOOK = False
ORDERBOOK_SYMBOLS = ["ETHUSDC"]
ORDERBOOK_SECONDS = 60          # 0 = run until Ctrl+C
ORDERBOOK_INTERVAL_SEC = 10.0
ORDERBOOK_LIMIT = 1000
ORDERBOOK_DEPTH_BPS = [1, 5, 10, 50]
ORDERBOOK_SAVE_FULL_SNAPSHOTS = True


# =====================================================================
# OPTIONAL PRESET FOR FULL IMPORT
# =====================================================================

def apply_full_history_preset() -> dict:
    """
    Full historical preset.

    Use with:
        python main_import_cex.py --full-history

    Notes:
        - Uses monthly files to avoid thousands of daily downloads.
        - ETHUSDC begins later than 2022; missing earlier files will be
          recorded as missing_404 in the manifest.
        - You may set END_DATE to 2026-04-30 for your thesis data cut.
    """
    return {
        "start_date": "2022-01-01",
        "end_date": "2026-04-30",
        "granularity": "monthly",
        "symbols": ["ETHUSDC", "ETHUSDT", "USDCUSDT"],
        "intervals": ["1m", "5m"],
        "download_klines": True,
        "download_agg_trades": True,
    }


# =====================================================================
# SCRIPT LOGIC
# =====================================================================

def print_configuration(cfg: dict) -> None:
    print("=" * 72)
    print("CEX IMPORT CONFIGURATION")
    print("=" * 72)
    for key, value in cfg.items():
        print(f"{key:>30}: {value}")
    print("=" * 72)


def run_import(cfg: dict) -> None:
    print_configuration(cfg)

    output_root = Path(cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    if cfg["download_klines"]:
        print("\n--- Downloading Binance klines ---")
        download_binance_klines(
            symbols=cfg["symbols"],
            intervals=cfg["intervals"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            granularity=cfg["granularity"],
            output_root=output_root,
            overwrite=cfg["overwrite"],
            extract=cfg["extract_zip"],
            keep_zip=cfg["keep_zip"],
            sleep_seconds=cfg["sleep_between_downloads_seconds"],
        )

    if cfg["download_agg_trades"]:
        print("\n--- Downloading Binance aggTrades ---")
        download_binance_agg_trades(
            symbols=cfg["symbols"],
            start_date=cfg["start_date"],
            end_date=cfg["end_date"],
            granularity=cfg["granularity"],
            output_root=output_root,
            overwrite=cfg["overwrite"],
            extract=cfg["extract_zip"],
            keep_zip=cfg["keep_zip"],
            sleep_seconds=cfg["sleep_between_downloads_seconds"],
        )

    if cfg["collect_live_orderbook"]:
        print("\n--- Collecting live Binance order-book snapshots ---")
        collect_orderbook_snapshots(
            symbols=cfg["orderbook_symbols"],
            output_root=output_root,
            seconds=cfg["orderbook_seconds"],
            interval_sec=cfg["orderbook_interval_sec"],
            limit=cfg["orderbook_limit"],
            depth_bps=cfg["orderbook_depth_bps"],
            save_full_snapshots=cfg["orderbook_save_full_snapshots"],
        )

    print("\nDone.")
    print(f"Raw data folder: {output_root}")
    print(f"Manifest: {output_root / 'download_manifest.csv'}")


def parse_csv_arg(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def build_config_from_user_parameters() -> dict:
    return {
        "output_root": OUTPUT_ROOT,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "granularity": ARCHIVE_GRANULARITY,
        "symbols": SYMBOLS,
        "intervals": INTERVALS,
        "download_klines": DOWNLOAD_KLINES,
        "download_agg_trades": DOWNLOAD_AGG_TRADES,
        "extract_zip": EXTRACT_ZIP,
        "keep_zip": KEEP_ZIP,
        "overwrite": OVERWRITE,
        "sleep_between_downloads_seconds": SLEEP_BETWEEN_DOWNLOADS_SECONDS,
        "collect_live_orderbook": COLLECT_LIVE_ORDERBOOK,
        "orderbook_symbols": ORDERBOOK_SYMBOLS,
        "orderbook_seconds": ORDERBOOK_SECONDS,
        "orderbook_interval_sec": ORDERBOOK_INTERVAL_SEC,
        "orderbook_limit": ORDERBOOK_LIMIT,
        "orderbook_depth_bps": ORDERBOOK_DEPTH_BPS,
        "orderbook_save_full_snapshots": ORDERBOOK_SAVE_FULL_SNAPSHOTS,
    }


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.full_history:
        preset = apply_full_history_preset()
        cfg.update(preset)

    if args.start is not None:
        cfg["start_date"] = args.start
    if args.end is not None:
        cfg["end_date"] = args.end
    if args.granularity is not None:
        cfg["granularity"] = args.granularity
    if args.symbols is not None:
        cfg["symbols"] = parse_csv_arg(args.symbols)
    if args.intervals is not None:
        cfg["intervals"] = parse_csv_arg(args.intervals)
    if args.output_root is not None:
        cfg["output_root"] = Path(args.output_root)

    if args.no_klines:
        cfg["download_klines"] = False
    if args.no_agg_trades:
        cfg["download_agg_trades"] = False
    if args.overwrite:
        cfg["overwrite"] = True
    if args.no_extract:
        cfg["extract_zip"] = False
    if args.delete_zip_after_extract:
        cfg["keep_zip"] = False
    if args.collect_orderbook:
        cfg["collect_live_orderbook"] = True

    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Main import script for free CEX data used in the AMM thesis."
    )

    parser.add_argument("--full-history", action="store_true", help="Use the full historical preset.")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD; inclusive for daily files")
    parser.add_argument("--granularity", choices=["daily", "monthly"], default=None)
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols, e.g. ETHUSDC,ETHUSDT")
    parser.add_argument("--intervals", default=None, help="Comma-separated kline intervals, e.g. 1m,5m")
    parser.add_argument("--output-root", default=None)

    parser.add_argument("--no-klines", action="store_true")
    parser.add_argument("--no-agg-trades", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--delete-zip-after-extract", action="store_true")

    parser.add_argument("--collect-orderbook", action="store_true")

    args = parser.parse_args()

    cfg = build_config_from_user_parameters()
    cfg = apply_cli_overrides(cfg, args)

    run_import(cfg)


if __name__ == "__main__":
    main()
