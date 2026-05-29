"""
Run the full CEX import (Binance klines + aggTrades, 2022-01-01 to 2026-04-30).

Usage:
    python scripts/import_data/run_cex.py               # full history (default)
    python scripts/import_data/run_cex.py --orderbook   # also collect live order-book snapshots

Symbols downloaded:
    ETHUSDC, ETHUSDT, USDCUSDT  (monthly files, 1m + 5m klines)

Note:
    ETHUSDC was listed on Binance in August 2022. Missing months before that
    are recorded as 404 in data_raw/CEX/binance/download_manifest.csv — this is expected.
    The processing script uses ETHUSDT / USDCUSDT as a synthetic proxy for earlier months.

Output:
    data_raw/CEX/binance/
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # propagate UTF-8 to child processes

CEX_DIR = Path(__file__).parent / "CEX"
MAIN_SCRIPT = CEX_DIR / "main_import_cex.py"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full CEX data import from Binance.")
    parser.add_argument(
        "--orderbook",
        action="store_true",
        help="Also collect live order-book depth snapshots (runs until Ctrl+C or --ob-seconds elapses)",
    )
    parser.add_argument(
        "--ob-seconds",
        type=int,
        default=0,
        help="How many seconds to collect order-book snapshots (0 = until Ctrl+C)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist locally",
    )
    args = parser.parse_args()

    cmd = [sys.executable, str(MAIN_SCRIPT), "--full-history"]
    if args.overwrite:
        cmd.append("--overwrite")
    if args.orderbook:
        cmd.append("--collect-orderbook")

    print("=" * 60)
    print("  CEX Import — Binance full history")
    print(f"  {MAIN_SCRIPT}")
    print("=" * 60)

    result = subprocess.run(cmd, check=False)

    print(f"\n{'='*60}")
    if result.returncode == 0:
        print("CEX IMPORT COMPLETED SUCCESSFULLY")
    else:
        print(f"[FAILED] exited with code {result.returncode}")
        sys.exit(result.returncode)
    print("=" * 60)


if __name__ == "__main__":
    main()
