"""
Run all DEX import scripts in the correct order.

Usage:
    python scripts/import_data/run_dex.py
    python scripts/import_data/run_dex.py --skip-ticks   # skip tick snapshots (slow)

Requires:
    THEGRAPH_API_KEY environment variable (set with: setx THEGRAPH_API_KEY "your_key")

Order:
    1. Pool time series   — hourly/daily OHLCV, TVL, fees  (no deps)
    2. Monthly events     — swaps, mints, burns             (no deps)
    3. Extra events       — collects, flashes               (no deps)
    4. Positions          — current LP positions            (no deps)
    5. Tick snapshots     — month-end liquidity snapshots   (slow; uses swap CSVs for block lookup)

Output:
    data_raw/DEX/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent / "DEX"

PIPELINE: list[tuple[str, str]] = [
    ("fetch_uniswap_pool_timeseries.py", "Pool time series (hourly/daily OHLCV, TVL, fees)"),
    ("fetch_all_monthly.py",             "Monthly swaps, mints, burns"),
    ("fetch_uniswap_extra_events.py",    "Collects and flashes"),
    ("fetch_uniswap_positions.py",       "Current LP positions"),
    ("fetch_uniswap_tick_snapshots.py",  "Month-end tick liquidity snapshots (slow)"),
]


def run_script(filename: str, label: str) -> int:
    path = SCRIPT_DIR / filename
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        print(f"\n[FAILED] {filename} exited with code {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all DEX import scripts.")
    parser.add_argument(
        "--skip-ticks",
        action="store_true",
        help="Skip tick snapshots (the slowest step; safe to run separately later)",
    )
    args = parser.parse_args()

    import os
    if not os.environ.get("THEGRAPH_API_KEY"):
        print("[ERROR] THEGRAPH_API_KEY is not set.")
        print("  Run: setx THEGRAPH_API_KEY \"your_key\"  then restart your terminal.")
        sys.exit(1)

    pipeline = [
        (f, l) for f, l in PIPELINE
        if not (args.skip_ticks and "tick_snapshots" in f)
    ]

    failed = []
    for filename, label in pipeline:
        rc = run_script(filename, label)
        if rc != 0:
            failed.append(filename)

    print(f"\n{'='*60}")
    if failed:
        print(f"COMPLETED WITH ERRORS — {len(failed)} script(s) failed:")
        for f in failed:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(pipeline)} DEX IMPORT SCRIPTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
