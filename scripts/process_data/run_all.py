"""
Run the full data processing pipeline in the correct dependency order.

Usage:
    python scripts/process_data/run_all.py
    python scripts/process_data/run_all.py --skip-merged   # skip if CEX data not yet ready

Order:
    1. CEX price panel        (no dependencies)
    2. CEX order-book summary (no dependencies)
    3. DEX pool hourly/daily  (no dependencies)
    4. DEX swap panel         (no dependencies)
    5. DEX LP positions       (no dependencies)
    6. Merged DEX+CEX panel   (requires 1 and 3)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # propagate UTF-8 to child processes

SCRIPT_DIR = Path(__file__).parent

PIPELINE: list[tuple[str, str]] = [
    # CEX (no dependencies)
    ("CEX/process_cex_price.py",           "CEX price panel (1m + hourly)"),
    ("CEX/process_cex_orderbook.py",       "CEX order-book daily summary"),
    # DEX (no dependencies)
    ("DEX/process_dex_pool_hourly.py",     "DEX pool hourly/daily"),
    ("DEX/process_dex_swaps.py",           "DEX swap panel"),
    ("DEX/process_dex_lp_positions.py",    "DEX LP position P&L"),
    # Merged (requires CEX + DEX pool)
    ("merged/process_merged_panel.py",     "Merged DEX+CEX hourly panel"),
    # Derived analytics (requires DEX pool + CEX price)
    ("DEX/process_dex_lvr.py",             "DEX LVR (Loss-Versus-Rebalancing)"),
    # Calibration (requires all of the above)
    ("calibration/process_calibration.py", "Simulation calibration parameters"),
]


def run_script(rel_path: str, label: str) -> int:
    path = SCRIPT_DIR / rel_path
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  {path}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        print(f"\n[FAILED] {rel_path} exited with code {result.returncode}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all processing scripts.")
    parser.add_argument(
        "--skip-merged",
        action="store_true",
        help="Skip merged panel (useful when CEX data not yet collected)",
    )
    args = parser.parse_args()

    pipeline = [
        (p, l) for p, l in PIPELINE
        if not (args.skip_merged and "merged" in p)
    ]

    failed = []
    for rel_path, label in pipeline:
        rc = run_script(rel_path, label)
        if rc != 0:
            failed.append(rel_path)

    print(f"\n{'='*60}")
    if failed:
        print(f"COMPLETED WITH ERRORS - {len(failed)} script(s) failed:")
        for f in failed:
            print(f"  * {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(pipeline)} SCRIPTS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
