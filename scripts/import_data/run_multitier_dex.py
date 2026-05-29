"""
Run multi-fee-tier DEX import scripts.

Downloads hourly and daily data for the USDC/WETH 0.01%, 0.30%, and 1.00% pools.
The 0.05% pool is already handled by run_dex.py.

Usage:
    py -3 scripts/import_data/run_multitier_dex.py           # all three pools
    py -3 scripts/import_data/run_multitier_dex.py --001     # 0.01% only
    py -3 scripts/import_data/run_multitier_dex.py --030     # 0.30% only
    py -3 scripts/import_data/run_multitier_dex.py --100     # 1.00% only

Requires:
    THEGRAPH_API_KEY environment variable (or a .env file at the project root)

Output:
    data_raw/multitier/fee_100/pool_hour_data.csv
    data_raw/multitier/fee_100/pool_day_data.csv
    data_raw/multitier/fee_100/pool_metadata_current.csv
    data_raw/multitier/fee_3000/pool_hour_data.csv
    data_raw/multitier/fee_3000/pool_day_data.csv
    data_raw/multitier/fee_3000/pool_metadata_current.csv
    data_raw/multitier/fee_10000/pool_hour_data.csv
    data_raw/multitier/fee_10000/pool_day_data.csv
    data_raw/multitier/fee_10000/pool_metadata_current.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# ── Load .env (same pattern as run_dex.py) ───────────────────────────────────

def _load_env() -> None:
    if os.environ.get("THEGRAPH_API_KEY"):
        return
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


_load_env()
os.environ.setdefault("PYTHONIOENCODING", "utf-8")  # propagate UTF-8 to child processes

SCRIPT_DIR = Path(__file__).parent / "DEX"

PIPELINE: list[tuple[str, str]] = [
    ("fetch_uniswap_pool_001pct_timeseries.py", "USDC/WETH 0.01%%  ->  data_raw/multitier/fee_100/"),
    ("fetch_uniswap_pool_030pct_timeseries.py", "USDC/WETH 0.30%%  ->  data_raw/multitier/fee_3000/"),
    ("fetch_uniswap_pool_100pct_timeseries.py", "USDC/WETH 1.00%%  ->  data_raw/multitier/fee_10000/"),
]


def run_script(filename: str, label: str) -> int:
    path = SCRIPT_DIR / filename
    print(f"\n{'='*60}")
    print(f"  {label.replace('%%', '%')}")
    print(f"  {path}")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, str(path)], check=False)
    if result.returncode != 0:
        print(f"\n[FAILED] {filename} exited with code {result.returncode}")
    return result.returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download USDC/WETH 0.01%%, 0.30%%, and 1.00%% pool time series"
    )
    p.add_argument("--001", dest="only_001", action="store_true", help="Download 0.01%% pool only")
    p.add_argument("--030", dest="only_030", action="store_true", help="Download 0.30%% pool only")
    p.add_argument("--100", dest="only_100", action="store_true", help="Download 1.00%% pool only")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.environ.get("THEGRAPH_API_KEY"):
        print("[ERROR] THEGRAPH_API_KEY is not set.")
        print("  Run: setx THEGRAPH_API_KEY \"your_key\"  then restart your terminal.")
        print("  Or add  THEGRAPH_API_KEY=your_key  to a .env file at the project root.")
        sys.exit(1)

    if args.only_001:
        pipeline = [PIPELINE[0]]
    elif args.only_030:
        pipeline = [PIPELINE[1]]
    elif args.only_100:
        pipeline = [PIPELINE[2]]
    else:
        pipeline = PIPELINE

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
        print(f"ALL {len(pipeline)} POOL(S) DOWNLOADED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()
