"""
Run all analysis scripts in order.

Usage:
    python scripts/analysis/run_all.py

Outputs land in:
    output/figures/*.pdf
    output/tables/*.csv  and  output/tables/*.tex

Requires processed data in data_processed/ (run scripts/process_data/run_all.py first).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

PIPELINE: list[tuple[str, str]] = [
    ("descriptive_stats.py",   "Descriptive statistics (all variables)"),
    ("price_dynamics.py",      "Price dynamics (GBM, ACF, vol series)"),
    ("lp_profitability.py",    "LP profitability (P&L by type, fee APR)"),
    ("dex_cex_basis.py",       "DEX-CEX basis (persistence, arbitrage)"),
    ("lvr_analysis.py",        "LVR analysis (vs fee income, by regime)"),
    ("trade_microstructure.py","Trade microstructure (size, gas, direction)"),
    ("volatility_regimes.py",  "Volatility regimes (stress, metrics by regime)"),
]


def run(script: str, label: str) -> int:
    path = SCRIPT_DIR / script
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, str(path)],
        check=False,
        cwd=str(SCRIPT_DIR),   # working dir = scripts/analysis/ so relative imports work
    )
    if result.returncode != 0:
        print(f"  [FAILED] {script} exited with code {result.returncode}")
    return result.returncode


def main() -> None:
    failed = []
    for script, label in PIPELINE:
        rc = run(script, label)
        if rc != 0:
            failed.append(script)

    print(f"\n{'='*60}")
    if failed:
        print(f"COMPLETED WITH ERRORS — {len(failed)} script(s) failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print(f"ALL {len(PIPELINE)} ANALYSIS SCRIPTS COMPLETED")
        print("Figures -> output/figures/")
        print("Tables  -> output/tables/")
    print("=" * 60)


if __name__ == "__main__":
    main()
