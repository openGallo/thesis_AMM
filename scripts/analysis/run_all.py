"""
Run all analysis scripts in order.

Usage:
    python scripts/analysis/run_all.py           # all scripts
    python scripts/analysis/run_all.py fast       # sets FAST_MODE=1 env-var (reserved for future use)

Note: FAST_MODE is forwarded to every sub-script as an environment variable.
Individual scripts may check os.environ.get("FAST_MODE") to skip heavy
computations (GARCH fitting, bootstrap CIs). Currently none do — all scripts
run their full pipeline regardless. This flag is reserved for future implementation.

Outputs land in:
    output/figures/*.pdf  (+ .png previews)
    output/tables/*.csv   and  output/tables/*.tex

Requires processed data in data_processed/ (run scripts/process_data/run_all.py first).

Script execution order and dependencies:
    1. descriptive_stats    - data coverage + correlations + panel timeline
    2. price_dynamics       - GBM params, GARCH, HAR-RV, tail risk, seasonality
    3. dex_cex_basis        - arbitrage persistence, VECM, determinants
    4. lvr_analysis         - LVR theory test, GARCH-implied, rolling coefs
    5. trade_microstructure - swap size, gas, price impact, Amihud
    6. volatility_regimes   - regime transitions, GARCH overlay, forecast accuracy
    7. lp_profitability     - LP P&L by type / range / regime, fee APR
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

PIPELINE: list[tuple[str, str]] = [
    ("descriptive_stats.py",    "1/7  Descriptive statistics"),
    ("price_dynamics.py",       "2/7  Price dynamics (GBM, GARCH, HAR-RV, tail risk)"),
    ("dex_cex_basis.py",        "3/7  DEX-CEX basis (arbitrage, VECM, determinants)"),
    ("lvr_analysis.py",         "4/7  LVR analysis (theory test, rolling, GARCH)"),
    ("trade_microstructure.py", "5/7  Trade microstructure (size, gas, price impact, Amihud)"),
    ("volatility_regimes.py",   "6/7  Volatility regimes (transitions, forecast accuracy)"),
    ("lp_profitability.py",     "7/7  LP profitability (P&L by type, range, regime)"),
]


def run(script: str, label: str, env_extra: dict | None = None) -> tuple[int, float]:
    path = SCRIPT_DIR / script
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.time()
    env = None
    if env_extra:
        import os
        env = {**os.environ, **env_extra}
    result = subprocess.run(
        [sys.executable, str(path)],
        check=False,
        cwd=str(SCRIPT_DIR),
        env=env,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"  [FAILED] {script} exited with code {result.returncode}")
    else:
        print(f"  [OK]    {script} ({elapsed:.1f}s)")
    return result.returncode, elapsed


def main() -> None:
    fast_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "fast"
    if fast_mode:
        print("[FAST MODE] FAST_MODE=1 forwarded to sub-scripts.")
        print("  NOTE: No script currently skips GARCH/bootstraps based on this flag.")
        print("  All 7 analysis scripts will run their full pipeline.")

    env_extra = {"FAST_MODE": "1"} if fast_mode else None

    t_start = time.time()
    results = []
    for script, label in PIPELINE:
        rc, elapsed = run(script, label, env_extra=env_extra)
        results.append((script, rc, elapsed))

    total = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"ANALYSIS PIPELINE COMPLETE  (total: {total:.0f}s)")
    print(f"{'='*60}")

    failed = [(s, rc) for s, rc, _ in results if rc != 0]
    if failed:
        print(f"\nFAILED ({len(failed)}):")
        for s, rc in failed:
            print(f"  - {s}  (exit {rc})")
        print()

    print("Timing summary:")
    for s, rc, el in results:
        status = "OK" if rc == 0 else "FAIL"
        print(f"  [{status}] {s:<35} {el:>6.1f}s")

    print(f"\nFigures: output/figures/  ({len(list((SCRIPT_DIR.parents[1] / 'output' / 'figures').glob('*.pdf')))} PDFs)")
    print(f"Tables:  output/tables/   ({len(list((SCRIPT_DIR.parents[1] / 'output' / 'tables').glob('*.csv')))} CSVs)")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
