"""
Causal Inference Pipeline Orchestrator.

Runs all 10 causal inference scripts (DiD / RD designs) in sequence.

Usage:
    python scripts/analysis/causal_inference/run_causal.py
    python scripts/analysis/causal_inference/run_causal.py fast   # forward FAST_MODE=1

Output directories (relative to project root):
    output/figures/causal/   – PDFs and PNG previews
    output/tables/causal/    – CSVs and LaTeX tables

Each sub-script changes cwd to the project root via sys.path manipulation so
that load() / savefig() / savetable() in analysis_utils.py resolve correctly.

Script execution order (by identification strength / chronological event order):
    ── Ethereum base-layer events ──────────────────────────────────────────────
     1. did_eip1559_gas           EIP-1559 (Aug 2021): gas market reform
     2. did_merge_impact          The Merge (Sep 2022): PoW→PoS transition
     3. did_shapella              Shapella (Apr 2023): ETH withdrawal unlock
    ── Macro crypto market shocks ──────────────────────────────────────────────
     4. did_terra_luna            Terra/LUNA collapse (May 2022)
     5. did_ftx_lp_withdrawal     FTX/Alameda collapse (Nov 2022)
     6. did_usdc_depeg            USDC de-peg episode (Mar 2023)
    ── Pool mechanism RD designs ───────────────────────────────────────────────
     7. rd_arbitrage_trigger      Sharp RD: gas-adjusted arb break-even
     8. rd_tick_crossing          Fuzzy RD: tick-boundary liquidity response
     9. rd_lp_profitability_threshold  Sharp RD: fee-APR → LP net P&L
    10. rd_vol_lvr_breakeven      Sharp RD: volatility LVR break-even

Ranking (see ranking.py for full criteria):
    #1  rd_arbitrage_trigger      23/25
    #2  rd_vol_lvr_breakeven      23/25
    #3  did_ftx_lp_withdrawal     22/25
    #4  rd_tick_crossing          22/25
    #5  did_usdc_depeg            21/25
    #6  did_merge_impact          20/25
    #7  did_terra_luna            19/25
    #8  did_shapella              19/25
    #9  rd_lp_profitability_threshold  19/25
   #10  did_eip1559_gas           15/25
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────────
CAUSAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CAUSAL_DIR.parent.parent.parent  # thesis_AMM/

PIPELINE: list[tuple[str, str]] = [
    # ── Ethereum base-layer events ────────────────────────────────────────────
    ("did_eip1559_gas.py",                " 1/10  DiD/ITS: EIP-1559 — gas cost and LP economics   [Aug 2021]"),
    ("did_merge_impact.py",               " 2/10  DiD/ITS: Ethereum Merge — PoW→PoS transition      [Sep 2022]"),
    ("did_shapella.py",                   " 3/10  Split-ITS: Shapella — ETH withdrawal unlock       [Apr 2023]"),
    # ── Macro crypto market shocks ────────────────────────────────────────────
    ("did_terra_luna.py",                 " 4/10  DiD/ITS: Terra/LUNA cascade collapse              [May 2022]"),
    ("did_ftx_lp_withdrawal.py",          " 5/10  DiD: FTX/Alameda collapse — LP exit behaviour     [Nov 2022]"),
    ("did_usdc_depeg.py",                 " 6/10  DiD/ITS: USDC de-peg episode                     [Mar 2023]"),
    # ── Pool mechanism RD designs ─────────────────────────────────────────────
    ("rd_arbitrage_trigger.py",           " 7/10  Sharp RD: gas-adjusted arbitrage break-even"),
    ("rd_tick_crossing.py",               " 8/10  Fuzzy RD: tick-boundary liquidity response"),
    ("rd_lp_profitability_threshold.py",  " 9/10  Sharp RD: fee-APR threshold → LP net P&L"),
    ("rd_vol_lvr_breakeven.py",           "10/10  Sharp RD: volatility LVR break-even cutoff"),
]


def run_script(script: str, label: str, env_extra: dict | None = None) -> tuple[int, float]:
    """Execute a single causal inference script as a subprocess."""
    path = CAUSAL_DIR / script
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    env = {**os.environ}
    if env_extra:
        env.update(env_extra)
    # sub-scripts resolve project root via their own sys.path fix; cwd = project root
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(path)],
        check=False,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  [FAILED] {script} exited with code {result.returncode}")
    else:
        print(f"\n  [OK]    {script}  ({elapsed:.1f}s)")

    return result.returncode, elapsed


def main() -> None:
    fast_mode = len(sys.argv) > 1 and sys.argv[1].lower() == "fast"
    if fast_mode:
        print("[FAST MODE] FAST_MODE=1 forwarded to sub-scripts.")
        print("  Scripts may skip bootstrap CIs and heavy GARCH fits if implemented.")

    env_extra = {"FAST_MODE": "1"} if fast_mode else None

    print(f"\nCausal Inference Pipeline  ({len(PIPELINE)} scripts)")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Scripts dir  : {CAUSAL_DIR}")

    t_total = time.time()
    results: list[tuple[str, int, float]] = []

    for script, label in PIPELINE:
        rc, elapsed = run_script(script, label, env_extra=env_extra)
        results.append((script, rc, elapsed))

    total_elapsed = time.time() - t_total

    # ── summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"CAUSAL INFERENCE PIPELINE COMPLETE  (total: {total_elapsed:.0f}s)")
    print(f"{'='*70}\n")

    failed = [(s, rc) for s, rc, _ in results if rc != 0]
    if failed:
        print(f"FAILED ({len(failed)} / {len(PIPELINE)}):")
        for s, rc in failed:
            print(f"  ✗  {s}  (exit {rc})")
        print()

    print("Timing summary:")
    for script, rc, el in results:
        status = "✓" if rc == 0 else "✗"
        print(f"  [{status}] {script:<45} {el:>7.1f}s")

    # output inventory
    fig_dir = PROJECT_ROOT / "output" / "figures"
    tbl_dir = PROJECT_ROOT / "output" / "tables"
    n_pdf = len(list(fig_dir.glob("*.pdf"))) if fig_dir.exists() else 0
    n_csv = len(list(tbl_dir.glob("*.csv"))) if tbl_dir.exists() else 0
    print(f"\nFigures: output/figures/  ({n_pdf} PDFs total)")
    print(f"Tables:  output/tables/   ({n_csv} CSVs total)")
    print("\nNext: python scripts/analysis/causal_inference/ranking.py")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
