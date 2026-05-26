"""
Data integrity check for all raw imported data.

Run this after completing all import scripts to verify coverage,
row counts, column completeness, and detect obvious gaps or failures.

Usage:
    python scripts/check_data_integrity.py

Output: console report with [OK] / [WARN] / [FAIL] per check.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW     = PROJECT_ROOT / "data_raw"
CEX_ROOT     = DATA_RAW / "CEX" / "binance"
DEX_ROOT     = DATA_RAW / "DEX"

# ── Expected coverage ─────────────────────────────────────────────────────────

# DEX pool inception -> thesis data cut-off
DEX_START  = (2021,  5)
DEX_END    = (2026,  4)

# Binance full history
CEX_START  = (2022,  1)
CEX_END    = (2026,  4)

# ETHUSDC listed on Binance August 2022
ETHUSDC_KLINE_START = (2022, 8)

CEX_SYMBOLS   = ["ETHUSDC", "ETHUSDT", "USDCUSDT"]
CEX_INTERVALS = ["1m", "5m"]

# Minimum rows considered non-empty for a monthly 1m kline file
MIN_ROWS_1M = 20_000   # shortest month has 28d*24h*60m = 40,320; be lenient
MIN_ROWS_5M =  4_000

# ── Helpers ───────────────────────────────────────────────────────────────────

issues:   list[str] = []
warnings: list[str] = []


def ok(msg: str)   -> None: print(f"  [OK]   {msg}")
def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")
    warnings.append(msg)
def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    issues.append(msg)


def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months = []
    y, m = start
    while (y, m) <= end:
        months.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def check_csv_readable(path: Path, min_rows: int = 1,
                       required_cols: list[str] | None = None) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, low_memory=False, nrows=min_rows + 1)
    except Exception as exc:
        fail(f"Cannot read {path.name}: {exc}")
        return None
    if len(df) < min_rows:
        fail(f"{path.name}: only {len(df):,} rows (expected ≥ {min_rows:,})")
        return None
    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            fail(f"{path.name}: missing columns {missing}")
    return df


# ── CEX: manifest ─────────────────────────────────────────────────────────────

def check_cex_manifest() -> None:
    print("\n[CEX] Download manifest")
    path = CEX_ROOT / "download_manifest.csv"
    if not path.exists():
        fail("download_manifest.csv not found — CEX import may not have run")
        return

    df = pd.read_csv(path, low_memory=False)
    counts = df["status"].value_counts().to_dict()

    ok(f"Manifest: {len(df):,} entries  |  statuses: {counts}")

    failed = df[~df["status"].isin(["downloaded_extracted", "downloaded", "missing_404", "skipped"])]
    if not failed.empty:
        for _, row in failed.iterrows():
            fail(f"Download failed: {row.get('symbol','')} {row.get('interval','')} "
                 f"{row.get('period','')}  status={row['status']}  error={row.get('error','')}")
    else:
        ok("No failed downloads")

    n404 = counts.get("missing_404", 0)
    if n404 > 0:
        ok(f"{n404} missing_404 entries (expected: ETHUSDC before Aug 2022)")


# ── CEX: klines coverage ──────────────────────────────────────────────────────

def check_cex_klines() -> None:
    print("\n[CEX] Klines coverage and row counts")

    for symbol in CEX_SYMBOLS:
        start = ETHUSDC_KLINE_START if symbol == "ETHUSDC" else CEX_START
        expected = month_range(start, CEX_END)

        for interval in CEX_INTERVALS:
            kline_dir = CEX_ROOT / "spot" / "monthly" / "klines" / symbol / interval
            if not kline_dir.exists():
                fail(f"{symbol}/{interval}: directory missing")
                continue

            found = {
                (int(f.stem.split("-")[-2]), int(f.stem.split("-")[-1]))
                for f in kline_dir.glob(f"{symbol}-{interval}-*.csv")
            }
            missing = [m for m in expected if m not in found]
            extra   = sorted(found - set(expected))

            label = f"{symbol}/{interval}"
            if not missing:
                ok(f"{label}: {len(found)}/{len(expected)} months  "
                   f"({start[0]}-{start[1]:02d} -> {CEX_END[0]}-{CEX_END[1]:02d})")
            else:
                ms = ", ".join(f"{y}-{m:02d}" for y, m in sorted(missing)[:5])
                tail = f"... +{len(missing)-5} more" if len(missing) > 5 else ""
                warn(f"{label}: {len(missing)} months missing: {ms}{tail}")

            # Row-count spot check on most recent file present
            if found:
                newest = sorted(found)[-1]
                f = kline_dir / f"{symbol}-{interval}-{newest[0]}-{newest[1]:02d}.csv"
                try:
                    n = sum(1 for _ in open(f)) - 1  # fast line count
                    min_r = MIN_ROWS_1M if interval == "1m" else MIN_ROWS_5M
                    if n < min_r:
                        warn(f"{label} {newest[0]}-{newest[1]:02d}: {n:,} rows (expected ≥ {min_r:,})")
                    else:
                        ok(f"{label} newest month ({newest[0]}-{newest[1]:02d}): {n:,} rows")
                except Exception as exc:
                    warn(f"{label}: could not count rows in newest file: {exc}")


# ── CEX: aggTrades coverage ───────────────────────────────────────────────────

def check_cex_agg_trades() -> None:
    print("\n[CEX] aggTrades coverage")
    for symbol in CEX_SYMBOLS:
        start = ETHUSDC_KLINE_START if symbol == "ETHUSDC" else CEX_START
        expected = month_range(start, CEX_END)
        agg_dir  = CEX_ROOT / "spot" / "monthly" / "aggTrades" / symbol
        if not agg_dir.exists():
            warn(f"{symbol} aggTrades: directory missing (optional)")
            continue

        found = {
            (int(f.stem.split("-")[-2]), int(f.stem.split("-")[-1]))
            for f in agg_dir.glob(f"{symbol}-aggTrades-*.csv")
        }
        missing = [m for m in expected if m not in found]
        if not missing:
            ok(f"{symbol} aggTrades: {len(found)}/{len(expected)} months")
        else:
            warn(f"{symbol} aggTrades: {len(missing)} months missing")


# ── DEX: pool time series ─────────────────────────────────────────────────────

def check_dex_pool_timeseries() -> None:
    print("\n[DEX] Pool time series")

    # Hourly
    path = DEX_ROOT / "pool_hour_data.csv"
    if not path.exists():
        fail("pool_hour_data.csv missing — run fetch_uniswap_pool_timeseries.py")
        return

    df = pd.read_csv(path, low_memory=False)
    n  = len(df)
    expected_hours = len(month_range(DEX_START, DEX_END)) * 24 * 30  # rough
    ok(f"pool_hour_data.csv: {n:,} rows")

    required = ["period_start_unix", "tvl_usd", "volume_usd", "fees_usd", "tx_count",
                "liquidity", "sqrt_price", "tick"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        fail(f"pool_hour_data.csv: missing columns {missing_cols}")
    else:
        ok("pool_hour_data.csv: all required columns present")

    null_tvl = df["tvl_usd"].isna().sum() if "tvl_usd" in df.columns else None
    if null_tvl is not None and null_tvl > n * 0.05:
        warn(f"pool_hour_data.csv: {null_tvl:,} null tvl_usd ({null_tvl/n:.1%})")
    elif null_tvl is not None:
        ok(f"pool_hour_data.csv: tvl_usd null rate {null_tvl/n:.2%}")

    if n < 1000:
        fail(f"pool_hour_data.csv: suspiciously few rows ({n:,}); import may be incomplete")
    elif n < 10_000:
        warn(f"pool_hour_data.csv: {n:,} rows — import may be partially complete")

    # Daily
    path_d = DEX_ROOT / "pool_day_data.csv"
    if path_d.exists():
        nd = len(pd.read_csv(path_d, nrows=10_000))
        ok(f"pool_day_data.csv: {nd:,} rows")
    else:
        fail("pool_day_data.csv missing")


# ── DEX: monthly event files ──────────────────────────────────────────────────

def check_dex_monthly(event: str, required_cols: list[str]) -> None:
    expected = month_range(DEX_START, DEX_END)
    found_paths = {
        (int(f.stem.split("_")[-2]), int(f.stem.split("_")[-1])): f
        for f in DEX_ROOT.glob(f"{event}_*.csv")
    }
    missing = [m for m in expected if m not in found_paths]
    found   = sorted(found_paths.keys())

    if not found_paths:
        fail(f"{event}: no files found — run import scripts")
        return

    coverage = f"{found[0][0]}-{found[0][1]:02d} -> {found[-1][0]}-{found[-1][1]:02d}"
    if not missing:
        ok(f"{event}: {len(found)}/{len(expected)} months  ({coverage})")
    else:
        warn(f"{event}: {len(missing)} months missing  (have {len(found)}: {coverage})")

    # Spot-check newest file
    newest_path = found_paths[found[-1]]
    try:
        df = pd.read_csv(newest_path, low_memory=False, nrows=5001)
        n  = len(df)
        if n == 0:
            fail(f"{event} {found[-1]}: file is empty")
        else:
            ok(f"{event} newest ({found[-1][0]}-{found[-1][1]:02d}): {n:,}+ rows")

        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            fail(f"{event} newest: missing columns {missing_cols}")
    except Exception as exc:
        fail(f"{event} newest: cannot read — {exc}")


def check_dex_monthly_events() -> None:
    print("\n[DEX] Monthly event files")
    check_dex_monthly("swaps",    ["timestamp", "amount_usd", "gas_price_wei", "sqrt_price_x96"])
    check_dex_monthly("mints",    ["timestamp", "amount_usd", "tick_lower", "tick_upper"])
    check_dex_monthly("burns",    ["timestamp", "amount_usd", "tick_lower", "tick_upper"])
    check_dex_monthly("collects", ["timestamp", "amount_usd", "tick_lower", "tick_upper"])


# ── DEX: positions and metadata ───────────────────────────────────────────────

def check_dex_static_files() -> None:
    print("\n[DEX] Static / current-state files")

    checks = [
        ("positions_current.csv",  50,  ["owner", "tick_lower", "tick_upper", "liquidity"]),
        ("pool_metadata_current.csv", 1, ["pool_id", "tvl_usd", "fee_tier"]),
        ("bundle_current.csv",     1,   []),
    ]
    for fname, min_rows, req_cols in checks:
        p = DEX_ROOT / fname
        if not p.exists():
            warn(f"{fname}: not found (run fetch_uniswap_positions.py)")
            continue
        check_csv_readable(p, min_rows=min_rows, required_cols=req_cols)
        df = pd.read_csv(p, low_memory=False)
        ok(f"{fname}: {len(df):,} rows")


# ── DEX: sanity-check prices ──────────────────────────────────────────────────

def check_dex_price_sanity() -> None:
    print("\n[DEX] Price sanity check")
    path = DEX_ROOT / "pool_hour_data.csv"
    if not path.exists():
        return

    df = pd.read_csv(path, low_memory=False)
    if "token0_price" not in df.columns:
        warn("pool_hour_data.csv: token0_price column missing, skipping price check")
        return

    price = pd.to_numeric(df["token0_price"], errors="coerce").dropna()
    p_min, p_max = float(price.min()), float(price.max())
    ok(f"ETH/USDC price range: ${p_min:,.0f} -> ${p_max:,.0f}")
    if p_min < 50 or p_max > 20_000:
        warn(f"Price range looks unusual: ${p_min:.0f} -> ${p_max:.0f} (expected ~$100–$10,000)")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary() -> None:
    print(f"\n{'='*60}")
    total = len(issues) + len(warnings)
    if total == 0:
        print("ALL CHECKS PASSED — data looks complete")
    else:
        if issues:
            print(f"FAILURES ({len(issues)}):")
            for i in issues:
                print(f"  [FAIL] {i}")
        if warnings:
            print(f"WARNINGS ({len(warnings)}):")
            for w in warnings:
                print(f"  [WARN] {w}")
    print("=" * 60)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Data Integrity Check")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Checked at   : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    check_cex_manifest()
    check_cex_klines()
    check_cex_agg_trades()
    check_dex_pool_timeseries()
    check_dex_monthly_events()
    check_dex_static_files()
    check_dex_price_sanity()
    print_summary()

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
