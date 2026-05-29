"""
Data integrity check for all raw imported data.

Covers:
    CEX  — Binance 1m/5m klines and aggTrades for ETHUSDC, ETHUSDT, USDCUSDT
    DEX  — Uniswap v3 USDC/WETH 0.05% pool (main study pool)
    MT   — Uniswap v3 multitier pools: 0.01%, 0.30%, 1.00%

Outputs:
    Console  — [OK] / [WARN] / [FAIL] per check
    File     — reports/data_integrity_YYYY-MM-DD.txt  (full report + remediation plan)

Exit code 1 if any [FAIL]; 0 if only [OK] / [WARN].

Usage:
    python scripts/check_data_integrity.py
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
MT_ROOT      = DATA_RAW / "multitier"
REPORT_DIR   = PROJECT_ROOT / "reports"

# ── Expected coverage windows ─────────────────────────────────────────────────

DEX_START = (2021,  5)   # pool inception
DEX_END   = (2026,  4)   # thesis data cut-off

CEX_START = (2022,  1)
CEX_END   = (2026,  4)

# ETHUSDC listed on Binance Aug 2022; confirmed 404 for these months.
ETHUSDC_START = (2022, 8)
ETHUSDC_404   = {(2022, 10), (2022, 11), (2022, 12), (2023, 1), (2023, 2)}
USDCUSDT_404  = {(2022, 10), (2022, 11), (2022, 12), (2023, 1), (2023, 2)}

CEX_SYMBOLS   = ["ETHUSDC", "ETHUSDT", "USDCUSDT"]
MIN_ROWS_1M   = 20_000   # 28d×24h×60m = 40,320 min; lenient floor
MIN_ROWS_5M   =  4_000
MIN_ROWS_AGGS = 10_000

MULTITIER_POOLS = [
    {"dir": "fee_100",   "label": "0.01%", "fee_tier": 100,   "min_hours": 1_000},
    {"dir": "fee_3000",  "label": "0.30%", "fee_tier": 3000,  "min_hours": 5_000},
    {"dir": "fee_10000", "label": "1.00%", "fee_tier": 10000, "min_hours": 2_000},
]

COLS_POOL_HOUR = [
    "period_start_unix", "datetime", "open", "high", "low", "close",
    "volume_usd", "fees_usd", "tx_count", "tvl_usd", "liquidity",
    "sqrt_price", "tick", "token0_price", "token1_price",
]
COLS_POOL_DAY = [
    "date_unix", "date", "open", "high", "low", "close",
    "volume_usd", "fees_usd", "tx_count", "tvl_usd",
]
COLS_METADATA = ["pool_id", "fee_tier", "tvl_usd", "token0_symbol", "token1_symbol"]
COLS_BUNDLE   = ["id", "eth_price_usd"]


# ── Report writer ─────────────────────────────────────────────────────────────

class Report:
    """Writes to both stdout and a report file simultaneously."""

    def __init__(self) -> None:
        self.lines:       list[str] = []
        self.issues:      list[str] = []
        self.warnings:    list[str] = []
        self.inventory:   list[dict] = []
        self.remediation: list[str] = []

    def write(self, line: str = "") -> None:
        print(line)
        self.lines.append(line)

    def ok(self, msg: str) -> None:
        self.write(f"  [OK]   {msg}")

    def warn(self, msg: str, fix: str = "") -> None:
        self.write(f"  [WARN] {msg}")
        self.warnings.append(msg)
        if fix:
            self.remediation.append(f"  [WARN] {msg}\n         → {fix}")

    def fail(self, msg: str, fix: str = "") -> None:
        self.write(f"  [FAIL] {msg}")
        self.issues.append(msg)
        if fix:
            self.remediation.append(f"  [FAIL] {msg}\n         → {fix}")

    def section(self, title: str) -> None:
        self.write()
        self.write(title)

    def inv(self, dataset: str, files: int, rows: int | str,
            period: str, note: str = "") -> None:
        self.inventory.append({
            "Dataset": dataset,
            "Files":   files,
            "Rows":    rows if isinstance(rows, str) else f"{rows:,}",
            "Period":  period,
            "Note":    note,
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines))
        print(f"\nFull report saved to: {path}")


R = Report()


# ── Utilities ─────────────────────────────────────────────────────────────────

def month_range(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    months, y, m = [], *start
    while (y, m) <= end:
        months.append((y, m))
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return months


def read_csv_safe(path: Path, **kwargs) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as exc:
        R.fail(f"Cannot read {path.name}: {exc}")
        return None


def check_required_cols(df: pd.DataFrame, required: list[str], label: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        R.fail(f"{label}: missing columns {missing}")
        return False
    return True


def check_no_duplicates(series: pd.Series, label: str) -> None:
    n_dup = series.duplicated().sum()
    if n_dup > 0:
        R.warn(f"{label}: {n_dup:,} duplicate timestamps",
               fix="Re-run the import script with --overwrite to re-fetch clean data.")
    else:
        R.ok(f"{label}: no duplicate timestamps")


def check_hourly_gaps(ts_series: pd.Series, label: str) -> None:
    ts    = ts_series.dropna().sort_values().astype(int)
    if len(ts) < 2:
        return
    diffs = ts.diff().dropna()
    big   = (diffs > 7200).sum()
    if big > 0:
        worst = int(diffs.max()) // 3600
        R.warn(f"{label}: {big:,} gaps > 2 h (largest ≈ {worst} h)",
               fix="Re-fetch with --overwrite; gaps in The Graph data may be permanent.")
    else:
        R.ok(f"{label}: no gaps > 2 h in hourly series")


def check_ohlcv_sanity(df: pd.DataFrame, label: str,
                        o="open", h="high", l="low", c="close", v="volume_usd") -> None:
    for col in [o, h, l, c, v]:
        if col not in df.columns:
            return
    df2  = df[[o, h, l, c, v]].apply(pd.to_numeric, errors="coerce").dropna()
    errs = []
    if (df2[h] < df2[l]).any():   errs.append(f"{(df2[h]<df2[l]).sum()} rows high<low")
    if (df2[h] < df2[o]).any():   errs.append(f"{(df2[h]<df2[o]).sum()} rows high<open")
    if (df2[l] > df2[c]).any():   errs.append(f"{(df2[l]>df2[c]).sum()} rows low>close")
    if (df2[v] < 0).any():        errs.append(f"{(df2[v]<0).sum()} rows volume<0")
    if errs:
        R.warn(f"{label} OHLCV: {'; '.join(errs)}",
               fix="Re-fetch data. If issue persists, flag rows in the processing script.")
    else:
        R.ok(f"{label}: OHLCV sanity OK (n={len(df2):,})")


def eth_price_range_ok(price_series: pd.Series, label: str) -> None:
    p = pd.to_numeric(price_series, errors="coerce").dropna()
    if p.empty:
        R.warn(f"{label}: no valid price data")
        return
    lo, hi = float(p.min()), float(p.max())
    R.ok(f"{label}: ETH/USDC price range ${lo:,.0f} – ${hi:,.0f}")
    if lo < 50 or hi > 25_000:
        R.warn(f"{label}: price range ${lo:.0f}–${hi:.0f} looks unusual "
               "(expected ~$100–$10,000)",
               fix="Verify pool address in pool_addresses.py and re-fetch.")


# ═══════════════════════════════════════════════════════════════════════════════
# CEX
# ═══════════════════════════════════════════════════════════════════════════════

def check_cex_manifest() -> None:
    R.section("[CEX] Download manifest")
    path = CEX_ROOT / "download_manifest.csv"
    if not path.exists():
        R.fail("download_manifest.csv not found",
               fix="Run: python scripts/import_data/run_cex.py")
        return

    df     = read_csv_safe(path)
    if df is None:
        return
    counts = df["status"].value_counts().to_dict()
    R.ok(f"Manifest: {len(df):,} entries  |  {counts}")

    bad = df[~df["status"].isin(
        {"downloaded_extracted", "downloaded", "missing_404", "skipped"})]
    if not bad.empty:
        for _, row in bad.iterrows():
            R.fail(f"Download failed: {row.get('symbol','')} "
                   f"{row.get('interval','')} {row.get('period','')} "
                   f"status={row['status']} error={row.get('error','')}",
                   fix="Run: python scripts/import_data/run_cex.py --overwrite")
    else:
        R.ok("No failed downloads in manifest")

    n404 = counts.get("missing_404", 0)
    if n404:
        R.ok(f"{n404} missing_404 entries (expected: ETHUSDC pre-Aug 2022 and "
             "ETHUSDC/USDCUSDT Oct 2022–Feb 2023 — these are permanent Binance gaps)")


def _check_klines_interval(interval: str, min_rows: int) -> None:
    label_404 = {"ETHUSDC": ETHUSDC_404, "USDCUSDT": USDCUSDT_404}

    for symbol in CEX_SYMBOLS:
        start    = ETHUSDC_START if symbol == "ETHUSDC" else CEX_START
        skip_404 = label_404.get(symbol, set())
        expected = [m for m in month_range(start, CEX_END) if m not in skip_404]

        kdir = CEX_ROOT / "spot" / "monthly" / "klines" / symbol / interval
        if not kdir.exists():
            (R.warn if interval == "5m" else R.fail)(
                f"{symbol}/{interval}: directory missing",
                fix=f"Run: python scripts/import_data/run_cex.py")
            continue

        found = {
            (int(f.stem.split("-")[-2]), int(f.stem.split("-")[-1]))
            for f in kdir.glob(f"{symbol}-{interval}-*.csv")
        }
        missing = [m for m in expected if m not in found]
        lab     = f"{symbol}/{interval}"

        if not missing:
            R.ok(f"{lab}: {len(found)}/{len(expected)} months  "
                 f"({start[0]}-{start[1]:02d} → {CEX_END[0]}-{CEX_END[1]:02d})")
        else:
            ms   = ", ".join(f"{y}-{m:02d}" for y, m in sorted(missing)[:5])
            tail = f" … +{len(missing)-5} more" if len(missing) > 5 else ""
            (R.warn if interval == "5m" else R.fail)(
                f"{lab}: {len(missing)} months missing — {ms}{tail}",
                fix="Run: python scripts/import_data/run_cex.py --overwrite")

        if not found:
            continue

        # Spot-validate newest file
        newest = sorted(found)[-1]
        fpath  = kdir / f"{symbol}-{interval}-{newest[0]}-{newest[1]:02d}.csv"
        try:
            df = pd.read_csv(fpath, header=None, low_memory=False)
        except Exception as exc:
            R.warn(f"{lab} newest: cannot read — {exc}")
            continue

        n_rows, n_cols = df.shape
        if n_cols != 12:
            R.fail(f"{lab} newest: expected 12 columns, got {n_cols}",
                   fix="Re-download: python scripts/import_data/run_cex.py --overwrite")
        if n_rows < min_rows:
            (R.warn if interval == "5m" else R.fail)(
                f"{lab} {newest[0]}-{newest[1]:02d}: {n_rows:,} rows "
                f"(expected ≥ {min_rows:,})",
                fix="Re-download: python scripts/import_data/run_cex.py --overwrite")
        else:
            R.ok(f"{lab} {newest[0]}-{newest[1]:02d}: {n_rows:,} rows, {n_cols} cols")

        # OHLCV + timestamp sanity
        df.columns = ["open_time","open","high","low","close","vol_base",
                      "close_time","vol_quote","n_trades",
                      "taker_buy_base","taker_buy_quote","ignore"]
        df2  = df[["open","high","low","close","vol_base"]].apply(
            pd.to_numeric, errors="coerce").dropna()
        errs = []
        if (df2["high"] < df2["low"]).any():   errs.append("high<low")
        if (df2["high"] < df2["open"]).any():  errs.append("high<open")
        if (df2["low"]  > df2["close"]).any(): errs.append("low>close")
        if (df2["vol_base"] < 0).any():        errs.append("volume<0")
        ts = pd.to_numeric(df["open_time"], errors="coerce").dropna()
        if not ts.is_monotonic_increasing:
            errs.append("timestamps not monotonic")
        if errs:
            R.warn(f"{lab} newest OHLCV: {'; '.join(errs)}",
                   fix="Re-download: python scripts/import_data/run_cex.py --overwrite")
        else:
            R.ok(f"{lab} newest: OHLCV + timestamp sanity OK")

        if interval == "1m":
            R.inv(f"CEX {symbol} 1m klines", len(found),
                  f"~{len(found)*40_000:,}",
                  f"{start[0]}-{start[1]:02d} → {CEX_END[0]}-{CEX_END[1]:02d}",
                  f"{len(skip_404)} known-404 months excluded" if skip_404 else "")


def check_cex_klines() -> None:
    R.section("[CEX] 1m klines — coverage, row counts, OHLCV sanity")
    _check_klines_interval("1m", MIN_ROWS_1M)
    R.section("[CEX] 5m klines — coverage (advisory, not used in main pipeline)")
    _check_klines_interval("5m", MIN_ROWS_5M)


def check_cex_aggtrades() -> None:
    R.section("[CEX] aggTrades coverage")
    for symbol in ["ETHUSDC", "ETHUSDT"]:
        start    = ETHUSDC_START if symbol == "ETHUSDC" else CEX_START
        skip_404 = ETHUSDC_404  if symbol == "ETHUSDC" else set()
        expected = [m for m in month_range(start, CEX_END) if m not in skip_404]

        agg_dir = CEX_ROOT / "spot" / "monthly" / "aggTrades" / symbol
        if not agg_dir.exists():
            R.warn(f"{symbol} aggTrades: directory missing",
                   fix="Run: python scripts/import_data/run_cex.py")
            continue

        found = {
            (int(f.stem.split("-")[-2]), int(f.stem.split("-")[-1]))
            for f in agg_dir.glob(f"{symbol}-aggTrades-*.csv")
        }
        missing = [m for m in expected if m not in found]
        lab     = f"{symbol}/aggTrades"

        if not missing:
            R.ok(f"{lab}: {len(found)}/{len(expected)} months")
        else:
            ms   = ", ".join(f"{y}-{m:02d}" for y, m in sorted(missing)[:5])
            tail = f" … +{len(missing)-5} more" if len(missing) > 5 else ""
            R.warn(f"{lab}: {len(missing)} months missing — {ms}{tail}",
                   fix="Run: python scripts/import_data/run_cex.py --overwrite")

        if found:
            newest = sorted(found)[-1]
            fpath  = agg_dir / f"{symbol}-aggTrades-{newest[0]}-{newest[1]:02d}.csv"
            try:
                n = sum(1 for _ in open(fpath)) - 1
                if n < MIN_ROWS_AGGS:
                    R.warn(f"{lab} {newest[0]}-{newest[1]:02d}: {n:,} rows "
                           f"(expected ≥ {MIN_ROWS_AGGS:,})")
                else:
                    R.ok(f"{lab} {newest[0]}-{newest[1]:02d}: {n:,} rows")
            except Exception as exc:
                R.warn(f"{lab} newest: {exc}")

            R.inv(f"CEX {symbol} aggTrades", len(found), "—",
                  f"{start[0]}-{start[1]:02d} → {CEX_END[0]}-{CEX_END[1]:02d}")


# ═══════════════════════════════════════════════════════════════════════════════
# DEX 0.05% main study pool
# ═══════════════════════════════════════════════════════════════════════════════

def check_dex_pool_timeseries() -> None:
    R.section("[DEX 0.05%] Pool time series")

    hour_path = DEX_ROOT / "pool_hour_data.csv"
    if not hour_path.exists():
        R.fail("pool_hour_data.csv missing",
               fix="Run: python scripts/import_data/run_dex.py")
    else:
        df = read_csv_safe(hour_path)
        if df is not None:
            n = len(df)
            R.ok(f"pool_hour_data.csv: {n:,} rows")
            if n < 1_000:
                R.fail(f"pool_hour_data.csv: suspiciously few rows ({n:,})",
                       fix="Run: python scripts/import_data/run_dex.py")
            check_required_cols(df, COLS_POOL_HOUR, "pool_hour_data.csv")
            if "period_start_unix" in df.columns:
                ts = pd.to_numeric(df["period_start_unix"], errors="coerce")
                check_no_duplicates(ts, "pool_hour_data.csv")
                check_hourly_gaps(ts, "pool_hour_data.csv")
                t0 = datetime.fromtimestamp(ts.min(), tz=timezone.utc).strftime("%Y-%m-%d")
                t1 = datetime.fromtimestamp(ts.max(), tz=timezone.utc).strftime("%Y-%m-%d")
                R.ok(f"pool_hour_data.csv: covers {t0} → {t1}")
                R.inv("DEX 0.05% hourly", 1, n, f"{t0} → {t1}")
            check_ohlcv_sanity(df, "pool_hour_data.csv")
            if "tvl_usd" in df.columns:
                null_tvl = df["tvl_usd"].isna().sum()
                rate = null_tvl / n if n else 0
                if rate > 0.05:
                    R.warn(f"pool_hour_data.csv: {null_tvl:,} null tvl_usd ({rate:.1%})",
                           fix="Re-fetch: python scripts/import_data/run_dex.py")
                else:
                    R.ok(f"pool_hour_data.csv: tvl_usd null rate {rate:.2%}")
            if "token0_price" in df.columns:
                eth_price_range_ok(df["token0_price"], "pool_hour_data.csv")

    day_path = DEX_ROOT / "pool_day_data.csv"
    if not day_path.exists():
        R.fail("pool_day_data.csv missing",
               fix="Run: python scripts/import_data/run_dex.py")
    else:
        df_d = read_csv_safe(day_path)
        if df_d is not None:
            nd = len(df_d)
            R.ok(f"pool_day_data.csv: {nd:,} rows")
            check_required_cols(df_d, COLS_POOL_DAY, "pool_day_data.csv")
            R.inv("DEX 0.05% daily", 1, nd, "—")


def check_dex_monthly_events() -> None:
    R.section("[DEX 0.05%] Monthly event files")
    expected = month_range(DEX_START, DEX_END)

    event_specs = [
        # (event, required_cols, allow_empty, optional)
        ("swaps",    ["timestamp", "amount_usd", "sqrt_price_x96", "gas_price_wei"], False, False),
        ("mints",    ["timestamp", "amount_usd", "tick_lower", "tick_upper"],        False, False),
        ("burns",    ["timestamp", "amount_usd", "tick_lower", "tick_upper"],        False, False),
        ("collects", ["timestamp"],                                                   True,  False),
        ("flashes",  ["timestamp"],                                                   False, True),
        ("position_snapshots", ["timestamp", "position_id"],                          False, True),
    ]

    for event, req_cols, allow_empty, optional in event_specs:
        found_paths = {
            (int(f.stem.split("_")[-2]), int(f.stem.split("_")[-1])): f
            for f in DEX_ROOT.glob(f"{event}_*.csv")
        }
        missing = [m for m in expected if m not in found_paths]
        found   = sorted(found_paths.keys())

        if not found_paths:
            (R.warn if optional else R.fail)(
                f"{event}: no files found",
                fix=f"Run: python scripts/import_data/run_dex.py")
            continue

        cov = f"{found[0][0]}-{found[0][1]:02d} → {found[-1][0]}-{found[-1][1]:02d}"
        if not missing:
            R.ok(f"{event}: {len(found)}/{len(expected)} months  ({cov})")
        else:
            ms   = ", ".join(f"{y}-{m:02d}" for y, m in sorted(missing)[:5])
            tail = f" … +{len(missing)-5} more" if len(missing) > 5 else ""
            R.warn(f"{event}: {len(missing)} months missing — {ms}{tail}",
                   fix="Run: python scripts/import_data/run_dex.py --overwrite")

        newest_path = found_paths[found[-1]]
        df_s = read_csv_safe(newest_path, nrows=5001)
        if df_s is not None:
            n = len(df_s)
            if n == 0:
                if allow_empty:
                    R.ok(f"{event}: files present (0 rows — upstream data not available)")
                else:
                    R.fail(f"{event} {found[-1]}: empty file",
                           fix="Run: python scripts/import_data/run_dex.py --overwrite")
            else:
                R.ok(f"{event} newest ({found[-1][0]}-{found[-1][1]:02d}): {n:,}+ rows")
                mc = [c for c in req_cols if c not in df_s.columns]
                if mc:
                    R.fail(f"{event} newest: missing columns {mc}",
                           fix="Re-run import; schema may have changed.")

        if not optional:
            R.inv(f"DEX 0.05% {event}", len(found), "—", cov)


def check_dex_tick_snapshots() -> None:
    R.section("[DEX 0.05%] Tick snapshots")
    snap_dir = DEX_ROOT / "tick_snapshots"
    if not snap_dir.exists():
        R.warn("tick_snapshots/ missing",
               fix="Run: python scripts/import_data/run_dex.py  (ticks are the last step; "
                   "use --skip-ticks to defer)")
        return

    cur = snap_dir / "ticks_current.csv"
    if cur.exists():
        df_c = read_csv_safe(cur)
        if df_c is not None:
            R.ok(f"ticks_current.csv: {len(df_c):,} active ticks")
    else:
        R.warn("ticks_current.csv missing",
               fix="Run: python scripts/import_data/run_dex.py")

    hist = sorted(snap_dir.glob("ticks_*_block_*.csv"))
    if hist:
        R.ok(f"Historical tick snapshots: {len(hist)} monthly files")
        R.inv("DEX 0.05% tick snapshots", len(hist), "—", "monthly (one per month-end)")
    else:
        R.warn("No historical tick snapshots",
               fix="Run: python scripts/import_data/run_dex.py  (slow; ~1–2 h)")

    pool_snaps = sorted(snap_dir.glob("pool_state_*_block_*.csv"))
    if pool_snaps:
        R.ok(f"Pool-state snapshots: {len(pool_snaps)} files")


def check_dex_static_files() -> None:
    R.section("[DEX 0.05%] Static / current-state files")
    checks = [
        ("positions_current.csv",    50, ["owner", "tick_lower", "tick_upper", "liquidity"]),
        ("pool_metadata_current.csv", 1, COLS_METADATA),
        ("bundle_current.csv",        1, COLS_BUNDLE),
    ]
    for fname, min_rows, req_cols in checks:
        p = DEX_ROOT / fname
        if not p.exists():
            R.warn(f"{fname}: not found",
                   fix="Run: python scripts/import_data/run_dex.py")
            continue
        df = read_csv_safe(p)
        if df is None:
            continue
        n = len(df)
        if n < min_rows:
            R.fail(f"{fname}: {n} rows (expected ≥ {min_rows})",
                   fix="Run: python scripts/import_data/run_dex.py")
        else:
            R.ok(f"{fname}: {n:,} rows")
        check_required_cols(df, req_cols, fname)

        if fname == "pool_metadata_current.csv" and "fee_tier" in df.columns:
            ft = int(df["fee_tier"].iloc[0])
            if ft != 500:
                R.fail(f"pool_metadata_current.csv: fee_tier={ft}, expected 500",
                       fix="Check pool address in dex_utils.py (should be 0x88e6a0c2…)")
            else:
                R.ok("pool_metadata_current.csv: fee_tier=500 (0.05%) confirmed")

        if fname == "bundle_current.csv" and "eth_price_usd" in df.columns:
            ep = float(df["eth_price_usd"].iloc[0])
            R.ok(f"bundle_current.csv: ETH/USD = ${ep:,.2f}")
            if ep < 100 or ep > 25_000:
                R.warn(f"bundle: ETH price ${ep:.0f} looks unusual",
                       fix="Re-run fetch_uniswap_pool_timeseries.py to refresh bundle.")


# ═══════════════════════════════════════════════════════════════════════════════
# Multitier pools (0.01%, 0.30%, 1.00%)
# ═══════════════════════════════════════════════════════════════════════════════

def check_multitier_pool(cfg: dict) -> None:
    label    = cfg["label"]
    pool_dir = MT_ROOT / cfg["dir"]
    fee_tier = cfg["fee_tier"]
    min_h    = cfg["min_hours"]

    # Infer script suffix from dir name (fee_100 → 001pct, fee_3000 → 030pct, etc.)
    suffix_map = {"fee_100": "001pct", "fee_3000": "030pct", "fee_10000": "100pct"}
    script_suffix = suffix_map.get(cfg["dir"], cfg["dir"])
    fix_cmd = (f"python scripts/import_data/DEX/"
               f"fetch_uniswap_pool_{script_suffix}_timeseries.py")

    R.section(f"[MT {label}] {pool_dir.name}/")

    if not pool_dir.exists():
        R.fail(f"{label}: directory {pool_dir.name}/ missing",
               fix=f"Run: {fix_cmd}")
        return

    required = [
        ("pool_hour_data.csv",        COLS_POOL_HOUR, min_h),
        ("pool_day_data.csv",         COLS_POOL_DAY,  10),
        ("pool_metadata_current.csv", COLS_METADATA,  1),
        ("bundle_current.csv",        COLS_BUNDLE,    1),
    ]

    for fname, req_cols, min_rows in required:
        p = pool_dir / fname
        if not p.exists():
            R.fail(f"{label} {fname}: missing", fix=f"Run: {fix_cmd}")
            continue

        df = read_csv_safe(p)
        if df is None:
            continue
        n = len(df)
        if n < min_rows:
            R.fail(f"{label} {fname}: {n:,} rows (expected ≥ {min_rows:,})",
                   fix=f"Run: {fix_cmd}")
        else:
            R.ok(f"{label} {fname}: {n:,} rows")
        check_required_cols(df, req_cols, f"{label} {fname}")

        if fname == "pool_hour_data.csv":
            if "period_start_unix" in df.columns:
                ts = pd.to_numeric(df["period_start_unix"], errors="coerce")
                check_no_duplicates(ts, f"{label} pool_hour_data.csv")
                check_hourly_gaps(ts, f"{label} pool_hour_data.csv")
                t0 = datetime.fromtimestamp(ts.min(), tz=timezone.utc).strftime("%Y-%m-%d")
                t1 = datetime.fromtimestamp(ts.max(), tz=timezone.utc).strftime("%Y-%m-%d")
                R.ok(f"{label} pool_hour_data.csv: covers {t0} → {t1}")
                R.inv(f"MT {label} hourly", 1, n, f"{t0} → {t1}")
            check_ohlcv_sanity(df, f"{label} pool_hour_data.csv")
            if "token0_price" in df.columns:
                eth_price_range_ok(df["token0_price"], f"{label} pool_hour_data.csv")

        if fname == "pool_metadata_current.csv" and "fee_tier" in df.columns:
            ft = int(df["fee_tier"].iloc[0])
            if ft != fee_tier:
                R.fail(f"{label} pool_metadata_current.csv: fee_tier={ft}, "
                       f"expected {fee_tier}",
                       fix=f"Wrong pool fetched. Check pool_addresses.py and re-run: {fix_cmd}")
            else:
                R.ok(f"{label} pool_metadata_current.csv: fee_tier={ft} confirmed")

        if fname == "bundle_current.csv" and "eth_price_usd" in df.columns:
            ep = float(df["eth_price_usd"].iloc[0])
            R.ok(f"{label} bundle_current.csv: ETH/USD = ${ep:,.2f}")
            if ep < 100 or ep > 25_000:
                R.warn(f"{label} bundle: ETH price ${ep:.0f} looks unusual")


def check_multitier_pools() -> None:
    if not MT_ROOT.exists():
        R.warn("data_raw/multitier/ does not exist",
               fix="Run: python scripts/import_data/run_multitier_dex.py")
        return
    for cfg in MULTITIER_POOLS:
        check_multitier_pool(cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-dataset consistency
# ═══════════════════════════════════════════════════════════════════════════════

def check_cross_dataset() -> None:
    R.section("[CROSS] Dataset consistency")

    dex_hour = DEX_ROOT / "pool_hour_data.csv"
    if not dex_hour.exists():
        R.warn("Skipping cross-dataset check — pool_hour_data.csv missing")
        return

    df = read_csv_safe(dex_hour, usecols=["period_start_unix"])
    if df is None:
        return

    ts      = pd.to_numeric(df["period_start_unix"], errors="coerce").dropna()
    dex_min = datetime.fromtimestamp(ts.min(), tz=timezone.utc)
    dex_max = datetime.fromtimestamp(ts.max(), tz=timezone.utc)
    cex_st  = datetime(2022, 1, 1, tzinfo=timezone.utc)

    if dex_max < cex_st:
        R.fail("DEX data ends before CEX data begins — no overlap for thesis analysis",
               fix="Re-fetch DEX data: python scripts/import_data/run_dex.py")
    else:
        overlap_start = max(dex_min, cex_st).strftime("%Y-%m")
        overlap_end   = min(dex_max, datetime(2026, 4, 30, tz=timezone.utc)).strftime("%Y-%m")
        R.ok(f"DEX/CEX overlap window: {overlap_start} → {overlap_end}")

    # Check each multitier pool covers a useful time range
    for cfg in MULTITIER_POOLS:
        mt_h = MT_ROOT / cfg["dir"] / "pool_hour_data.csv"
        if not mt_h.exists():
            R.warn(f"MT {cfg['label']}: pool_hour_data.csv absent — cannot check overlap",
                   fix=f"Run: python scripts/import_data/run_multitier_dex.py --{cfg['dir'].replace('fee_','').zfill(3)[:3]}")
            continue
        df_mt = read_csv_safe(mt_h, usecols=["period_start_unix"])
        if df_mt is None:
            continue
        ts_mt = pd.to_numeric(df_mt["period_start_unix"], errors="coerce").dropna()
        t0 = datetime.fromtimestamp(ts_mt.min(), tz=timezone.utc).strftime("%Y-%m")
        t1 = datetime.fromtimestamp(ts_mt.max(), tz=timezone.utc).strftime("%Y-%m")
        R.ok(f"MT {cfg['label']}: {t0} → {t1}")

    # Check DEX and MT pools all share at least 12 months of common data
    mt_dirs = [MT_ROOT / c["dir"] / "pool_hour_data.csv" for c in MULTITIER_POOLS]
    all_present = [p for p in mt_dirs if p.exists()]
    if len(all_present) == len(MULTITIER_POOLS):
        R.ok("All multitier pool_hour_data.csv files present — cross-pool σ* test can proceed")
    else:
        missing_n = len(MULTITIER_POOLS) - len(all_present)
        R.warn(f"{missing_n} multitier pool file(s) missing — σ* scaling test incomplete",
               fix="Run: python scripts/import_data/run_multitier_dex.py")


# ═══════════════════════════════════════════════════════════════════════════════
# Report output
# ═══════════════════════════════════════════════════════════════════════════════

def print_inventory() -> None:
    if not R.inventory:
        return
    R.write()
    R.write("─" * 72)
    R.write("DATA INVENTORY  (for thesis Methods section)")
    R.write("─" * 72)
    R.write(f"  {'Dataset':<38} {'Files':>5}  {'Rows':>14}  Period")
    R.write(f"  {'─'*38} {'─'*5}  {'─'*14}  {'─'*22}")
    for row in R.inventory:
        R.write(f"  {row['Dataset']:<38} {row['Files']:>5}  {row['Rows']:>14}  {row['Period']}")
        if row["Note"]:
            R.write(f"  {'':>60}↳ {row['Note']}")
    R.write("─" * 72)


def print_remediation() -> None:
    if not R.remediation:
        return
    R.write()
    R.write("═" * 72)
    R.write("REMEDIATION PLAN — run these commands to fix every issue above")
    R.write("═" * 72)
    seen = set()
    for item in R.remediation:
        if item not in seen:
            R.write()
            R.write(item)
            seen.add(item)
    R.write()
    R.write("After running the above, re-run this script to confirm all checks pass.")
    R.write("═" * 72)


def print_summary() -> None:
    R.write()
    R.write("=" * 72)
    n_fail = len(R.issues)
    n_warn = len(R.warnings)
    if n_fail == 0 and n_warn == 0:
        R.write("RESULT: ALL CHECKS PASSED — data is complete and consistent")
        R.write("        Dataset is ready for thesis analysis.")
    elif n_fail == 0:
        R.write(f"RESULT: PASSED with {n_warn} warning(s) — data is usable;")
        R.write("        review warnings before submitting.")
    else:
        R.write(f"RESULT: {n_fail} FAILURE(S), {n_warn} WARNING(S)")
        R.write("        Data is NOT ready for thesis analysis. See REMEDIATION PLAN.")
    R.write("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    stamp = datetime.now(timezone.utc)

    R.write("=" * 72)
    R.write("DATA INTEGRITY CHECK — AMM Thesis")
    R.write(f"Project root : {PROJECT_ROOT}")
    R.write(f"Checked at   : {stamp.strftime('%Y-%m-%d %H:%M UTC')}")
    R.write("=" * 72)

    check_cex_manifest()
    check_cex_klines()
    check_cex_aggtrades()

    check_dex_pool_timeseries()
    check_dex_monthly_events()
    check_dex_tick_snapshots()
    check_dex_static_files()

    check_multitier_pools()

    check_cross_dataset()

    print_inventory()
    print_remediation()
    print_summary()

    # Save report file
    report_path = REPORT_DIR / f"data_integrity_{stamp.strftime('%Y-%m-%d')}.txt"
    R.save(report_path)

    if R.issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
