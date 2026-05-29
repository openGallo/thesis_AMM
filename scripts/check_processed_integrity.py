"""
Processed data integrity check.

Validates all outputs of scripts/process_data/* for:
  - Existence and non-zero file size
  - Required schema (expected columns present)
  - Row counts and temporal coverage
  - Value ranges (ETH price, realized vol, APR, LVR)
  - No duplicate primary keys / timestamps
  - OHLC consistency (high >= open/close, low <= open/close)
  - Cross-file consistency: DEX-CEX price correlation, LVR formula,
    basis stationarity
  - Calibration parameter plausibility

Outputs:
  Console  — [OK] / [WARN] / [FAIL] per check
  File     — reports/processed_integrity_YYYY-MM-DD.txt

Exit code 1 if any [FAIL]; 0 if only [OK] / [WARN].

Usage:
    python scripts/check_processed_integrity.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROC    = PROJECT_ROOT / "data_processed"
REPORT_DIR   = PROJECT_ROOT / "reports"

# ── Thresholds ────────────────────────────────────────────────────────────────

ETH_PRICE_MIN    = 50.0
ETH_PRICE_MAX    = 25_000.0
VOL_ANN_MAX      = 5.0        # 500% — extreme but not impossible
CORR_DEX_CEX_MIN = 0.99       # DEX and CEX ETH price must co-move tightly
FEE_RATE_005     = 0.0005
FEE_RATE_TOL     = 1e-9

# Minimum row counts
MIN_1M_ROWS      = 1_200_000  # ~2.3 years × 365 × 24 × 60 with gaps
MIN_HOURLY_ROWS  = 20_000     # thesis: N=22,368 for main study period
MIN_DAILY_ROWS   = 600
MIN_SWAP_ROWS    = 50_000
MIN_LVR_ROWS     = 15_000
MIN_MERGED_ROWS  = 20_000

# Coverage window (main study)
STUDY_START = pd.Timestamp("2022-01-01", tz="UTC")
STUDY_END   = pd.Timestamp("2024-12-31", tz="UTC")

VALID_DIRECTIONS   = {"buy_eth", "sell_eth"}
VALID_SIZE_BUCKETS = {"<1k", "1k-10k", "10k-100k", ">100k"}
VALID_POS_TYPES    = {"narrow", "medium", "wide"}


# ── Report writer ─────────────────────────────────────────────────────────────

class Report:
    def __init__(self) -> None:
        self.lines:       list[str] = []
        self.issues:      list[str] = []
        self.warnings:    list[str] = []
        self.inventory:   list[dict] = []
        self.remediation: list[str] = []

    def _emit(self, line: str) -> None:
        print(line)
        self.lines.append(line)

    def section(self, title: str) -> None:
        sep = "=" * 70
        self._emit(f"\n{sep}")
        self._emit(f"  {title}")
        self._emit(sep)

    def subsection(self, title: str) -> None:
        self._emit(f"\n  -- {title} --")

    def ok(self, msg: str) -> None:
        self._emit(f"  [OK]   {msg}")

    def warn(self, msg: str, fix: str = "") -> None:
        self._emit(f"  [WARN] {msg}")
        self.warnings.append(msg)
        if fix:
            self.remediation.append(f"[WARN] {msg}\n       Fix: {fix}")

    def fail(self, msg: str, fix: str = "") -> None:
        self._emit(f"  [FAIL] {msg}")
        self.issues.append(msg)
        if fix:
            self.remediation.append(f"[FAIL] {msg}\n       Fix: {fix}")

    def info(self, msg: str) -> None:
        self._emit(f"         {msg}")

    def add_inventory(self, name: str, rows: int | str, cols: int | str,
                      period: str, size_mb: float) -> None:
        self.inventory.append({
            "file":     name,
            "rows":     rows,
            "cols":     cols,
            "period":   period,
            "size_mb":  f"{size_mb:.1f}",
        })

    def save(self) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = REPORT_DIR / f"processed_integrity_{date_str}.txt"

        with open(path, "w", encoding="utf-8") as fh:
            for line in self.lines:
                fh.write(line + "\n")

            fh.write("\n" + "=" * 70 + "\n")
            fh.write("  DATA INVENTORY\n")
            fh.write("=" * 70 + "\n")
            if self.inventory:
                header = f"  {'File':<42} {'Rows':>10} {'Cols':>5} {'Size MB':>8}  Period"
                fh.write(header + "\n")
                fh.write("  " + "-" * 80 + "\n")
                for item in self.inventory:
                    fh.write(
                        f"  {item['file']:<42} {str(item['rows']):>10} "
                        f"{str(item['cols']):>5} {item['size_mb']:>8}  {item['period']}\n"
                    )

            if self.remediation:
                fh.write("\n" + "=" * 70 + "\n")
                fh.write("  REMEDIATION PLAN\n")
                fh.write("=" * 70 + "\n")
                for i, r in enumerate(self.remediation, 1):
                    fh.write(f"\n[{i}] {r}\n")

            fh.write("\n" + "=" * 70 + "\n")
            fh.write(f"  SUMMARY: {len(self.issues)} FAIL  |  {len(self.warnings)} WARN\n")
            fh.write("=" * 70 + "\n")

        return path


# ── Helper functions ──────────────────────────────────────────────────────────

def load_csv(path: Path, index_col=None, parse_dates=False,
             parse_index_utc=False) -> pd.DataFrame | None:
    """Load CSV; return None if file doesn't exist."""
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates,
                     low_memory=False)
    if parse_index_utc and df.index is not None:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1_000_000
    except OSError:
        return 0.0


def check_exists(R: Report, path: Path, label: str,
                 fix: str = "") -> bool:
    """Returns True if file exists and is non-empty."""
    if not path.exists():
        R.fail(f"{label} missing: {path.name}",
               fix=fix or "python scripts/process_data/run_all.py")
        return False
    if path.stat().st_size == 0:
        R.fail(f"{label} exists but is empty: {path.name}",
               fix=fix or "python scripts/process_data/run_all.py")
        return False
    return True


def check_columns(R: Report, df: pd.DataFrame, required: list[str],
                  label: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        R.fail(f"{label}: missing columns {missing}")
        return False
    R.ok(f"{label}: all {len(required)} required columns present")
    return True


def check_row_count(R: Report, df: pd.DataFrame, min_rows: int, label: str,
                    fix: str = "") -> None:
    n = len(df)
    if n < min_rows:
        R.fail(f"{label}: only {n:,} rows (expected >= {min_rows:,})", fix=fix)
    else:
        R.ok(f"{label}: {n:,} rows")


def check_no_duplicate_index(R: Report, df: pd.DataFrame, label: str) -> None:
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        R.fail(f"{label}: {n_dup:,} duplicate index values")
    else:
        R.ok(f"{label}: no duplicate index values")


def check_no_duplicate_cols(R: Report, df: pd.DataFrame, key_cols: list[str],
                             label: str) -> None:
    if not all(c in df.columns for c in key_cols):
        R.warn(f"{label}: key columns {key_cols} not all present — dup check skipped")
        return
    n_dup = df.duplicated(subset=key_cols).sum()
    if n_dup > 0:
        R.fail(f"{label}: {n_dup:,} duplicate rows on key {key_cols}")
    else:
        R.ok(f"{label}: no duplicate rows on key {key_cols}")


def check_price_range(R: Report, series: pd.Series, label: str) -> None:
    s = series.dropna()
    if s.empty:
        R.warn(f"{label}: all NaN — cannot check price range")
        return
    lo, hi = float(s.min()), float(s.max())
    bad = ((s < ETH_PRICE_MIN) | (s > ETH_PRICE_MAX)).sum()
    if bad > 0:
        R.fail(f"{label}: {bad:,} prices outside [{ETH_PRICE_MIN}, {ETH_PRICE_MAX}]  "
               f"(min={lo:.1f}, max={hi:.1f})")
    else:
        R.ok(f"{label}: price range OK  min={lo:.1f}  max={hi:.1f}")


def check_ohlcv_sanity(R: Report, df: pd.DataFrame, prefix: str,
                        label: str) -> None:
    o = f"{prefix}open"; h = f"{prefix}high"
    l = f"{prefix}low";  c = f"{prefix}close"
    if not all(col in df.columns for col in (o, h, l, c)):
        R.warn(f"{label}: OHLC columns not found — skipped")
        return
    sub = df[[o, h, l, c]].dropna()
    if sub.empty:
        R.warn(f"{label}: OHLC all NaN")
        return
    bad_high = ((sub[h] < sub[o]) | (sub[h] < sub[c])).sum()
    bad_low  = ((sub[l] > sub[o]) | (sub[l] > sub[c])).sum()
    bad_hl   = (sub[h] < sub[l]).sum()
    if bad_high + bad_low + bad_hl > 0:
        R.fail(f"{label}: OHLC violations — "
               f"high<open/close: {bad_high}, low>open/close: {bad_low}, "
               f"high<low: {bad_hl}")
    else:
        R.ok(f"{label}: OHLC sanity OK ({len(sub):,} bars checked)")


def check_nan_rate(R: Report, series: pd.Series, label: str,
                   warn_pct: float = 30.0, fail_pct: float = 60.0) -> None:
    nan_pct = series.isna().mean() * 100
    if nan_pct >= fail_pct:
        R.fail(f"{label}: {nan_pct:.1f}% NaN (threshold {fail_pct:.0f}%)")
    elif nan_pct >= warn_pct:
        R.warn(f"{label}: {nan_pct:.1f}% NaN (threshold {warn_pct:.0f}%)")
    else:
        R.ok(f"{label}: {nan_pct:.1f}% NaN — OK")


def check_non_negative(R: Report, series: pd.Series, label: str,
                        allow_zero: bool = True) -> None:
    s = series.dropna()
    if s.empty:
        return
    threshold = 0.0 if allow_zero else 1e-12
    neg = (s < threshold).sum()
    if neg > 0:
        R.warn(f"{label}: {neg:,} negative values (min={s.min():.6f})")
    else:
        R.ok(f"{label}: all non-negative")


def check_hourly_gaps(R: Report, idx: pd.DatetimeIndex, label: str,
                      max_gap_hours: float = 6.0) -> None:
    if len(idx) < 2:
        return
    gaps = idx.to_series().diff().dropna()
    large = gaps[gaps > pd.Timedelta(hours=max_gap_hours)]
    if large.empty:
        R.ok(f"{label}: no gaps > {max_gap_hours:.0f}h")
    else:
        R.warn(f"{label}: {len(large)} gap(s) > {max_gap_hours:.0f}h  "
               f"(largest: {large.max()})")


def check_coverage(R: Report, idx: pd.DatetimeIndex, label: str) -> None:
    if idx.empty:
        R.fail(f"{label}: no timestamps")
        return
    t_min, t_max = idx.min(), idx.max()
    if t_min > STUDY_START + pd.Timedelta(days=90):
        R.warn(f"{label}: starts {t_min.date()} — missing early study period "
               f"(expected <= {STUDY_START.date()})")
    else:
        R.ok(f"{label}: coverage starts {t_min.date()}")

    if t_max < STUDY_END - pd.Timedelta(days=90):
        R.warn(f"{label}: ends {t_max.date()} — short of study end "
               f"(expected >= {STUDY_END.date()})")
    else:
        R.ok(f"{label}: coverage ends   {t_max.date()}")


# ── Per-file check functions ──────────────────────────────────────────────────

def check_cex_price_1m(R: Report) -> pd.DataFrame | None:
    R.section("CEX Price 1m  (cex_price_1m.csv)")
    path = DATA_PROC / "CEX" / "cex_price_1m.csv"

    if not check_exists(R, path, "cex_price_1m",
                        fix="python scripts/process_data/CEX/process_cex_price.py"):
        return None

    df = load_csv(path, index_col=0, parse_index_utc=True)
    size = file_size_mb(path)
    period = f"{df.index.min()} → {df.index.max()}" if not df.empty else "empty"
    R.add_inventory("CEX/cex_price_1m.csv", len(df), len(df.columns), period, size)

    required = ["eth_usdc_price", "eth_usdc_synthetic",
                "log_return_1m", "realized_vol_1h_ann", "realized_vol_24h_ann"]
    if not check_columns(R, df, required, "cex_price_1m"):
        return df

    check_row_count(R, df, MIN_1M_ROWS, "cex_price_1m",
                    fix="python scripts/process_data/CEX/process_cex_price.py")
    check_no_duplicate_index(R, df, "cex_price_1m")
    check_nan_rate(R, df["eth_usdc_price"], "eth_usdc_price", warn_pct=30, fail_pct=60)
    check_price_range(R, df["eth_usdc_price"], "eth_usdc_price (1m)")

    lr = df["log_return_1m"].dropna()
    if not lr.empty:
        mean_lr = float(lr.mean())
        if abs(mean_lr) > 0.001:
            R.warn(f"log_return_1m: mean={mean_lr:.6f} seems large for a 1m log-return")
        else:
            R.ok(f"log_return_1m: mean={mean_lr:.7f} — OK")

    vol = df["realized_vol_24h_ann"].dropna()
    if not vol.empty:
        bad_vol = ((vol < 0) | (vol > VOL_ANN_MAX)).sum()
        if bad_vol > 0:
            R.warn(f"realized_vol_24h_ann: {bad_vol} values outside [0, {VOL_ANN_MAX}]")
        else:
            R.ok(f"realized_vol_24h_ann: range [{vol.min():.3f}, {vol.max():.3f}] — OK")

    check_coverage(R, df.index, "cex_price_1m")
    return df


def check_cex_price_hourly(R: Report) -> pd.DataFrame | None:
    R.section("CEX Price Hourly  (cex_price_hourly.csv)")
    path = DATA_PROC / "CEX" / "cex_price_hourly.csv"

    if not check_exists(R, path, "cex_price_hourly",
                        fix="python scripts/process_data/CEX/process_cex_price.py"):
        return None

    df = load_csv(path, index_col=0, parse_index_utc=True)
    size = file_size_mb(path)
    period = f"{df.index.min()} → {df.index.max()}" if not df.empty else "empty"
    R.add_inventory("CEX/cex_price_hourly.csv", len(df), len(df.columns), period, size)

    required = [
        "eth_usdc_open", "eth_usdc_high", "eth_usdc_low", "eth_usdc_close",
        "log_return_1h", "realized_vol_24h_ann",
    ]
    if not check_columns(R, df, required, "cex_price_hourly"):
        return df

    check_row_count(R, df, MIN_HOURLY_ROWS, "cex_price_hourly",
                    fix="python scripts/process_data/CEX/process_cex_price.py")
    check_no_duplicate_index(R, df, "cex_price_hourly")
    check_nan_rate(R, df["eth_usdc_close"], "eth_usdc_close (hourly)",
                   warn_pct=20, fail_pct=40)
    check_ohlcv_sanity(R, df, "eth_usdc_", "cex_price_hourly")
    check_price_range(R, df["eth_usdc_close"], "eth_usdc_close (hourly)")
    check_non_negative(R, df["realized_vol_24h_ann"], "realized_vol_24h_ann")
    check_hourly_gaps(R, df.index, "cex_price_hourly", max_gap_hours=6)
    check_coverage(R, df.index, "cex_price_hourly")

    vol = df["realized_vol_24h_ann"].dropna()
    if not vol.empty:
        R.info(f"realized_vol_24h_ann: mean={vol.mean():.3f}  "
               f"p5={vol.quantile(0.05):.3f}  p95={vol.quantile(0.95):.3f}  "
               f"max={vol.max():.3f}")

    return df


def check_cex_orderbook(R: Report) -> None:
    R.section("CEX Order-Book Daily  (cex_orderbook_daily.csv)")
    path = DATA_PROC / "CEX" / "cex_orderbook_daily.csv"

    if not path.exists():
        R.warn("cex_orderbook_daily.csv not found — optional; "
               "only produced if collect_binance_orderbook_rest.py was run")
        return

    df = load_csv(path)
    size = file_size_mb(path)
    R.add_inventory("CEX/cex_orderbook_daily.csv", len(df), len(df.columns), "—", size)

    required = ["date", "symbol", "n_snapshots", "spread_bps_mean"]
    if not check_columns(R, df, required, "cex_orderbook_daily"):
        return

    check_row_count(R, df, 1, "cex_orderbook_daily")

    if "spread_bps_mean" in df.columns:
        spread = pd.to_numeric(df["spread_bps_mean"], errors="coerce").dropna()
        if (spread <= 0).any():
            R.fail("cex_orderbook_daily: non-positive spread_bps_mean values")
        else:
            R.ok(f"cex_orderbook_daily: spread_bps_mean "
                 f"min={spread.min():.3f}  max={spread.max():.3f} — OK")


def check_dex_pool_hourly(R: Report) -> pd.DataFrame | None:
    R.section("DEX Pool Hourly  (dex_pool_hourly.csv)")
    path = DATA_PROC / "DEX" / "dex_pool_hourly.csv"

    if not check_exists(R, path, "dex_pool_hourly",
                        fix="python scripts/process_data/DEX/process_dex_pool_hourly.py"):
        return None

    df = load_csv(path, index_col=0, parse_index_utc=True)
    size = file_size_mb(path)
    period = f"{df.index.min()} → {df.index.max()}" if not df.empty else "empty"
    R.add_inventory("DEX/dex_pool_hourly.csv", len(df), len(df.columns), period, size)

    required = [
        "eth_usdc_price", "eth_usdc_open", "eth_usdc_high", "eth_usdc_low",
        "eth_usdc_close", "volume_usd", "fees_usd", "tvl_usd", "tx_count",
        "fee_rate", "fee_apr_ann", "vol_over_tvl", "log_return_1h",
    ]
    if not check_columns(R, df, required, "dex_pool_hourly"):
        return df

    check_row_count(R, df, MIN_HOURLY_ROWS, "dex_pool_hourly",
                    fix="python scripts/process_data/DEX/process_dex_pool_hourly.py")
    check_no_duplicate_index(R, df, "dex_pool_hourly")
    check_ohlcv_sanity(R, df, "eth_usdc_", "dex_pool_hourly")
    check_price_range(R, df["eth_usdc_close"], "dex eth_usdc_close")
    check_hourly_gaps(R, df.index, "dex_pool_hourly", max_gap_hours=6)
    check_coverage(R, df.index, "dex_pool_hourly")

    # fee_rate must be exactly 0.0005 everywhere
    if "fee_rate" in df.columns:
        bad_fee = (df["fee_rate"] - FEE_RATE_005).abs() > FEE_RATE_TOL
        n_bad = int(bad_fee.sum())
        if n_bad > 0:
            R.fail(f"dex_pool_hourly: {n_bad} rows with fee_rate != {FEE_RATE_005}")
        else:
            R.ok(f"dex_pool_hourly: fee_rate = {FEE_RATE_005} everywhere")

    # fee_apr_ann: should be positive and finite; warn if implausibly high
    apr = df["fee_apr_ann"].dropna()
    if not apr.empty:
        bad = (apr < 0).sum()
        extreme = (apr > 50).sum()
        if bad > 0:
            R.fail(f"dex_pool_hourly: {bad} rows with fee_apr_ann < 0")
        else:
            R.ok(f"dex_pool_hourly: fee_apr_ann all non-negative  "
                 f"mean={apr.mean():.3%}  median={apr.median():.3%}")
        if extreme > 0:
            R.warn(f"dex_pool_hourly: {extreme} rows with fee_apr_ann > 5000%  "
                   "(may be periods of zero/near-zero TVL)")

    check_non_negative(R, df["tvl_usd"], "tvl_usd")
    check_non_negative(R, df["volume_usd"], "volume_usd")
    check_non_negative(R, df["fees_usd"], "fees_usd")

    # tx_count sanity
    tx = df["tx_count"].dropna()
    if not tx.empty:
        if (tx < 0).any():
            R.fail("dex_pool_hourly: negative tx_count values")
        else:
            R.ok(f"dex_pool_hourly: tx_count  mean={tx.mean():.1f}  "
                 f"median={tx.median():.0f}")

    return df


def check_dex_pool_daily(R: Report) -> None:
    R.section("DEX Pool Daily  (dex_pool_daily.csv)")
    path = DATA_PROC / "DEX" / "dex_pool_daily.csv"

    if not check_exists(R, path, "dex_pool_daily",
                        fix="python scripts/process_data/DEX/process_dex_pool_hourly.py"):
        return

    df = load_csv(path)
    size = file_size_mb(path)
    R.add_inventory("DEX/dex_pool_daily.csv", len(df), len(df.columns), "—", size)

    required = ["eth_usdc_price", "volume_usd", "fees_usd", "tvl_usd", "fee_rate"]
    check_columns(R, df, required, "dex_pool_daily")
    check_row_count(R, df, MIN_DAILY_ROWS, "dex_pool_daily",
                    fix="python scripts/process_data/DEX/process_dex_pool_hourly.py")

    if "fee_rate" in df.columns:
        bad_fee = (pd.to_numeric(df["fee_rate"], errors="coerce") - FEE_RATE_005).abs() > FEE_RATE_TOL
        n_bad = int(bad_fee.sum())
        if n_bad > 0:
            R.fail(f"dex_pool_daily: {n_bad} rows with fee_rate != {FEE_RATE_005}")
        else:
            R.ok(f"dex_pool_daily: fee_rate = {FEE_RATE_005} everywhere")

    check_ohlcv_sanity(R, df, "eth_usdc_", "dex_pool_daily")
    check_price_range(R, pd.to_numeric(df.get("eth_usdc_close", pd.Series(dtype=float)),
                                       errors="coerce"), "dex_pool_daily price")


def check_dex_swaps(R: Report) -> pd.DataFrame | None:
    R.section("DEX Swaps  (dex_swaps.csv)")
    path = DATA_PROC / "DEX" / "dex_swaps.csv"

    if not check_exists(R, path, "dex_swaps",
                        fix="python scripts/process_data/DEX/process_dex_swaps.py"):
        return None

    df = load_csv(path)
    size = file_size_mb(path)
    R.add_inventory("DEX/dex_swaps.csv", len(df), len(df.columns), "—", size)

    required = [
        "block_number", "log_index", "timestamp",
        "eth_usdc_price", "eth_usdc_price_x96",
        "direction", "trade_size_bucket",
    ]
    if not check_columns(R, df, required, "dex_swaps"):
        return df

    check_row_count(R, df, MIN_SWAP_ROWS, "dex_swaps",
                    fix="python scripts/process_data/DEX/process_dex_swaps.py")

    # Direction values
    if "direction" in df.columns:
        bad_dir = ~df["direction"].isin(VALID_DIRECTIONS) & df["direction"].notna()
        if bad_dir.any():
            R.fail(f"dex_swaps: unexpected direction values: "
                   f"{df.loc[bad_dir, 'direction'].unique()[:5]}")
        else:
            vc = df["direction"].value_counts()
            R.ok(f"dex_swaps: direction values OK  "
                 f"buy_eth={vc.get('buy_eth', 0):,}  sell_eth={vc.get('sell_eth', 0):,}")

    # trade_size_bucket
    if "trade_size_bucket" in df.columns:
        vals = set(df["trade_size_bucket"].dropna().astype(str).unique())
        unexpected = vals - VALID_SIZE_BUCKETS
        if unexpected:
            R.fail(f"dex_swaps: unexpected trade_size_bucket values: {unexpected}")
        else:
            R.ok(f"dex_swaps: trade_size_bucket values OK")

    check_price_range(R, pd.to_numeric(df["eth_usdc_price"], errors="coerce"),
                      "dex_swaps eth_usdc_price")

    # Cross-check x96 vs primary price (should correlate extremely closely)
    p_main = pd.to_numeric(df["eth_usdc_price"],     errors="coerce")
    p_x96  = pd.to_numeric(df["eth_usdc_price_x96"], errors="coerce")
    valid  = p_main.notna() & p_x96.notna() & (p_main > 0) & (p_x96 > 0)
    if valid.sum() > 100:
        # Both columns should be identical (primary is assigned from x96 in the script)
        diff_pct = ((p_main - p_x96).abs() / p_x96)[valid]
        max_diff = float(diff_pct.max())
        if max_diff > 1e-6:
            R.warn(f"dex_swaps: eth_usdc_price vs x96 max diff = {max_diff:.2e} "
                   "(expected 0 — they should be identical)")
        else:
            R.ok(f"dex_swaps: eth_usdc_price == eth_usdc_price_x96 everywhere")

    # Timestamp coverage
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            R.info(f"swap period: {ts.min().date()} → {ts.max().date()}  "
                   f"({len(ts):,} swaps)")

    # Block number monotonicity
    if "block_number" in df.columns:
        bn = pd.to_numeric(df["block_number"], errors="coerce")
        if not bn.is_monotonic_increasing:
            R.fail("dex_swaps: block_number is not monotonically increasing")
        else:
            R.ok("dex_swaps: block_number sorted ascending")

    # Gas cost sanity
    if "gas_cost_usd" in df.columns:
        gas = pd.to_numeric(df["gas_cost_usd"], errors="coerce").dropna()
        gas_pos = gas[gas > 0]
        if len(gas_pos) < len(gas) * 0.9:
            R.warn(f"dex_swaps: only {len(gas_pos)/len(gas)*100:.1f}% of swaps "
                   "have positive gas_cost_usd")
        else:
            R.ok(f"dex_swaps: gas_cost_usd  mean=${gas_pos.mean():.2f}  "
                 f"median=${gas_pos.median():.2f}")

    return df


def check_dex_lp_positions(R: Report) -> None:
    R.section("DEX LP Positions  (dex_lp_positions.csv)")
    path = DATA_PROC / "DEX" / "dex_lp_positions.csv"

    if not check_exists(R, path, "dex_lp_positions",
                        fix="python scripts/process_data/DEX/process_dex_lp_positions.py"):
        return

    df = load_csv(path)
    size = file_size_mb(path)
    R.add_inventory("DEX/dex_lp_positions.csv", len(df), len(df.columns), "—", size)

    required = [
        "owner", "tick_lower", "tick_upper",
        "total_minted_usd", "net_pnl_usd",
        "range_width_ticks", "range_width_pct", "position_type",
    ]
    if not check_columns(R, df, required, "dex_lp_positions"):
        return

    check_row_count(R, df, 100, "dex_lp_positions",
                    fix="python scripts/process_data/DEX/process_dex_lp_positions.py")
    check_no_duplicate_cols(R, df, ["owner", "tick_lower", "tick_upper"],
                            "dex_lp_positions")

    # Range width sanity
    rw = pd.to_numeric(df["range_width_ticks"], errors="coerce").dropna()
    if (rw <= 0).any():
        R.fail(f"dex_lp_positions: {(rw <= 0).sum()} positions with "
               "range_width_ticks <= 0")
    else:
        R.ok(f"dex_lp_positions: range_width_ticks all positive  "
             f"median={rw.median():.0f}")

    rp = pd.to_numeric(df["range_width_pct"], errors="coerce").dropna()
    bad_rp = ((rp < 0) | (rp > 500)).sum()
    if bad_rp > 0:
        R.fail(f"dex_lp_positions: {bad_rp} positions with "
               "range_width_pct outside [0, 500]")
    else:
        R.ok(f"dex_lp_positions: range_width_pct in [0, 500] — OK  "
             f"median={rp.median():.2f}%")

    # position_type values
    if "position_type" in df.columns:
        vals = set(df["position_type"].dropna().astype(str).unique())
        unexpected = vals - VALID_POS_TYPES
        if unexpected:
            R.fail(f"dex_lp_positions: unexpected position_type: {unexpected}")
        else:
            vc = df["position_type"].value_counts()
            R.ok(f"dex_lp_positions: position_type OK  "
                 + "  ".join(f"{k}={v:,}" for k, v in vc.items()))

    # net_pnl_usd: large positive bias would suggest accounting error
    pnl = pd.to_numeric(df["net_pnl_usd"], errors="coerce").dropna()
    if not pnl.empty:
        R.info(f"net_pnl_usd: mean=${pnl.mean():,.0f}  "
               f"median=${pnl.median():,.0f}  "
               f"pct_positive={100*(pnl>0).mean():.1f}%")


def check_dex_lvr(R: Report, dex_hourly: pd.DataFrame | None) -> pd.DataFrame | None:
    R.section("DEX LVR  (dex_lvr_hourly.csv)")
    path = DATA_PROC / "DEX" / "dex_lvr_hourly.csv"

    if not check_exists(R, path, "dex_lvr_hourly",
                        fix="python scripts/process_data/DEX/process_dex_lvr.py"):
        return None

    df = load_csv(path, index_col=0, parse_index_utc=True)
    size = file_size_mb(path)
    period = f"{df.index.min()} → {df.index.max()}" if not df.empty else "empty"
    R.add_inventory("DEX/dex_lvr_hourly.csv", len(df), len(df.columns), period, size)

    required = [
        "tvl_usd", "realized_vol_24h_ann",
        "lvr_usd_tvl_approx", "lvr_rate_ann",
    ]
    if not check_columns(R, df, required, "dex_lvr_hourly"):
        return df

    check_row_count(R, df, MIN_LVR_ROWS, "dex_lvr_hourly",
                    fix="python scripts/process_data/DEX/process_dex_lvr.py")
    check_no_duplicate_index(R, df, "dex_lvr_hourly")
    check_non_negative(R, df["lvr_rate_ann"], "lvr_rate_ann")
    check_non_negative(R, df["lvr_usd_tvl_approx"], "lvr_usd_tvl_approx")

    # Verify LVR formula: lvr_rate_ann = σ² / 8
    sigma = pd.to_numeric(df["realized_vol_24h_ann"], errors="coerce")
    lvr   = pd.to_numeric(df["lvr_rate_ann"],         errors="coerce")
    valid = sigma.notna() & lvr.notna() & (sigma > 0)
    if valid.sum() > 100:
        expected_lvr = sigma[valid] ** 2 / 8.0
        rel_err = ((lvr[valid] - expected_lvr) / expected_lvr).abs()
        max_rel = float(rel_err.max())
        if max_rel > 0.01:
            R.fail(f"dex_lvr_hourly: LVR formula mismatch  "
                   f"max relative error = {max_rel:.4f}  "
                   f"(expected lvr_rate_ann = σ² / 8)")
        else:
            R.ok(f"dex_lvr_hourly: lvr_rate_ann = σ²/8 verified  "
                 f"(max rel error {max_rel:.2e})")

    # Verify lvr_usd_tvl_approx ≈ tvl_usd × σ² / 8 × dt
    tvl = pd.to_numeric(df["tvl_usd"], errors="coerce")
    lvr_usd = pd.to_numeric(df["lvr_usd_tvl_approx"], errors="coerce")
    dt = 1.0 / (365.25 * 24)
    valid2 = tvl.notna() & sigma.notna() & lvr_usd.notna() & (tvl > 0) & (sigma > 0)
    if valid2.sum() > 100:
        expected_usd = (tvl[valid2] / 8.0) * sigma[valid2] ** 2 * dt
        rel_err2 = ((lvr_usd[valid2] - expected_usd) / expected_usd).abs()
        max_rel2 = float(rel_err2.max())
        if max_rel2 > 0.01:
            R.fail(f"dex_lvr_hourly: lvr_usd_tvl_approx formula mismatch  "
                   f"max rel error = {max_rel2:.4f}")
        else:
            R.ok(f"dex_lvr_hourly: lvr_usd_tvl_approx = TVL×σ²/8×dt verified")

    # Summary
    lvr_r = df["lvr_rate_ann"].dropna()
    if not lvr_r.empty:
        R.info(f"lvr_rate_ann: mean={lvr_r.mean():.4%}  "
               f"median={lvr_r.median():.4%}  "
               f"p95={lvr_r.quantile(0.95):.4%}")

    if "lvr_to_fee_ratio" in df.columns:
        ratio = pd.to_numeric(df["lvr_to_fee_ratio"], errors="coerce").dropna()
        if not ratio.empty:
            R.info(f"lvr_to_fee_ratio: mean={ratio.mean():.2f}  "
                   f"median={ratio.median():.2f}")

    return df


def check_merged_hourly(R: Report,
                        dex_hourly: pd.DataFrame | None,
                        cex_hourly: pd.DataFrame | None) -> None:
    R.section("Merged DEX+CEX Hourly  (merged_hourly.csv)")
    path = DATA_PROC / "merged" / "merged_hourly.csv"

    if not check_exists(R, path, "merged_hourly",
                        fix="python scripts/process_data/merged/process_merged_panel.py"):
        return

    df = load_csv(path, index_col=0, parse_index_utc=True)
    size = file_size_mb(path)
    period = f"{df.index.min()} → {df.index.max()}" if not df.empty else "empty"
    R.add_inventory("merged/merged_hourly.csv", len(df), len(df.columns), period, size)

    required = [
        "dex_eth_usdc_close", "cex_eth_usdc_close",
        "dex_cex_basis_bps", "arbitrage_flag",
    ]
    if not check_columns(R, df, required, "merged_hourly"):
        return

    check_row_count(R, df, MIN_MERGED_ROWS, "merged_hourly",
                    fix="python scripts/process_data/merged/process_merged_panel.py")
    check_no_duplicate_index(R, df, "merged_hourly")
    check_coverage(R, df.index, "merged_hourly")

    # Overlap count
    overlap = df[["dex_eth_usdc_close", "cex_eth_usdc_close"]].dropna()
    R.info(f"DEX+CEX joint non-NaN rows: {len(overlap):,}")
    if len(overlap) < 20_000:
        R.warn(f"merged_hourly: only {len(overlap):,} hours with both DEX and CEX price — "
               "check import coverage")
    else:
        R.ok(f"merged_hourly: {len(overlap):,} hours have joint DEX+CEX data")

    # Price correlation
    if len(overlap) > 100:
        corr = float(overlap["dex_eth_usdc_close"].corr(overlap["cex_eth_usdc_close"]))
        if corr < CORR_DEX_CEX_MIN:
            R.fail(f"merged_hourly: DEX-CEX price correlation = {corr:.5f} "
                   f"(threshold {CORR_DEX_CEX_MIN}) — potential price formula error")
        else:
            R.ok(f"merged_hourly: DEX-CEX price correlation = {corr:.5f}")

    # Basis bps: should be stationary near 0
    basis = pd.to_numeric(df["dex_cex_basis_bps"], errors="coerce").dropna()
    if not basis.empty:
        abs_med = float(basis.abs().median())
        if abs_med > 20:
            R.warn(f"merged_hourly: |basis| median = {abs_med:.2f} bps — "
                   "larger than expected; check price formula alignment")
        else:
            R.ok(f"merged_hourly: basis_bps  median={basis.median():.2f}  "
                 f"|median|={abs_med:.2f}  p95={basis.quantile(0.95):.2f}")

        extreme_basis = (basis.abs() > 500).sum()
        if extreme_basis > 0:
            R.warn(f"merged_hourly: {extreme_basis} rows with |basis| > 500 bps "
                   "(likely thin DEX / stale price hours)")

    # Arbitrage flag
    if "arbitrage_flag" in df.columns:
        arb = df["arbitrage_flag"]
        if arb.dtype not in (bool, "bool"):
            arb = arb.astype(str).str.lower().map({"true": True, "false": False})
        arb_pct = arb.mean() * 100
        if arb_pct > 50:
            R.warn(f"merged_hourly: {arb_pct:.1f}% of hours flagged as arbitrage "
                   "(|basis| > 5 bps) — seems high")
        else:
            R.ok(f"merged_hourly: arbitrage_flag = True for {arb_pct:.1f}% of hours")

    # DEX realized vol cols forwarded from CEX
    if "cex_realized_vol_24h_ann" in df.columns:
        vol = pd.to_numeric(df["cex_realized_vol_24h_ann"], errors="coerce").dropna()
        if not vol.empty:
            R.info(f"cex_realized_vol_24h_ann: mean={vol.mean():.3f}  "
                   f"p95={vol.quantile(0.95):.3f}")


def check_calibration(R: Report) -> None:
    R.section("Calibration Parameters  (calibration_params.json)")
    json_path = DATA_PROC / "calibration" / "calibration_params.json"
    csv_path  = DATA_PROC / "calibration" / "calibration_summary.csv"

    # JSON
    if not check_exists(R, json_path, "calibration_params.json",
                        fix="python scripts/process_data/calibration/process_calibration.py"):
        return

    try:
        with open(json_path, encoding="utf-8") as fh:
            params = json.load(fh)
        R.ok("calibration_params.json: valid JSON")
    except json.JSONDecodeError as exc:
        R.fail(f"calibration_params.json: invalid JSON — {exc}")
        return

    size = file_size_mb(json_path)
    R.add_inventory("calibration/calibration_params.json", "—", "—", "—", size)

    expected_sections = [
        "price_dynamics", "pool_state", "trade_arrival",
        "trade_size", "gas_dynamics", "stress_regime", "lvr",
    ]
    missing_sections = [s for s in expected_sections if s not in params]
    if missing_sections:
        R.warn(f"calibration_params.json: missing sections {missing_sections} "
               "(re-run after all upstream data is ready)")
    else:
        R.ok(f"calibration_params.json: all {len(expected_sections)} sections present")

    # Price dynamics plausibility
    pd_sec = params.get("price_dynamics", {})
    sigma_daily = pd_sec.get("sigma_daily")
    mu_daily    = pd_sec.get("mu_daily")
    if sigma_daily is not None:
        if not (0.005 < sigma_daily < 0.5):
            R.fail(f"calibration: sigma_daily = {sigma_daily:.5f} outside "
                   "[0.005, 0.5] — implausible")
        else:
            R.ok(f"calibration: sigma_daily = {sigma_daily:.5f} — OK")
    if mu_daily is not None:
        if abs(mu_daily) > 0.05:
            R.warn(f"calibration: |mu_daily| = {abs(mu_daily):.5f} seems large")
        else:
            R.ok(f"calibration: mu_daily = {mu_daily:.6f} — OK")

    # Trade arrival
    ta = params.get("trade_arrival", {})
    lam_hour = ta.get("lambda_hour")
    if lam_hour is not None:
        if lam_hour < 1:
            R.fail(f"calibration: lambda_hour = {lam_hour:.2f} < 1 — "
                   "active pool should have many transactions per hour")
        else:
            R.ok(f"calibration: lambda_hour = {lam_hour:.1f}  "
                 f"lambda_day = {ta.get('lambda_day', '?')}")

    # Pool state
    ps = params.get("pool_state", {})
    tvl_med = ps.get("median_tvl_usd")
    if tvl_med is not None:
        if tvl_med < 1e5:
            R.warn(f"calibration: median_tvl_usd = ${tvl_med:,.0f} — "
                   "seems low for USDC/WETH 0.05% pool")
        else:
            R.ok(f"calibration: median_tvl_usd = ${tvl_med:,.0f}")

    # Gas dynamics
    gd = params.get("gas_dynamics", {})
    gwei_mean = gd.get("mean_gas_gwei")
    if gwei_mean is not None:
        if not (1 < gwei_mean < 1000):
            R.warn(f"calibration: mean_gas_gwei = {gwei_mean:.1f} — "
                   "outside typical [1, 1000] Gwei range")
        else:
            R.ok(f"calibration: mean_gas_gwei = {gwei_mean:.1f}")

    # Stress regime
    sr = params.get("stress_regime", {})
    stress_frac = sr.get("stress_fraction")
    if stress_frac is not None:
        if not (0.05 < stress_frac < 0.20):
            R.warn(f"calibration: stress_fraction = {stress_frac:.3f} outside "
                   "[0.05, 0.20] — stress threshold may be mis-calibrated")
        else:
            R.ok(f"calibration: stress_fraction = {stress_frac:.3f} (90th pct vol) — OK")

    # LVR
    lvr_sec = params.get("lvr", {})
    lvr_rate = lvr_sec.get("lvr_rate_ann_mean")
    if lvr_rate is not None:
        R.ok(f"calibration: lvr_rate_ann_mean = {lvr_rate:.4%}")

    # CSV version
    R.subsection("calibration_summary.csv")
    if not check_exists(R, csv_path, "calibration_summary.csv",
                        fix="python scripts/process_data/calibration/process_calibration.py"):
        return

    csv_df = load_csv(csv_path)
    R.add_inventory("calibration/calibration_summary.csv", len(csv_df), 3, "—",
                    file_size_mb(csv_path))
    if "section" not in csv_df.columns or "parameter" not in csv_df.columns:
        R.fail("calibration_summary.csv: missing section/parameter columns")
    else:
        R.ok(f"calibration_summary.csv: {len(csv_df)} parameter rows")


# ── Cross-file consistency checks ─────────────────────────────────────────────

def check_cross_consistency(R: Report,
                             dex_h: pd.DataFrame | None,
                             cex_h: pd.DataFrame | None,
                             lvr_h: pd.DataFrame | None) -> None:
    R.section("Cross-File Consistency Checks")

    # 1. DEX-CEX price alignment (independent of merged panel)
    if dex_h is not None and cex_h is not None:
        R.subsection("DEX-CEX price co-movement")
        dex_close = dex_h["eth_usdc_close"].dropna()
        cex_close = cex_h["eth_usdc_close"].dropna() if "eth_usdc_close" in cex_h.columns else pd.Series(dtype=float)

        common_idx = dex_close.index.intersection(cex_close.index)
        if len(common_idx) > 100:
            d = dex_close.reindex(common_idx)
            c = cex_close.reindex(common_idx)
            corr = float(d.corr(c))
            if corr < CORR_DEX_CEX_MIN:
                R.fail(f"DEX-CEX close price correlation = {corr:.5f} "
                       f"(min {CORR_DEX_CEX_MIN})")
            else:
                R.ok(f"DEX-CEX close price correlation = {corr:.5f}")

            mean_abs_diff_bps = float(((d - c).abs() / c * 10_000).median())
            R.info(f"Median |DEX-CEX| price deviation = {mean_abs_diff_bps:.2f} bps")
        else:
            R.warn("Not enough common timestamps for DEX-CEX correlation check")

    # 2. LVR σ² must match CEX realized vol used in computation
    if lvr_h is not None and cex_h is not None:
        R.subsection("LVR vs CEX vol consistency")
        cex_vol = cex_h.get("realized_vol_24h_ann") if cex_h is not None else None
        lvr_vol  = lvr_h.get("realized_vol_24h_ann") if lvr_h is not None else None

        if cex_vol is not None and lvr_vol is not None:
            common = cex_vol.dropna().index.intersection(lvr_vol.dropna().index)
            if len(common) > 100:
                diff = (cex_vol.reindex(common) - lvr_vol.reindex(common)).abs()
                if diff.max() > 1e-6:
                    R.warn(f"realized_vol_24h_ann differs between cex_price_hourly and "
                           f"dex_lvr_hourly (max diff {diff.max():.2e}) — "
                           "LVR may have been computed with a different CEX dataset")
                else:
                    R.ok("realized_vol_24h_ann matches between CEX hourly and LVR files")

    # 3. DEX-hourly row count vs thesis claim
    if dex_h is not None:
        R.subsection("DEX hourly row count vs thesis N=22,368")
        study_mask = (
            (dex_h.index >= STUDY_START) &
            (dex_h.index <= STUDY_END)
        )
        n_study = int(study_mask.sum())
        if n_study < 20_000:
            R.warn(f"DEX hourly rows in study period (2022–2024): {n_study:,} — "
                   "thesis states N=22,368; possible data gaps")
        else:
            R.ok(f"DEX hourly rows in study period (2022–2024): {n_study:,} "
                 f"(thesis N=22,368)")

    # 4. Calibration sigma_daily vs realized CEX vol
    if cex_h is not None:
        R.subsection("Calibration sigma vs CEX realized vol")
        cal_path = DATA_PROC / "calibration" / "calibration_params.json"
        if cal_path.exists():
            with open(cal_path, encoding="utf-8") as fh:
                params = json.load(fh)
            sigma_daily_cal = params.get("price_dynamics", {}).get("sigma_daily")
            if sigma_daily_cal is not None:
                cex_sigma_h = cex_h["log_return_1h"].dropna().std() if "log_return_1h" in cex_h.columns else None
                if cex_sigma_h is not None:
                    sigma_daily_live = float(cex_sigma_h * np.sqrt(24))
                    rel_diff = abs(sigma_daily_cal - sigma_daily_live) / sigma_daily_live
                    if rel_diff > 0.10:
                        R.warn(f"calibration sigma_daily ({sigma_daily_cal:.5f}) differs "
                               f"from current CEX data ({sigma_daily_live:.5f}) by "
                               f"{rel_diff*100:.1f}% — re-run process_calibration.py "
                               "if CEX data was updated")
                    else:
                        R.ok(f"calibration sigma_daily matches CEX data within "
                             f"{rel_diff*100:.1f}%")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(R: Report) -> None:
    R.section("SUMMARY")

    if R.inventory:
        R._emit(f"\n  {'File':<42} {'Rows':>10} {'Cols':>5} {'Size MB':>8}  Period")
        R._emit("  " + "-" * 85)
        for item in R.inventory:
            R._emit(f"  {item['file']:<42} {str(item['rows']):>10} "
                    f"{str(item['cols']):>5} {item['size_mb']:>8}  {item['period']}")

    R._emit(f"\n  FAIL:  {len(R.issues)}")
    R._emit(f"  WARN:  {len(R.warnings)}")

    if R.issues:
        R._emit("\n  Critical issues:")
        for i, issue in enumerate(R.issues, 1):
            R._emit(f"    {i}. {issue}")

    if R.remediation:
        R._emit("\n  Remediation plan:")
        for i, rem in enumerate(R.remediation, 1):
            R._emit(f"\n  [{i}] {rem}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    R = Report()

    R._emit("=" * 70)
    R._emit("  PROCESSED DATA INTEGRITY CHECK")
    R._emit(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    R._emit("=" * 70)
    R._emit(f"  Data root: {DATA_PROC}")

    # Run all checks; collect dataframes for cross-checks
    check_cex_price_1m(R)
    cex_hourly   = check_cex_price_hourly(R)
    check_cex_orderbook(R)
    dex_hourly   = check_dex_pool_hourly(R)
    check_dex_pool_daily(R)
    check_dex_swaps(R)
    check_dex_lp_positions(R)
    lvr_hourly   = check_dex_lvr(R, dex_hourly)
    check_merged_hourly(R, dex_hourly, cex_hourly)
    check_calibration(R)
    check_cross_consistency(R, dex_hourly, cex_hourly, lvr_hourly)

    print_summary(R)

    report_path = R.save()
    print(f"\nReport saved → {report_path}")

    sys.exit(1 if R.issues else 0)


if __name__ == "__main__":
    main()
