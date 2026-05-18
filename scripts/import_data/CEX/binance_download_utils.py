r"""
Utility functions for downloading Binance public historical data.

The functions in this module are shared by:
    - fetch_binance_klines.py
    - fetch_binance_agg_trades.py
    - main_import_cex.py

They download .zip files from data.binance.vision, optionally extract the
raw CSV inside, and append a manifest row for reproducibility.
"""

from __future__ import annotations

import csv
import hashlib
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests

from cex_config import (
    BINANCE_PUBLIC_DATA_BASE_URL,
    DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS,
    MAX_RETRIES,
    REQUEST_TIMEOUT_SECONDS,
    RETRY_SLEEP_SECONDS,
)


@dataclass(frozen=True)
class DownloadResult:
    exchange: str
    market_type: str
    data_type: str
    symbol: str
    interval: str
    granularity: str
    period: str
    url: str
    zip_path: Path
    extracted_dir: Optional[Path]
    status: str
    bytes_downloaded: int
    sha256: str
    error: str = ""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_yyyy_mm_dd(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid date '{value}'. Use YYYY-MM-DD.") from exc


def iter_days(start_date: str | date, end_date: str | date) -> Iterable[date]:
    if isinstance(start_date, str):
        start = parse_yyyy_mm_dd(start_date)
    else:
        start = start_date
    if isinstance(end_date, str):
        end = parse_yyyy_mm_dd(end_date)
    else:
        end = end_date

    if end < start:
        raise ValueError(f"end_date {end} is before start_date {start}")

    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def iter_months(start_date: str | date, end_date: str | date) -> Iterable[str]:
    """Yield YYYY-MM strings from the month of start_date to month of end_date."""
    if isinstance(start_date, str):
        start = parse_yyyy_mm_dd(start_date)
    else:
        start = start_date
    if isinstance(end_date, str):
        end = parse_yyyy_mm_dd(end_date)
    else:
        end = end_date

    if end < start:
        raise ValueError(f"end_date {end} is before start_date {start}")

    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        yield f"{year:04d}-{month:02d}"
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1


def periods_for_granularity(
    start_date: str | date,
    end_date: str | date,
    granularity: str,
) -> list[str]:
    granularity = granularity.lower().strip()
    if granularity == "daily":
        return [d.strftime("%Y-%m-%d") for d in iter_days(start_date, end_date)]
    if granularity == "monthly":
        return list(iter_months(start_date, end_date))
    raise ValueError("granularity must be 'daily' or 'monthly'")


def build_data_vision_url(
    data_type: str,
    symbol: str,
    period: str,
    granularity: str,
    interval: str = "",
) -> str:
    """
    Build Binance Data Vision URL.

    Examples:
        klines daily:
        /data/spot/daily/klines/ETHUSDC/1m/ETHUSDC-1m-2024-01-01.zip

        aggTrades daily:
        /data/spot/daily/aggTrades/ETHUSDC/ETHUSDC-aggTrades-2024-01-01.zip
    """
    data_type = data_type.strip()
    symbol = symbol.upper().strip()
    granularity = granularity.lower().strip()

    if data_type == "klines":
        if not interval:
            raise ValueError("interval is required for klines")
        filename = f"{symbol}-{interval}-{period}.zip"
        return (
            f"{BINANCE_PUBLIC_DATA_BASE_URL}/spot/{granularity}/klines/"
            f"{symbol}/{interval}/{filename}"
        )

    if data_type == "aggTrades":
        filename = f"{symbol}-aggTrades-{period}.zip"
        return (
            f"{BINANCE_PUBLIC_DATA_BASE_URL}/spot/{granularity}/aggTrades/"
            f"{symbol}/{filename}"
        )

    raise ValueError("data_type must be 'klines' or 'aggTrades'")


def local_paths(
    output_root: Path,
    data_type: str,
    symbol: str,
    period: str,
    granularity: str,
    interval: str = "",
) -> tuple[Path, Path]:
    """
    Return (zip_path, extract_dir) for one requested file.
    Raw extracted CSVs go under:
        output_root/binance/spot/<granularity>/<data_type>/<symbol>/<interval?>/
    ZIP archives go under:
        output_root/binance/zip_archives/spot/<granularity>/<data_type>/<symbol>/<interval?>/
    """
    symbol = symbol.upper().strip()
    granularity = granularity.lower().strip()
    data_type = data_type.strip()

    if data_type == "klines":
        if not interval:
            raise ValueError("interval is required for klines")
        zip_dir = (
            output_root
            / "zip_archives"
            / "spot"
            / granularity
            / data_type
            / symbol
            / interval
        )
        extract_dir = output_root / "spot" / granularity / data_type / symbol / interval
        zip_name = f"{symbol}-{interval}-{period}.zip"
    elif data_type == "aggTrades":
        zip_dir = output_root / "zip_archives" / "spot" / granularity / data_type / symbol
        extract_dir = output_root / "spot" / granularity / data_type / symbol
        zip_name = f"{symbol}-aggTrades-{period}.zip"
    else:
        raise ValueError("data_type must be 'klines' or 'aggTrades'")

    return zip_dir / zip_name, extract_dir


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download_zip(
    url: str,
    zip_path: Path,
    overwrite: bool = False,
    timeout_seconds: int = REQUEST_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
) -> tuple[str, int, str]:
    """
    Download a ZIP file.

    Returns:
        (status, bytes_downloaded, error)

    status:
        downloaded
        exists
        missing_404
        failed
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if zip_path.exists() and zip_path.stat().st_size > 0 and not overwrite:
        return "exists", zip_path.stat().st_size, ""

    tmp_path = zip_path.with_suffix(zip_path.suffix + ".tmp")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout_seconds) as response:
                if response.status_code == 404:
                    if tmp_path.exists():
                        tmp_path.unlink()
                    return "missing_404", 0, "404 Not Found"

                response.raise_for_status()

                bytes_written = 0
                with tmp_path.open("wb") as fh:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            bytes_written += len(chunk)

                if bytes_written == 0:
                    last_error = "Downloaded zero bytes"
                    continue

                tmp_path.replace(zip_path)
                return "downloaded", bytes_written, ""

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(RETRY_SLEEP_SECONDS)

    if tmp_path.exists():
        tmp_path.unlink()
    return "failed", 0, last_error


def extract_zip_csv(zip_path: Path, extract_dir: Path, overwrite: bool = False) -> list[Path]:
    """
    Extract CSV files from the ZIP into extract_dir.

    Raw Binance CSV files are kept exactly as downloaded.
    """
    extract_dir.mkdir(parents=True, exist_ok=True)

    extracted_files: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if not member.lower().endswith(".csv"):
                continue
            target_path = extract_dir / Path(member).name
            if target_path.exists() and not overwrite:
                extracted_files.append(target_path)
                continue
            with zf.open(member) as src, target_path.open("wb") as dst:
                dst.write(src.read())
            extracted_files.append(target_path)

    return extracted_files


def append_manifest_row(manifest_path: Path, result: DownloadResult) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp_utc": utc_now_iso(),
        "exchange": result.exchange,
        "market_type": result.market_type,
        "data_type": result.data_type,
        "symbol": result.symbol,
        "interval": result.interval,
        "granularity": result.granularity,
        "period": result.period,
        "url": result.url,
        "zip_path": str(result.zip_path),
        "extracted_dir": "" if result.extracted_dir is None else str(result.extracted_dir),
        "status": result.status,
        "bytes_downloaded": result.bytes_downloaded,
        "sha256": result.sha256,
        "error": result.error,
    }

    file_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def download_one_binance_public_file(
    output_root: Path,
    data_type: str,
    symbol: str,
    period: str,
    granularity: str,
    interval: str = "",
    overwrite: bool = False,
    extract: bool = True,
    keep_zip: bool = True,
    sleep_seconds: float = DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS,
) -> DownloadResult:
    r"""
    Download one Binance public-data ZIP and optionally extract its CSV.

    output_root should normally be:
        C:\Courses\thesis_AMM\data_raw\CEX\binance
    """
    url = build_data_vision_url(
        data_type=data_type,
        symbol=symbol,
        interval=interval,
        period=period,
        granularity=granularity,
    )
    zip_path, extract_dir = local_paths(
        output_root=output_root,
        data_type=data_type,
        symbol=symbol,
        interval=interval,
        period=period,
        granularity=granularity,
    )

    status, bytes_downloaded, error = download_zip(url, zip_path, overwrite=overwrite)

    extracted_dir: Optional[Path] = None
    sha256 = ""
    final_status = status
    final_error = error

    if status in {"downloaded", "exists"}:
        try:
            sha256 = sha256_file(zip_path)
            if extract:
                extracted_files = extract_zip_csv(zip_path, extract_dir, overwrite=overwrite)
                extracted_dir = extract_dir if extracted_files else None
                final_status = f"{status}_extracted" if extracted_files else f"{status}_no_csv"
            if not keep_zip and zip_path.exists():
                zip_path.unlink()
        except Exception as exc:
            final_status = "failed_extract"
            final_error = f"{type(exc).__name__}: {exc}"

    result = DownloadResult(
        exchange="binance",
        market_type="spot",
        data_type=data_type,
        symbol=symbol.upper().strip(),
        interval=interval,
        granularity=granularity,
        period=period,
        url=url,
        zip_path=zip_path,
        extracted_dir=extracted_dir,
        status=final_status,
        bytes_downloaded=bytes_downloaded,
        sha256=sha256,
        error=final_error,
    )

    append_manifest_row(output_root / "download_manifest.csv", result)

    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    return result
