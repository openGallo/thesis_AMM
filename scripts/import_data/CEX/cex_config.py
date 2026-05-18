r"""
Shared configuration for CEX raw-data imports.

Target project layout:
    C:\Courses\thesis_AMM\scripts\import_data\CEX
    C:\Courses\thesis_AMM\data_raw\CEX

This file is intentionally small. Change the high-level import parameters
in main_import_cex.py, not here.
"""

from pathlib import Path


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(r"C:\Courses\thesis_AMM")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data_raw" / "CEX"

BINANCE_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "binance"


# ---------------------------------------------------------------------
# Binance public-data endpoints
# ---------------------------------------------------------------------

BINANCE_PUBLIC_DATA_BASE_URL = "https://data.binance.vision/data"
BINANCE_SPOT_API_BASE_URL = "https://api.binance.com"


# ---------------------------------------------------------------------
# Download behavior
# ---------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_SLEEP_SECONDS = 2.0

# Small pause between public-data downloads. This is polite and avoids
# hammering the endpoint when importing many files.
DEFAULT_SLEEP_BETWEEN_DOWNLOADS_SECONDS = 0.25


# ---------------------------------------------------------------------
# Raw CSV column names for later processing
# ---------------------------------------------------------------------

# Binance data-vision raw CSVs are usually stored without a header.
# These names are not injected into raw files; they are provided here for
# your future scripts in scripts/process_data/CEX.
KLINE_COLUMNS = [
    "open_time_ms",
    "open",
    "high",
    "low",
    "close",
    "volume_base",
    "close_time_ms",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

AGG_TRADE_COLUMNS = [
    "agg_trade_id",
    "price",
    "quantity",
    "first_trade_id",
    "last_trade_id",
    "transact_time_ms",
    "is_buyer_maker",
    "is_best_match",
]
