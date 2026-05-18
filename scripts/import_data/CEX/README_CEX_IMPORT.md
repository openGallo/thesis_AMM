# CEX raw-data import scripts

Put these files in:

```text
C:\Courses\thesis_AMM\scripts\import_data\CEX
```

The scripts write raw CEX data to:

```text
C:\Courses\thesis_AMM\data_raw\CEX\binance
```

## Files

```text
main_import_cex.py
cex_config.py
binance_download_utils.py
fetch_binance_klines.py
fetch_binance_agg_trades.py
collect_binance_orderbook_rest.py
README_CEX_IMPORT.md
```

## Install dependency

Only `requests` is required for the import scripts.

```powershell
C:\Interpreters\python.exe -m pip install requests
```

## Test import

The main script is configured for a small test by default:

- symbol: `ETHUSDC`
- dates: `2024-01-01` to `2024-01-02`
- granularity: daily
- klines: `1m`, `5m`
- aggTrades: on

Run:

```powershell
cd C:\Courses\thesis_AMM\scripts\import_data\CEX
C:\Interpreters\python.exe main_import_cex.py
```

Expected output folder:

```text
C:\Courses\thesis_AMM\data_raw\CEX\binance
```

Expected manifest:

```text
C:\Courses\thesis_AMM\data_raw\CEX\binance\download_manifest.csv
```

## Change parameters

Edit the `USER PARAMETERS` section at the top of:

```text
main_import_cex.py
```

Most important parameters:

```python
START_DATE = "2024-01-01"
END_DATE = "2024-01-02"
ARCHIVE_GRANULARITY = "daily"
SYMBOLS = ["ETHUSDC"]
INTERVALS = ["1m", "5m"]
DOWNLOAD_KLINES = True
DOWNLOAD_AGG_TRADES = True
```

## Full historical import

Use monthly files for the full import:

```powershell
C:\Interpreters\python.exe main_import_cex.py --full-history
```

The full preset uses:

```python
start_date = "2022-01-01"
end_date = "2026-04-30"
granularity = "monthly"
symbols = ["ETHUSDC", "ETHUSDT", "USDCUSDT"]
intervals = ["1m", "5m"]
```

Note: `ETHUSDC` did not exist for the whole 2022 period on Binance. Missing files are not fatal. They are recorded as `missing_404` in the manifest.

## Custom command examples

Small test with ETHUSDT:

```powershell
C:\Interpreters\python.exe main_import_cex.py --symbols ETHUSDT --start 2024-01-01 --end 2024-01-03 --granularity daily
```

Download klines only:

```powershell
C:\Interpreters\python.exe main_import_cex.py --symbols ETHUSDC --start 2024-01-01 --end 2024-01-03 --no-agg-trades
```

Monthly import for the thesis benchmark:

```powershell
C:\Interpreters\python.exe main_import_cex.py --symbols ETHUSDC,ETHUSDT,USDCUSDT --start 2022-01-01 --end 2026-04-30 --granularity monthly
```

## Live order-book collection

Historical full L2 order-book archives are not imported here. This script collects order-book snapshots prospectively.

Run for 60 seconds:

```powershell
C:\Interpreters\python.exe collect_binance_orderbook_rest.py --symbols ETHUSDC --seconds 60 --interval-sec 10 --limit 1000
```

Run until Ctrl+C:

```powershell
C:\Interpreters\python.exe collect_binance_orderbook_rest.py --symbols ETHUSDC,ETHUSDT --seconds 0 --interval-sec 10 --limit 1000
```

Output:

```text
C:\Courses\thesis_AMM\data_raw\CEX\binance\live_orderbook\ETHUSDC\
    depth_summary_YYYYMMDD.csv
    depth_snapshots_YYYYMMDD.ndjson
```

## Raw output structure

```text
C:\Courses\thesis_AMM\data_raw\CEX\binance\
    download_manifest.csv

    spot\daily\klines\ETHUSDC\1m\*.csv
    spot\daily\klines\ETHUSDC\5m\*.csv
    spot\daily\aggTrades\ETHUSDC\*.csv

    zip_archives\spot\daily\klines\ETHUSDC\1m\*.zip
    zip_archives\spot\daily\aggTrades\ETHUSDC\*.zip
```

## Processing stage

Keep these scripts as raw import scripts. Later, create processing scripts in:

```text
C:\Courses\thesis_AMM\scripts\process_data\CEX
```

Those scripts should build:

```text
CEX reference price
synthetic ETH/USDC benchmark from ETHUSDT and USDCUSDT
realized volatility
CEX volume
CEX trade counts
trade-size distribution
bid-ask spread
depth within 1, 5, 10, and 50 bps
DEX-CEX price deviation
```
