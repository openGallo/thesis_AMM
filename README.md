# thesis_AMM — Public Repository

Empirical study of the USDC/WETH 0.05% pool (`0x88e6A0c2ddd26FEEb64F039a2c41296FcB3f5640`) over May 2021–December 2024.

> **Note:** Analysis scripts, thesis LaTeX source, and figures live in a **private companion repository** (`thesis-AMM-private`). This public repo contains only the data-import and processing pipeline.

### Key Results

| Quantity | Value |
|---|---|
| Break-even volatility σ\* | 63.4% (annualised) |
| Reduced-form jump (Δlog TVL +24h) | −0.00312 (SE = 0.00089, *p* < 0.001) |
| LATE (fuzzy RD / Wald) | −0.0167 (SE = 0.0051) |
| McCrary density test | *p* = 0.44 (no manipulation) |
| Sample | 31,258 hourly observations |

---

## Repository structure

```
thesis_AMM/
│
├── scripts/
│   ├── import_data/
│   │   ├── run_cex.py                  ← run all CEX imports
│   │   ├── run_dex.py                  ← run all DEX imports
│   │   ├── CEX/
│   │   │   ├── main_import_cex.py      Binance klines + aggTrades downloader
│   │   │   ├── fetch_binance_klines.py
│   │   │   ├── fetch_binance_agg_trades.py
│   │   │   ├── collect_binance_orderbook_rest.py
│   │   │   ├── binance_download_utils.py
│   │   │   └── cex_config.py
│   │   └── DEX/
│   │       ├── fetch_uniswap_pool_timeseries.py   hourly/daily pool OHLCV
│   │       ├── fetch_all_monthly.py               swaps, mints, burns
│   │       ├── fetch_uniswap_extra_events.py      collects, flashes
│   │       ├── fetch_uniswap_positions.py         current LP positions
│   │       ├── fetch_uniswap_tick_snapshots.py    month-end tick liquidity
│   │       └── dex_utils.py                       shared GraphQL helpers
│   │
│   └── process_data/
│       ├── run_all.py                  ← run full processing pipeline
│       ├── CEX/
│       │   ├── process_cex_price.py    1m + hourly price panel, realized vol
│       │   └── process_cex_orderbook.py  daily spread and depth summary
│       ├── DEX/
│       │   ├── process_dex_pool_hourly.py   pool OHLCV, fee APR, vol/TVL
│       │   ├── process_dex_swaps.py         swap panel, gas costs, direction
│       │   ├── process_dex_lp_positions.py  LP P&L, range metrics, duration
│       │   └── process_dex_lvr.py           Loss-Versus-Rebalancing estimates
│       ├── merged/
│       │   └── process_merged_panel.py  DEX+CEX join, basis bps, arb flags
│       └── calibration/
│           └── process_calibration.py   GBM, Poisson-lognormal, gas AR(1)
│
├── data_raw/          (gitignored — populated by import scripts)
├── data_processed/    (gitignored — populated by processing scripts)
└── output/
    ├── figures/
    └── tables/
```

---

## Data sources

| Source | What | Coverage |
|---|---|---|
| [Uniswap v3 subgraph](https://thegraph.com/explorer/) | Pool OHLCV, swaps, mints, burns, collects, positions | May 2021 → present |
| [Binance public archive](https://data.binance.vision) | ETHUSDC, ETHUSDT, USDCUSDT klines (1m, 5m) | Jan 2022 → Apr 2026 |

> ETHUSDC was listed on Binance in August 2022. For earlier months the processing script constructs a synthetic rate: `ETHUSDT / USDCUSDT`.

---

## Setup

**Requirements:** Python 3.13+

```powershell
pip install requests pandas numpy
```

**API key** (The Graph, required for DEX imports):

```powershell
setx THEGRAPH_API_KEY "your_key_here"
# Restart terminal after running setx
```

---

## How to run

Open a terminal in the project root. Run in this order:

### 1 — Import CEX data
```powershell
python scripts/import_data/run_cex.py
```
Downloads Binance klines for ETHUSDC, ETHUSDT, USDCUSDT (monthly files, 2022–2026).
**~20–60 min** depending on connection speed.

### 2 — Import DEX data
```powershell
python scripts/import_data/run_dex.py --skip-ticks
```
Fetches pool time series, swaps, mints, burns, collects, and LP positions from The Graph subgraph.
**~1–3 hours** (rate-limited API). Steps 1 and 2 are independent and can run in parallel.

### 3 — Process everything
```powershell
python scripts/process_data/run_all.py
```
Runs all processing scripts in dependency order. **~2–5 min.**

---

## Processed outputs

| File | Contents |
|---|---|
| `data_processed/CEX/cex_price_1m.csv` | 1-minute ETH/USDC price panel |
| `data_processed/CEX/cex_price_hourly.csv` | Hourly OHLC + realized vol (1h/24h/7d ann.) |
| `data_processed/CEX/cex_orderbook_daily.csv` | Daily bid-ask spread and depth summary |
| `data_processed/DEX/dex_pool_hourly.csv` | Pool OHLCV, TVL, fees, fee APR, vol/TVL |
| `data_processed/DEX/dex_pool_daily.csv` | Same, daily granularity |
| `data_processed/DEX/dex_swaps.csv` | Swap panel with direction, gas cost, trade size |
| `data_processed/DEX/dex_lp_positions.csv` | LP P&L by position, range width, duration |
| `data_processed/DEX/dex_lvr_hourly.csv` | LVR estimates (TVL approx + v3 active-liquidity) |
| `data_processed/merged/merged_hourly.csv` | DEX+CEX joined panel, basis bps, arb flag |
| `data_processed/calibration/calibration_params.json` | All simulation parameters (JSON) |
| `data_processed/calibration/calibration_summary.csv` | Same, flat CSV for quick inspection |

---

## Key variables

| Variable | Formula | Location |
|---|---|---|
| `eth_usdc_price` | `10¹² / (sqrtPriceX96 / 2⁹⁶)²` | dex_swaps, dex_pool_hourly |
| `fee_apr_ann` | `fees_usd / tvl_usd × 24 × 365.25` | dex_pool_hourly |
| `realized_vol_24h_ann` | `std(log_ret_1h) × √8760` (rolling 24h) | cex_price_hourly |
| `dex_cex_basis_bps` | `(dex_price − cex_price) / cex_price × 10000` | merged_hourly |
| `arbitrage_flag` | `\|basis_bps\| > 5` (exceeds 0.05% fee) | merged_hourly |
| `lvr_rate_ann` | `σ²_ann / 8` (fraction of TVL per year) | dex_lvr_hourly |
| `net_pnl_usd` | `burns_usd + collects_usd − mints_usd` | dex_lp_positions |

---

## Data policy

Raw and processed datasets are **not committed to GitHub** (gitignored).
The repository stores only code and documentation. All data can be fully reproduced by running the import and processing scripts above.
