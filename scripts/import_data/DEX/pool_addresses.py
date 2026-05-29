"""
pool_addresses.py
=================
Central registry of Uniswap v3 USDC/WETH pool addresses used in the thesis.

Addresses are lowercase (matching The Graph's id field format).
fee_tier: raw contract value (fee in units of 1/1,000,000, e.g. 500 = 0.05%).
fee_rate: decimal form (e.g. 0.0005).

Each pool has a dedicated fetch script and a data directory under data_raw/.

    Fee tier  |  Script                                   |  data_raw/ path
    ----------|-------------------------------------------|-----------------------------
    0.01%     |  fetch_uniswap_pool_001pct_timeseries.py  |  multitier/fee_100/
    0.05%     |  fetch_uniswap_pool_timeseries.py         |  DEX/
    0.30%     |  fetch_uniswap_pool_030pct_timeseries.py  |  multitier/fee_3000/
    1.00%     |  fetch_uniswap_pool_100pct_timeseries.py  |  multitier/fee_10000/
"""

from __future__ import annotations

# ── 0.01% pool — multi-tier extension ────────────────────────────────────────
POOL_001 = "0xe0554a476a092703abdb3ef35c80e0d76d32939f"

# ── 0.05% pool — main study pool ─────────────────────────────────────────────
POOL_005 = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"

# ── 0.30% pool — multi-tier extension ────────────────────────────────────────
POOL_030 = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"

# ── 1.00% pool — multi-tier extension ────────────────────────────────────────
POOL_100 = "0x7bea39867e4169dbe237d55c8242a8f2fcdcc387"

# ── Structured registry (for code that needs to iterate) ─────────────────────
MULTITIER_POOLS: list[dict] = [
    {
        "label":    "USDC/WETH 0.01%",
        "address":  POOL_001,
        "fee_tier": 100,
        "fee_rate": 0.0001,
        "data_dir": "multitier/fee_100",
        "script":   "fetch_uniswap_pool_001pct_timeseries.py",
    },
    {
        "label":    "USDC/WETH 0.30%",
        "address":  POOL_030,
        "fee_tier": 3000,
        "fee_rate": 0.003,
        "data_dir": "multitier/fee_3000",
        "script":   "fetch_uniswap_pool_030pct_timeseries.py",
    },
    {
        "label":    "USDC/WETH 1.00%",
        "address":  POOL_100,
        "fee_tier": 10000,
        "fee_rate": 0.01,
        "data_dir": "multitier/fee_10000",
        "script":   "fetch_uniswap_pool_100pct_timeseries.py",
    },
]
