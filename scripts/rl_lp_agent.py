"""
rl_lp_agent.py v3 — RL Liquidity Provider for Uniswap v3

Environment design: Design B (historical replay; sec:rl-env).
Small-LP assumption: agent capital << pool TVL.

Improvements over v2
--------------------
1.  Full gymnasium.Env subclass — enables EvalCallback, make_vec_env, SubprocVecEnv
2.  27-feature state exploiting ALL available data sources
        Market (8): multi-horizon returns + vol + OHLC range + DEX-CEX basis
        Pool   (5): fee APR + volume/TVL + tx count + LVR/fee ratio + CEX volume
        Position(6): range metrics + cumulative episode P&L
        Cost   (7): gas + LVR rate + intra-day + day-of-week seasonality
        Arb    (1): arbitrage flag (|basis| > 5 bps)
3.  3D action space: MultiDiscrete([8 widths, 5 capital fracs, 5 center offsets])
        Center offset shifts range up/down relative to price (directional bets)
4.  Exact v3 concentration factor: sqrt(p_b/p_a) / (sqrt(p_b/p_a) - 1)
5.  Fee income uses actual pool fees_usd when available (not just APR formula)
6.  LVR cost uses actual lvr_usd_tvl_approx when available (not just formula)
7.  70 / 10 / 20 train / val / test split; EvalCallback on validation set
8.  Linear LR schedule; best model selected by validation P&L across seeds
9.  HAR-vol benchmark: w = k* x (vol_1h + vol_24h + vol_7d) / 3 (calibrated)
10. Calibration params loaded from data_processed/calibration/calibration_params.json

Honest-reporting checklist (sec:rl-honest)
------------------------------------------
  Environment design:      Design B (zero price impact; documented)
  Capital as % of TVL:     reported in evaluation table
  Reward (train):          Form 2 (decomposed) + optional Form 4 (risk-adjusted)
  Reward (report):         raw P&L components
  Decomposed P&L:          fees, LVR, gas, swap cost
  Benchmarks:              7 (passive wide/narrow/mid, vol-scaled, HAR-vol, CDM, hold)
  Risk metrics:            Sharpe, Sortino, CVaR5, max drawdown
  Significance:            block-bootstrap 95% CI (24h blocks, 1000 resamples)
  Seeds:                   configurable (default 3)

Usage
-----
    pip install gymnasium stable-baselines3 torch
    python scripts/rl_lp_agent.py [--timesteps 500000] [--seeds 3] [--eval-only]
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm as sp_norm

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROC    = PROJECT_ROOT / "data_processed"
OUTPUT_FIGS  = PROJECT_ROOT / "output" / "figures"
OUTPUT_TABS  = PROJECT_ROOT / "output" / "tables"
MODEL_DIR    = PROJECT_ROOT / "output" / "rl"
for _d in [OUTPUT_FIGS, OUTPUT_TABS, MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Simulation constants ───────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000     # USD
GAS_UNITS         = 200_000     # gas per rebalance (burn + mint)
POOL_FEE_RATE     = 0.0005      # 0.05% tier
DEFAULT_ETH_PRICE = 2_000
EPISODE_HOURS     = 720         # 30-day episodes
NORM_WINDOW       = 168         # 7-day rolling window for z-scores
TRAIN_FRAC        = 0.70
VAL_FRAC          = 0.10        # validation set for EvalCallback
REWARD_SCALE      = 1_000
SEED              = 42
N_ENVS            = 4           # parallel environments for PPO

# ── Action space ──────────────────────────────────────────────────────────────
# Axis 0: range half-width (fraction of price)
RANGE_WIDTHS   = [0.005, 0.010, 0.020, 0.040, 0.060, 0.100, 0.150, 0.200]
# Axis 1: fraction of capital to deploy
CAPITAL_FRACS  = [0.00,  0.25,  0.50,  0.75,  1.00]
# Axis 2: range center offset in multiples of hourly sigma
#   negative = bearish shift (range below current price)
#   positive = bullish shift (range above current price)
CENTER_OFFSETS = [-2.0, -1.0, 0.0, 1.0, 2.0]

# ── Columns that receive rolling z-score normalization ────────────────────────
NORM_COLS = [
    "log_return_1h", "log_return_6h", "log_return_24h",
    "realized_vol_1h_ann", "realized_vol_24h_ann", "realized_vol_7d_ann",
    "ohlc_range_1h",
    "dex_cex_basis_bps",
    "fee_apr_ann", "vol_over_tvl", "tx_count",
    "lvr_to_fee_ratio",
    "cex_vol_1h",
    "gas_gwei", "lvr_rate_ann",
]


# ── Calibration params ────────────────────────────────────────────────────────

def load_calibration() -> dict:
    """Load fitted params from process_calibration.py (empty dict if not ready)."""
    cal = DATA_PROC / "calibration" / "calibration_params.json"
    if cal.exists():
        with open(cal, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ── Data loading ──────────────────────────────────────────────────────────────

def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False)
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
    return df.sort_index()


def load_data() -> pd.DataFrame:
    """
    Build full hourly panel from all processed data sources.

    Features added beyond base DEX panel:
        CEX:    realized_vol_1h/24h/7d_ann, cex_vol_1h
        LVR:    lvr_rate_ann, lvr_to_fee_ratio, lvr_usd_tvl_approx (actual $/hr)
        Merged: dex_cex_basis_bps, arbitrage_flag
        Swaps:  gas_gwei (hourly median from swap-level gas_price_wei)
        Derived: ohlc_range_1h = (high-low)/close, log_return_6h/24h

    Rolling normalization stats computed with shift(1) — no look-ahead.
    """
    dex    = _read_csv(DATA_PROC / "DEX"    / "dex_pool_hourly.csv")
    cex    = _read_csv(DATA_PROC / "CEX"    / "cex_price_hourly.csv")
    lvr    = _read_csv(DATA_PROC / "DEX"    / "dex_lvr_hourly.csv")
    merged = _read_csv(DATA_PROC / "merged" / "merged_hourly.csv")

    if dex is None:
        raise FileNotFoundError(
            "data_processed/DEX/dex_pool_hourly.csv not found — "
            "run scripts/process_data/run_all.py first.")

    panel = dex.copy()

    # ── CEX features ──────────────────────────────────────────────────────────
    if cex is not None:
        for col in ["log_return_1h", "realized_vol_1h_ann",
                    "realized_vol_24h_ann", "realized_vol_7d_ann",
                    "vol_base_1h_ethusdt"]:
            if col in cex.columns and col not in panel.columns:
                panel[col] = cex[col].reindex(panel.index)

    # ── LVR features ─────────────────────────────────────────────────────────
    if lvr is not None:
        for col in ["lvr_rate_ann", "lvr_to_fee_ratio", "lvr_usd_tvl_approx"]:
            if col in lvr.columns:
                panel[col] = lvr[col].reindex(panel.index)

    # ── Merged features ───────────────────────────────────────────────────────
    if merged is not None:
        for col in ["dex_cex_basis_bps", "arbitrage_flag", "dex_cex_vol_ratio"]:
            if col in merged.columns:
                panel[col] = merged[col].reindex(panel.index)

    # ── Gas from swaps (hourly median) ────────────────────────────────────────
    swaps_path = DATA_PROC / "DEX" / "dex_swaps.csv"
    if swaps_path.exists():
        try:
            # Load ALL swaps for gas (only 2 columns → ~200 MB for 10.9M rows).
            # Previously nrows=2_000_000 was used, which only covered the first
            # ~11 months of data and left most of the panel at the 50 Gwei default.
            gas_raw = pd.read_csv(
                swaps_path, usecols=["timestamp", "gas_price_wei"],
                low_memory=False)
            gas_raw["timestamp"] = pd.to_datetime(
                gas_raw["timestamp"], utc=True, errors="coerce")
            gas_raw["gas_gwei"] = (
                pd.to_numeric(gas_raw["gas_price_wei"], errors="coerce") / 1e9)
            gas_h = (gas_raw.set_index("timestamp")["gas_gwei"]
                     .resample("1h").median())
            panel["gas_gwei"] = gas_h.reindex(panel.index)
        except Exception as exc:
            print(f"  [WARN] gas from swaps: {exc}")

    # ── Intra-hour OHLC realized range ────────────────────────────────────────
    for h_col, l_col in [("eth_usdc_high", "eth_usdc_low"),
                          ("eth_usdc_open", "eth_usdc_close")]:
        if h_col in panel.columns and l_col in panel.columns:
            ref = pd.to_numeric(
                panel.get("eth_usdc_close", panel.get("eth_usdc_price")),
                errors="coerce").replace(0, np.nan)
            hi  = pd.to_numeric(panel[h_col], errors="coerce")
            lo  = pd.to_numeric(panel[l_col], errors="coerce")
            panel["ohlc_range_1h"] = ((hi - lo) / ref).clip(0.0, 0.5)
            break

    # ── Rename CEX volume ─────────────────────────────────────────────────────
    if "vol_base_1h_ethusdt" in panel.columns:
        panel["cex_vol_1h"] = pd.to_numeric(
            panel["vol_base_1h_ethusdt"], errors="coerce")

    # ── Resolve price column ──────────────────────────────────────────────────
    for col in ["eth_usdc_price", "eth_usdc_close"]:
        if col in panel.columns:
            panel["price"] = pd.to_numeric(panel[col], errors="coerce")
            break
    if "price" not in panel.columns:
        raise ValueError("No ETH/USDC price column found in DEX data.")
    panel = panel[panel["price"].notna() & (panel["price"] > 0)].copy()

    # ── Multi-horizon returns ─────────────────────────────────────────────────
    if "log_return_1h" not in panel.columns:
        panel["log_return_1h"] = np.log(
            panel["price"] / panel["price"].shift(1))
    panel["log_return_6h"]  = panel["log_return_1h"].rolling(6).sum()
    panel["log_return_24h"] = panel["log_return_1h"].rolling(24).sum()

    # ── Fallback defaults for critical columns ────────────────────────────────
    _defaults: dict[str, float] = {
        "gas_gwei":             50.0,
        "dex_cex_basis_bps":     0.0,
        "arbitrage_flag":        0.0,
        "realized_vol_1h_ann":   0.5,
        "realized_vol_24h_ann":  0.5,
        "realized_vol_7d_ann":   0.5,
        "ohlc_range_1h":         0.0,
        "lvr_rate_ann":          np.nan,
        "lvr_to_fee_ratio":      np.nan,
        "lvr_usd_tvl_approx":    np.nan,
        "cex_vol_1h":            np.nan,
        "fees_usd":              np.nan,
        "tvl_usd":               1e8,
    }
    for col, default in _defaults.items():
        if col not in panel.columns:
            panel[col] = default
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
        if not np.isnan(default):
            panel[col] = panel[col].fillna(default)

    # ── Rolling z-score stats (no look-ahead: shift by 1) ────────────────────
    for col in NORM_COLS:
        if col in panel.columns:
            s = pd.to_numeric(panel[col], errors="coerce")
            panel[f"{col}__mu"] = (s.rolling(NORM_WINDOW, min_periods=24)
                                    .mean().shift(1))
            panel[f"{col}__sd"] = (s.rolling(NORM_WINDOW, min_periods=24)
                                    .std().shift(1).clip(lower=1e-8))

    # ── Gas percentile (rolling rank, shifted) ────────────────────────────────
    panel["gas_pctile"] = (
        panel["gas_gwei"]
        .rolling(NORM_WINDOW, min_periods=24).rank(pct=True)
        .shift(1).fillna(0.5)
    )

    panel = panel.apply(pd.to_numeric, errors="coerce")
    panel = panel.dropna(subset=[f"{NORM_COLS[0]}__mu"]).copy()

    print(f"  Panel: {len(panel):,} hours "
          f"({panel.index[0].date()} – {panel.index[-1].date()})")
    _report_coverage(panel)
    return panel


def _report_coverage(panel: pd.DataFrame) -> None:
    key_cols = ["fees_usd", "lvr_usd_tvl_approx", "lvr_to_fee_ratio",
                "cex_vol_1h", "ohlc_range_1h", "dex_cex_basis_bps",
                "gas_gwei", "arbitrage_flag"]
    for col in key_cols:
        if col in panel.columns:
            pct = panel[col].notna().mean() * 100
            print(f"    {col}: {pct:.0f}% coverage")


# ── Uniswap v3 mathematics ────────────────────────────────────────────────────

def concentration_factor(p_lower: float, p_upper: float) -> float:
    """
    Exact concentration of a v3 position relative to full-range v2.

    Derivation: for unit liquidity L over [p_a, p_b] the capital
    efficiency relative to v2 is sqrt(p_b/p_a) / (sqrt(p_b/p_a) - 1).
    Approximation 2/w holds for small w but this is exact.
    """
    if p_lower <= 0 or p_upper <= p_lower:
        return 1.0
    r = float(np.sqrt(p_upper / p_lower))
    return r / max(r - 1.0, 1e-9)


# ── Gymnasium environment ─────────────────────────────────────────────────────

try:
    import gymnasium as gym
    _BaseEnv = gym.Env
    HAS_GYMNASIUM = True
except ImportError:
    _BaseEnv = object
    HAS_GYMNASIUM = False


class UniswapV3LPEnv(_BaseEnv):
    """
    Uniswap v3 LP environment (Design B — historical replay; sec:rl-env).

    State (27 features, float32):
    ┌─ Market  (8) ─┐  log_return_1h/6h/24h, realized_vol_1h/24h/7d,
    │               │  ohlc_range_1h, dex_cex_basis_bps
    ├─ Pool    (5) ─┤  fee_apr_ann, vol_over_tvl, tx_count,
    │               │  lvr_to_fee_ratio, cex_vol_1h
    ├─ Position(6) ─┤  dist_from_center, in_range, range_width_norm,
    │               │  capital_deployed_frac, time_in_pos_norm, cum_pnl_norm
    ├─ Cost    (7) ─┤  gas_gwei_z, gas_pctile, lvr_rate_z,
    │               │  hour_sin/cos, dow_sin/cos
    └─ Arb     (1) ─┘  arbitrage_flag

    Action: MultiDiscrete([8, 5, 5])
        [range_width_idx, capital_frac_idx, center_offset_idx]
    Center offset shifts range center by k * sigma_h from current price,
    enabling directional bets on price movement.

    Reward (Form 2; sec:rl-reward):
        fee_income − LVR_cost − gas_cost − swap_cost   / REWARD_SCALE
    Form 4 (optional): subtract quadratic penalty on P&L deviation.

    Fee income uses actual pool fees_usd when available.
    LVR cost uses actual lvr_usd_tvl_approx when available.
    """

    metadata  = {"render_modes": []}
    N_OBS     = 27

    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = INITIAL_CAPITAL,
        episode_hours: int = EPISODE_HOURS,
        train_mode: bool = True,
        risk_aversion: float = 0.0,
    ):
        if HAS_GYMNASIUM:
            super().__init__()

        # Store timestamps separately for fast O(1) lookup
        self._timestamps     = pd.DatetimeIndex(data.index)
        self.data            = data.reset_index(drop=True)
        self.initial_capital = initial_capital
        self.episode_hours   = episode_hours
        self.train_mode      = train_mode
        self.eta             = risk_aversion
        self.n_rows          = len(data)
        self._np_random      = np.random.default_rng(SEED)

        if HAS_GYMNASIUM:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.N_OBS,), dtype=np.float32)
            self.action_space = gym.spaces.MultiDiscrete(
                [len(RANGE_WIDTHS), len(CAPITAL_FRACS), len(CENTER_OFFSETS)])

        self._reset_position()
        self.t = 0
        self.t_end = min(self.episode_hours, self.n_rows - 1)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _row(self) -> pd.Series:
        return self.data.iloc[self.t]

    def _get(self, col: str, default: float = 0.0) -> float:
        v = self._row().get(col, default)
        return float(v) if pd.notna(v) else float(default)

    def _z(self, col: str) -> float:
        """Look-ahead-free z-score using pre-computed rolling stats."""
        val = self._get(col, 0.0)
        mu  = self._get(f"{col}__mu", 0.0)
        sd  = max(self._get(f"{col}__sd", 1.0), 1e-8)
        return float(np.clip((val - mu) / sd, -4.0, 4.0))

    def _in_range(self, price: float) -> bool:
        return (self.p_lower is not None
                and self.p_lower <= price <= self.p_upper)

    def _reset_position(self) -> None:
        self.capital          = self.initial_capital
        self.p_lower          = None
        self.p_upper          = None
        self.entry_price      = None
        self.range_width      = 0.0
        self.center_offset_k  = 0.0
        self.capital_deployed = 0.0
        self.time_in_pos      = 0
        self.cum_pnl          = 0.0
        self._pnl_history: list[float] = []

    # ── Observation ───────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        price = self._get("price", DEFAULT_ETH_PRICE)
        ts    = self._timestamps[self.t]
        hour  = ts.hour
        dow   = ts.dayofweek

        # Range distance: signed, in units of range half-width
        if self.p_lower is not None and self.p_upper is not None:
            mid  = (self.p_lower + self.p_upper) / 2.0
            hw   = mid * max(self.range_width / 2.0, 1e-9)
            dist = float(np.clip((price - mid) / hw, -4.0, 4.0))
        else:
            dist = 0.0

        cum_pnl_norm = float(np.clip(
            self.cum_pnl / max(self.initial_capital, 1.0), -2.0, 2.0))

        market = [
            self._z("log_return_1h"),
            self._z("log_return_6h"),
            self._z("log_return_24h"),
            self._z("realized_vol_1h_ann"),
            self._z("realized_vol_24h_ann"),
            self._z("realized_vol_7d_ann"),
            self._z("ohlc_range_1h"),
            self._z("dex_cex_basis_bps"),
        ]
        pool = [
            self._z("fee_apr_ann"),
            self._z("vol_over_tvl"),
            self._z("tx_count"),
            self._z("lvr_to_fee_ratio"),
            self._z("cex_vol_1h"),
        ]
        position = [
            dist,
            float(self._in_range(price)),
            float(self.range_width / RANGE_WIDTHS[-1]),
            float(self.capital_deployed / max(self.initial_capital, 1.0)),
            float(np.clip(self.time_in_pos / 168.0, 0.0, 4.0)),
            cum_pnl_norm,
        ]
        cost = [
            self._z("gas_gwei"),
            float(self._get("gas_pctile", 0.5)),
            self._z("lvr_rate_ann"),
            float(np.sin(2 * np.pi * hour / 24.0)),
            float(np.cos(2 * np.pi * hour / 24.0)),
            float(np.sin(2 * np.pi * dow  / 7.0)),
            float(np.cos(2 * np.pi * dow  / 7.0)),
        ]
        arb = [float(self._get("arbitrage_flag", 0.0))]

        obs = np.array(market + pool + position + cost + arb, dtype=np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=4.0, neginf=-4.0)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options=None):
        if HAS_GYMNASIUM:
            super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.default_rng(seed)
        max_start = max(1, self.n_rows - self.episode_hours - 1)
        self.t     = (int(self._np_random.integers(0, max_start))
                      if self.train_mode else 0)
        self.t_end = min(self.t + self.episode_hours, self.n_rows - 1)
        self._reset_position()
        return self._obs(), {}

    def step(self, action):
        price           = self._get("price",               DEFAULT_ETH_PRICE)
        vol_ann         = max(self._get("realized_vol_24h_ann", 0.5), 0.01)
        fee_apr         = max(self._get("fee_apr_ann",      0.10), 0.0)
        gas_gwei        = max(self._get("gas_gwei",          50.0), 1.0)
        tvl_usd         = max(self._get("tvl_usd",            1e8), 1e6)
        fees_usd_actual = max(self._get("fees_usd",            0.0), 0.0)
        lvr_actual      = self._get("lvr_usd_tvl_approx", np.nan)

        # ── Parse 3D action ───────────────────────────────────────────────────
        rw_idx     = int(action[0])
        cf_idx     = int(action[1])
        co_idx     = int(action[2])
        new_width  = RANGE_WIDTHS[rw_idx]
        new_cap    = self.capital * CAPITAL_FRACS[cf_idx]
        offset_k   = CENTER_OFFSETS[co_idx]

        # Range boundaries: center offset in units of hourly sigma
        sigma_h  = vol_ann / np.sqrt(8_760)
        p_center = price * np.exp(float(offset_k) * sigma_h)
        new_lower = p_center * (1.0 - new_width / 2.0)
        new_upper = p_center * (1.0 + new_width / 2.0)

        # ── Rebalance decision ────────────────────────────────────────────────
        out_of_range = (self.p_lower is not None
                        and not self._in_range(price)
                        and self.capital_deployed > 0)
        width_change  = abs(new_width  - self.range_width)   > 1e-9
        cap_change    = abs(new_cap    - self.capital_deployed) > 1.0
        offset_change = abs(offset_k  - self.center_offset_k) > 0.1
        changing      = width_change or cap_change or offset_change
        rebalance     = out_of_range or (
            changing and (new_cap > 0 or self.capital_deployed > 0))

        # ── Transaction costs ─────────────────────────────────────────────────
        gas_usd = swap_cost = 0.0
        if rebalance and (new_cap > 0 or self.capital_deployed > 0):
            gas_usd   = GAS_UNITS * gas_gwei * 1e9 * price / 1e18
            gas_usd   = min(gas_usd, self.capital * 0.005)
            active_c  = max(self.capital_deployed, new_cap)
            swap_cost = active_c * POOL_FEE_RATE * 2.0

        # ── P&L from existing position ─────────────────────────────────────
        fee_income = lvr_cost = 0.0
        conc = 1.0

        if self.capital_deployed > 0 and self._in_range(price):
            conc = concentration_factor(self.p_lower, self.p_upper)

            # Fee income: actual pool fees when available; APR formula fallback
            if fees_usd_actual > 0 and tvl_usd > 1e4:
                fee_income = fees_usd_actual * (
                    self.capital_deployed * conc / tvl_usd)
            else:
                fee_income = fee_apr * self.capital_deployed / 8_760 * conc

            # LVR: actual lvr_usd_tvl_approx when available; formula fallback
            if pd.notna(lvr_actual) and lvr_actual > 0 and tvl_usd > 1e4:
                lvr_cost = lvr_actual * (self.capital_deployed / tvl_usd) * conc
            else:
                sigma_sq_h = (vol_ann ** 2) / 8_760
                lvr_cost   = (sigma_sq_h / 8.0) * self.capital_deployed * conc

            # Sanity caps
            fee_income = min(fee_income, self.capital_deployed * 0.02)
            lvr_cost   = min(lvr_cost,   fee_income * 10.0)

        # ── Net P&L and reward ────────────────────────────────────────────────
        net_pnl = fee_income - lvr_cost - gas_usd - swap_cost
        reward  = net_pnl

        # Form 4: risk-adjusted reward with quadratic P&L penalty
        if self.eta > 0 and len(self._pnl_history) >= 24:
            mu_24 = float(np.mean(self._pnl_history[-24:]))
            reward -= self.eta / 2.0 * (net_pnl - mu_24) ** 2
        self._pnl_history.append(net_pnl)
        self.cum_pnl += net_pnl
        self.capital  = max(self.capital + net_pnl, 1.0)

        # ── Apply action ──────────────────────────────────────────────────────
        if rebalance:
            if new_cap > 0:
                self.p_lower          = new_lower
                self.p_upper          = new_upper
                self.range_width      = new_width
                self.center_offset_k  = offset_k
                self.entry_price      = price
                self.capital_deployed = min(new_cap, self.capital)
                self.time_in_pos      = 0
            else:
                self.p_lower = self.p_upper = self.entry_price = None
                self.range_width = self.capital_deployed = 0.0
                self.center_offset_k = 0.0
                self.time_in_pos     = 0

        self.time_in_pos += 1
        self.t += 1
        truncated  = self.t >= self.t_end
        terminated = self.capital < self.initial_capital * 0.01  # near-bankruptcy

        info = {
            "fee_income":       fee_income,
            "lvr_cost":         lvr_cost,
            "gas_usd":          gas_usd,
            "swap_cost":        swap_cost,
            "net_pnl":          net_pnl,
            "in_range":         self._in_range(price),
            "capital":          self.capital,
            "range_width":      self.range_width,
            "center_offset_k":  self.center_offset_k,
            "rebalanced":       rebalance,
            "tvl_frac":         self.capital / max(tvl_usd, 1.0),
            "vol_ann":          vol_ann,
            "conc":             conc,
            "price":            price,
        }
        return self._obs(), reward / REWARD_SCALE, terminated, truncated, info

    def close(self):
        pass


# ── Training helpers ──────────────────────────────────────────────────────────

def linear_schedule(lr_start: float):
    """Linearly decay learning rate from lr_start to 0."""
    def schedule(progress_remaining: float) -> float:
        return max(lr_start * progress_remaining, 1e-6)
    return schedule


# ── Benchmark policies ────────────────────────────────────────────────────────

def _fixed_policy(rw_idx: int, cf_idx: int, co_idx: int = 2):
    """Constant (range_width, capital_frac, center_offset=0) policy."""
    action = np.array([rw_idx, cf_idx, co_idx])
    return lambda env, obs: action


def cdm_policy(env: UniswapV3LPEnv, obs: np.ndarray) -> np.ndarray:
    """
    CDM-inspired one-period analytical benchmark (sec:rl-benchmarks).

    For each candidate width w, evaluates:
        f(w) = conc(w) × P_in(w) × (fee_rate_h − sigma_h²/8) − gas_rate
    where P_in = Φ(log(1+w/2)/σ_h) − Φ(log(1-w/2)/σ_h) under GBM.
    Deploys 100% capital, symmetric range (offset=0).
    """
    vol_ann  = max(env._get("realized_vol_24h_ann", 0.5), 0.01)
    fee_apr  = max(env._get("fee_apr_ann",          0.10), 0.0)
    gas_gwei = max(env._get("gas_gwei",              50.0), 1.0)
    price    = max(env._get("price",    DEFAULT_ETH_PRICE), 1.0)

    sigma_h  = vol_ann / np.sqrt(8_760)
    gas_usd  = GAS_UNITS * gas_gwei * 1e9 * price / 1e18

    best_w, best_val = RANGE_WIDTHS[-1], -np.inf
    for w in RANGE_WIDTHS:
        p_lo = 1.0 - w / 2.0
        p_hi = 1.0 + w / 2.0
        p_in = (sp_norm.cdf(np.log(p_hi) / (sigma_h + 1e-9))
                - sp_norm.cdf(np.log(max(p_lo, 1e-9)) / (sigma_h + 1e-9)))
        p_in = max(p_in, 1e-6)
        conc = concentration_factor(price * p_lo, price * p_hi)
        val  = (conc * p_in * (fee_apr / 8_760 - sigma_h ** 2 / 8.0)
                - gas_usd * (1.0 - p_in) / max(env.capital, 1.0))
        if val > best_val:
            best_val, best_w = val, w

    return np.array([RANGE_WIDTHS.index(best_w),
                     len(CAPITAL_FRACS) - 1,
                     CENTER_OFFSETS.index(0.0)])


def calibrate_vol_scale(train_data: pd.DataFrame) -> float:
    """Grid-search k* = argmax_k P&L on training set; width = k × sigma_24h."""
    best_k, best_pnl = RANGE_WIDTHS[0], -np.inf
    for k in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]:
        def _pol(env, obs, _k=k):
            v   = max(env._get("realized_vol_24h_ann", 0.5), 0.01)
            idx = int(np.argmin([abs(w - _k * v) for w in RANGE_WIDTHS]))
            return np.array([idx, len(CAPITAL_FRACS) - 1, CENTER_OFFSETS.index(0.0)])
        env  = UniswapV3LPEnv(train_data, train_mode=False,
                               episode_hours=len(train_data))
        pnl  = sum(r["net_pnl"] for r in run_episode(env, _pol))
        if pnl > best_pnl:
            best_pnl, best_k = pnl, k
    print(f"  Vol-scaled:    k* = {best_k:.2f}  (train P&L = ${best_pnl:,.0f})")
    return best_k


def calibrate_har_vol_policy(train_data: pd.DataFrame) -> tuple[float, tuple]:
    """
    Grid-search k*, beta* for HAR-vol range policy.
    Width = k × (beta[0]*vol_1h + beta[1]*vol_24h + beta[2]*vol_7d).
    Beta choices mirror typical HAR-RV coefficient ratios (Corsi 2009).
    """
    k_grid = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    beta_grid = [
        (1/3,  1/3,  1/3),    # equal weights
        (0.10, 0.55, 0.35),   # HAR-RV typical (daily dominates)
        (0.05, 0.60, 0.35),   # daily + monthly heavy
        (0.20, 0.50, 0.30),   # more weight on short-term
    ]
    best_k, best_beta, best_pnl = k_grid[0], beta_grid[0], -np.inf
    for k in k_grid:
        for beta in beta_grid:
            def _pol(env, obs, _k=k, _b=beta):
                v1  = max(env._get("realized_vol_1h_ann",  0.5), 0.01)
                v24 = max(env._get("realized_vol_24h_ann", 0.5), 0.01)
                v7d = max(env._get("realized_vol_7d_ann",  0.5), 0.01)
                v_hat = _b[0] * v1 + _b[1] * v24 + _b[2] * v7d
                idx   = int(np.argmin([abs(w - _k * v_hat) for w in RANGE_WIDTHS]))
                return np.array([idx, len(CAPITAL_FRACS) - 1,
                                  CENTER_OFFSETS.index(0.0)])
            env = UniswapV3LPEnv(train_data, train_mode=False,
                                  episode_hours=len(train_data))
            pnl = sum(r["net_pnl"] for r in run_episode(env, _pol))
            if pnl > best_pnl:
                best_pnl, best_k, best_beta = pnl, k, beta
    print(f"  HAR-vol:       k* = {best_k:.2f}  beta={best_beta}  "
          f"(train P&L = ${best_pnl:,.0f})")
    return best_k, best_beta


def make_vol_scaled_policy(k: float):
    def _pol(env: UniswapV3LPEnv, obs: np.ndarray) -> np.ndarray:
        v   = max(env._get("realized_vol_24h_ann", 0.5), 0.01)
        idx = int(np.argmin([abs(w - k * v) for w in RANGE_WIDTHS]))
        return np.array([idx, len(CAPITAL_FRACS) - 1, CENTER_OFFSETS.index(0.0)])
    return _pol


def make_har_vol_policy(k: float, beta: tuple):
    def _pol(env: UniswapV3LPEnv, obs: np.ndarray) -> np.ndarray:
        v1  = max(env._get("realized_vol_1h_ann",  0.5), 0.01)
        v24 = max(env._get("realized_vol_24h_ann", 0.5), 0.01)
        v7d = max(env._get("realized_vol_7d_ann",  0.5), 0.01)
        v_hat = beta[0] * v1 + beta[1] * v24 + beta[2] * v7d
        idx   = int(np.argmin([abs(w - k * v_hat) for w in RANGE_WIDTHS]))
        return np.array([idx, len(CAPITAL_FRACS) - 1, CENTER_OFFSETS.index(0.0)])
    return _pol


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(env: UniswapV3LPEnv, policy_fn) -> list[dict]:
    obs, _ = env.reset(seed=SEED)
    records, done = [], False
    while not done:
        action = policy_fn(env, obs)
        obs, _, terminated, truncated, info = env.step(action)
        records.append(info)
        done = terminated or truncated
    return records


# ── Statistical helpers ────────────────────────────────────────────────────────

def block_bootstrap_ci(
    series: np.ndarray,
    block_length: int = 24,
    n_boot: int = 1_000,
    alpha: float = 0.05,
    seed: int = SEED,
) -> tuple[float, float]:
    """Block-bootstrap 95% CI for the mean (24h blocks capture intra-day auto.)."""
    rng, n = np.random.default_rng(seed), len(series)
    n_blocks = int(np.ceil(n / block_length))
    boot = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block_length), size=n_blocks)
        sample = np.concatenate([series[s: s + block_length] for s in starts])[:n]
        boot.append(float(np.mean(sample)))
    return tuple(np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def compute_metrics(records: list[dict], label: str = "") -> dict:
    """
    Full evaluation suite (sec:rl-evaluation, Table tab:rl-evaluation).
    Decomposed P&L + Sharpe/Sortino/CVaR/drawdown + position diagnostics.
    """
    df   = pd.DataFrame(records)
    pnl  = np.array(df["net_pnl"])
    cap  = np.array(df["capital"])

    hourly_mean = float(np.mean(pnl))
    hourly_std  = float(np.std(pnl)) + 1e-9
    sharpe      = hourly_mean / hourly_std * np.sqrt(8_760)

    downside  = pnl[pnl < 0]
    sd_down   = (float(np.sqrt(np.mean(downside ** 2))) + 1e-9
                 if len(downside) > 0 else 1e-9)
    sortino   = hourly_mean / sd_down * np.sqrt(8_760)

    run_max   = np.maximum.accumulate(cap)
    dd_pct    = float(((run_max - cap) / (run_max + 1e-9)).max()) * 100

    p5   = np.percentile(pnl, 5)
    cvar = float(np.mean(pnl[pnl <= p5]))

    ci_lo, ci_hi = block_bootstrap_ci(pnl)

    n_rebal      = int(df["rebalanced"].sum()) if "rebalanced" in df else 0
    avg_rw_pct   = float(df["range_width"].mean()) * 100
    pct_in       = float(df["in_range"].mean()) * 100
    avg_tvl_frac = float(df["tvl_frac"].mean()) * 100 if "tvl_frac" in df else None
    avg_offset   = float(df["center_offset_k"].mean()) if "center_offset_k" in df else 0.0
    avg_conc     = float(df["conc"].mean()) if "conc" in df else None

    return {
        "Strategy":             label,
        "Final_capital":        round(float(cap[-1]), 0),
        "Total_net_pnl":        round(float(pnl.sum()), 0),
        "Total_fees":           round(float(df["fee_income"].sum()), 0),
        "Total_LVR":            round(float(df["lvr_cost"].sum()), 0),
        "Total_gas":            round(float(df["gas_usd"].sum()), 0),
        "Total_swap_cost":      round(float(df["swap_cost"].sum()), 0),
        "Sharpe_ann":           round(sharpe, 3),
        "Sortino_ann":          round(sortino, 3),
        "Max_drawdown_pct":     round(dd_pct, 2),
        "CVaR_5pct":            round(cvar, 2),
        "CI95_lo":              round(ci_lo, 2),
        "CI95_hi":              round(ci_hi, 2),
        "Pct_in_range":         round(pct_in, 1),
        "N_rebalances":         n_rebal,
        "Avg_range_width_pct":  round(avg_rw_pct, 2),
        "Avg_center_offset_k":  round(avg_offset, 2),
        "Avg_concentration":    round(avg_conc, 1) if avg_conc else None,
        "Avg_capital_pct_TVL":  round(avg_tvl_frac, 3) if avg_tvl_frac else None,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_all(
    model,
    test_data: pd.DataFrame,
    vol_scale_k: float,
    har_k: float,
    har_beta: tuple,
) -> dict[str, list[dict]]:
    """
    Run 7 benchmarks + PPO agent on full test set.

    Benchmarks (sec:rl-benchmarks):
        1. passive_wide_20pct  — fixed 20% range, 100% capital
        2. static_narrow_1pct — fixed 1% range, 100% capital
        3. static_mid_5pct    — fixed 5% range, 100% capital
        4. vol_scaled          — w = k* × sigma_24h (calibrated on train)
        5. har_vol_scaled      — w = k* × HAR(sigma_1h, 24h, 7d) [NEW]
        6. CDM_inspired        — one-period static optimum (CDM criterion)
        7. hold_cash           — 0% deployed (zero fee, zero LVR)
    """
    n = len(test_data)
    benchmarks = {
        "passive_wide_20pct": _fixed_policy(7, 4),        # 20% range, 100% cap
        "static_narrow_1pct": _fixed_policy(1, 4),        # 1%  range, 100% cap
        "static_mid_5pct":    _fixed_policy(4, 4),        # 6%  range, 100% cap
        "vol_scaled":         make_vol_scaled_policy(vol_scale_k),
        "HAR_vol_scaled":     make_har_vol_policy(har_k, har_beta),
        "CDM_inspired":       cdm_policy,
        "hold_cash":          _fixed_policy(0, 0),
    }

    all_records: dict[str, list[dict]] = {}
    for label, pol in benchmarks.items():
        env = UniswapV3LPEnv(test_data, train_mode=False, episode_hours=n)
        all_records[label] = run_episode(env, pol)
        pnl = sum(r["net_pnl"] for r in all_records[label])
        print(f"    {label:25s}  P&L = ${pnl:>10,.0f}")

    if model is not None:
        env = UniswapV3LPEnv(test_data, train_mode=False, episode_hours=n)
        obs, _ = env.reset(seed=SEED)
        records, done = [], False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            records.append(info)
            done = terminated or truncated
        all_records["PPO_agent"] = records
        pnl = sum(r["net_pnl"] for r in records)
        print(f"    {'PPO_agent':25s}  P&L = ${pnl:>10,.0f}")

    return all_records


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
          "#7c3aed", "#0891b2", "#374151", "#be185d"]


def plot_results(all_records: dict[str, list[dict]]) -> None:
    # ── Figure 1: Capital over time ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 4))
    for (label, recs), color in zip(all_records.items(), COLORS):
        cap = [r["capital"] for r in recs]
        lw  = 2.0 if label == "PPO_agent" else 0.9
        ax.plot(cap, label=label.replace("_", " "), lw=lw, color=color)
    ax.axhline(INITIAL_CAPITAL, color="black", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel("Hour")
    ax.set_ylabel("Capital (USD)")
    ax.set_title("LP Agent Capital: PPO vs Benchmarks (test set)")
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(OUTPUT_FIGS / "rl_capital_comparison.pdf",
                bbox_inches="tight", dpi=300)
    plt.close()

    # ── Figure 2: PPO agent decomposition ────────────────────────────────────
    if "PPO_agent" in all_records:
        df  = pd.DataFrame(all_records["PPO_agent"])
        win = min(24 * 7, len(df) // 4)
        fig, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

        axes[0].fill_between(range(len(df)),
                             df["fee_income"].rolling(win).mean(),
                             color=COLORS[2], alpha=0.7, label="Fee income (7d MA)")
        axes[0].fill_between(range(len(df)),
                             -df["lvr_cost"].rolling(win).mean(),
                             color=COLORS[1], alpha=0.7, label="-LVR cost (7d MA)")
        axes[0].axhline(0, color="black", lw=0.6)
        axes[0].set_ylabel("USD/h")
        axes[0].legend(fontsize=7)
        axes[0].set_title("PPO Agent — Decomposed P&L (Form 2)")

        # Cumulative P&L vs key benchmarks
        for label, color, ls in [
            ("PPO_agent",     COLORS[0], "-"),
            ("CDM_inspired",  COLORS[3], "--"),
            ("HAR_vol_scaled",COLORS[4], "--"),
            ("vol_scaled",    COLORS[5], ":"),
        ]:
            if label in all_records:
                axes[1].plot(
                    np.cumsum([r["net_pnl"] for r in all_records[label]]),
                    label=label.replace("_", " "), lw=1.2, color=color, ls=ls)
        axes[1].axhline(0, color="black", lw=0.5, ls="--")
        axes[1].set_ylabel("Cumulative P&L (USD)")
        axes[1].legend(fontsize=7)

        # Range width chosen over time
        axes[2].plot(df["range_width"].rolling(24).mean() * 100,
                     color=COLORS[5], lw=0.8)
        axes[2].set_ylabel("Range width (%, 24h MA)")
        axes[2].set_title("Range width — policy interpretation")

        # Center offset (directional betting behaviour)
        axes[3].plot(df["center_offset_k"].rolling(24).mean(),
                     color=COLORS[7], lw=0.8)
        axes[3].axhline(0, color="black", lw=0.6, ls="--")
        axes[3].set_ylabel("Center offset k (24h MA)")
        axes[3].set_title("Directional bias (+ = bullish)")

        # % in-range
        axes[4].plot(df["in_range"].rolling(24 * 7).mean() * 100,
                     color=COLORS[6], lw=0.9)
        axes[4].set_ylabel("% time in range (7d MA)")
        axes[4].set_xlabel("Hour")

        fig.savefig(OUTPUT_FIGS / "rl_agent_decomposition.pdf",
                    bbox_inches="tight", dpi=300)
        plt.close()
        print("  Saved rl_agent_decomposition.pdf")

    # ── Figure 3: Risk-adjusted comparison (Sharpe + CVaR) ───────────────────
    metrics_list = [compute_metrics(recs, label)
                    for label, recs in all_records.items()]
    labels  = [m["Strategy"] for m in metrics_list]
    sharpes = [m["Sharpe_ann"] for m in metrics_list]
    cvars   = [m["CVaR_5pct"] for m in metrics_list]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    x = np.arange(len(labels))
    axes[0].bar(x, sharpes, color=COLORS[:len(labels)], alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
    axes[0].axhline(0, color="black", lw=0.6)
    axes[0].set_title("Annualised Sharpe Ratio")

    axes[1].bar(x, cvars, color=COLORS[:len(labels)], alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=7)
    axes[1].set_title("CVaR₅ (USD / hour)")

    fig.suptitle("Risk-Adjusted Performance: PPO vs Benchmarks", y=1.01)
    fig.savefig(OUTPUT_FIGS / "rl_risk_comparison.pdf",
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  Saved rl_risk_comparison.pdf")

    print("  Saved rl_capital_comparison.pdf")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    total_timesteps: int,
    n_seeds: int = 3,
    n_envs: int = N_ENVS,
    risk_aversion: float = 0.0,
):
    """
    Train PPO across n_seeds seeds; select best model on validation P&L.

    Uses EvalCallback for checkpoint-based early-stopping signal.
    Linear LR decay: 3e-4 → 0.
    n_envs parallel envs (DummyVecEnv; SubprocVecEnv for speedup on Linux).

    Walk-forward note: single 70/10/20 split used here. Full rolling walk-forward
    (e.g., expanding window with 3 folds) is the more rigorous alternative and
    is flagged as future work (sec:rl-training item 3).

    PPO hyperparameters (Table tab:rl-algorithm):
        lr=3e-4 (linear decay), n_steps=2048, batch=256, epochs=10,
        gamma=0.999, gae_lambda=0.95, clip=0.2, ent_coef=0.01.
    Grid search over {lr, clip_range} on val P&L recommended before final report.
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.callbacks import EvalCallback
    except ImportError:
        raise ImportError(
            "Run: pip install gymnasium stable-baselines3 torch")

    val_len   = len(val_data)
    best_model, best_pnl = None, -np.inf

    for s in range(n_seeds):
        print(f"  Seed {s + 1}/{n_seeds}...")

        def _make_train():
            return UniswapV3LPEnv(train_data, train_mode=True,
                                   risk_aversion=risk_aversion)

        def _make_val():
            return UniswapV3LPEnv(val_data, train_mode=False,
                                   episode_hours=val_len)

        vec_train = make_vec_env(_make_train, n_envs=n_envs,
                                  seed=SEED + s * 100)
        vec_val   = make_vec_env(_make_val,   n_envs=1,
                                  seed=SEED + 999 + s)

        eval_freq = max(total_timesteps // 20 // n_envs, 2048)
        eval_cb   = EvalCallback(
            vec_val,
            best_model_save_path=str(MODEL_DIR / f"seed_{s}"),
            log_path=str(MODEL_DIR / f"seed_{s}"),
            eval_freq=eval_freq,
            n_eval_episodes=3,
            deterministic=True,
            verbose=0,
        )

        model = PPO(
            "MlpPolicy",
            vec_train,
            n_steps       = 2_048,
            batch_size    = 256,
            n_epochs      = 10,
            learning_rate = linear_schedule(3e-4),
            gamma         = 0.999,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            ent_coef      = 0.01,
            policy_kwargs = {"net_arch": [256, 256]},
            verbose       = 0,
            seed          = SEED + s,
        )
        model.learn(total_timesteps=total_timesteps, callback=eval_cb)

        # Load best checkpoint; fall back to final model if callback saved nothing
        try:
            from stable_baselines3 import PPO as _PPO
            candidate = _PPO.load(str(MODEL_DIR / f"seed_{s}" / "best_model"))
        except Exception:
            candidate = model

        # Evaluate on val set using raw P&L (not reward signal)
        val_env  = UniswapV3LPEnv(val_data, train_mode=False,
                                   episode_hours=val_len)
        val_recs = run_episode(
            val_env,
            lambda env, obs: candidate.predict(obs, deterministic=True)[0])
        val_pnl  = sum(r["net_pnl"] for r in val_recs)
        print(f"    Validation P&L = ${val_pnl:,.0f}")

        if val_pnl > best_pnl:
            best_pnl, best_model = val_pnl, candidate

        vec_train.close()
        vec_val.close()

    best_model.save(str(MODEL_DIR / "ppo_lp_best"))
    print(f"  Best model saved  val P&L = ${best_pnl:,.0f}")
    return best_model


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RL LP Agent v3 — Uniswap v3")
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="PPO steps per seed (use 500k+ for production)")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--n-envs", type=int, default=N_ENVS)
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; load saved model from output/rl/")
    parser.add_argument("--risk-aversion", type=float, default=0.0,
                        help="eta for Form 4 reward (0 = Form 2 only)")
    args = parser.parse_args()

    print("=" * 60)
    print("RL LP Agent v3 — Uniswap v3 (sec:rl-framework)")
    print("=" * 60)

    # ── [1] Load data ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading and preprocessing data...")
    _cal = load_calibration()
    if _cal:
        print(f"  Calibration params loaded "
              f"(fee_tier={_cal.get('fee_tier')}, "
              f"pool={_cal.get('pool_address', 'unknown')[:10]}...)")
    data  = load_data()
    n     = len(data)
    n_tr  = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_data = data.iloc[:n_tr]
    val_data   = data.iloc[n_tr : n_tr + n_val]
    test_data  = data.iloc[n_tr + n_val :]
    print(f"  Train: {len(train_data):,}h  "
          f"Val: {len(val_data):,}h  "
          f"Test: {len(test_data):,}h")

    # ── [2] Calibrate benchmarks on training set ───────────────────────────────
    print("\n[2/4] Calibrating benchmarks on training set...")
    vol_scale_k = calibrate_vol_scale(train_data)
    har_k, har_beta = calibrate_har_vol_policy(train_data)

    # ── [3] Train PPO ─────────────────────────────────────────────────────────
    model = None
    if not args.eval_only:
        print(f"\n[3/4] Training PPO "
              f"({args.timesteps:,} steps × {args.seeds} seeds × "
              f"{args.n_envs} envs)...")
        model = train(train_data, val_data,
                      args.timesteps, args.seeds,
                      args.n_envs, args.risk_aversion)
    else:
        print("\n[3/4] Loading saved model...")
        try:
            from stable_baselines3 import PPO
            model = PPO.load(str(MODEL_DIR / "ppo_lp_best"))
            print("  Loaded output/rl/ppo_lp_best.zip")
        except Exception as exc:
            print(f"  [WARN] {exc} — running benchmarks only")

    # ── [4] Evaluate on test set ──────────────────────────────────────────────
    print("\n[4/4] Evaluating on test set...")
    all_records = evaluate_all(model, test_data, vol_scale_k, har_k, har_beta)

    metrics_list = [compute_metrics(recs, label)
                    for label, recs in all_records.items()]
    tab = pd.DataFrame(metrics_list).set_index("Strategy")

    tab.to_csv(OUTPUT_TABS / "rl_evaluation.csv")
    try:
        (OUTPUT_TABS / "rl_evaluation.tex").write_text(
            tab.to_latex(float_format="%.2f", na_rep="--"), encoding="utf-8")
    except Exception:
        pass

    print("\n" + tab.to_string())
    print(f"\n  Saved output/tables/rl_evaluation.csv + .tex")

    plot_results(all_records)

    # Edge over CDM with block-bootstrap CI
    if "PPO_agent" in all_records and "CDM_inspired" in all_records:
        ppo = np.array([r["net_pnl"] for r in all_records["PPO_agent"]])
        cdm = np.array([r["net_pnl"] for r in all_records["CDM_inspired"]])
        n   = min(len(ppo), len(cdm))
        edge = ppo[:n] - cdm[:n]
        ci_lo, ci_hi = block_bootstrap_ci(edge)
        print(f"\n  PPO edge over CDM:     "
              f"mean ${edge.mean():+,.2f}/h  "
              f"[95% CI: ${ci_lo:,.2f}, ${ci_hi:,.2f}]")

    if "PPO_agent" in all_records and "HAR_vol_scaled" in all_records:
        ppo = np.array([r["net_pnl"] for r in all_records["PPO_agent"]])
        har = np.array([r["net_pnl"] for r in all_records["HAR_vol_scaled"]])
        n   = min(len(ppo), len(har))
        edge = ppo[:n] - har[:n]
        ci_lo, ci_hi = block_bootstrap_ci(edge)
        print(f"  PPO edge over HAR-vol: "
              f"mean ${edge.mean():+,.2f}/h  "
              f"[95% CI: ${ci_lo:,.2f}, ${ci_hi:,.2f}]")

    print("\n" + "=" * 60)
    print("DONE")
    print("  Model:   output/rl/ppo_lp_best.zip")
    print("  Figures: output/figures/rl_*.pdf")
    print("  Tables:  output/tables/rl_evaluation.csv / .tex")
    print("=" * 60)


if __name__ == "__main__":
    main()
