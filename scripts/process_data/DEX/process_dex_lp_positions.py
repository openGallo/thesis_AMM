"""
Build an LP position P&L table from raw mints, burns, collects, and positions_current.

Position key (pool-level): (owner, tick_lower, tick_upper)
Note: if the owner is the NonfungiblePositionManager contract, all NFT-wrapped
positions at the same range are merged — this is still valid for aggregate analysis.

Cash-flow accounting:
    total_minted_usd    — USDC value of liquidity deposited (cost)
    total_burned_usd    — USDC value of tokens returned on burn (principal out)
    total_collected_usd — USDC value of collect events (fees + returned principal)
    net_pnl_usd         — total_burned_usd + total_collected_usd - total_minted_usd
    fee_income_usd      — total_collected_usd - total_burned_usd (approximate fee income)

Range metrics:
    range_width_ticks   — tick_upper - tick_lower
    range_width_pct     — (1.0001^range_width_ticks - 1) * 100  (price range width)
    position_type       — narrow (<5%) | medium (5-20%) | wide (>20%)

Inputs:
    data_raw/DEX/mints_YYYY_MM.csv      (all months)
    data_raw/DEX/burns_YYYY_MM.csv      (all months)
    data_raw/DEX/collects_YYYY_MM.csv   (all months)
    data_raw/DEX/positions_current.csv  (optional; adds is_active flag)

Output:
    data_processed/DEX/dex_lp_positions.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_RAW     = PROJECT_ROOT / "data_raw" / "DEX"
DATA_OUT     = PROJECT_ROOT / "data_processed" / "DEX"

_KEY = ["owner", "tick_lower", "tick_upper"]


def load_monthly(pattern: str) -> pd.DataFrame:
    files = sorted(DATA_RAW.glob(pattern))
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f, low_memory=False))
        except Exception as exc:
            print(f"  [WARN] {f.name}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def coerce_events(df: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and ticks; drop rows with missing key fields."""
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for col in ("tick_lower", "tick_upper"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount_usd" in df.columns:
        df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce")
    return df.dropna(subset=[c for c in _KEY if c in df.columns])


def aggregate_cashflows(
    mints: pd.DataFrame,
    burns: pd.DataFrame,
    collects: pd.DataFrame,
) -> pd.DataFrame:
    def grp_agg(df: pd.DataFrame, sum_col: str, sum_name: str,
                ts_min: str | None = None, ts_max: str | None = None) -> pd.DataFrame:
        if df.empty or sum_col not in df.columns:
            return pd.DataFrame(columns=_KEY)
        agg: dict = {sum_name: (sum_col, "sum")}
        if ts_min and "timestamp" in df.columns:
            agg[ts_min] = ("timestamp", "min")
        if ts_max and "timestamp" in df.columns:
            agg[ts_max] = ("timestamp", "max")
        return (
            df.dropna(subset=_KEY)
            .groupby(_KEY, as_index=False)
            .agg(**agg)
        )

    m  = grp_agg(mints,    "amount_usd", "total_minted_usd",    "first_mint_utc", "last_mint_utc")
    b  = grp_agg(burns,    "amount_usd", "total_burned_usd",    ts_max="last_burn_utc")
    c  = grp_agg(collects, "amount_usd", "total_collected_usd", ts_max="last_collect_utc")

    # Outer join: keep all positions that appear in any event type
    result = m
    for other in (b, c):
        if not other.empty:
            result = result.merge(other, on=_KEY, how="outer")

    return result


def main() -> None:
    DATA_OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DEX LP Position Processing")
    print("=" * 60)

    print("Loading mints...")
    mints = coerce_events(load_monthly("mints_*.csv"))
    print(f"  {len(mints):,} mint events")

    print("Loading burns...")
    burns = coerce_events(load_monthly("burns_*.csv"))
    print(f"  {len(burns):,} burn events")

    print("Loading collects...")
    collects = coerce_events(load_monthly("collects_*.csv"))
    print(f"  {len(collects):,} collect events")

    if mints.empty and burns.empty and collects.empty:
        print("[ERROR] No event data found. Run import scripts first.")
        return

    # ── Aggregate by position key ─────────────────────────────────────────────
    print("Aggregating by position (owner, tick_lower, tick_upper)...")
    cf = aggregate_cashflows(mints, burns, collects)

    for col in ("total_minted_usd", "total_burned_usd", "total_collected_usd"):
        if col not in cf.columns:
            cf[col] = 0.0
        else:
            cf[col] = pd.to_numeric(cf[col], errors="coerce").fillna(0.0)

    cf["net_pnl_usd"]    = cf["total_burned_usd"] + cf["total_collected_usd"] - cf["total_minted_usd"]
    # Fees ≈ collected value minus the USD value of principal returned via burn
    cf["fee_income_usd"] = cf["total_collected_usd"] - cf["total_burned_usd"]

    # ── Range metrics ─────────────────────────────────────────────────────────
    tl = pd.to_numeric(cf["tick_lower"], errors="coerce")
    tu = pd.to_numeric(cf["tick_upper"], errors="coerce")
    cf["range_width_ticks"] = (tu - tl).astype("Int64")
    cf["range_width_pct"]   = (1.0001 ** (tu - tl) - 1) * 100
    cf["position_type"]     = pd.cut(
        cf["range_width_pct"],
        bins=[-np.inf, 5, 20, np.inf],
        labels=["narrow", "medium", "wide"],
    )

    # ── Duration ──────────────────────────────────────────────────────────────
    ts_cols = [c for c in ("last_mint_utc", "last_burn_utc", "last_collect_utc") if c in cf.columns]
    if ts_cols and "first_mint_utc" in cf.columns:
        for c in ts_cols:
            cf[c] = pd.to_datetime(cf[c], utc=True, errors="coerce")
        cf["first_mint_utc"] = pd.to_datetime(cf["first_mint_utc"], utc=True, errors="coerce")
        cf["last_event_utc"] = cf[ts_cols].max(axis=1)
        cf["duration_days"]  = (
            (cf["last_event_utc"] - cf["first_mint_utc"]).dt.total_seconds() / 86400
        )

    # ── Active status from positions_current.csv ──────────────────────────────
    pos_path = DATA_RAW / "positions_current.csv"
    if pos_path.exists():
        print("Loading positions_current.csv for is_active flag...")
        pos = pd.read_csv(pos_path, low_memory=False)
        # Normalise tick column names
        rename = {}
        for c in pos.columns:
            lc = c.lower()
            if "tick_lower" in lc and c != "tick_lower":
                rename[c] = "tick_lower"
            elif "tick_upper" in lc and c != "tick_upper":
                rename[c] = "tick_upper"
        pos = pos.rename(columns=rename)

        if {"owner", "tick_lower", "tick_upper", "liquidity"}.issubset(pos.columns):
            pos["tick_lower"] = pd.to_numeric(pos["tick_lower"], errors="coerce")
            pos["tick_upper"] = pd.to_numeric(pos["tick_upper"], errors="coerce")
            active = frozenset(
                tuple(r)
                for r in pos.loc[pos["liquidity"].astype(str).ne("0"), _KEY]
                .dropna()
                .itertuples(index=False, name=None)
            )
            cf["is_active"] = cf.apply(
                lambda r: (r["owner"], r["tick_lower"], r["tick_upper"]) in active,
                axis=1,
            )
        else:
            print("  [WARN] positions_current.csv missing required columns — is_active set to NA.")
            cf["is_active"] = pd.NA
    else:
        print("[WARN] positions_current.csv not found — is_active set to NA.")
        cf["is_active"] = pd.NA

    out = DATA_OUT / "dex_lp_positions.csv"
    cf.to_csv(out, index=False)
    print(f"\nSaved {out}  ({len(cf):,} positions)")

    if "position_type" in cf.columns:
        for pt, grp in cf.groupby("position_type", observed=True):
            print(f"  {pt}: {len(grp):,} positions  |  "
                  f"median net P&L: ${grp['net_pnl_usd'].median():,.0f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
