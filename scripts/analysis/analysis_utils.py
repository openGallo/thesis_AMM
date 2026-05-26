"""Shared utilities for all analysis scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PROC    = PROJECT_ROOT / "data_processed"
FIG_DIR      = PROJECT_ROOT / "output" / "figures"
TAB_DIR      = PROJECT_ROOT / "output" / "tables"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

# ── Matplotlib style ──────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "figure.constrained_layout.use": True,
})

COLORS = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]


def savefig(name: str) -> None:
    path = FIG_DIR / f"{name}.pdf"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.relative_to(PROJECT_ROOT)}")


def savetable(df: pd.DataFrame, name: str) -> None:
    csv_path = TAB_DIR / f"{name}.csv"
    df.to_csv(csv_path)
    try:
        tex = df.to_latex(float_format="%.4f", na_rep="--")
        (TAB_DIR / f"{name}.tex").write_text(tex, encoding="utf-8")
    except Exception:
        pass
    print(f"  Saved {csv_path.relative_to(PROJECT_ROOT)}")


def load(rel: str, **kwargs) -> pd.DataFrame | None:
    path = DATA_PROC / rel
    if not path.exists():
        print(f"  [SKIP] {rel} not found — run processing pipeline first")
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True, low_memory=False, **kwargs)
    if hasattr(df.index, "tz") and df.index.tz is None:
        try:
            df.index = pd.to_datetime(df.index, utc=True)
        except Exception:
            pass
    return df


def vol_regime(series: pd.Series, low_q: float = 0.33,
               high_q: float = 0.67) -> pd.Series:
    """Label each observation as low / normal / high volatility."""
    lo = series.quantile(low_q)
    hi = series.quantile(high_q)
    return pd.cut(series, bins=[-float("inf"), lo, hi, float("inf")],
                  labels=["low", "normal", "high"])
