"""
Causal Inference Scripts — Academic Ranking.

Scores all 10 DiD/RD scripts on five criteria and outputs:
  • output/tables/causal_ranking.csv
  • output/tables/causal_ranking.tex    (LaTeX booktabs table)
  • output/tables/causal_ranking_detail.csv

Ranking criteria (each 1–5 points, equal weight → max 25):
─────────────────────────────────────────────────────────────
  IC  Identification Credibility
        5 = quasi-experimental gold standard with no plausible
            competing explanation, placebo tests pass
        4 = strong design with one credibly addressed threat
        3 = reasonable design with residual confounding risk
        2 = design is underpowered or confounded
        1 = mostly descriptive, causal claim very weak

  SP  Statistical Power
        5 = N > 50 000, effect well-determined, tight SE
        4 = adequate N (10 000–50 000) or wide CI but detectable
        3 = moderate N or reliance on asymptotic approximations
        2 = small N (<500 effective obs), test likely underpowered
        1 = too few observations for any meaningful inference

  ER  Economic Relevance
        5 = directly maps to a core thesis contribution (LVR,
            LP profitability, arb mechanics)
        4 = strong tie to AMM economics
        3 = relevant but somewhat tangential
        2 = weakly related to thesis RQs
        1 = tangential / peripheral

  DF  Data Fit
        5 = columns exist, correctly scaled, no proxy needed
        4 = minor proxy needed (e.g., gas estimated) but defensible
        3 = proxy + matching across datasets with non-trivial error
        2 = outcome variable is a weak proxy for the theoretical construct
        1 = data largely unavailable; simulation required

  LN  Literature Novelty
        5 = first paper to address this exact mechanism with this data
        4 = novel application of established method to AMM context
        3 = replicates existing literature with minor extension
        2 = direct replication with different data
        1 = already well-established, no novel contribution

Usage:
    python scripts/analysis/causal_inference/ranking.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path fix ──────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis_utils import savetable  # noqa: E402

# ── Scoring table ─────────────────────────────────────────────────────────────
#
# Each row: (file, short_name, event/design, IC, SP, ER, DF, LN, notes)
#
# Scores reflect the *improved* versions in scripts/analysis/causal_inference/
# (with BH correction, donut RD, cluster-SE caveats, ETH price control, etc.)
#
SCORES: list[dict] = [
    {
        "file":       "rd_arbitrage_trigger.py",
        "name":       "RD: Arb Break-even",
        "type":       "Sharp RD",
        "event":      "Gas-adj. arb threshold c* (bps)",
        "IC": 4, "SP": 5, "ER": 5, "DF": 4, "LN": 5,
        "notes": (
            "Running variable (DEX-CEX spread in bps) is continuous and not "
            "manipulable by individual LPs. McCrary density test rules out "
            "strategic sorting. IK-optimal bandwidth + donut RD for robustness. "
            "Directly identifies the gas-cost floor that triggers arbitrage — "
            "core to the thesis LVR mechanism. Novel: first RD on AMM arb trigger."
        ),
    },
    {
        "file":       "rd_vol_lvr_breakeven.py",
        "name":       "RD: Vol LVR Break-even",
        "type":       "Sharp RD",
        "event":      "Realised vol σ* where LVR = fee",
        "IC": 4, "SP": 5, "ER": 5, "DF": 4, "LN": 5,
        "notes": (
            "Running variable (24h realised vol) jumps in TVL / liquidity provision "
            "at σ* where the LP break-even condition changes sign. Two independent "
            "cutoff estimators (bin-crossing + quantile split) cross-validate. "
            "Reverse-causality note: LP withdrawal at σ* could raise vol (addressed "
            "in McCrary bias paragraph). Novel: empirical estimation of LVR-break-even "
            "vol threshold; literature uses only theoretical σ* from Black-Scholes."
        ),
    },
    {
        "file":       "did_ftx_lp_withdrawal.py",
        "name":       "DiD: FTX LP Exit",
        "type":       "DiD (panel)",
        "event":      "FTX collapse, Nov 8 2022",
        "IC": 5, "SP": 4, "ER": 5, "DF": 4, "LN": 4,
        "notes": (
            "Exogenous crypto-sector shock unrelated to USDC/WETH pool mechanics. "
            "Narrow vs wide LP positions as treatment/control. Parallel-trends "
            "assumption checked over ±7d pre-period. Kaplan-Meier survival + "
            "log-rank test. HAC SE; wild-bootstrap caveat with 2 groups noted "
            "(Bertrand et al. 2004; Cameron et al. 2008). ATT estimand explicit."
        ),
    },
    {
        "file":       "rd_tick_crossing.py",
        "name":       "RD: Tick Crossing",
        "type":       "Fuzzy RD",
        "event":      "Tick-spacing boundary (TICK=10)",
        "IC": 4, "SP": 5, "ER": 4, "DF": 4, "LN": 5,
        "notes": (
            "Fuzzy design because tick-crossing probability jumps but is not "
            "deterministic (pool can skip ticks). Wald (LATE) = rf / fs with "
            "first-stage F reported; weak-instrument check (Staiger-Stock 1997). "
            "Empirical tick-spacing verification guards against miscalssification. "
            "Novel: first RD exploiting Uniswap v3 discrete tick architecture."
        ),
    },
    {
        "file":       "did_usdc_depeg.py",
        "name":       "DiD: USDC De-peg",
        "type":       "DiD + ITS",
        "event":      "USDC de-peg, Mar 10–14 2023",
        "IC": 5, "SP": 2, "ER": 5, "DF": 4, "LN": 5,
        "notes": (
            "Narrow event window (3 days) and SVB-linked exogenous trigger provide "
            "exceptional identification. N is small (depeg episode only), so "
            "Wilcoxon rank-sum (non-parametric) used for the episode study. "
            "BH correction excludes contaminated abs_basis outcome. Novel: first "
            "causal study of a stable-coin de-peg effect on Uniswap v3 LP economics. "
            "SP penalised: only ~72 hourly obs in the depeg window."
        ),
    },
    {
        "file":       "did_merge_impact.py",
        "name":       "DiD/ITS: The Merge",
        "type":       "ITS (5 outcomes)",
        "event":      "Ethereum Merge, Sep 15 2022",
        "IC": 4, "SP": 3, "ER": 5, "DF": 4, "LN": 4,
        "notes": (
            "Ethereum's PoW→PoS transition is exogenous to USDC/WETH pool. "
            "Confound: bear market overlap and prior FTX/Terra shocks. ±7d "
            "exclusion window reduces but cannot fully eliminate macro confounds. "
            "BH correction across 5 outcomes. HAC SE with 168-lag (1-week). "
            "ITS design identifies level + slope breaks but not staggered timing."
        ),
    },
    {
        "file":       "did_terra_luna.py",
        "name":       "DiD/ITS: Terra/LUNA",
        "type":       "ITS + DiD",
        "event":      "LUNA cascade, May 9 2022",
        "IC": 4, "SP": 3, "ER": 4, "DF": 4, "LN": 4,
        "notes": (
            "Event is exogenous to USDC/WETH pool. Confounder risk: 3AC/Celsius "
            "failure at day 62 post-Terra → DiD window capped at 30d. "
            "Multi-event comparison function jointly estimates Terra vs FTX "
            "ITS and performs approximate Wald test for H0: β₂_Terra = β₂_FTX. "
            "Sensitivity to May 8 start date. BH correction across 5 outcomes."
        ),
    },
    {
        "file":       "did_shapella.py",
        "name":       "Split-ITS: Shapella",
        "type":       "Split ITS",
        "event":      "Shapella upgrade, Apr 12 2023",
        "IC": 4, "SP": 3, "ER": 4, "DF": 3, "LN": 5,
        "notes": (
            "Unique split-ITS design separates short (0–30d) from long (31–120d) "
            "post-Shapella adjustment paths — motivated by the theoretical claim "
            "that ETH staking inflows reallocate across DeFi with a lag. Pre-trend "
            "test and Levene variance-shift test. Data fit penalised: staking-yield "
            "proxy is indirect (ETH staking flows not directly observed in pool data). "
            "Novel: first split-ITS applied to an Ethereum upgrade."
        ),
    },
    {
        "file":       "rd_lp_profitability_threshold.py",
        "name":       "RD: LP Profitability",
        "type":       "Sharp RD",
        "event":      "Fee-APR at mint vs LVR cutoff",
        "IC": 3, "SP": 4, "ER": 5, "DF": 3, "LN": 4,
        "notes": (
            "Running variable (ex-ante fee-APR at mint hour) determines ex-post "
            "net P&L. Cutoff at p50(LVR rate at mint). Identification concern: "
            "LPs with high ex-ante fee-APR may differ on unobservables (experience, "
            "risk appetite). Heterogeneity: separate RDs for narrow vs wide positions. "
            "Data fit: fee-APR and LVR rate are only approximations at mint hour "
            "(pool-level, not position-level). IC penalised for self-selection threat."
        ),
    },
    {
        "file":       "did_eip1559_gas.py",
        "name":       "DiD/ITS: EIP-1559",
        "type":       "ITS + DiD",
        "event":      "EIP-1559, Aug 5 2021",
        "IC": 3, "SP": 2, "ER": 4, "DF": 3, "LN": 3,
        "notes": (
            "EIP-1559 changed Ethereum gas market mechanics. IC limited: "
            "Aug–Nov 2021 was a major bull market; ETH price control added "
            "to reduce this confounder but residual confounding remains. "
            "SP limited: USDC/WETH 0.05% pool was launched in May 2021 "
            "→ only ~3 months of pre-period data (≈2160 hourly obs). "
            "Gas variables measured in the pool data are indirect proxies "
            "for the mechanism. Ranked lowest: most confounded and data-thin."
        ),
    },
]

# ── Criteria labels and descriptions ─────────────────────────────────────────
CRITERIA = {
    "IC": "Identification Credibility",
    "SP": "Statistical Power",
    "ER": "Economic Relevance",
    "DF": "Data Fit",
    "LN": "Literature Novelty",
}

SCORE_SCALE = {
    5: "Excellent",
    4: "Good",
    3: "Fair",
    2: "Weak",
    1: "Poor",
}


def build_ranking_df() -> pd.DataFrame:
    rows = []
    for entry in SCORES:
        total = entry["IC"] + entry["SP"] + entry["ER"] + entry["DF"] + entry["LN"]
        stars = "★" * (total // 5) + ("☆" * (5 - total // 5))
        rows.append({
            "File":               entry["file"],
            "Name":               entry["name"],
            "Design":             entry["type"],
            "Event / Running var": entry["event"],
            "IC": entry["IC"],
            "SP": entry["SP"],
            "ER": entry["ER"],
            "DF": entry["DF"],
            "LN": entry["LN"],
            "Total": total,
            "Stars": stars,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Total", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def build_detail_df() -> pd.DataFrame:
    rows = []
    for entry in SCORES:
        rows.append({
            "File":   entry["file"],
            "Name":   entry["name"],
            "Design": entry["type"],
            "IC": entry["IC"],
            "SP": entry["SP"],
            "ER": entry["ER"],
            "DF": entry["DF"],
            "LN": entry["LN"],
            "Total": entry["IC"] + entry["SP"] + entry["ER"] + entry["DF"] + entry["LN"],
            "Rationale": entry["notes"],
        })
    df = pd.DataFrame(rows).sort_values("Total", ascending=False).reset_index(drop=True)
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


def to_latex(df: pd.DataFrame) -> str:
    """
    Render the ranking DataFrame as a LaTeX booktabs table suitable for
    copy-pasting into a master thesis.
    """
    cols_display = ["Rank", "Name", "Design", "IC", "SP", "ER", "DF", "LN", "Total"]
    sub = df[cols_display].copy()

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Academic ranking of causal inference designs. "
        r"Criteria scored 1–5: IC = Identification Credibility, "
        r"SP = Statistical Power, ER = Economic Relevance, "
        r"DF = Data Fit, LN = Literature Novelty (total max = 25).}",
        r"\label{tab:causal_ranking}",
        r"\begin{tabular}{rllcccccr}",
        r"\toprule",
        r"\# & Script & Design & IC & SP & ER & DF & LN & Total \\",
        r"\midrule",
    ]

    for _, row in sub.iterrows():
        rank   = int(row["Rank"])
        name   = row["Name"].replace("&", r"\&")
        design = row["Design"].replace("&", r"\&")
        ic, sp, er, df_, ln, tot = (int(row[c]) for c in ["IC", "SP", "ER", "DF", "LN", "Total"])
        lines.append(
            rf"{rank} & {name} & {design} & {ic} & {sp} & {er} & {df_} & {ln} & \textbf{{{tot}}} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_report(df: pd.DataFrame, detail: pd.DataFrame) -> None:
    sep = "─" * 78
    print(f"\n{'═'*78}")
    print("  CAUSAL INFERENCE SCRIPTS — ACADEMIC RANKING")
    print(f"  Scored on 5 criteria × 10 scripts (max = 25 per script)")
    print(f"{'═'*78}\n")

    for _, row in df.iterrows():
        rank  = int(row["Rank"])
        total = int(row["Total"])
        print(f"  #{rank:2d}  [{total:2d}/25]  {row['Name']}")
        print(f"        Design : {row['Design']}")
        print(f"        Scores : IC={row['IC']} SP={row['SP']} ER={row['ER']} "
              f"DF={row['DF']} LN={row['LN']}")
        # grab notes from detail
        note = detail.loc[detail["File"] == row["File"], "Rationale"].iloc[0]
        # word-wrap at 72 chars
        words = note.split()
        line, wrapped = "        ", []
        for w in words:
            if len(line) + len(w) + 1 > 78:
                wrapped.append(line)
                line = "        " + w
            else:
                line += (" " if line.strip() else "") + w
        if line.strip():
            wrapped.append(line)
        print("\n".join(wrapped))
        print()

    print(sep)
    print("Criteria legend:")
    for code, label in CRITERIA.items():
        print(f"  {code} = {label}")
    print("\nScale: 5=Excellent  4=Good  3=Fair  2=Weak  1=Poor")
    print(sep)

    # summary stats
    score_cols = ["IC", "SP", "ER", "DF", "LN", "Total"]
    stats = df[score_cols].agg(["mean", "min", "max"]).T
    stats.columns = ["Mean", "Min", "Max"]
    stats = stats.round(2)
    print("\nCriteria summary (across 10 scripts):")
    for col, row2 in stats.iterrows():
        print(f"  {col:6s}  mean={row2['Mean']:.1f}  min={row2['Min']:.0f}  max={row2['Max']:.0f}")

    print(f"\nTop RD script  : #{df[df['Design'].str.contains('RD')].iloc[0]['Rank']:d}"
          f"  {df[df['Design'].str.contains('RD')].iloc[0]['Name']}")
    print(f"Top DiD script : #{df[df['Design'].str.contains('DiD|ITS|Split')].iloc[0]['Rank']:d}"
          f"  {df[df['Design'].str.contains('DiD|ITS|Split')].iloc[0]['Name']}")


def main() -> None:
    df     = build_ranking_df()
    detail = build_detail_df()

    print_report(df, detail)

    # ── save outputs ──────────────────────────────────────────────────────────
    # Main ranking (wide)
    savetable(df, "causal_ranking")

    # Detailed with rationale
    savetable(detail, "causal_ranking_detail")

    # LaTeX
    tex = to_latex(df)
    # find output dir via analysis_utils conventions
    out_dir = Path(__file__).resolve().parent.parent.parent.parent / "output" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / "causal_ranking.tex"
    tex_path.write_text(tex, encoding="utf-8")
    print(f"\n  [saved]  {tex_path.relative_to(Path(__file__).resolve().parents[4])}")
    print(f"  [saved]  output/tables/causal_ranking.csv")
    print(f"  [saved]  output/tables/causal_ranking_detail.csv")
    print("\nTip: \\input{tables/causal_ranking.tex} in your thesis LaTeX document.")


if __name__ == "__main__":
    main()
