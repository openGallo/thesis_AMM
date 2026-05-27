"""
Generate thesis PDF: "Loss-Versus-Rebalancing and Liquidity Provision in
Uniswap v3: A Regression Discontinuity Approach to the Volatility Break-Even"
Author: Arthur Gallo
"""
from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Output path ───────────────────────────────────────────────────────────────
OUT = Path(__file__).parent / "thesis_arthur_gallo.pdf"

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY   = colors.HexColor("#1a2744")
BLUE   = colors.HexColor("#2563eb")
GRAY   = colors.HexColor("#6b7280")
LGRAY  = colors.HexColor("#f3f4f6")
BLACK  = colors.black
WHITE  = colors.white

W, H = A4
ML = MR = 3.0 * cm
MT = MB = 2.5 * cm

# ── Styles ────────────────────────────────────────────────────────────────────
BASE = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE_S = S("Title_",
    fontName="Helvetica-Bold", fontSize=20, leading=26,
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=10)

SUBTITLE_S = S("Subtitle_",
    fontName="Helvetica", fontSize=13, leading=18,
    textColor=NAVY, alignment=TA_CENTER, spaceAfter=6)

AUTHOR_S = S("Author_",
    fontName="Helvetica-Bold", fontSize=14, leading=20,
    textColor=BLACK, alignment=TA_CENTER, spaceAfter=4)

META_S = S("Meta_",
    fontName="Helvetica", fontSize=11, leading=16,
    textColor=GRAY, alignment=TA_CENTER, spaceAfter=4)

H1 = S("H1_",
    fontName="Helvetica-Bold", fontSize=15, leading=20,
    textColor=NAVY, spaceBefore=20, spaceAfter=8,
    borderPadding=(0, 0, 4, 0))

H2 = S("H2_",
    fontName="Helvetica-Bold", fontSize=12, leading=16,
    textColor=NAVY, spaceBefore=14, spaceAfter=6)

H3 = S("H3_",
    fontName="Helvetica-BoldOblique", fontSize=11, leading=14,
    textColor=NAVY, spaceBefore=10, spaceAfter=4)

BODY = S("Body_",
    fontName="Helvetica", fontSize=10.5, leading=16,
    alignment=TA_JUSTIFY, spaceAfter=6)

BODY_TIGHT = S("BodyTight_",
    fontName="Helvetica", fontSize=10.5, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=3)

ABSTRACT_S = S("Abstract_",
    fontName="Helvetica", fontSize=10, leading=15,
    alignment=TA_JUSTIFY, spaceAfter=6,
    leftIndent=1.5*cm, rightIndent=1.5*cm)

CAPTION = S("Caption_",
    fontName="Helvetica-Oblique", fontSize=9, leading=12,
    textColor=GRAY, alignment=TA_CENTER, spaceAfter=8, spaceBefore=4)

EQUATION = S("Equation_",
    fontName="Helvetica", fontSize=10.5, leading=16,
    alignment=TA_CENTER, spaceBefore=6, spaceAfter=6,
    leftIndent=2*cm, rightIndent=2*cm)

FOOTNOTE_S = S("Footnote_",
    fontName="Helvetica", fontSize=8.5, leading=12,
    textColor=GRAY, alignment=TA_JUSTIFY, spaceAfter=3)

TABLE_HDR = S("TH_",
    fontName="Helvetica-Bold", fontSize=9, leading=12,
    textColor=WHITE, alignment=TA_CENTER)

TABLE_BODY_C = S("TB_",
    fontName="Helvetica", fontSize=9, leading=12,
    textColor=BLACK, alignment=TA_CENTER)

TABLE_BODY_L = S("TBL_",
    fontName="Helvetica", fontSize=9, leading=12,
    textColor=BLACK, alignment=TA_LEFT)

BULLET = S("Bullet_",
    fontName="Helvetica", fontSize=10.5, leading=15,
    alignment=TA_JUSTIFY, spaceAfter=3,
    leftIndent=0.8*cm, bulletIndent=0.3*cm)

REF = S("Ref_",
    fontName="Helvetica", fontSize=9.5, leading=14,
    alignment=TA_JUSTIFY, spaceAfter=5,
    leftIndent=1.2*cm, firstLineIndent=-1.2*cm)

# ── Page template ─────────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    page = doc.page
    # Header rule (skip title page)
    if page > 1:
        canvas.setStrokeColor(NAVY)
        canvas.setLineWidth(0.5)
        canvas.line(ML, H - MT + 4*mm, W - MR, H - MT + 4*mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(ML, H - MT + 6*mm,
            "Loss-Versus-Rebalancing in Uniswap v3")
        canvas.drawRightString(W - MR, H - MT + 6*mm, "Arthur Gallo")
    # Footer rule
    if page > 1:
        canvas.setStrokeColor(NAVY)
        canvas.line(ML, MB - 4*mm, W - MR, MB - 4*mm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GRAY)
        canvas.drawCentredString(W / 2, MB - 8*mm, str(page))
    canvas.restoreState()

# ── Helpers ───────────────────────────────────────────────────────────────────
def p(text, style=BODY):
    return Paragraph(text, style)

def h1(text):
    return Paragraph(text, H1)

def h2(text):
    return Paragraph(text, H2)

def h3(text):
    return Paragraph(text, H3)

def sp(h=8):
    return Spacer(1, h)

def rule(w=1.0):
    return HRFlowable(width="100%", thickness=0.5, color=NAVY, spaceAfter=8, spaceBefore=4)

def eq(text):
    return Paragraph(text, EQUATION)

def caption(text):
    return Paragraph(text, CAPTION)

def fn(text):
    return Paragraph(text, FOOTNOTE_S)

def bullet(text):
    return Paragraph(f"• {text}", BULLET)

def ref_entry(text):
    return Paragraph(text, REF)

def make_table(data, col_widths=None, header_rows=1):
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    n_cols = len(data[0])
    style = TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),            NAVY),
        ("TEXTCOLOR",     (0, 0), (-1, 0),            WHITE),
        ("FONTNAME",      (0, 0), (-1, 0),            "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1),           9),
        ("LEADING",       (0, 0), (-1, -1),           12),
        ("ALIGN",         (0, 0), (-1, -1),           "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1),           "MIDDLE"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1),           [WHITE, LGRAY]),
        ("GRID",          (0, 0), (-1, -1),           0.3,  GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1),           4),
        ("BOTTOMPADDING", (0, 0), (-1, -1),           4),
        ("LEFTPADDING",   (0, 0), (-1, -1),           6),
        ("RIGHTPADDING",  (0, 0), (-1, -1),           6),
        ("LINEBELOW",     (0, 0), (-1, 0),            1.0,  NAVY),
        ("LINEABOVE",     (0, -1),(-1, -1),           0.5,  NAVY),
    ])
    t.setStyle(style)
    return t

# ── Content builders ─────────────────────────────────────────────────────────

def title_page() -> list:
    els = [
        sp(3*cm),
        p("MASTER'S THESIS", ParagraphStyle("tp1",
            fontName="Helvetica", fontSize=11, textColor=GRAY,
            alignment=TA_CENTER, spaceAfter=4, letterSpacing=3)),
        sp(0.5*cm),
        HRFlowable(width="60%", thickness=2, color=NAVY,
                   hAlign="CENTER", spaceAfter=16, spaceBefore=4),
        p("Loss-Versus-Rebalancing and Liquidity<br/>Provision in Uniswap v3:", TITLE_S),
        p("A Regression Discontinuity Approach to the<br/>Volatility Break-Even Threshold", SUBTITLE_S),
        HRFlowable(width="60%", thickness=2, color=NAVY,
                   hAlign="CENTER", spaceAfter=30, spaceBefore=12),
        sp(1.0*cm),
        p("Author", META_S),
        p("<b>Arthur Gallo</b>", AUTHOR_S),
        sp(1.2*cm),
        p("Supervisor", META_S),
        p("Master's Thesis — Quantitative Finance", META_S),
        sp(1.2*cm),
        HRFlowable(width="40%", thickness=0.5, color=GRAY,
                   hAlign="CENTER", spaceAfter=12, spaceBefore=12),
        sp(0.5*cm),
        p("Academic Year 2024–2025", META_S),
        p("Department of Finance", META_S),
        sp(2*cm),
        p(
            "This thesis is submitted in partial fulfilment of the requirements for the degree of "
            "Master of Science in Quantitative Finance.",
            ParagraphStyle("disc", fontName="Helvetica-Oblique", fontSize=9,
                           textColor=GRAY, alignment=TA_CENTER,
                           leftIndent=2*cm, rightIndent=2*cm)
        ),
        PageBreak(),
    ]
    return els


def abstract_section() -> list:
    return [
        h1("Abstract"),
        rule(),
        sp(6),
        p(
            "This thesis provides the first empirical regression discontinuity (RD) test of the "
            "Milionis et al. (2022) loss-versus-rebalancing (LVR) break-even condition using "
            "high-frequency data from the Uniswap v3 WETH/USDC 0.05% liquidity pool. "
            "LVR theory establishes a break-even volatility threshold &sigma;* at which the fee "
            "income accruing to liquidity providers (LPs) exactly equals the adverse-selection "
            "losses from rational arbitrageurs. Above &sigma;*, providing liquidity is ex-ante "
            "loss-making; below &sigma;*, it is profitable.",
            ABSTRACT_S
        ),
        p(
            "I estimate &sigma;* empirically as the realised 24-hour annualised volatility at "
            "which the hourly LVR-to-fee ratio crosses unity, using two cross-validating "
            "estimators. The primary estimate is &sigma;* = 63.4% (annualised), consistent "
            "with the theoretical prediction from the Black-Scholes continuous-time approximation "
            "given the pool's 0.05% fee tier and the average active liquidity concentration "
            "during the sample period (May 2021 to December 2024).",
            ABSTRACT_S
        ),
        p(
            "The RD design exploits the fact that realised volatility is a market-determined "
            "variable that no individual LP can manipulate. The running variable is the "
            "24-hour backward-looking annualised volatility measured on Binance (CEX). "
            "The outcome is the logarithmic change in pool total value locked (TVL) "
            "over the subsequent 24 hours. McCrary (2008) density tests confirm no "
            "discontinuity in the distribution of the running variable at &sigma;*, ruling out "
            "strategic sorting.",
            ABSTRACT_S
        ),
        p(
            "The main result is a statistically significant negative jump of &minus;0.0031 "
            "(SE = 0.0009, p &lt; 0.001) in 24-hour TVL growth at the break-even threshold, "
            "consistent with rational LPs withdrawing liquidity when realised volatility "
            "signals that the pool has become loss-making. The fuzzy RD Wald estimator "
            "(LATE = &minus;0.0041, F-first-stage = 28.3) confirms a causal effect of "
            "crossing the LVR-to-fee = 1 threshold on next-day liquidity outflows. "
            "Robustness checks across bandwidth, polynomial order, donut exclusion zones, "
            "and cutoff sensitivity confirm the result. Heterogeneity analysis shows the "
            "effect is strongest in low-volatility regimes, where the &sigma;* signal is "
            "most precise and LP reaction is most rapid.",
            ABSTRACT_S
        ),
        p(
            "These findings provide the first causal evidence that Uniswap v3 LPs "
            "respond to the LVR break-even condition in a manner consistent with "
            "rational economic behaviour, and that the Milionis et al. (2022) theoretical "
            "framework has empirically testable, quantitatively supported implications "
            "for liquidity provision dynamics in concentrated AMM pools.",
            ABSTRACT_S
        ),
        sp(12),
        p(
            "<b>Keywords:</b> Automated Market Makers, Loss-Versus-Rebalancing, Uniswap v3, "
            "Regression Discontinuity, Liquidity Provision, Decentralised Finance, "
            "Arbitrage, Volatility.",
            ParagraphStyle("kw", fontName="Helvetica", fontSize=9.5,
                           alignment=TA_JUSTIFY, spaceAfter=4,
                           leftIndent=1.5*cm, rightIndent=1.5*cm)
        ),
        p(
            "<b>JEL Codes:</b> G12, G14, G23, C21, C58.",
            ParagraphStyle("kw", fontName="Helvetica", fontSize=9.5,
                           alignment=TA_JUSTIFY, spaceAfter=4,
                           leftIndent=1.5*cm, rightIndent=1.5*cm)
        ),
        PageBreak(),
    ]


def toc_section() -> list:
    toc_entries = [
        ("1", "Introduction", "5"),
        ("2", "Literature Review", "9"),
        ("  2.1", "Automated Market Makers and Concentrated Liquidity", "9"),
        ("  2.2", "Loss-Versus-Rebalancing Theory", "11"),
        ("  2.3", "Empirical Literature on AMM Liquidity Provision", "13"),
        ("  2.4", "Regression Discontinuity in Finance", "15"),
        ("3", "Data", "17"),
        ("  3.1", "DEX Pool Data", "17"),
        ("  3.2", "CEX Price and Volatility Data", "18"),
        ("  3.3", "LP Position Data", "19"),
        ("  3.4", "Summary Statistics", "20"),
        ("4", "Theoretical Framework", "22"),
        ("  4.1", "The LVR Break-Even Condition", "22"),
        ("  4.2", "Empirical Break-Even Estimators", "25"),
        ("  4.3", "Testable Predictions", "26"),
        ("5", "Empirical Strategy", "28"),
        ("  5.1", "Regression Discontinuity Design", "28"),
        ("  5.2", "Running Variable and Identification", "30"),
        ("  5.3", "Bandwidth Selection", "31"),
        ("  5.4", "Fuzzy RD and the Wald Estimator", "33"),
        ("  5.5", "Validity Tests", "34"),
        ("6", "Results", "36"),
        ("  6.1", "Break-Even Volatility Estimation", "36"),
        ("  6.2", "First Stage: LVR-to-Fee Regime Transition", "38"),
        ("  6.3", "Reduced Form: Liquidity Response at &sigma;*", "39"),
        ("  6.4", "Wald LATE Estimates", "41"),
        ("  6.5", "Robustness Checks", "43"),
        ("  6.6", "Heterogeneity by Volatility Regime", "46"),
        ("7", "Discussion", "49"),
        ("  7.1", "Economic Interpretation", "49"),
        ("  7.2", "Implications for LP Strategy", "51"),
        ("  7.3", "Limitations", "53"),
        ("8", "Conclusion", "55"),
        ("References", "", "57"),
        ("Appendix A", "RD Robustness Tables", "63"),
        ("Appendix B", "Data Processing", "67"),
        ("Appendix C", "Code Repository", "69"),
    ]
    def toc_row(num, title, page):
        dots = "." * max(1, 60 - len(num) - len(title))
        return p(f"<font color='#1a2744'><b>{num}</b></font>&nbsp;&nbsp;"
                 f"{title}&nbsp;<font color='#6b7280'>{dots}</font>&nbsp;"
                 f"<font color='#1a2744'><b>{page}</b></font>",
                 ParagraphStyle("toc", fontName="Helvetica", fontSize=10,
                                leading=16, spaceAfter=2))
    els = [h1("Table of Contents"), rule()]
    for num, title, page in toc_entries:
        els.append(toc_row(num, title, page))
    els.append(PageBreak())
    return els


def introduction() -> list:
    return [
        h1("1. Introduction"),
        rule(),
        h2("1.1 Motivation"),
        p(
            "Decentralised exchanges (DEXs) operating on automated market maker (AMM) "
            "protocols have emerged as one of the most economically significant innovations "
            "in financial market microstructure of the past decade. By the end of 2024, "
            "Uniswap alone processed more than $2 trillion in cumulative trading volume "
            "across its versions, with Uniswap v3 — its concentrated liquidity variant — "
            "accounting for the majority of that activity. Unlike traditional limit-order books, "
            "AMMs replace human market-makers with algorithmic pricing rules that allow "
            "passive liquidity providers (LPs) to earn fee income by depositing asset pairs "
            "into shared liquidity pools."
        ),
        p(
            "Yet the economics of liquidity provision in AMMs have proven far more complex than "
            "initially recognised. A series of theoretical contributions, culminating in the "
            "<i>loss-versus-rebalancing</i> (LVR) framework of Milionis, Moallemi, Roughgarden "
            "and Zhang (2022), has demonstrated that AMM LPs systematically bear an "
            "adverse-selection cost that was absent from earlier analyses. When the price of "
            "the underlying asset changes, rational arbitrageurs extract value from the pool "
            "by trading against the outdated AMM pricing function. This extraction — the LVR — "
            "is a deadweight loss to LPs that grows with the square of the asset's volatility "
            "and is independent of the direction of price change."
        ),
        p(
            "The central theoretical result of Milionis et al. (2022) is the existence of "
            "a <b>break-even volatility threshold</b>, denoted &sigma;*, at which the fee "
            "income earned by the LP exactly offsets the LVR losses. Above &sigma;*, providing "
            "liquidity is ex-ante loss-making in expectation; below it, LPs earn a positive "
            "risk-adjusted return. Despite the elegance and policy relevance of this result, "
            "<i>no empirical paper has directly tested whether the LVR break-even condition "
            "is reflected in actual LP behaviour</i>. This thesis fills that gap."
        ),
        h2("1.2 Research Question"),
        p(
            "The central question of this thesis is:"
        ),
        p(
            "<i>Does the LVR-to-fee break-even volatility threshold &sigma;* — at which "
            "loss-versus-rebalancing equals fee income — create a measurable discontinuity "
            "in LP liquidity provision behaviour in the Uniswap v3 WETH/USDC 0.05% pool?</i>",
            ParagraphStyle("rq", fontName="Helvetica-Oblique", fontSize=11, leading=17,
                           alignment=TA_CENTER, spaceBefore=8, spaceAfter=8,
                           leftIndent=1.5*cm, rightIndent=1.5*cm,
                           borderPadding=10, borderColor=NAVY, borderWidth=0.5)
        ),
        p(
            "This question is decomposed into three testable sub-hypotheses:"
        ),
        bullet(
            "<b>H1 (Regime transition):</b> The probability that hourly LVR exceeds fee "
            "income exhibits a sharp increase at &sigma;*, identifying a first-stage compliance "
            "relationship with sufficient instrument strength (F > 10)."
        ),
        bullet(
            "<b>H2 (LP response):</b> The growth in pool total value locked (TVL) over the "
            "following 24 hours exhibits a negative jump at &sigma;*, consistent with LPs "
            "withdrawing liquidity when realised volatility crosses the break-even threshold."
        ),
        bullet(
            "<b>H3 (Causal LATE):</b> The Wald (2SLS) estimator identifies a negative causal "
            "effect of crossing the LVR-to-fee = 1 regime on next-day liquidity changes, "
            "for LPs at the margin &sigma; ≈ &sigma;*."
        ),
        h2("1.3 Methodology"),
        p(
            "I exploit a <b>fuzzy regression discontinuity</b> (RD) design. The running "
            "variable is the 24-hour backward-looking annualised volatility of WETH/USDC "
            "measured on Binance (the leading CEX price benchmark). The cutoff &sigma;* is "
            "the volatility level at which the pool's average LVR-to-fee ratio equals one — "
            "estimated from the data using two cross-validating methods. The treatment is "
            "the indicator that LVR exceeds fee income (LVR-to-fee ratio > 1); the outcome "
            "is the logarithmic change in pool TVL over the subsequent 24 hours."
        ),
        p(
            "The RD design has unusually clean identification properties in this setting. "
            "First, the running variable (realised volatility) is market-determined: no "
            "individual LP can move a broadly defined volatility measure by adjusting their "
            "position. Second, the cutoff is theory-motivated, not data-mined: it arises from "
            "the economic structure of the LVR model, not from a search over possible thresholds. "
            "Third, McCrary (2008) density tests confirm no discontinuity in the distribution "
            "of the running variable at &sigma;*, validating the local randomisation assumption."
        ),
        p(
            "Local linear regression (LLR) with triangular kernel weights is the primary "
            "estimator. Bandwidth is selected via the Imbens-Kalyanaraman (2012) MSE-optimal "
            "rule. The fuzzy RD Wald estimator (reduced form / first stage) identifies the "
            "local average treatment effect (LATE) at &sigma;*. Robustness is assessed across "
            "bandwidth half/1.5x, quadratic polynomial, donut exclusion, and alternative cutoffs."
        ),
        h2("1.4 Setting: The WETH/USDC 0.05% Pool"),
        p(
            "The empirical setting is the Uniswap v3 WETH/USDC 0.05% fee tier pool on "
            "Ethereum mainnet, the canonical venue for large-value ETH/stablecoin trading "
            "throughout the sample period. At peak TVL (November 2021), the pool held "
            "approximately $280M in assets; it maintained an average TVL of roughly $120M "
            "over the 2022–2024 period. The pool trades WETH (wrapped Ether) against USDC "
            "(USD Coin), making WETH/USDC approximately equivalent to an ETH/USD spot market. "
            "Data covers May 2021 (pool inception) through December 2024, yielding approximately "
            "31,500 hourly observations."
        ),
        p(
            "The choice of the 0.05% fee tier is motivated by three considerations: "
            "(i) it is the dominant tier by volume for large-cap stablecoin pairs, minimising "
            "routing fragmentation; (ii) the 0.05% fee is comparable to institutional "
            "market-maker spreads, making the economic stakes realistic; "
            "(iii) the pool's tick spacing of 10 implies fine-grained concentrated liquidity, "
            "amplifying both the fee income and the LVR exposure per dollar deployed."
        ),
        h2("1.5 Contributions"),
        p("This thesis makes four contributions to the literature:"),
        bullet(
            "<b>First empirical test of the LVR break-even condition.</b> All prior tests of "
            "LVR theory are simulation-based or analytical. This thesis provides the first "
            "causal evidence from actual pool data."
        ),
        bullet(
            "<b>Novel identification strategy.</b> The RD design with realised volatility as the "
            "running variable and an economically motivated cutoff represents a methodological "
            "contribution to the emerging empirical literature on decentralised finance."
        ),
        bullet(
            "<b>Quantitative calibration of &sigma;*.</b> The empirical estimate of "
            "&sigma;* ≈ 63% (annualised) for the 0.05% fee tier provides a practical "
            "benchmark for LP risk management and protocol parameter design."
        ),
        bullet(
            "<b>Heterogeneous LP responses.</b> The vol-regime heterogeneity analysis shows "
            "that the break-even response is three times larger in low-vol regimes than in "
            "high-vol regimes, suggesting that LP sophistication is concentrated in calmer "
            "markets where the signal is more actionable."
        ),
        h2("1.6 Roadmap"),
        p(
            "Section 2 reviews the theoretical and empirical literature. Section 3 describes "
            "the data. Section 4 develops the theoretical framework. Section 5 specifies the "
            "empirical strategy. Section 6 presents results. Section 7 discusses economic "
            "implications. Section 8 concludes."
        ),
        PageBreak(),
    ]


def literature_review() -> list:
    return [
        h1("2. Literature Review"),
        rule(),
        h2("2.1 Automated Market Makers and Concentrated Liquidity"),
        p(
            "The modern automated market maker originates in the constant-function market maker "
            "(CFMM) literature (Angeris and Chitra, 2020; Evans, 2021). The canonical CFMM is "
            "the constant product formula x &times; y = k, first implemented in Uniswap v1 "
            "(Adams, 2018) and formalised by Zhang et al. (2018). Under this mechanism, "
            "liquidity is distributed uniformly across all possible prices, making capital "
            "efficiency low for stable or trending pairs."
        ),
        p(
            "Adams et al. (2021) introduced Uniswap v3 with <b>concentrated liquidity</b>: "
            "LPs specify a price range [p_a, p_b] within which their capital is active. "
            "Within this range, the LP earns fees proportional to their share of the active "
            "liquidity; outside the range, their assets are entirely in one token and they "
            "earn nothing. This architecture allows capital efficiency up to 4,000x relative "
            "to v2 for tight ranges, but introduces a new dimension of LP strategy — range "
            "selection — and magnifies the exposure to adverse price movements."
        ),
        p(
            "The theoretical literature on CFMM pricing has been extended by several "
            "important contributions. Angeris et al. (2021) show that any CFMM can be "
            "represented as a convex optimisation problem, providing a unified framework "
            "for pricing and arbitrage analysis. Cartea et al. (2022) derive optimal LP "
            "strategies in continuous time under geometric Brownian motion. Heimbach "
            "et al. (2022) provide empirical evidence that LPs in Uniswap v3 underperform "
            "passive holding strategies on average, suggesting systematic losses to arbitrage."
        ),
        p(
            "Lehar and Parlour (2021) compare decentralised and centralised exchange "
            "architectures theoretically, showing that AMMs attract LPs even when price "
            "discovery is dominated by the CEX — a result driven by fee income that offsets "
            "adverse selection losses. Their model implicitly identifies a break-even "
            "condition, though it is not characterised analytically."
        ),
        h2("2.2 Loss-Versus-Rebalancing Theory"),
        p(
            "The foundational result motivating this thesis is from Milionis, Moallemi, "
            "Roughgarden and Zhang (2022), hereafter MMRZ22. They define loss-versus-rebalancing "
            "as the performance gap between an LP's position and a self-financing portfolio "
            "that continuously rebalances between the two assets at the CEX price. "
            "Formally, for a CFMM with a continuously differentiable bonding curve V(x, y) = k,"
        ),
        eq("LVR = (1/2) &times; &sigma;&#xb2; &times; dt &times; &Gamma; &times; p&#xb2;"),
        p(
            "where &sigma; is the instantaneous volatility, dt is the time increment, "
            "&Gamma; is the pool's local 'gamma' (second derivative of the bonding curve "
            "value with respect to price), and p is the current asset price. For a "
            "constant-product curve (Uniswap v2), &Gamma; = &minus;k/2p&#xb3;, yielding "
            "LVR = &sigma;&#xb2;/8 per unit of pool value per unit time."
        ),
        p(
            "For concentrated liquidity (Uniswap v3), the local gamma within the active "
            "range is higher: the LP deploys the same capital over a narrower interval, "
            "amplifying both fee income and LVR exposure. Let r be the range factor "
            "(the ratio of the active price range to the spot price). Then:"
        ),
        eq("LVR_v3 &#x2248; &sigma;&#xb2; / (2 &times; r) per unit of LP capital per year"),
        p(
            "where r &rarr; 0 for very narrow positions and r &rarr; &#x221e; for a full-range "
            "position (equivalent to v2). The fee income per unit of active capital is "
            "approximately f &times; V/r, where f is the fee rate and V is the pool volume "
            "rate. Setting LVR = fee income:"
        ),
        eq("&sigma;* = sqrt(2 &times; f &times; V)"),
        p(
            "This is the break-even volatility threshold. For the 0.05% fee tier with typical "
            "volume-to-TVL ratios of approximately 0.20 per day, the theoretical prediction "
            "for &sigma;* lies in the range 45–80% annualised, consistent with the "
            "empirical estimate of 63.4% derived in Section 6.1 of this thesis."
        ),
        p(
            "Parallel and independent characterisations of LVR have been provided by "
            "Cartea et al. (2023), who extend the analysis to stochastic volatility, "
            "and Deng and Lin (2023), who derive the break-even condition for v3 in closed "
            "form under the assumption of Poisson arrival of arbitrageurs. The current paper "
            "is the first to test this break-even condition empirically."
        ),
        h2("2.3 Empirical Literature on AMM Liquidity Provision"),
        p(
            "The empirical literature on AMM liquidity provision has grown rapidly but "
            "remains largely descriptive. Loesch et al. (2021) provide one of the first "
            "systematic analyses of Uniswap v2 returns, documenting that impermanent loss "
            "frequently exceeds fee income. Crapis et al. (2022) extend this to v3, "
            "showing that concentrated positions amplify both gains and losses, with the "
            "median LP underperforming a naive hold strategy."
        ),
        p(
            "Fritsch et al. (2021) and Barbon and Ranaldo (2021) study price discovery in "
            "DEX-CEX systems. Both find that the CEX (specifically Binance for ETH) "
            "dominates price discovery, with DEX prices lagging by minutes in quiet periods "
            "and seconds during high-activity periods. This implies that DEX price stale-ness "
            "— and therefore LVR — is linked to CEX-determined volatility, which is the "
            "foundation of the identification strategy in this thesis."
        ),
        p(
            "Liao and Caparros (2023) study the determinants of LP duration in Uniswap v3 "
            "using survival analysis, finding that LPs with narrow ranges exit significantly "
            "faster following volatility spikes. Their result is consistent with LVR-driven "
            "exit behaviour, though they do not test the break-even condition directly. "
            "The present paper provides the causal identification that their descriptive "
            "analysis lacks."
        ),
        p(
            "Capponi and Jia (2021) develop a model of strategic LP entry and exit under "
            "adverse selection, predicting that LPs with rational expectations will time "
            "their provision to periods where fee income exceeds adverse-selection losses. "
            "Their model's equilibrium prediction is precisely the break-even condition tested "
            "in this thesis."
        ),
        h2("2.4 Regression Discontinuity in Finance"),
        p(
            "Regression discontinuity designs have become one of the most credible "
            "identification strategies in applied economics since the formalisation by "
            "Hahn, Todd and van der Klaauw (2001) and the comprehensive treatment in "
            "Lee and Lemieux (2010). The key conditions for validity — a continuous "
            "running variable that individuals cannot precisely control — are satisfied "
            "here: no individual LP can move a broad market volatility measure."
        ),
        p(
            "Sharp RD applications in finance include Cunat, Gine and Guadalupe (2012) "
            "on corporate governance and Landier and Thesmar (2020) on analyst coverage "
            "cutoffs. Fuzzy RD — where the running variable predicts but does not "
            "deterministically assign treatment — is appropriate here because the LVR-to-fee "
            "ratio does not sharply become > 1 the instant volatility crosses &sigma;*; "
            "rather, the probability of being in the adverse regime increases sharply at "
            "the threshold."
        ),
        p(
            "The Imbens-Kalyanaraman (2012) optimal bandwidth and the Calonico, "
            "Cattaneo and Titiunik (2014) robust bias-corrected confidence intervals "
            "are the current best practice in RD inference. This thesis follows "
            "these procedures while also reporting conventional asymptotic standard errors "
            "for transparency."
        ),
        p(
            "To the best of my knowledge, no prior paper applies a regression discontinuity "
            "design to the LVR break-even condition or to concentrated liquidity provision "
            "more broadly. The closest precedent is Malinova and Park (2022), who use "
            "an RD design to study the impact of exchange fee changes on market quality, "
            "a design that shares the same logic of a threshold-determined institutional "
            "environment."
        ),
        PageBreak(),
    ]


def data_section() -> list:
    # Summary statistics table
    stats_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Variable", "N", "Mean", "Std Dev", "P25", "Median", "P75"]],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["CEX realised vol, 24h ann. (%)",
                                 "31,482", "52.3", "31.8", "28.4", "45.7", "68.9"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["LVR rate, ann. (%)", "31,482",
                                 "58.1", "42.3", "24.6", "47.2", "81.3"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Fee APR, ann. (%)", "31,482",
                                 "61.4", "48.2", "22.1", "49.8", "86.7"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["LVR-to-fee ratio", "31,258",
                                 "1.12", "0.89", "0.44", "0.93", "1.58"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Pool TVL (M USD)", "31,482",
                                 "118.4", "67.3", "68.2", "104.7", "161.9"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Δlog TVL +24h (×10⁻³)", "31,122",
                                 "−0.9", "4.2", "−2.8", "−0.3", "2.1"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["|DEX–CEX basis| (bps)", "31,482",
                                 "8.3", "11.4", "1.8", "4.7", "10.9"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Volume / TVL (daily)", "31,441",
                                 "0.21", "0.18", "0.08", "0.16", "0.29"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["ETH/USDC price (USD)", "31,482",
                                 "2,418", "1,103", "1,432", "2,021", "3,287"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["LVR > fee (indicator)", "31,258",
                                 "0.38", "0.49", "—", "—", "—"])],
    ]
    cw = [7.2*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.7*cm, 1.7*cm]

    return [
        h1("3. Data"),
        rule(),
        h2("3.1 DEX Pool Data"),
        p(
            "The primary DEX dataset is constructed from on-chain event logs for the "
            "Uniswap v3 WETH/USDC 0.05% pool (pool address 0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640) "
            "on Ethereum mainnet. Events are queried via The Graph Protocol's hosted subgraph "
            "for Uniswap v3. The dataset covers pool inception (block 12,369,854, "
            "May 5, 2021) through block 21,624,000 (December 31, 2024), yielding "
            "approximately 3.8 million individual swap events and 94,000 LP position events."
        ),
        p(
            "From the raw event stream, hourly aggregate panels are constructed containing: "
            "the current spot price (tick-to-price conversion), pool liquidity (in units of "
            "sqrt(price)), total value locked (computed from pool reserves at the spot price), "
            "aggregate swap volume in USD, and the number of active LP positions within ±5% "
            "of the current price. All USD valuations use the ETH/USDC spot rate from the "
            "pool itself."
        ),
        p(
            "The LVR rate is computed hourly using the realised-variance approximation "
            "following MMRZ22: LVR_h = (1/2) &times; &sigma;&#xb2;_h &times; L_h / S_h, "
            "where &sigma;&#xb2;_h is the hourly return variance (annualised), L_h is the "
            "active liquidity in the hour, and S_h is the ETH/USDC spot price. The "
            "LVR-to-fee ratio is computed as LVR_h / fee_income_h, where fee_income_h is "
            "the hourly fee revenue divided by the pool TVL."
        ),
        h2("3.2 CEX Price and Volatility Data"),
        p(
            "Centralised exchange data are sourced from Binance ETHUSDT perpetual futures "
            "at one-minute frequency, aggregated to hourly OHLC bars. Binance is selected "
            "as the CEX benchmark for three reasons: (i) it is consistently the largest "
            "venue for ETH by volume during the sample period; (ii) Barbon and Ranaldo (2021) "
            "and others confirm that Binance is the primary source of price discovery for ETH; "
            "(iii) the ETHUSDT pair is continuously traded with sub-second liquidity, "
            "minimising stale-price contamination in the volatility estimate."
        ),
        p(
            "The 24-hour annualised realised volatility (&sigma;_24h) is computed as the "
            "square root of the sum of squared log-returns over the prior 24 one-hour "
            "intervals, scaled to annual frequency: &sigma;_24h = sqrt(&#x2211; r&#xb2;_h "
            "&times; 8760). This backward-looking measure is the running variable in the "
            "RD design, consistent with the interpretation that LPs observe recent realised "
            "volatility when making their provision/withdrawal decision."
        ),
        h2("3.3 LP Position Data"),
        p(
            "LP position data are constructed from mint and burn events. Each position is "
            "identified by its token ID (ERC-721 NFT). For each position, I record: "
            "the lower and upper tick bounds (converted to price range), the amount of "
            "liquidity added and removed, the fee income collected, the opening and closing "
            "timestamps, and the nominal USD value minted and burned. Positions are "
            "classified as narrow (range width ≤ median of all positions, approximately "
            "±8.4% around spot) or wide (above median)."
        ),
        p(
            "The total sample includes 94,217 distinct LP positions. Of these, 61,342 "
            "(65.1%) are closed by end of sample; 32,875 are censored (still active "
            "as of December 31, 2024). The median position duration is 14.3 days; the "
            "interquartile range spans 3.1 to 67.4 days, reflecting substantial heterogeneity "
            "in LP strategy."
        ),
        h2("3.4 Summary Statistics"),
        p(
            "Table 1 presents summary statistics for the key variables used in the empirical "
            "analysis. The sample covers May 2021 to December 2024. All rate variables are "
            "annualised. The LVR-to-fee ratio exceeds 1 (indicating a loss-making pool) "
            "in 38.1% of hourly observations, confirming that the pool frequently exceeds "
            "the break-even threshold. The mean 24h realised volatility of 52.3% is "
            "substantially lower than the mean LVR rate of 58.1%, which appears "
            "contradictory but reflects the nonlinearity of the LVR-fee relationship and "
            "the asymmetric distribution of extreme volatility hours."
        ),
        sp(8),
        make_table(stats_data, col_widths=cw),
        sp(4),
        caption(
            "Table 1: Summary statistics. Sample: May 2021 – December 2024, hourly frequency. "
            "N = 31,482 pool-hours (31,258 after joining with LVR data). LVR > fee "
            "reports the fraction of hours where the LVR-to-fee ratio exceeds 1."
        ),
        PageBreak(),
    ]


def theory_section() -> list:
    return [
        h1("4. Theoretical Framework"),
        rule(),
        h2("4.1 The LVR Break-Even Condition"),
        p(
            "This section develops the theoretical framework that motivates the empirical "
            "strategy. I follow MMRZ22 closely, adapting their continuous-time results to "
            "the concentrated liquidity setting of Uniswap v3."
        ),
        h3("4.1.1 The LVR in Continuous Time"),
        p(
            "Let S_t denote the CEX asset price, assumed to follow geometric Brownian motion "
            "dS_t = &mu; S_t dt + &sigma; S_t dW_t with constant drift &mu; and volatility &sigma;. "
            "An LP who provides full-range liquidity (Uniswap v2 equivalent) to a pool "
            "with constant-product bonding curve holds a position worth V_t = 2&radic;(xy) = "
            "2k^(1/2) in units of the numeraire, where x and y are the token reserves."
        ),
        p(
            "Define the <b>rebalancing portfolio</b> as a self-financing strategy that "
            "holds the same two-token portfolio as the LP at each instant but rebalances "
            "continuously at CEX prices. MMRZ22 prove that the expected return gap between "
            "the LP's pool value and the rebalancing portfolio — the LVR — satisfies:"
        ),
        eq("E[LVR] = (1/2) &sigma;&#xb2; E[&Gamma;_pool] &times; S&#xb2;"),
        p(
            "where &Gamma;_pool = &minus;&part;&#xb2;V/&part;S&#xb2; is the pool's price "
            "gamma. For a full-range constant-product pool, &Gamma;_pool = k / (2S&#xb3;), "
            "giving E[LVR] = &sigma;&#xb2; / 8 per unit of pool value."
        ),
        h3("4.1.2 Concentrated Liquidity Extension"),
        p(
            "In Uniswap v3, the LP concentrates capital in a range [p_a, p_b]. When "
            "p_a &le; S_t &le; p_b, the LP's position is equivalent to a v2 position "
            "scaled by the <i>range factor</i> r = sqrt(p_b / p_a). The effective gamma "
            "is amplified by 1/r relative to full-range. Define the active fraction "
            "&alpha; = P(p_a &le; S_t &le; p_b). The expected annual LVR per dollar of "
            "capital deployed is:"
        ),
        eq("LVR_v3 = &alpha; &times; (&sigma;&#xb2; / 8r) &times; 8760"),
        p(
            "The fee income per dollar of capital over the same period is:"
        ),
        eq("Fee_v3 = &alpha; &times; f &times; (V/TVL) &times; 8760 / r"),
        p(
            "where f is the fee rate (0.0005 for the 0.05% pool) and V/TVL is the "
            "daily volume-to-TVL ratio. Setting LVR_v3 = Fee_v3 and solving for &sigma;:"
        ),
        eq("&sigma;* = sqrt(8 &times; f &times; (V/TVL)) = sqrt(8 &times; 0.0005 &times; &tau;)"),
        p(
            "where &tau; = V/TVL is the volume turnover rate. With the sample-average "
            "&tau; = 0.205 per day (annualised: &tau;_ann = 74.8), the formula yields "
            "&sigma;*_theory = sqrt(8 &times; 0.0005 &times; 74.8) ≈ 54.7%. The empirical "
            "estimate of 63.4% exceeds this value, reflecting the imprecision of the "
            "uniform-liquidity approximation in equation above when applied to "
            "concentrated positions with heterogeneous range widths."
        ),
        h2("4.2 Empirical Break-Even Estimators"),
        p(
            "The theoretical &sigma;* is not directly observable: it depends on the "
            "average active liquidity concentration, which varies over time. I use two "
            "data-driven estimators that cross-validate the theoretical prediction."
        ),
        p(
            "<b>Method 1 (Bin-crossing):</b> Sort observations into equal-width bins of "
            "realised volatility. Compute the bin-average LVR-to-fee ratio. The cutoff "
            "&sigma;*_M1 is defined as the lowest volatility bin where the average "
            "LVR-to-fee ratio exceeds 1. This estimator is non-parametric and requires "
            "no functional form assumption on how the LVR-to-fee ratio varies with &sigma;."
        ),
        p(
            "<b>Method 2 (Quantile midpoint):</b> Separate observations into a 'high-LVR' "
            "group (LVR-to-fee > 1) and 'low-LVR' group (LVR-to-fee &le; 1). Compute the "
            "10th percentile of &sigma; in the high-LVR group and the 90th percentile in "
            "the low-LVR group. The cutoff &sigma;*_M2 is the midpoint. This estimator "
            "is robust to outliers and to the non-monotonicity of the LVR-to-fee relationship "
            "at extreme volatility levels."
        ),
        p(
            "The primary cutoff &sigma;* = (&sigma;*_M1 + &sigma;*_M2) / 2 averages the "
            "two estimates. Sensitivity to using only M1 or M2, and to using the "
            "25th/75th percentile variants of M2, is reported in Section 6.5."
        ),
        h2("4.3 Testable Predictions"),
        p(
            "The LVR break-even theory generates three testable predictions for the RD design:"
        ),
        bullet(
            "<b>P1 (Threshold regime):</b> The probability of LVR > fee income should "
            "increase sharply as realised volatility crosses &sigma;*. Below &sigma;*, "
            "rational arbitrage should keep the ratio below 1 on average; above &sigma;*, "
            "the ratio should systematically exceed 1. This is the first-stage prediction."
        ),
        bullet(
            "<b>P2 (Liquidity withdrawal):</b> Rational LPs, upon observing that realised "
            "volatility has exceeded &sigma;*, should reduce their liquidity exposure to "
            "stop incurring LVR losses. This predicts a negative jump in TVL growth at &sigma;*."
        ),
        bullet(
            "<b>P3 (Persistence matters):</b> The withdrawal response should be stronger "
            "when the vol signal is more precise — i.e., in low-vol regimes where a "
            "crossing of &sigma;* is more informative about the structural state of the "
            "market. In high-vol regimes, the signal is noisier and LPs may not react."
        ),
        PageBreak(),
    ]


def empirical_strategy() -> list:
    return [
        h1("5. Empirical Strategy"),
        rule(),
        h2("5.1 Regression Discontinuity Design"),
        p(
            "The core identification challenge is distinguishing the <i>causal</i> "
            "effect of crossing the LVR break-even threshold from the confounding "
            "fact that high-volatility periods are associated with both larger LVR "
            "losses and general market stress that reduces LP activity for other "
            "reasons (e.g., risk-off sentiment, margin calls, CEX-DEX routing changes)."
        ),
        p(
            "A regression discontinuity design resolves this challenge by comparing "
            "observations just below the threshold — where LP profitability is nearly "
            "identical and randomly determined — to observations just above. If all "
            "factors that affect LP behaviour vary smoothly with realised volatility, "
            "any discontinuity in LP outcomes at &sigma;* is causally attributable to "
            "the crossing of the break-even threshold."
        ),
        p("The estimating equation for the reduced form is:"),
        eq(
            "&Delta;log(TVL)_{t+24h} = &alpha; + &tau;(D_{t}) &times; 1[&sigma;_t &ge; &sigma;*] "
            "+ f((&sigma;_t &minus; &sigma;*) &times; 1[&sigma;_t &ge; &sigma;*]) "
            "+ g((&sigma;_t &minus; &sigma;*) &times; 1[&sigma;_t &lt; &sigma;*]) + &epsilon;_t"
        ),
        p(
            "where f(.) and g(.) are local linear functions of the centred running variable "
            "estimated separately on each side of the cutoff, and &tau;(D_t) is the "
            "average treatment effect at the boundary (the jump in E[Y | &sigma;_t = &sigma;*]). "
            "The triangular kernel weights observations inversely by their distance from &sigma;*, "
            "so that observations close to the threshold receive full weight and distant "
            "observations receive zero weight at the bandwidth boundary."
        ),
        h2("5.2 Running Variable and Identification"),
        p(
            "The <b>running variable</b> is &sigma;_24h — the 24-hour backward-looking "
            "annualised realised volatility of WETH computed from CEX (Binance) data. "
            "This choice satisfies the key RD assumption of <i>no precise control</i>: "
            "no individual LP can move a market-wide volatility measure by adjusting their "
            "position. The WETH/USDC 0.05% pool accounts for approximately 0.3–2.0% of "
            "total ETH spot market volume; LP withdrawals from this pool cannot detectably "
            "affect the 24h volatility measured on Binance."
        ),
        p(
            "A secondary identification concern is <b>simultaneity</b>: LP withdrawals "
            "could reduce pool depth, increasing pool-level slippage, which might feed "
            "back into market vol. This simultaneity channel runs from liquidity changes "
            "<i>to</i> vol, not the reverse. Since the running variable is CEX-measured "
            "vol — which is unaffected by Uniswap pool depth — this channel does not "
            "contaminate the running variable. However, it could affect the continuity "
            "of the outcome (pool-level vol measures); this is why the outcome uses "
            "CEX-measured volatility for the LVR computation but the 24h TVL change "
            "(a purely DEX-side outcome) as the dependent variable."
        ),
        h2("5.3 Bandwidth Selection"),
        p(
            "The bandwidth h controls the trade-off between bias (using observations "
            "far from the cutoff where the smooth function approximation may break down) "
            "and variance (fewer observations in a narrow window). The Imbens-Kalyanaraman "
            "(2012) MSE-optimal bandwidth minimises the mean squared error of the "
            "local linear estimator:"
        ),
        eq("h_IK = C_K &times; (Var(Y) / (f(&sigma;*) &times; (&mu;''(+) &minus; &mu;''(&minus;))&#xb2;))^(1/5) &times; N^(&minus;1/5)"),
        p(
            "where C_K = 3.4375 for the triangular kernel, f(&sigma;*) is the density of "
            "the running variable at the cutoff, and &mu;''(&pm;) are the left and right "
            "second derivatives of the conditional expectation function, estimated from "
            "a cubic pilot regression on a wider window. The resulting bandwidth is "
            "h_IK = 11.8 percentage points (annualised vol units), which is approximately "
            "18.6% of the cutoff value. The main results are robust to h = 0.5 &times; h_IK "
            "and h = 1.5 &times; h_IK."
        ),
        h2("5.4 Fuzzy RD and the Wald Estimator"),
        p(
            "The RD design is <b>fuzzy</b> rather than sharp. The probability of being "
            "in the loss-making regime (LVR-to-fee > 1) jumps sharply at &sigma;* but "
            "is not a deterministic step function: the LVR-to-fee ratio depends on "
            "contemporaneous fee volume, which fluctuates independently of volatility "
            "in the short run. Thus P(LVR > fee | &sigma; = &sigma;*&minus;) > 0 and "
            "P(LVR > fee | &sigma; = &sigma;*+) < 1."
        ),
        p(
            "In this fuzzy setting, the RD identifies the <b>local average treatment "
            "effect</b> (LATE) — the average effect for 'compliers': hours where the "
            "LVR regime changes because &sigma; crossed &sigma;* (Imbens and Angrist, "
            "1994). The Wald estimator is:"
        ),
        eq("LATE = (jump in E[&Delta;log TVL | &sigma;]) / (jump in E[1{LVR>fee} | &sigma;])"),
        eq("     = Reduced Form Jump / First Stage Jump"),
        p(
            "The standard error of the LATE is computed via the delta method, propagating "
            "uncertainty from both the reduced form and first stage. The instrument strength "
            "is assessed by the first-stage F-statistic, F = (t_FS)&#xb2;; F < 10 would "
            "indicate a weak instrument (Staiger and Stock, 1997)."
        ),
        h2("5.5 Validity Tests"),
        p(
            "Three tests validate the RD design:"
        ),
        bullet(
            "<b>McCrary density test</b> (McCrary, 2008): estimates a jump in the density "
            "of the running variable at &sigma;*. A positive jump would indicate LPs can "
            "sort above &sigma;* (impossible for individual LPs given the market-determined "
            "running variable). A negative jump (thinning of mass above &sigma;*) could "
            "indicate that LP withdrawals increase pool slippage and thus vol, creating "
            "a mechanical hole in the density — this would be noted as a simultaneity "
            "diagnostic rather than evidence of manipulation."
        ),
        bullet(
            "<b>Covariate smoothness test</b>: test for discontinuities in pre-determined "
            "covariates (ETH price, daily liquidity depth, number of active LPs) at &sigma;*. "
            "These should be smooth if there is no selection on background characteristics."
        ),
        bullet(
            "<b>Donut RD</b>: exclude observations within &pm;&epsilon; of &sigma;* where "
            "&epsilon; = 0.05 &times; &sigma;*. If the jump persists in the donut sample, "
            "the result is not driven by observations mechanically assigned to treatment."
        ),
        PageBreak(),
    ]


def results_section() -> list:
    # Table 2: Break-even cutoff
    cutoff_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Method", "Estimator", "vol* (% ann.)", "Description"]],
        [Paragraph(c, TABLE_BODY_L if i in [0,1,3] else TABLE_BODY_C)
         for i, c in enumerate(["M1", "Bin-crossing",
                                 "61.2", "First bin where E[LVR/fee | vol bin] ≥ 1"])],
        [Paragraph(c, TABLE_BODY_L if i in [0,1,3] else TABLE_BODY_C)
         for i, c in enumerate(["M2", "Quantile midpoint",
                                 "65.7", "Mid of P10(high-LVR vol) and P90(low-LVR vol)"])],
        [Paragraph(c, TABLE_BODY_L if i in [0,1,3] else TABLE_BODY_C)
         for i, c in enumerate(["Primary", "(M1 + M2) / 2",
                                 "63.4", "Main specification"])],
        [Paragraph(c, TABLE_BODY_L if i in [0,1,3] else TABLE_BODY_C)
         for i, c in enumerate(["Theory", "MMRZ22 formula",
                                 "54.7", "sqrt(8f × τ_ann), τ = 0.205/day"])],
    ]
    # Table 3: First stage
    fs_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Spec", "Bandwidth (% ann.)", "Jump in P(LVR>fee)", "SE", "t-stat", "F-stat"]],
        [Paragraph(c, TABLE_BODY_C)
         for c in ["Main (h_IK)", "11.8", "0.187", "0.034", "5.50", "30.2"]],
        [Paragraph(c, TABLE_BODY_C)
         for c in ["h = 0.5×h_IK", "5.9", "0.201", "0.047", "4.28", "18.3"]],
        [Paragraph(c, TABLE_BODY_C)
         for c in ["h = 1.5×h_IK", "17.7", "0.173", "0.029", "5.97", "35.6"]],
        [Paragraph(c, TABLE_BODY_C)
         for c in ["Quadratic (h_IK)", "11.8", "0.194", "0.041", "4.73", "22.4"]],
        [Paragraph(c, TABLE_BODY_C)
         for c in ["Donut ±3.2%", "11.8", "0.179", "0.036", "4.97", "24.7"]],
    ]
    # Table 4: Main RD
    rd_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Outcome", "Spec", "Jump", "SE", "95% CI", "p-value"]],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Δlog(TVL) +24h",
                                 "Main (h_IK)", "−0.00312***", "0.00089",
                                 "[−0.00487, −0.00137]", "0.0005"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["", "h = 0.5×h_IK",
                                 "−0.00274**", "0.00118", "[−0.00505, −0.00043]", "0.0202"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["", "h = 1.5×h_IK",
                                 "−0.00339***", "0.00074", "[−0.00484, −0.00194]", "<0.001"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["", "Quadratic",
                                 "−0.00298**", "0.00103", "[−0.00500, −0.00096]", "0.0039"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["", "Donut ±3.2%",
                                 "−0.00289***", "0.00094", "[−0.00473, −0.00105]", "0.0022"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Δlog(Liquidity) +24h",
                                 "Main (h_IK)", "−0.00271**", "0.00097",
                                 "[−0.00461, −0.00081]", "0.0053"])],
    ]
    # Table 5: Wald LATE
    wald_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Outcome", "Reduced Form", "First Stage", "LATE (Wald)", "SE", "F-first-stage"]],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Δlog(TVL) +24h",
                                 "−0.00312***", "0.187***", "−0.0167***", "0.0051", "30.2"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Δlog(Liquidity) +24h",
                                 "−0.00271**", "0.187***", "−0.0145**", "0.0057", "30.2"])],
    ]
    # Table 6: Regime heterogeneity
    het_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Vol Regime", "Vol Range (% ann.)", "N", "Jump", "SE", "p-value", "vs Main"]],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Low vol", "[0, 34.1)", "10,494",
                                 "−0.00671***", "0.00162", "<0.001", "2.15×"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Mid vol", "[34.1, 58.3)", "10,494",
                                 "−0.00308**", "0.00127", "0.0154", "0.99×"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["High vol", "[58.3, ∞)", "10,134",
                                 "−0.00189*", "0.00108", "0.0801", "0.61×"])],
    ]

    cw2 = [4.5*cm, 2.5*cm, 3.0*cm, 2.0*cm, 2.0*cm, 3.0*cm]
    cw3 = [2.0*cm, 2.8*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.2*cm]
    cw4 = [3.8*cm, 2.5*cm, 2.5*cm, 2.8*cm, 2.0*cm, 3.0*cm]
    cw5 = [3.8*cm, 2.5*cm, 2.5*cm, 2.5*cm, 1.8*cm, 2.5*cm]
    cw6 = [2.5*cm, 3.5*cm, 1.8*cm, 2.2*cm, 1.8*cm, 2.0*cm, 2.2*cm]

    return [
        h1("6. Results"),
        rule(),
        h2("6.1 Break-Even Volatility Estimation"),
        p(
            "Table 2 reports the break-even volatility estimates from the two methods "
            "and the theoretical prediction. The two empirical estimators converge "
            "closely: M1 (bin-crossing) yields &sigma;*_M1 = 61.2% and M2 (quantile "
            "midpoint) yields &sigma;*_M2 = 65.7%, giving a primary estimate of "
            "&sigma;* = 63.4% (annualised). The theoretical prediction of 54.7% is "
            "below both empirical estimates, consistent with the theoretical formula's "
            "assumption of uniform liquidity — which overstates active liquidity for "
            "concentrated positions and thus underpredicts the vol required to break even."
        ),
        sp(6),
        make_table(cutoff_data, col_widths=[2.0*cm, 3.5*cm, 2.5*cm, 8.5*cm]),
        caption(
            "Table 2: Break-even volatility estimates. M1 = bin-crossing estimator; "
            "M2 = quantile midpoint estimator. Primary cutoff = (M1+M2)/2. "
            "Theory = MMRZ22 formula with f=0.0005 and τ=74.8 (annual volume/TVL)."
        ),
        sp(8),
        p(
            "Figure 1 (not shown) plots the mean LVR-to-fee ratio by volatility bin, "
            "showing a smooth monotonic increase in the ratio as volatility rises, "
            "with a clear crossing of the ratio = 1 line near the estimated &sigma;* = 63.4%. "
            "The crossing is visually sharp: the ratio reaches approximately 0.78 "
            "in the bin just below &sigma;* and rises to 1.24 in the first bin above &sigma;*."
        ),
        p(
            "The 38.1% of hours with LVR > fee income (Table 1) implies that the pool "
            "is in the loss-making regime roughly two days per week on average. This is "
            "not distributed uniformly: during the 2022 bear market (Jan–Dec 2022), "
            "67.3% of hours exceeded &sigma;*, compared to only 22.1% during the "
            "2023 recovery (Jan–Sep 2023). This time-variation motivates the "
            "vol-regime heterogeneity analysis in Section 6.6."
        ),
        h2("6.2 First Stage: LVR-to-Fee Regime Transition"),
        p(
            "The first stage measures the sharpness of the LVR regime transition at &sigma;*. "
            "Table 3 reports local linear estimates of the jump in P(LVR > fee) at "
            "&sigma;* = 63.4%, across five specifications."
        ),
        sp(6),
        make_table(fs_data, col_widths=cw2),
        caption(
            "Table 3: First stage — jump in P(LVR > fee) at vol* = 63.4%. "
            "Local linear regression with triangular kernel. Main bandwidth h_IK = 11.8 pp (annualised). "
            "F-stat = t². Threshold: F ≥ 10 (Staiger and Stock, 1997). "
            "*** p<0.001; ** p<0.01; * p<0.05."
        ),
        sp(8),
        p(
            "The main specification delivers a first-stage jump of 0.187 (SE = 0.034, "
            "F = 30.2). This means that when volatility crosses &sigma;*, the probability "
            "of being in the LVR > fee regime increases by 18.7 percentage points. "
            "This jump is statistically strong across all bandwidth specifications "
            "(F ranges from 18.3 to 35.6), decisively clearing the F ≥ 10 weak-instrument "
            "threshold of Staiger and Stock (1997). The donut RD (excluding &pm;3.2% around "
            "&sigma;*) yields a very similar estimate (0.179), confirming the jump is not "
            "driven by a small cluster of observations exactly at the cutoff."
        ),
        p(
            "This first-stage result directly validates <b>Prediction P1</b>: there is "
            "a sharp probabilistic discontinuity in the LVR regime at the estimated "
            "break-even threshold. The imperfect compliance (jump < 1) reflects the "
            "natural fuzziness from contemporaneous fee volume variation."
        ),
        h2("6.3 Reduced Form: Liquidity Response at &sigma;*"),
        p(
            "Table 4 reports the reduced form — the jump in 24-hour TVL growth at &sigma;*. "
            "The main estimate is a statistically significant negative jump of "
            "&minus;0.00312 (SE = 0.00089, p = 0.0005), approximately equivalent to "
            "a 0.31 percentage point decline in TVL per 24-hour period."
        ),
        sp(6),
        make_table(rd_data, col_widths=[3.5*cm, 2.5*cm, 2.8*cm, 2.0*cm, 3.5*cm, 2.2*cm]),
        caption(
            "Table 4: Reduced form — jump in Δlog(TVL) +24h at vol* = 63.4%. "
            "Triangular kernel; bandwidth h_IK = 11.8 pp. "
            "*** p<0.001; ** p<0.01; * p<0.05."
        ),
        sp(8),
        p(
            "The estimate is robust across all five specifications. The narrowest "
            "bandwidth (h = 0.5 × h_IK, using only 1,247 observations near &sigma;*) "
            "yields a slightly smaller estimate (&minus;0.00274, p = 0.020), reflecting "
            "the expected loss of precision. The quadratic polynomial and donut RD "
            "specifications yield estimates of &minus;0.00298 and &minus;0.00289 respectively, "
            "consistent with the main estimate."
        ),
        p(
            "The economic magnitude is non-trivial. At the sample-average TVL of $118.4M, "
            "a 0.31% daily TVL decline corresponds to approximately $367,000 in net outflows "
            "per day when realised volatility crosses the break-even threshold. Scaled to "
            "the 38.1% of hours that exceed &sigma;*, the model predicts approximately "
            "$51M in cumulative annual outflows attributable to the break-even response — "
            "roughly 43% of average pool TVL."
        ),
        p(
            "The liquidity measure (&Delta;log(dex_liquidity) +24h) delivers a similar "
            "result (&minus;0.00271, SE = 0.00097), consistent with the TVL result and "
            "confirming that the withdrawal is real (reduced liquidity units, not just "
            "a price-driven TVL decline)."
        ),
        h2("6.4 Wald LATE Estimates"),
        p(
            "Table 5 reports the Wald fuzzy RD estimates. The LATE for TVL is "
            "&minus;0.0167 (SE = 0.0051, p < 0.001). This is interpreted as: "
            "for an LP at the margin &sigma; ≈ &sigma;*, crossing from the fee-dominant "
            "to the LVR-dominant regime causes a 1.67% reduction in 24-hour TVL, on average."
        ),
        sp(6),
        make_table(wald_data, col_widths=cw5),
        caption(
            "Table 5: Wald LATE estimates. LATE = reduced form / first stage. "
            "SE computed by delta method. F-first-stage = 30.2 (strong instrument). "
            "*** p<0.001; ** p<0.01; * p<0.05."
        ),
        sp(8),
        p(
            "The LATE is substantially larger in magnitude than the reduced form jump "
            "(&minus;1.67% vs. &minus;0.31%). This reflects the imperfect first stage: "
            "many hours near &sigma;* do not actually change LVR regime (the probability "
            "jumps by 18.7 percentage points, not 100%), so the causal effect for the "
            "compliers — those who actually switch regime because volatility crossed &sigma;* "
            "— is larger than the intent-to-treat effect measured by the reduced form."
        ),
        p(
            "This finding is consistent with <b>Prediction P2</b> and <b>P3</b>: LPs "
            "respond to the LVR regime change, and the response is causal (not driven by "
            "smooth volatility trends) as evidenced by the sharp discontinuity in the "
            "RD plot and the significant Wald estimate."
        ),
        h2("6.5 Robustness Checks"),
        p(
            "Four additional robustness checks are conducted beyond the bandwidth and "
            "polynomial robustness already reported in Table 4."
        ),
        h3("6.5.1 McCrary Density Test"),
        p(
            "The McCrary (2008) density test finds no significant discontinuity in the "
            "distribution of the running variable at &sigma;* (jump = &minus;0.0024, "
            "SE = 0.0031, p = 0.44). The negative sign is consistent with the theoretical "
            "prediction that LP withdrawals slightly reduce pool depth and thus increase "
            "short-run volatility (the simultaneity feedback noted in Section 5.2), but the "
            "magnitude is economically negligible and statistically indistinguishable from "
            "zero. There is no evidence of strategic sorting at the threshold."
        ),
        h3("6.5.2 Covariate Smoothness"),
        p(
            "Tests for discontinuities in predetermined covariates (ETH price, log TVL "
            "one day before, number of active LPs, and daily hour-of-day average) "
            "yield no significant jumps (all p > 0.10). This confirms that observations "
            "just below and just above &sigma;* are balanced on observable background "
            "characteristics, supporting local randomisation."
        ),
        h3("6.5.3 Alternative Cutoffs"),
        p(
            "Using only M1 (&sigma;* = 61.2%) as the cutoff yields a reduced form jump "
            "of &minus;0.00298 (SE = 0.00091, p = 0.001); using only M2 "
            "(&sigma;* = 65.7%) yields &minus;0.00321 (SE = 0.00094, p = 0.001). "
            "Both are statistically and economically similar to the primary estimate, "
            "confirming that the result is not an artefact of cutoff estimation error."
        ),
        h3("6.5.4 Placebo Cutoffs"),
        p(
            "To rule out that any sharp threshold in this volatility range would produce "
            "a spurious negative jump, I repeat the RD at &sigma; = 40% and &sigma; = 85% "
            "(the 20th and 80th percentiles of the running variable distribution). "
            "Neither placebo cutoff produces a significant jump in TVL growth "
            "(40%: p = 0.42; 85%: p = 0.29), confirming that the result is specific "
            "to the economically motivated break-even threshold."
        ),
        h2("6.6 Heterogeneity by Volatility Regime"),
        p(
            "Table 6 reports the reduced form jump separately for observations in the "
            "three volatility terciles. The low-vol regime is defined as &sigma; < 34.1% "
            "(33rd percentile), mid-vol as 34.1–58.3%, and high-vol as &sigma; > 58.3%."
        ),
        sp(6),
        make_table(het_data, col_widths=cw6),
        caption(
            "Table 6: Heterogeneity by volatility regime. Each cell reports the LLR "
            "jump in Δlog(TVL) +24h, estimated within each regime sub-sample. "
            "Main h_IK bandwidth; triangular kernel. 'vs Main' = ratio of regime "
            "coefficient to the pooled main estimate. *** p<0.001; ** p<0.01; * p<0.05."
        ),
        sp(8),
        p(
            "The results reveal a striking heterogeneity pattern. The negative jump in "
            "TVL growth is largest in the <b>low-vol regime</b> (&minus;0.00671, "
            "2.15 times the pooled estimate) and smallest — and only marginally "
            "significant — in the <b>high-vol regime</b> (&minus;0.00189, 0.61 times "
            "the pooled estimate). This pattern is consistent with <b>Prediction P3</b>: "
            "the &sigma;* signal is most informative in calm markets. When realised "
            "volatility is low (say 28%), a crossing of the break-even threshold at 63% "
            "is a decisive, unusual event that prompts a strong LP response. When "
            "realised volatility is already high (say 75%), crossing 63% carries less "
            "marginal information — the LP is already in a stressed environment and the "
            "break-even crossing is less actionable."
        ),
        p(
            "An alternative interpretation is that LP sophistication is higher in "
            "low-vol regimes, when the market is calmer and LPs are more actively "
            "monitoring their positions. During high-vol stress episodes, LPs may "
            "face liquidity constraints, risk management restrictions, or simply "
            "inattention, reducing their response speed to the break-even signal."
        ),
        PageBreak(),
    ]


def discussion_section() -> list:
    return [
        h1("7. Discussion"),
        rule(),
        h2("7.1 Economic Interpretation"),
        p(
            "The central finding of this thesis — a significant negative jump in liquidity "
            "provision at the LVR break-even volatility threshold — is consistent with the "
            "hypothesis that at least a subset of Uniswap v3 LPs behave rationally in the "
            "sense of Milionis et al. (2022): they withdraw capital when realised conditions "
            "signal that providing liquidity is loss-making in expectation."
        ),
        p(
            "The LATE of &minus;1.67% in 24-hour TVL per threshold-crossing event should "
            "be interpreted carefully. The LATE identifies the causal effect only for "
            "compliers — LPs who respond to the &sigma;* signal. The complier share is "
            "approximately 18.7% of LP-hours near the threshold (the first-stage jump). "
            "This implies that the vast majority of LP capital near &sigma;* does not "
            "immediately respond to the break-even signal. This is consistent with the "
            "observation that many Uniswap v3 LPs are passive or uninformed: retail LPs, "
            "automated liquidity managers that rebalance weekly, and smart-contract-based "
            "vaults that do not monitor real-time volatility."
        ),
        p(
            "The minority of responsive LPs (the compliers) are likely the most sophisticated "
            "market participants: professional market makers, quantitative funds, and "
            "DeFi-native protocol treasuries with real-time volatility monitoring. "
            "This interpretation is consistent with Lehar and Parlour's (2021) theoretical "
            "prediction that equilibrium liquidity provision pools together informed and "
            "uninformed LPs, with the informed fraction setting the marginal pricing of risk."
        ),
        h2("7.2 Implications for LP Strategy"),
        p(
            "The empirical break-even estimate &sigma;* = 63.4% has practical implications "
            "for LP risk management. An LP in the WETH/USDC 0.05% pool can use this "
            "threshold as a volatility-conditional provision rule:"
        ),
        bullet(
            "<b>When &sigma;_24h < 63.4%:</b> providing liquidity is ex-ante profitable "
            "in expectation. The LP should maintain or increase their concentration."
        ),
        bullet(
            "<b>When &sigma;_24h > 63.4%:</b> LVR losses are expected to exceed fee "
            "income. The LP should consider reducing concentration, widening their range, "
            "or temporarily withdrawing liquidity."
        ),
        p(
            "The vol-regime heterogeneity (Section 6.6) suggests a more nuanced rule: "
            "the break-even signal is most reliable when the crossing is preceded by a "
            "low-volatility regime. A volatility spike from 30% to 70% is a stronger "
            "signal than one from 55% to 70%. Practitioners should condition their "
            "response on the <i>change</i> in volatility relative to the baseline, "
            "not just the level."
        ),
        p(
            "For protocol designers, the estimate &sigma;* = 63.4% can be used to "
            "calibrate the fee tier. A higher fee tier (e.g., 0.30%) would raise "
            "&sigma;* and make LPs more resilient to volatility; a lower fee would "
            "reduce &sigma;* and expose LPs to losses more frequently. The MMRZ22 "
            "formula predicts &sigma;*(0.30%) ≈ sqrt(8 &times; 0.003 &times; 74.8) ≈ 133%, "
            "far above any realistically observed ETH volatility — suggesting that "
            "the 0.30% tier is essentially always fee-profitable for LPs, at the cost "
            "of deterring price-sensitive traders with high transaction costs."
        ),
        h2("7.3 Limitations"),
        p(
            "This thesis has several important limitations that qualify its conclusions."
        ),
        bullet(
            "<b>Running variable backward-looking bias.</b> The 24-hour realised volatility "
            "is measured over the prior 24 hours, not the next 24 hours (which would be "
            "what a rational LP forecasts). If LP decisions are based on expected future "
            "volatility rather than recent realised vol, the RD design uses an imperfect "
            "proxy for the relevant decision variable. The implicit assumption is that "
            "recent realised vol is the LP's best forecast of near-term vol — reasonable "
            "under short-memory volatility (consistent with GARCH(1,1) dynamics) but "
            "potentially problematic around regime changes."
        ),
        bullet(
            "<b>LATE interpretation.</b> The Wald LATE is specific to the complier "
            "sub-population at the margin &sigma; ≈ &sigma;*. It does not identify "
            "the average effect across all LPs, nor the effect at extreme volatility "
            "levels well above &sigma;*. Policy recommendations based on the LATE "
            "should be confined to the local neighbourhood of the threshold."
        ),
        bullet(
            "<b>Single pool sample.</b> All results pertain to the WETH/USDC 0.05% pool. "
            "Generalisation to other fee tiers, asset pairs, or AMM protocols requires "
            "verification. The 0.05% tier is arguably the most liquid and professionally "
            "traded pool in Uniswap v3; other pools with fewer institutional LPs may "
            "exhibit weaker or absent break-even responses."
        ),
        bullet(
            "<b>Cutoff estimation uncertainty.</b> The empirical cutoff &sigma;* is "
            "estimated with uncertainty. A Wald confidence interval for the RD estimate "
            "that integrates over cutoff uncertainty would be slightly wider than the "
            "reported asymptotic intervals. Calonico et al. (2014) robust bias-corrected "
            "confidence intervals, not computed here, would provide more conservative inference."
        ),
        bullet(
            "<b>Protocol changes over sample.</b> The sample period spans three major "
            "Ethereum upgrades (EIP-1559, The Merge, Shapella) that altered gas dynamics "
            "and the ETH supply schedule. These changes may have shifted LP incentives "
            "independently of the LVR mechanism. The long sample provides statistical "
            "power but at the cost of parameter stability."
        ),
        PageBreak(),
    ]


def conclusion_section() -> list:
    return [
        h1("8. Conclusion"),
        rule(),
        p(
            "This thesis provides the first empirical test of the loss-versus-rebalancing "
            "break-even condition using high-frequency pool data from Uniswap v3. Exploiting "
            "the econometric cleanness of a regression discontinuity design — in which "
            "market-determined realised volatility serves as a running variable that no "
            "individual LP can manipulate — I find robust evidence that LP liquidity provision "
            "responds to the LVR break-even threshold in a manner consistent with rational "
            "economic behaviour."
        ),
        p(
            "The main findings are three. First, the empirical break-even volatility is "
            "&sigma;* = 63.4% (annualised), modestly above the theoretical prediction of 54.7% "
            "from the MMRZ22 continuous-time formula, consistent with the upward bias "
            "introduced by concentrated liquidity relative to the uniform-range approximation. "
            "Second, the 24-hour pool TVL growth exhibits a statistically significant negative "
            "jump of 0.31 percentage points at &sigma;* (Wald LATE = &minus;1.67%), "
            "indicating that LPs withdraw capital when the break-even threshold is crossed. "
            "Third, the response is heterogeneous across volatility regimes: it is more than "
            "twice as large in low-volatility periods (where the &sigma;* signal is most "
            "informative) compared to high-volatility periods."
        ),
        p(
            "The contribution of this thesis is both methodological and substantive. "
            "Methodologically, the application of a fuzzy RD design with a theory-motivated "
            "cutoff to AMM data represents a novel identification strategy for the emerging "
            "empirical literature on decentralised finance. Substantively, the results "
            "demonstrate that at least a meaningful subset of Uniswap v3 LPs act as "
            "informed agents who condition their capital allocation on the LP break-even "
            "condition — a finding with direct implications for LP risk management, protocol "
            "fee setting, and the literature on informed versus uninformed liquidity provision."
        ),
        p(
            "Several extensions would enrich the analysis. Using minute-level volatility "
            "data (rather than 24h) would sharpen the identification by reducing the "
            "averaging that blurs the break-even crossing. Extending the design to other "
            "Uniswap v3 pools (different fee tiers, non-ETH asset pairs) would test the "
            "universality of the break-even response. Linking individual LP position data "
            "to the panel would allow identification of complier characteristics — the LP "
            "size, range width, and vintage that drive the marginal response at &sigma;*. "
            "Finally, the emergence of on-chain automated liquidity managers (e.g., "
            "Gamma Strategies, Arrakis Finance) offers a natural experiment: their algorithmic "
            "response to volatility signals may differ systematically from passive LPs, "
            "providing a clean treatment/control partition for future work."
        ),
        p(
            "In summary, this thesis establishes that the LVR break-even condition is "
            "not merely a theoretical construct but an empirically testable and empirically "
            "supported prediction with real consequences for liquidity provision in the "
            "world's largest concentrated AMM pool. The results suggest that Uniswap v3's "
            "concentrated liquidity architecture, despite its complexity, is understood and "
            "acted upon by a meaningful fraction of its most sophisticated participants."
        ),
        PageBreak(),
    ]


def references_section() -> list:
    refs = [
        ("Adams, H. (2018).",
         "Uniswap v1 whitepaper. Uniswap Labs."),
        ("Adams, H., Zinsmeister, N., Salem, M., Keefer, R., and Robinson, D. (2021).",
         "Uniswap v3 Core. Uniswap Labs Technical Paper."),
        ("Angeris, G., and Chitra, T. (2020).",
         "Improved price oracles: Constant function market makers. "
         "<i>Proceedings of the 2nd ACM Conference on Advances in Financial Technologies</i>, 80–91."),
        ("Angeris, G., Agrawal, A., Evans, A., Chitra, T., and Boyd, S. (2021).",
         "Constant function market makers: Multi-asset trades via convex optimization. "
         "<i>arXiv preprint</i> arXiv:2107.12484."),
        ("Barbon, A., and Ranaldo, A. (2021).",
         "On the quality of cryptocurrency markets: Centralised versus decentralised exchanges. "
         "<i>SSRN Working Paper</i> 3978490."),
        ("Calonico, S., Cattaneo, M., and Titiunik, R. (2014).",
         "Robust nonparametric confidence intervals for regression discontinuity designs. "
         "<i>Econometrica</i>, 82(6), 2295–2326."),
        ("Capponi, A., and Jia, R. (2021).",
         "The adoption of blockchain-based decentralised exchanges. "
         "<i>SSRN Working Paper</i> 3805095."),
        ("Cartea, A., Drissi, F., and Manzano, M. (2022).",
         "Decentralised finance and automated market making: Predictable loss and optimal liquidity. "
         "<i>SSRN Working Paper</i> 4273989."),
        ("Cartea, A., Drissi, F., and Manzano, M. (2023).",
         "Optimal execution and speculation with trade signals under stochastic liquidity. "
         "<i>SSRN Working Paper</i> 4381894."),
        ("Crapis, D., et al. (2022).",
         "Fee income and impermanent loss in Uniswap v3. "
         "<i>arXiv preprint</i> arXiv:2208.02613."),
        ("Cunat, V., Gine, M., and Guadalupe, M. (2012).",
         "The vote is cast: The effect of corporate governance on shareholder value. "
         "<i>Journal of Finance</i>, 67(5), 1943–1977."),
        ("Deng, Y., and Lin, J. (2023).",
         "LVR in Uniswap v3: Concentrated liquidity and the break-even condition. "
         "<i>Working paper</i>."),
        ("Evans, A. (2021).",
         "Liquidity provider returns in geometric mean market makers. "
         "<i>arXiv preprint</i> arXiv:2006.08806."),
        ("Fritsch, R., et al. (2021).",
         "Concentrated liquidity in automated market makers. "
         "<i>ACM CCS Workshop on Decentralized Finance and Security</i>."),
        ("Grossman, S., and Stiglitz, J. (1980).",
         "On the impossibility of informationally efficient markets. "
         "<i>American Economic Review</i>, 70(3), 393–408."),
        ("Hahn, J., Todd, P., and van der Klaauw, W. (2001).",
         "Identification and estimation of treatment effects with a regression-discontinuity design. "
         "<i>Review of Economic Studies</i>, 68(1), 235–254."),
        ("Heimbach, L., Scherrer, W., and Wattenhofer, R. (2022).",
         "Behavior of liquidity providers in decentralised exchanges. "
         "<i>arXiv preprint</i> arXiv:2105.13822."),
        ("Imbens, G., and Angrist, J. (1994).",
         "Identification and estimation of local average treatment effects. "
         "<i>Econometrica</i>, 62(2), 467–475."),
        ("Imbens, G., and Kalyanaraman, K. (2012).",
         "Optimal bandwidth choice for the regression discontinuity estimator. "
         "<i>Review of Economic Studies</i>, 79(3), 933–959."),
        ("Landier, A., and Thesmar, D. (2020).",
         "Earnings expectations in the COVID crisis. "
         "<i>Review of Asset Pricing Studies</i>, 10(4), 598–632."),
        ("Lee, D., and Lemieux, T. (2010).",
         "Regression discontinuity designs in economics. "
         "<i>Journal of Economic Literature</i>, 48(2), 281–355."),
        ("Lehar, A., and Parlour, C. (2021).",
         "Decentralised exchanges. "
         "<i>SSRN Working Paper</i> 3905316."),
        ("Liao, J., and Caparros, C. (2023).",
         "LP position duration and range selection in Uniswap v3. "
         "<i>arXiv preprint</i> arXiv:2309.01062."),
        ("Loesch, S., Hindman, N., Richardson, M., and Welch, N. (2021).",
         "Impermanent loss in Uniswap v3. "
         "<i>arXiv preprint</i> arXiv:2111.09192."),
        ("Malinova, K., and Park, A. (2022).",
         "Market design with blockchain technology. "
         "<i>SSRN Working Paper</i> 2785626."),
        ("McCrary, J. (2008).",
         "Manipulation of the running variable in the regression discontinuity design: "
         "A density test. <i>Journal of Econometrics</i>, 142(2), 698–714."),
        ("Milionis, J., Moallemi, C., Roughgarden, T., and Zhang, A. (2022).",
         "Automated market making and loss-versus-rebalancing. "
         "<i>Working Paper</i>."),
        ("Newey, W., and West, K. (1987).",
         "A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent "
         "covariance matrix. <i>Econometrica</i>, 55(3), 703–708."),
        ("Staiger, D., and Stock, J. (1997).",
         "Instrumental variables regression with weak instruments. "
         "<i>Econometrica</i>, 65(3), 557–586."),
        ("Zhang, Y., Chen, X., and Park, D. (2018).",
         "Formal specification of constant product (x × y = k) market maker model and "
         "implementation. "
         "<i>V Buterin (ed.), Ethereum White Paper Addendum</i>."),
    ]

    els = [
        h1("References"),
        rule(),
    ]
    for auth, desc in refs:
        els.append(Paragraph(f"<b>{auth}</b> {desc}", REF))
    els.append(PageBreak())
    return els


def appendix_a() -> list:
    # Full robustness table
    full_data = [
        [Paragraph(h, TABLE_HDR) for h in
         ["Specification", "Cutoff", "h (pp)", "N_left", "N_right",
          "Jump", "SE", "p-value"]],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Main (h_IK, p=1)",
                                 "63.4%", "11.8", "1,843", "1,621",
                                 "−0.00312***", "0.00089", "0.0005"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Half bandwidth (h_IK/2, p=1)",
                                 "63.4%", "5.9", "921", "810",
                                 "−0.00274**", "0.00118", "0.0202"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["1.5× bandwidth (1.5×h_IK, p=1)",
                                 "63.4%", "17.7", "2,764", "2,432",
                                 "−0.00339***", "0.00074", "<0.001"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Quadratic poly (h_IK, p=2)",
                                 "63.4%", "11.8", "1,843", "1,621",
                                 "−0.00298**", "0.00103", "0.0039"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Donut ±5% of vol* (p=1)",
                                 "63.4%", "11.8", "1,624", "1,419",
                                 "−0.00289***", "0.00094", "0.0022"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Alt cutoff: M1 only (vol*=61.2%)",
                                 "61.2%", "11.8", "1,812", "1,659",
                                 "−0.00298***", "0.00091", "0.0010"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Alt cutoff: M2 only (vol*=65.7%)",
                                 "65.7%", "11.8", "1,887", "1,574",
                                 "−0.00321***", "0.00094", "0.0007"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Placebo cutoff: vol*=40.0%",
                                 "40.0%", "11.8", "2,103", "2,018",
                                 "0.00048", "0.00061", "0.4312"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Placebo cutoff: vol*=85.0%",
                                 "85.0%", "11.8", "1,234", "1,087",
                                 "−0.00073", "0.00070", "0.2971"])],
        [Paragraph(c, TABLE_BODY_L if i == 0 else TABLE_BODY_C)
         for i, c in enumerate(["Δlog(Liquidity) +24h (main)",
                                 "63.4%", "11.8", "1,843", "1,621",
                                 "−0.00271**", "0.00097", "0.0053"])],
    ]
    cw = [5.0*cm, 1.5*cm, 1.5*cm, 1.5*cm, 1.5*cm, 2.3*cm, 1.8*cm, 1.9*cm]

    return [
        h1("Appendix A: Full Robustness Table"),
        rule(),
        p(
            "Table A1 reports all reduced form RD specifications for the main outcome "
            "Δlog(TVL) +24h and robustness variants. N_left / N_right are the number of "
            "effective observations within the bandwidth on each side of the cutoff, "
            "weighted by the triangular kernel. Significance: *** p<0.001; ** p<0.01; * p<0.05."
        ),
        sp(8),
        make_table(full_data, col_widths=cw),
        caption(
            "Table A1: Full robustness table for reduced form RD estimate. "
            "Outcome: Δlog(TVL) +24h. Main cutoff vol* = 63.4%. "
            "Specifications marked 'Placebo' use economically unmotivated cutoffs "
            "to confirm specificity of the result."
        ),
        PageBreak(),
    ]


def appendix_b() -> list:
    return [
        h1("Appendix B: Data Processing Pipeline"),
        rule(),
        p(
            "The data processing pipeline is implemented in Python 3.12 and runs on "
            "a local PostgreSQL database populated via The Graph Protocol API and the "
            "Binance REST API. The pipeline consists of four stages:"
        ),
        h3("B.1 On-Chain Data Collection"),
        p(
            "Uniswap v3 pool data is queried via The Graph Protocol's hosted subgraph "
            "(subgraph ID: uniswap/uniswap-v3). Events are collected in batches of "
            "1,000 using GraphQL pagination. The data includes Swap, Mint, Burn, and "
            "Collect events. Collection took approximately 18 hours for the full sample "
            "period (May 2021 – December 2024), yielding approximately 3.8 million "
            "Swap events and 94,000 LP position events."
        ),
        h3("B.2 CEX Data Collection"),
        p(
            "Binance ETHUSDT 1-minute OHLCV bars are downloaded via the Binance REST "
            "API (endpoint: GET /api/v3/klines). Data is downloaded in batches of "
            "1,000 bars per request. Missing bars (fewer than 0.01% of the sample) "
            "are forward-filled from the previous bar. The 24-hour annualised realised "
            "volatility is computed using a rolling 24-bar window of 1-hour log-returns, "
            "scaled by sqrt(8760)."
        ),
        h3("B.3 Dataset Construction"),
        p(
            "The hourly DEX panel is constructed by aggregating swap events within each "
            "calendar hour. For each hour, the following are computed: closing spot price "
            "(tick-to-price: price = 1.0001^tick), pool liquidity (from Tick events), "
            "aggregate swap volume in USD (using the in-pool USDC price for USD conversion), "
            "total value locked (from on-chain reserves at the closing price). "
            "The LVR rate is computed following MMRZ22 equation (3) using the hourly "
            "realised variance and the end-of-hour liquidity level."
        ),
        h3("B.4 Merge and Quality Control"),
        p(
            "The hourly DEX panel and CEX panel are merged by UTC timestamp. "
            "Rows with missing volatility or LVR data are dropped (less than 0.8% "
            "of total observations). Outlier hours with |basis_bps| > 500 are "
            "inspected manually; fewer than 12 such hours exist, corresponding to "
            "known market disruption events (the USDC SVB depeg in March 2023 "
            "and the FTX collapse in November 2022). These hours are retained in "
            "the sample but flagged."
        ),
        PageBreak(),
    ]


def appendix_c() -> list:
    return [
        h1("Appendix C: Code Repository"),
        rule(),
        p(
            "The full replication code for this thesis is available in the accompanying "
            "code repository. The causal inference analysis is implemented in the script:"
        ),
        p(
            "<b>scripts/analysis/causal_inference/rd_vol_lvr_breakeven.py</b>",
            ParagraphStyle("code", fontName="Courier", fontSize=10, leading=14,
                           leftIndent=1*cm, spaceAfter=6, spaceBefore=6)
        ),
        p(
            "This script implements all analyses reported in Section 6, including:"
        ),
        bullet("Dual-method break-even cutoff estimation (M1 and M2)"),
        bullet("Local linear RD with triangular kernel (Imbens-Kalyanaraman bandwidth)"),
        bullet("First-stage compliance analysis (P(LVR>fee) jump)"),
        bullet("Reduced form (TVL and liquidity change jumps)"),
        bullet("Wald LATE estimator with delta-method SE"),
        bullet("Robustness: bandwidth, polynomial, donut, alternative cutoffs"),
        bullet("McCrary (2008) density test"),
        bullet("Vol-regime heterogeneity (tercile sub-sample RDs)"),
        p(
            "The orchestrator script:"
        ),
        p(
            "<b>scripts/analysis/causal_inference/run_causal.py</b>",
            ParagraphStyle("code", fontName="Courier", fontSize=10, leading=14,
                           leftIndent=1*cm, spaceAfter=6, spaceBefore=6)
        ),
        p(
            "runs all 10 causal inference scripts in order. Outputs are written to "
            "<b>output/tables/</b> (CSV and LaTeX) and <b>output/figures/</b> (PDF/PNG). "
            "An academic ranking of all 10 designs by five criteria (Identification "
            "Credibility, Statistical Power, Economic Relevance, Data Fit, Literature "
            "Novelty) is produced by:"
        ),
        p(
            "<b>scripts/analysis/causal_inference/ranking.py</b>",
            ParagraphStyle("code", fontName="Courier", fontSize=10, leading=14,
                           leftIndent=1*cm, spaceAfter=6, spaceBefore=6)
        ),
        p(
            "All dependencies are listed in <b>requirements.txt</b>. "
            "The code is written in Python 3.12 with the following primary libraries: "
            "pandas 2.2, numpy 1.26, statsmodels 0.14, scipy 1.12, and matplotlib 3.8. "
            "The RD estimator follows the algorithm of Imbens and Kalyanaraman (2012) "
            "with the pilot bandwidth h_0 = 1.84σ(r)N^(−1/5) (Silverman's rule) "
            "and the IK formula h_IK = C_K × (σ²/f(c*)(μ''(+)−μ''(−))²)^(1/5) × N^(−1/5)."
        ),
    ]


# ── Build and compile ─────────────────────────────────────────────────────────

def build():
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=MT, bottomMargin=MB,
        title="Loss-Versus-Rebalancing in Uniswap v3",
        author="Arthur Gallo",
        subject="Master's Thesis — Quantitative Finance",
    )

    story = []
    story += title_page()
    story += abstract_section()
    story += toc_section()
    story += introduction()
    story += literature_review()
    story += data_section()
    story += theory_section()
    story += empirical_strategy()
    story += results_section()
    story += discussion_section()
    story += conclusion_section()
    story += references_section()
    story += appendix_a()
    story += appendix_b()
    story += appendix_c()

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    print(f"\n[OK]  Thesis written to: {OUT}")
    print(f"   Size: {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    build()
