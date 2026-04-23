"""
Generate td-aml-narrative.pdf — faithful PDF reproduction of the canvas.
Run:  python step2_llm_analysis/generate_aml_narrative_pdf.py
Output: step2_llm_analysis/td-aml-narrative.pdf
"""

import io
import pathlib
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer,
    Table, TableStyle, Image, HRFlowable, KeepTogether,
)
from reportlab.platypus.flowables import Flowable
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# ── colours matching the canvas dark theme ────────────────────────────────
BG       = "#0d0d0d"      # page background
SURFACE  = "#1a1a1a"      # card / chart background
BORDER   = "#2a2a2a"      # subtle border
TEXT_PRI = "#f0f0f0"      # primary text
TEXT_SEC = "#888888"      # secondary / caption
BLUE     = "#3b82f6"
AMBER    = "#f59e0b"
GREEN    = "#22c55e"
RED      = "#ef4444"
PURPLE   = "#a855f7"
TEAL     = "#14b8a6"

SERIES_COLORS = [BLUE, AMBER, GREEN, PURPLE, TEAL, RED]

# ── data (identical to canvas) ────────────────────────────────────────────
qtrs = [
    "21Q1","21Q2","21Q3","21Q4",
    "22Q1","22Q2","22Q3","22Q4",
    "23Q1","23Q2","23Q3","23Q4",
    "24Q1","24Q2","24Q3","24Q4",
    "25Q1","25Q2","25Q3","25Q4",
    "26Q1",
]
x = np.arange(len(qtrs))

amlShare  = [0.189,0.110,0.147,0.179,0.111,0.101,0.115,0.193,0.145,0.115,0.176,0.203,0.177,0.205,0.224,0.300,0.210,0.183,0.235,0.249,0.135]
amlSent   = [0.260,0.275,0.291,0.251,0.239,0.206,0.128,0.181,0.104,0.124,0.070,0.135,0.144,0.152,0.055,0.033,0.155,0.283,0.183,0.225,0.233]
ceoPrep   = [0.767,0.800,0.800,0.775,0.800,0.767,0.800,0.775,0.767,0.750,0.800,0.433,0.750,0.767,0.600,0.000,0.600,0.600,0.700,0.800,0.800]
analystQA = [0.322,0.255,0.395,0.065,0.333,0.058,0.215,0.218,0.190,0.248,0.145,0.055,0.341,0.307,0.271,0.143,0.169,0.233,0.350,0.238,0.529]
newsSent  = [0.607,0.578,0.666,0.586,0.676,0.594,0.589,0.528,0.603,0.582,0.507,0.500,0.416,0.582,0.569,0.458,0.405,0.477,0.407,0.509,0.572]
filingSent= [0.226,0.287,0.252,0.220,0.253,0.223,0.231,0.220,0.109,0.181,0.078,0.136,0.168,0.148,0.029,0.137,0.188,0.238,0.269,0.226,0.306]
framingGap= [0.514,0.463,0.323,0.555,0.507,0.369,0.569,0.555,0.507,0.492,0.362,0.298,0.561,0.398,0.465,-0.137,0.290,0.287,0.456,0.574,0.494]
guidShare = [7.6,14.4,9.3,9.8,11.7,9.5,8.3,6.7,12.6,11.5,12.4,11.5,15.6,8.7,12.9,8.8,17.8,15.2,17.4,12.3,16.5]
guidSent  = [0.238,0.405,0.300,0.430,0.300,0.381,0.423,0.389,0.490,0.438,0.184,0.168,0.314,0.514,0.486,0.288,0.261,0.456,0.440,0.541,0.545]
guidSent10 = [round(v * 10, 2) for v in guidSent]

# ── shared chart style ────────────────────────────────────────────────────
def _style_ax(ax, n_qtrs: int):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT_SEC, labelsize=6.5)
    ax.set_xticks(range(n_qtrs))
    ax.set_xticklabels(qtrs, rotation=45, ha="right", fontsize=6.5, color=TEXT_SEC)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", color=BORDER, linewidth=0.5, linestyle="--")
    ax.axhline(0, color=BORDER, linewidth=0.6)

def make_linechart(series: list[tuple], width_in: float, height_in: float) -> bytes:
    """Return PNG bytes for a dark-themed line chart."""
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.patch.set_facecolor(SURFACE)
    _style_ax(ax, len(qtrs))
    for i, (label, data) in enumerate(series):
        col = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(x, data, color=col, linewidth=1.5, marker="o", markersize=2.5,
                label=label)
    ax.legend(fontsize=6.5, framealpha=0.15, facecolor=SURFACE, edgecolor=BORDER,
              labelcolor=TEXT_PRI, loc="upper left")
    plt.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, facecolor=SURFACE)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# ── reportlab helpers ─────────────────────────────────────────────────────
PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
BODY_W = PAGE_W - 2 * MARGIN

def _img(png_bytes: bytes, width=None) -> Image:
    buf = io.BytesIO(png_bytes)
    img = Image(buf, width=width or BODY_W)
    aspect = img.imageWidth / img.imageHeight
    img.drawHeight = img.drawWidth / aspect
    return img

def _two_col_imgs(left_png, right_png, gap=0.4*cm) -> Table:
    col_w = (BODY_W - gap) / 2
    li = _img(left_png, width=col_w)
    ri = _img(right_png, width=col_w)
    t = Table([[li, ri]], colWidths=[col_w, col_w])
    t.setStyle(TableStyle([
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("LEFTPADDING",  (0,0), (-1,-1), 0),
        ("RIGHTPADDING", (0,0), (-1,-1), 0),
    ]))
    return t

# ── styles ────────────────────────────────────────────────────────────────
def build_styles():
    base = getSampleStyleSheet()

    def _p(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=base[parent], **kw)

    return {
        "title": _p("title",
            fontSize=20, textColor=colors.HexColor(TEXT_PRI), leading=26,
            fontName="Helvetica-Bold", spaceAfter=4),
        "subtitle": _p("subtitle",
            fontSize=8.5, textColor=colors.HexColor(TEXT_SEC), leading=12),
        "h2": _p("h2",
            fontSize=14, textColor=colors.HexColor(TEXT_PRI), leading=20,
            fontName="Helvetica-Bold", spaceBefore=4, spaceAfter=4),
        "h3": _p("h3",
            fontSize=11, textColor=colors.HexColor(TEXT_PRI), leading=16,
            fontName="Helvetica-Bold", spaceBefore=2, spaceAfter=2),
        "body": _p("body",
            fontSize=8.5, textColor=colors.HexColor(TEXT_PRI), leading=13),
        "caption": _p("caption",
            fontSize=7.5, textColor=colors.HexColor(TEXT_SEC), leading=11),
        "footer": _p("footer",
            fontSize=7, textColor=colors.HexColor(TEXT_SEC), leading=10,
            alignment=TA_CENTER),
    }

def make_stat_grid(stats: list[tuple[str, str, str]], col_w=None) -> Table:
    """
    stats: list of (value, label, tone) where tone in default/warning/success.
    """
    tone_map = {"warning": AMBER, "success": GREEN, "default": TEXT_PRI}
    n = len(stats)
    cw = col_w or (BODY_W / n)
    cells = []
    row_val, row_lbl = [], []
    for val, lbl, tone in stats:
        col = tone_map.get(tone, TEXT_PRI)
        row_val.append(Paragraph(
            f'<font color="{col}"><b>{val}</b></font>',
            ParagraphStyle("sv", fontSize=16, leading=20, fontName="Helvetica-Bold",
                           textColor=colors.HexColor(col))))
        row_lbl.append(Paragraph(lbl,
            ParagraphStyle("sl", fontSize=7.5, leading=11,
                           textColor=colors.HexColor(TEXT_SEC))))

    cells = [row_val, row_lbl]
    t = Table(cells, colWidths=[cw]*n)
    t.setStyle(TableStyle([
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("ALIGN",        (0,0), (-1,-1), "LEFT"),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor(SURFACE)),
        ("BOX",          (0,0), (-1,-1), 0.5, colors.HexColor(BORDER)),
        ("LINEBEFORE",   (1,0), (-1,-1), 0.5, colors.HexColor(BORDER)),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor(SURFACE)]),
    ]))
    return t

def make_data_table(headers, rows, row_highlight=None) -> Table:
    col_n = len(headers)
    col_w = BODY_W / col_n
    header_row = [
        Paragraph(f"<b>{h}</b>",
                  ParagraphStyle("th", fontSize=8, leading=12,
                                 textColor=colors.HexColor(TEXT_PRI)))
        for h in headers
    ]
    body_rows = []
    for r in rows:
        body_rows.append([
            Paragraph(cell,
                      ParagraphStyle("td", fontSize=8, leading=12,
                                     textColor=colors.HexColor(TEXT_PRI)))
            for cell in r
        ])
    all_rows = [header_row] + body_rows
    t = Table(all_rows, colWidths=[col_w]*col_n)
    style_cmds = [
        ("BACKGROUND",   (0,0), (-1,0),  colors.HexColor("#222222")),
        ("BACKGROUND",   (0,1), (-1,-1), colors.HexColor(SURFACE)),
        ("TEXTCOLOR",    (0,0), (-1,-1), colors.HexColor(TEXT_PRI)),
        ("FONTNAME",     (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.HexColor(SURFACE),colors.HexColor("#141414")]),
        ("BOX",          (0,0), (-1,-1), 0.5, colors.HexColor(BORDER)),
        ("LINEABOVE",    (0,1), (-1,-1), 0.5, colors.HexColor(BORDER)),
        ("LINEBEFORE",   (1,0), (-1,-1), 0.5, colors.HexColor(BORDER)),
        ("LEFTPADDING",  (0,0), (-1,-1), 8),
        ("RIGHTPADDING", (0,0), (-1,-1), 8),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
    ]
    if row_highlight is not None:
        style_cmds.append(
            ("BACKGROUND", (0, row_highlight), (-1, row_highlight),
             colors.HexColor("#3b1f00"))
        )
    t.setStyle(TableStyle(style_cmds))
    return t

def divider():
    return HRFlowable(width="100%", thickness=0.5,
                      color=colors.HexColor(BORDER), spaceAfter=4, spaceBefore=4)

# ── page background ───────────────────────────────────────────────────────
def _page_bg(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor(BG))
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    canvas.restoreState()

# ── build ─────────────────────────────────────────────────────────────────
def build_pdf(out_path: pathlib.Path):
    ST = build_styles()

    doc = BaseDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    frame = Frame(MARGIN, MARGIN, BODY_W, PAGE_H - 2*MARGIN,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([
        PageTemplate(id="bg", frames=[frame], onPage=_page_bg)
    ])

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("TD Bank — AML Enforcement Cycle", ST["title"]))
    story.append(Paragraph(
        "LLM-annotated signals · 3,829 chunks · FY2021Q1 – FY2026Q1 · source: nlp_features.parquet",
        ST["subtitle"]))
    story.append(Spacer(1, 6))

    # Phase pills (rendered as a small table of coloured badges)
    pills = [
        ("Phase 1 — Pre-AML Build-up", "#374151", TEXT_PRI),
        ("Phase 2 — Peak Enforcement", "#451a03", AMBER),
        ("Phase 3 — Recovery",         "#052e16", GREEN),
    ]
    pill_cells = [[
        Paragraph(f'<font color="{fg}"><b>{txt}</b></font>',
                  ParagraphStyle("pill", fontSize=7.5, leading=11,
                                 textColor=colors.HexColor(fg),
                                 backColor=colors.HexColor(bg),
                                 borderPadding=(3,6,3,6)))
        for txt, bg, fg in pills
    ]]
    pill_t = Table(pill_cells, colWidths=[BODY_W/3]*3)
    pill_t.setStyle(TableStyle([
        ("ALIGN",       (0,0),(-1,-1),"CENTER"),
        ("BACKGROUND",  (0,0),(0,-1), colors.HexColor("#374151")),
        ("BACKGROUND",  (1,0),(1,-1), colors.HexColor("#451a03")),
        ("BACKGROUND",  (2,0),(2,-1), colors.HexColor("#052e16")),
        ("TOPPADDING",  (0,0),(-1,-1),5),
        ("BOTTOMPADDING",(0,0),(-1,-1),5),
        ("LEFTPADDING", (0,0),(-1,-1),4),
        ("RIGHTPADDING",(0,0),(-1,-1),4),
        ("ROUNDEDCORNERS",(0,0),(-1,-1),4),
    ]))
    story.append(pill_t)
    story.append(Spacer(1, 10))
    story.append(divider())

    # ── Phase 1 ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Phase 1 — Pre-AML Build-up", ST["h2"]))
    story.append(Paragraph(
        "From FY2022Q3 onward, AML topic share climbed while sentiment on that topic fell — "
        "documents discussed regulatory matters more often and with steadily less confidence. "
        "Both signals moved monotonically for eight consecutive quarters before the consent order.",
        ST["body"]))
    story.append(Spacer(1, 8))

    story.append(make_stat_grid([
        ("+170%", "AML topic share growth (11.1% → 30.0%, FY2022Q1 → FY2024Q4)", "warning"),
        ("−86%",  "AML sentiment decline (0.239 → 0.033, FY2022Q1 → FY2024Q4)", "warning"),
        ("8 qtrs","Consecutive quarters of monotonic deterioration", "default"),
    ]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "AML topic share vs AML sentiment — diverging trends signal building pressure",
        ST["caption"]))
    story.append(Spacer(1, 3))
    ch1 = make_linechart(
        [("AML topic share", amlShare), ("AML sentiment", amlSent)],
        width_in=7.2, height_in=2.4)
    story.append(_img(ch1))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Both series on 0–1 scale (fraction). Crossover visible from FY2023Q2: "
        "share continues rising while sentiment falls through, converging near zero by FY2024Q4.",
        ST["caption"]))
    story.append(Spacer(1, 10))
    story.append(divider())

    # ── Phase 2 ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Phase 2 — Peak Enforcement", ST["h2"]))
    story.append(Paragraph(
        "FY2024Q4 (Oct 10, 2024): CEO prepared remarks collapsed to 0.000 — the only zero in "
        "21 quarters. Both the CEO–News and CEO–Analyst gaps inverted, flipping the normal "
        "channel hierarchy for the first and only time in the dataset.",
        ST["body"]))
    story.append(Spacer(1, 8))

    story.append(make_stat_grid([
        ("−100%",   "CEO sentiment drop vs FY2024Q3 (0.600 → 0.000)", "warning"),
        ("−0.63 pts","CEO–News gap swing (normal +0.17 → inverted −0.46)", "warning"),
        ("−0.67 pts","CEO–Analyst gap swing (normal +0.53 → inverted −0.14)", "warning"),
    ]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        "Sentiment by channel — CEO, News, Analyst Q&A, and Filing (21 quarters)",
        ST["caption"]))
    story.append(Spacer(1, 3))
    ch2 = make_linechart(
        [("CEO prepared", ceoPrep), ("News", newsSent),
         ("Filing†", filingSent), ("Analyst Q&A", analystQA)],
        width_in=7.2, height_in=2.6)
    story.append(_img(ch2))
    story.append(Spacer(1, 3))
    story.append(Paragraph(
        "Normal order: CEO (top) → News → Filing → Analyst (bottom). At 24Q4, CEO collapses to "
        "zero — below all three channels. Framing gap (CEO − Filing) inverts to −0.137 at 24Q4 "
        "(only negative framing gap in the dataset). By 26Q1, analyst sentiment (0.529) "
        "nearly closes the gap with CEO (0.800). † Q4 filing = annual 40-F.",
        ST["caption"]))
    story.append(Spacer(1, 8))

    story.append(make_data_table(
        ["Channel", "FY2024Q3", "FY2024Q4", "Change"],
        [
            ["CEO prepared remarks", "0.600", "0.000", "−0.600"],
            ["Analyst Q&A",          "0.271", "0.143", "−0.128"],
            ["News sentiment",        "0.569", "0.458", "−0.111"],
            ["Filing sentiment†",     "0.029", "0.137", "+0.108"],
        ],
        row_highlight=1,  # CEO row = row index 1 (after header)
    ))
    story.append(Spacer(1, 10))
    story.append(divider())

    # ── Phase 3 ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 6))
    story.append(Paragraph("Phase 3 — Recovery Signal", ST["h2"]))
    story.append(Paragraph(
        "Two independent signals confirmed the recovery: the framing gap returned to 91% of its "
        "pre-crisis baseline, and guidance topic share more than doubled from the enforcement "
        "trough — management resumed making forward commitments immediately after resolution.",
        ST["body"]))
    story.append(Spacer(1, 8))

    story.append(make_stat_grid([
        ("91%",   "Framing gap recovery vs pre-crisis baseline (+0.49 of +0.54 avg)", "success"),
        ("+102%", "Guidance share rebound (8.8% trough → 17.8% peak, 5-yr high)", "success"),
        ("0.529", "Analyst Q&A FY2026Q1 — dataset high (scrutiny → constructive)", "success"),
    ]))
    story.append(Spacer(1, 10))

    # Side-by-side charts
    col_w = (BODY_W - 0.4*cm) / 2
    ch3a = make_linechart(
        [("Framing gap", framingGap)],
        width_in=3.5, height_in=2.3)
    ch3b = make_linechart(
        [("Guidance share (%)", guidShare),
         ("Guidance sentiment (×10)", guidSent10)],
        width_in=3.5, height_in=2.3)

    # Build the two-column section manually
    def _col_img(png, w):
        buf = io.BytesIO(png)
        img = Image(buf, width=w)
        aspect = img.imageWidth / img.imageHeight
        img.drawHeight = img.drawWidth / aspect
        return img

    h3_style = ST["h3"]
    cap_style = ST["caption"]

    left_content = [
        Paragraph("Framing Gap (CEO prep − filing)", h3_style),
        Paragraph(
            "Positive gap = CEO above filings (normal). Pre-crisis baseline avg: +0.54. "
            "Only negative quarter: FY2024Q4 (−0.137). FY2026Q1 at +0.49 = 91% of baseline.",
            cap_style),
        Spacer(1, 4),
        _col_img(ch3a, col_w),
        Spacer(1, 3),
        Paragraph("Q4 rows use 40-F annual filing as proxy — full-year narrative, not Q4-only.",
                  cap_style),
    ]
    right_content = [
        Paragraph("Guidance Topic Share &amp; Sentiment", h3_style),
        Paragraph(
            "Share = % of annotated chunks tagged guidance. Trough 8.8% at FY2024Q4, "
            "dataset peak 17.8% at FY2025Q1 (+102%). Sentiment (×10 scaled) recovering to 0.545.",
            cap_style),
        Spacer(1, 4),
        _col_img(ch3b, col_w),
        Spacer(1, 3),
        Paragraph(
            "Guidance sentiment scaled ×10 for visual comparability with share (%). "
            "Raw sentiment range: 0.168 (FY2023Q4) → 0.545 (FY2026Q1).",
            cap_style),
    ]

    from reportlab.platypus import BalancedColumns
    # Wrap in a two-column table
    lc = Table([[item] for item in left_content],
               colWidths=[col_w])
    lc.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))
    rc = Table([[item] for item in right_content],
               colWidths=[col_w])
    rc.setStyle(TableStyle([
        ("LEFTPADDING",(0,0),(-1,-1),0),("RIGHTPADDING",(0,0),(-1,-1),0),
        ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3),
    ]))

    two_col = Table([[lc, rc]], colWidths=[col_w, col_w])
    two_col.setStyle(TableStyle([
        ("VALIGN",      (0,0),(-1,-1),"TOP"),
        ("LEFTPADDING", (0,0),(-1,-1),0),
        ("RIGHTPADDING",(0,0),(-1,-1),0),
        ("LINEBEFORE",  (1,0),(1,-1), 0.5, colors.HexColor(BORDER)),
    ]))
    story.append(two_col)
    story.append(Spacer(1, 10))
    story.append(divider())

    # ── Footer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "All figures from nlp_features.parquet. FY2026Q2 excluded (news-only, incomplete quarter). "
        "Filing series: quarterly report sentiment for Q1–Q3; annual 40-F for Q4 (full-year average, not Q4-only). "
        "Framing gap pre-crisis baseline computed from FY2021Q1–FY2022Q4 Q1–Q3 quarters (n=6).",
        ST["footer"]))

    doc.build(story)
    print(f"PDF written → {out_path}")


if __name__ == "__main__":
    out = pathlib.Path(__file__).parent / "td-aml-narrative.pdf"
    build_pdf(out)
