"""Build the paper-ready Results section PDF from RESULTS_SECTION.md (figures placed at [[FIG:...]] markers)."""
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).parent
F = "C:/Windows/Fonts/"
pdfmetrics.registerFont(TTFont("Arial", F + "arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", F + "arialbd.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Italic", F + "ariali.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BoldItalic", F + "arialbi.ttf"))
pdfmetrics.registerFontFamily("Arial", normal="Arial", bold="Arial-Bold", italic="Arial-Italic", boldItalic="Arial-BoldItalic")
pdfmetrics.registerFont(TTFont("Consolas", F + "consola.ttf"))

INK, MUTED, ACCENT, LINE, SOFT = (colors.HexColor(c) for c in ("#1d2329", "#5d6670", "#1f5f8b", "#cfd5db", "#eef3f7"))
base = ParagraphStyle("b", fontName="Arial", fontSize=9.5, leading=13.5, textColor=INK, alignment=TA_LEFT, spaceAfter=5)
S = {
    "h1": ParagraphStyle("h1", base, fontName="Arial-Bold", fontSize=17, leading=21, spaceAfter=8, textColor=ACCENT),
    "h2": ParagraphStyle("h2", base, fontName="Arial-Bold", fontSize=13, leading=17, spaceBefore=10, spaceAfter=6, textColor=ACCENT),
    "h3": ParagraphStyle("h3", base, fontName="Arial-Bold", fontSize=10.5, leading=14, spaceBefore=6, spaceAfter=4),
    "p": base,
    "li": ParagraphStyle("li", base, leftIndent=12, bulletIndent=2, spaceAfter=3),
    "quote": ParagraphStyle("q", base, leftIndent=10, textColor=INK, backColor=SOFT, borderPadding=6, spaceBefore=4, spaceAfter=8),
    "cell": ParagraphStyle("c", base, fontSize=7.4, leading=9.2, spaceAfter=0),
    "cellh": ParagraphStyle("ch", base, fontName="Arial-Bold", fontSize=7.4, leading=9.2, spaceAfter=0, textColor=MUTED),
    "cap": ParagraphStyle("cap", base, fontSize=8.5, leading=11.5, textColor=MUTED, spaceAfter=10),
}
WIDTH = A4[0] - 36 * mm


def inline(t):
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"`([^`]+)`", r'<font name="Consolas">\1</font>', t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<link href="\2" color="#1f5f8b">\1</link>', t)
    return t.replace("★", "*").replace("∈", "in")


def md_table(lines):
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines if not re.match(r"^\|?\s*:?-{2,}", l.strip())]
    data = [[Paragraph(inline(c), S["cellh" if i == 0 else "cell"]) for c in r] for i, r in enumerate(rows)]
    n = len(rows[0])
    lens = [max(len(re.sub(r"[*`]", "", r[j])) if j < len(r) else 0 for r in rows) for j in range(n)]
    words = [max((len(t) for r in rows if j < len(r) for t in re.sub(r"[*`]", "", r[j]).split()), default=4)
             for j in range(n)]
    w = [max(words[j] + 1.5, min(lens[j], 40) * 0.55) for j in range(n)]  # longest word always fits
    widths = [WIDTH * x / sum(w) for x in w]
    tb = Table(data, colWidths=widths, repeatRows=1)
    tb.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT), ("LINEBELOW", (0, 1), (-1, -1), 0.3, LINE),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 3),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return [tb, Spacer(1, 8)]


def md_to_flow(text, hooks=None):
    out, lines, i, para = [], text.splitlines(), 0, []

    def flush():
        if para:
            out.append(Paragraph(inline(" ".join(para)), S["p"]))
            para.clear()

    while i < len(lines):
        l = lines[i]
        s = l.strip()
        if s.startswith("|"):
            flush(); blk = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                blk.append(lines[i]); i += 1
            out += md_table(blk); continue
        if s.startswith(">"):
            flush(); blk = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                blk.append(lines[i].strip().lstrip(">").strip()); i += 1
            out.append(Paragraph(inline(" ".join(blk)), S["quote"])); continue
        m = re.match(r"^(#{1,3})\s+(.*)", s)
        if m:
            flush()
            if hooks and s in hooks:
                out += hooks[s]()
            out.append(Paragraph(inline(m.group(2)), S["h" + str(len(m.group(1)))]))
        elif re.match(r"^([-*]|\d+\.)\s+", s):
            flush()
            num = re.match(r"^(\d+)\.", s)
            body = re.sub(r"^([-*]|\d+\.)\s+", "", s)
            while i + 1 < len(lines) and lines[i + 1].startswith("  ") and lines[i + 1].strip() and not re.match(r"^\s*([-*]|\d+\.)\s", lines[i + 1]):
                i += 1; body += " " + lines[i].strip()
            out.append(Paragraph(inline(body), S["li"], bulletText=(num.group(1) + "." if num else "•")))
        elif s == "---":
            flush(); out.append(Spacer(1, 6))
        elif not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return out



def build(text):
    story = []
    for chunk in re.split(r"(\[\[FIG:[^\]]+\]\])", text):
        m = re.match(r"\[\[FIG:([^|]+)\|([0-9.]+)\|(.+)\]\]", chunk)
        if not m:
            story += md_to_flow(chunk)
            continue
        f, frac, cap = m.group(1), float(m.group(2)), m.group(3)
        img = Image(str(ROOT / "figures" / f))
        w = WIDTH * frac
        img.drawHeight, img.drawWidth = img.imageHeight * w / img.imageWidth, w
        story.append(KeepTogether([Spacer(1, 4), img, Spacer(1, 3), Paragraph(inline(cap), S["cap"])]))
    return story


def footer(c, d):
    c.saveState(); c.setFont("Arial", 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "Results section: Classical ML vs. Deep Learning for CVD Classification (NHANES 2011–2018)")
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {d.page}"); c.restoreState()


out = ROOT.parent / "Results_Section_NHANES_CVD.pdf"
doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm, bottomMargin=16 * mm,
                        title="Results section: Classical ML vs Deep Learning for CVD (NHANES)", author="AI RP team")
doc.build(build((ROOT / "RESULTS_SECTION.md").read_text(encoding="utf-8")), onFirstPage=footer, onLaterPages=footer)
print(out)
