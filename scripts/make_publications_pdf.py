"""Render RELATED_PUBLICATIONS.md to a PDF with reportlab.

Handles the markdown subset the document actually uses: headings (#/##/###),
horizontal rules, bullet lists, bold/italic, inline [text](url) links and bare
<url> autolinks. No external tooling (no pandoc/LaTeX) required.

Usage:
    python scripts/make_publications_pdf.py [in.md] [out.pdf]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
LINK = colors.HexColor("#1a5fb4")
RULE = colors.HexColor("#c8ccd4")

_AUTOLINK = re.compile(r"<((?:https?|mailto):[^>\s]+)>")
_MDLINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", re.S)


def inline(text: str) -> str:
    """Convert inline markdown to reportlab's mini-HTML."""
    # Pull links out before escaping so their URLs survive intact.
    stash: list[tuple[str, str]] = []

    def _stash(label: str, url: str) -> str:
        stash.append((label, url))
        return f"\x00{len(stash) - 1}\x00"

    text = _AUTOLINK.sub(lambda m: _stash(m.group(1), m.group(1)), text)
    text = _MDLINK.sub(lambda m: _stash(m.group(1), m.group(2)), text)

    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<i>\1</i>", text)

    def _unstash(m: re.Match[str]) -> str:
        label, url = stash[int(m.group(1))]
        label = label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        label = _BOLD.sub(r"<b>\1</b>", label)
        label = re.sub(r"`([^`]+)`", r"<i>\1</i>", label)
        url = url.replace("&", "&amp;")
        return f'<a href="{url}" color="#1a5fb4">{label}</a>'

    return re.sub(r"\x00(\d+)\x00", _unstash, text)


def register_fonts() -> tuple[str, str]:
    """Register a Unicode TTF so glyphs like ↔ ≈ γ ± → survive. Falls back to Helvetica."""
    candidates = [
        ("DejaVuSans", "DejaVuSans-Bold", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ("Arial", "Arial-Bold", "arial.ttf", "arialbd.ttf"),
    ]
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
    for reg, bold, reg_file, bold_file in candidates:
        for root in roots:
            if (root / reg_file).exists() and (root / bold_file).exists():
                pdfmetrics.registerFont(TTFont(reg, str(root / reg_file)))
                pdfmetrics.registerFont(TTFont(bold, str(root / bold_file)))
                italic = root / reg_file.replace(".ttf", "i.ttf")
                it_name = reg
                if italic.exists():
                    it_name = f"{reg}-Italic"
                    pdfmetrics.registerFont(TTFont(it_name, str(italic)))
                pdfmetrics.registerFontFamily(
                    reg, normal=reg, bold=bold, italic=it_name, boldItalic=bold
                )
                return reg, bold
    return "Helvetica", "Helvetica-Bold"


def build_styles():
    ss = getSampleStyleSheet()
    regular, bold = register_fonts()
    base = dict(fontName=regular, leading=13.5, spaceAfter=6)
    return {
        "title": ParagraphStyle(
            "t", ss["Title"], fontName=bold, fontSize=19,
            leading=24, spaceAfter=10, alignment=0,
        ),
        "h2": ParagraphStyle(
            "h2", ss["Normal"], fontName=bold, fontSize=14,
            leading=18, spaceBefore=16, spaceAfter=7,
            textColor=colors.HexColor("#14324f"),
        ),
        "h3": ParagraphStyle(
            "h3", ss["Normal"], fontName=bold, fontSize=11.5,
            leading=15, spaceBefore=12, spaceAfter=5,
            textColor=colors.HexColor("#1f5c8b"),
        ),
        "body": ParagraphStyle(
            "body", ss["Normal"], fontSize=9.6, alignment=TA_JUSTIFY, **base
        ),
        "bullet": ParagraphStyle(
            "bullet", ss["Normal"], fontSize=9.2, leading=12.5,
            fontName=regular, spaceAfter=3,
        ),
    }


def convert(md_path: Path, pdf_path: Path) -> None:
    styles = build_styles()
    lines = md_path.read_text(encoding="utf-8").splitlines()

    story: list = []
    bullets: list[ListItem] = []

    def flush_bullets() -> None:
        if bullets:
            story.append(
                ListFlowable(
                    list(bullets), bulletType="bullet", start="•",
                    leftIndent=12, bulletFontSize=7, spaceAfter=6,
                )
            )
            bullets.clear()

    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            flush_bullets()
            continue
        if line.startswith("- "):
            bullets.append(ListItem(Paragraph(inline(line[2:]), styles["bullet"])))
            continue
        flush_bullets()
        if line.startswith("---"):
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.6, color=RULE))
            story.append(Spacer(1, 4))
        elif line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), styles["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), styles["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), styles["title"]))
        else:
            story.append(Paragraph(inline(line), styles["body"]))
    flush_bullets()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(styles["body"].fontName, 7.5)
        canvas.setFillColor(colors.HexColor("#7a8290"))
        canvas.drawString(
            18 * mm, 12 * mm,
            "Project Varuna / FloodTwin — Related Publications & Patents",
        )
        canvas.drawRightString(A4[0] - 18 * mm, 12 * mm, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=18 * mm,
        title="Project Varuna / FloodTwin — Related Publications & Patents",
        author="Ansh Vivek",
    ).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "RELATED_PUBLICATIONS.md"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "RELATED_PUBLICATIONS.pdf"
    convert(src, dst)
    print(f"wrote {dst} ({dst.stat().st_size:,} bytes)")
