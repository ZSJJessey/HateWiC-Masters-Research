"""Build the English chapter stage document from its Markdown source.

Run with the bundled document Python runtime. This does not touch the thesis.
"""
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

BASE = Path(__file__).resolve().parent


def inline(paragraph, text):
    for i, piece in enumerate(re.split(r"(`[^`]+`)", text)):
        run = paragraph.add_run(piece[1:-1] if i % 2 else piece)
        if i % 2:
            run.font.name = "Courier New"
            run.font.size = Pt(9)


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(.75)
    section.left_margin = section.right_margin = Inches(.85)
    section.page_width, section.page_height = Inches(8.27), Inches(11.69)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.12
    for name, size in [("Title", 20), ("Heading 1", 14), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(6)
        borders = OxmlElement("w:pBdr")
        bottom = OxmlElement("w:bottom")
        bottom.set(qn("w:val"), "nil")
        borders.append(bottom)
        style.element.get_or_add_pPr().append(borders)

    lines = (BASE / "chapter4.md").read_text().splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        picture = re.fullmatch(r"!\[(.*?)\]\((.*?)\)", line)
        if picture:
            doc.add_picture(str(BASE / picture.group(2)), width=Inches(6.45))
            doc.paragraphs[-1].paragraph_format.keep_with_next = True
            caption = doc.add_paragraph(picture.group(1))
            for run in caption.runs:
                run.font.size = Pt(9)
                run.italic = True
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                    rows.append(cells)
                i += 1
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Table Grid"
            table.autofit = False
            widths = [2.9, 3.6] if len(rows[0]) == 2 else [2.7, 1.9, 1.9]
            for column, width in zip(table.columns, widths):
                column.width = Inches(width)
            for j, row in enumerate(rows):
                cells = table.add_row().cells
                for cell, value, width in zip(cells, row, widths):
                    cell.width = Inches(width)
                    inline(cell.paragraphs[0], value)
                    cell.paragraphs[0].paragraph_format.space_before = Pt(2)
                    cell.paragraphs[0].paragraph_format.space_after = Pt(2)
                    for run in cell.paragraphs[0].runs:
                        run.font.size = Pt(10)
                        run.bold = j == 0
                trpr = table.rows[-1]._tr.get_or_add_trPr()
                prevent = OxmlElement("w:cantSplit")
                trpr.append(prevent)
                if j == 0:
                    trpr.append(OxmlElement("w:tblHeader"))
            continue
        if line.startswith("# "):
            doc.add_paragraph(line[2:], style="Title")
        elif line.startswith("## "):
            doc.add_paragraph(line[3:], style="Heading 1")
        elif line.startswith("### "):
            doc.add_paragraph(line[4:], style="Heading 2")
        else:
            p = doc.add_paragraph()
            inline(p, line)
            p.paragraph_format.widow_control = True
        i += 1
    target = BASE / "HateWiC_Chapter4_Stage_Draft.docx"
    doc.core_properties.title = "HateWiC Chapter 4 Results and Discussion"
    doc.core_properties.subject = "Formal evaluation stage chapter"
    doc.core_properties.author = "Shijie Zhou"
    doc.save(target)
    print(target)


if __name__ == "__main__":
    build()
