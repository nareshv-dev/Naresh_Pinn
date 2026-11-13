import re
from copy import deepcopy

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

SOURCE = r"E:\Naresh_Projects\PINN_EV_battery\PINN_EV_Battery_Report_formatted.docx"
OUTPUT = SOURCE


def set_cell_borders(cell, color="808080", size="8"):
    """Apply visible single-line borders to all four sides of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge_name in ("top", "left", "bottom", "right"):
        edge = borders.find(qn(f"w:{edge_name}"))
        if edge is None:
            edge = OxmlElement(f"w:{edge_name}")
            borders.append(edge)
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), size)
        edge.set(qn("w:space"), "0")
        edge.set(qn("w:color"), color)


def first_page_number(value):
    """Keep the first displayed page number while retaining roman numerals."""
    return re.split(r"[\u2013\u2014\-]", value.strip(), maxsplit=1)[0].strip()


def add_serial_number_column(table):
    """Insert a leftmost S.No. column without altering existing cell content."""
    new_column = table.add_column(table.columns[0].width)
    for row_index, row in enumerate(table.rows):
        added_cell = row.cells[-1]
        added_cell._tc.getparent().remove(added_cell._tc)
        row._tr.insert(0, added_cell._tc)
        paragraph = added_cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run("S.No." if row_index == 0 else str(row_index))
        if row_index == 0:
            run.bold = True
        added_cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

    # Retain only the first page in each table-of-contents entry.
    for row in table.rows[1:]:
        row.cells[-1].text = first_page_number(row.cells[-1].text)
        for paragraph in row.cells[-1].paragraphs:
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER


def force_single_column_before(paragraph):
    """Make the section beginning after this paragraph a standard single-column section."""
    p_pr = paragraph._p.get_or_add_pPr()
    section_props = p_pr.sectPr
    if section_props is None:
        section_props = OxmlElement("w:sectPr")
        p_pr.append(section_props)
    columns = section_props.find(qn("w:cols"))
    if columns is None:
        columns = OxmlElement("w:cols")
        section_props.append(columns)
    columns.set(qn("w:num"), "1")
    columns.set(qn("w:space"), "720")
    for child in list(columns):
        columns.remove(child)


document = Document(SOURCE)

# The report's first table is the table of contents.
add_serial_number_column(document.tables[0])

# Give every existing Word table visible, structured grid borders.
for table in document.tables:
    for row in table.rows:
        for cell in row.cells:
            set_cell_borders(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

# The empty paragraph immediately before the reference entries owns their section settings.
reference_heading_index = next(
    index for index, paragraph in enumerate(document.paragraphs)
    if paragraph.text.strip().upper() == "REFERENCES"
)
force_single_column_before(document.paragraphs[reference_heading_index - 1])

document.save(OUTPUT)
