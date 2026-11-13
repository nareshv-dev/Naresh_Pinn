import zipfile
from docx import Document
from docx.oxml.ns import qn

PATH = r"E:\Naresh_Projects\PINN_EV_battery\PINN_EV_Battery_Report_formatted.docx"

with zipfile.ZipFile(PATH) as archive:
    assert archive.testzip() is None, "Corrupt DOCX package"

document = Document(PATH)
toc = document.tables[0]
assert len(toc.columns) == 3
assert [cell.text for cell in toc.rows[0].cells] == ["S.No.", "TITLE", "PAGE"]
for index, row in enumerate(toc.rows[1:], start=1):
    assert row.cells[0].text == str(index)
    assert "–" not in row.cells[-1].text and "—" not in row.cells[-1].text

for table in document.tables:
    for row in table.rows:
        for cell in row.cells:
            borders = cell._tc.tcPr.first_child_found_in("w:tcBorders")
            assert borders is not None
            for side in ("top", "left", "bottom", "right"):
                border = borders.find(qn(f"w:{side}"))
                assert border is not None and border.get(qn("w:val")) == "single"

reference_index = next(
    index for index, paragraph in enumerate(document.paragraphs)
    if paragraph.text.strip().upper() == "REFERENCES"
)
props = document.paragraphs[reference_index - 1]._p.pPr.sectPr
columns = props.find(qn("w:cols"))
assert columns is not None and columns.get(qn("w:num")) == "1"

print("DOCX package and requested formatting checks passed.")
