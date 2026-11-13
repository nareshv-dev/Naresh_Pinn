from docx import Document
from docx.oxml.ns import qn

path = r"E:\Naresh_Projects\PINN_EV_battery\PINN_EV_Battery_Report_formatted.docx"
document = Document(path)

for index, paragraph in enumerate(document.paragraphs):
    props = paragraph._p.pPr
    section_props = props.sectPr if props is not None else None
    if section_props is not None:
        columns = section_props.find(qn("w:cols"))
        count = columns.get(qn("w:num")) if columns is not None else "1"
        print("section break after", index, "columns", count, "text", repr(paragraph.text[:80]))

print("near refs:")
for index in range(575, 615):
    paragraph = document.paragraphs[index]
    print(index, repr(paragraph.text[:200]), paragraph.style.name)
