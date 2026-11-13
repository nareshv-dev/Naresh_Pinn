import pypdfium2 as pdfium
from pathlib import Path

source = Path(r"C:\Users\naresh v\Downloads\naresh _v1.pdf")
out = Path(r"E:\Naresh_Projects\PINN_EV_battery\output\similarity_report_pages")
out.mkdir(parents=True, exist_ok=True)
doc = pdfium.PdfDocument(str(source))
for page_number in range(5, 13):
    image = doc[page_number - 1].render(scale=1.6).to_pil().convert("RGB")
    image.save(out / f"page_{page_number}.png")
print(out)
