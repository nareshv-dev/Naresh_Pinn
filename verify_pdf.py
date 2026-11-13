import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pathlib import Path

pdf_path = Path(r"E:\Naresh_Projects\PINN_EV_battery\output\pdf\battery_pinn_ieee_conference_table10_neat.pdf")
doc = pdfium.PdfDocument(str(pdf_path))
print("pages", len(doc), "size_bytes", pdf_path.stat().st_size)

output_dir = pdf_path.parent / "rendered"
output_dir.mkdir(exist_ok=True)
thumbnails = []

for index in range(len(doc)):
    image = doc[index].render(scale=1.25).to_pil().convert("RGB")
    image.save(output_dir / f"page_{index + 1}.png")
    thumb = image.copy()
    thumb.thumbnail((510, 660))
    canvas = Image.new("RGB", (530, 700), "white")
    canvas.paste(thumb, ((530 - thumb.width) // 2, 25))
    ImageDraw.Draw(canvas).text((15, 5), f"Page {index + 1}", fill="black")
    thumbnails.append(canvas)

rows = (len(thumbnails) + 1) // 2
sheet = Image.new("RGB", (1060, rows * 700), "#777777")
for index, thumbnail in enumerate(thumbnails):
    sheet.paste(thumbnail, ((index % 2) * 530, (index // 2) * 700))

sheet_path = output_dir / "contact_sheet.png"
sheet.save(sheet_path)
print(sheet_path)
