import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pathlib import Path

pdf_path = Path(r"E:\Naresh_Projects\PINN_EV_battery\output\originality_revision\battery_pinn_ieee_conference_originality_revision.pdf")
doc = pdfium.PdfDocument(str(pdf_path))
print("pages", len(doc), "size_bytes", pdf_path.stat().st_size)
out = pdf_path.parent / "rendered"
out.mkdir(exist_ok=True)
tiles = []
for i in range(len(doc)):
    page = doc[i].render(scale=1.25).to_pil().convert("RGB")
    page.save(out / f"page_{i+1}.png")
    page.thumbnail((510, 660))
    tile = Image.new("RGB", (530, 700), "white")
    tile.paste(page, ((530-page.width)//2, 25))
    ImageDraw.Draw(tile).text((15, 5), f"Page {i+1}", fill="black")
    tiles.append(tile)
rows = (len(tiles)+1)//2
sheet = Image.new("RGB", (1060, rows*700), "#777777")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i % 2)*530, (i//2)*700))
sheet.save(out / "contact_sheet.png")
print(out / "contact_sheet.png")
