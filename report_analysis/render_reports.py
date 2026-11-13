from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw

files = [
    Path(r"C:\Users\naresh v\Downloads\Physics_Informed_Neural_Networks_EV_Battery_Phase_I_Report.pdf"),
    Path(r"C:\Users\naresh v\Downloads\NutriGuard_AI_Phase_I_FINAL_SUBMISSION_READY_FOR_SIGNATURES.pdf"),
]
root = Path(r"E:\Naresh_Projects\PINN_EV_battery\report_analysis\renders")
for source in files:
    doc = pdfium.PdfDocument(str(source))
    out = root / source.stem
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in range(len(doc)):
        page = doc[i].render(scale=1.0).to_pil().convert("RGB")
        page.save(out / f"page_{i+1:02d}.png")
        page.thumbnail((360, 510))
        tile = Image.new("RGB", (380, 550), "white")
        tile.paste(page, ((380-page.width)//2, 25))
        ImageDraw.Draw(tile).text((10, 5), f"Page {i+1}", fill="black")
        tiles.append(tile)
    cols = 4
    rows = (len(tiles)+cols-1)//cols
    sheet = Image.new("RGB", (cols*380, rows*550), "#888888")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % cols)*380, (i//cols)*550))
    sheet.save(out / "contact_sheet.png")
    print(source.name, len(doc), out / "contact_sheet.png")
