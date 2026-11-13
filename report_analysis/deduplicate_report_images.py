from pathlib import Path
from hashlib import sha256
from PIL import Image, ImageDraw

root = Path(r"E:\Naresh_Projects\PINN_EV_battery\report_analysis\extracted_images")
out = Path(r"E:\Naresh_Projects\PINN_EV_battery\report_analysis\unique_images")
out.mkdir(exist_ok=True)
unique = {}
for path in root.glob("*.png"):
    digest = sha256(path.read_bytes()).hexdigest()
    unique.setdefault(digest, path)
tiles = []
for i, path in enumerate(unique.values()):
    image = Image.open(path).convert("RGB")
    target = out / f"unique_{i:02d}_{path.name}"
    image.save(target)
    image.thumbnail((420, 260))
    tile = Image.new("RGB", (440, 300), "white")
    tile.paste(image, ((440-image.width)//2, 30))
    ImageDraw.Draw(tile).text((10, 8), target.name, fill="black")
    tiles.append(tile)
cols = 3
rows = (len(tiles)+cols-1)//cols
sheet = Image.new("RGB", (cols*440, rows*300), "#888888")
for i, tile in enumerate(tiles):
    sheet.paste(tile, ((i%cols)*440, (i//cols)*300))
sheet.save(out / "contact_sheet.png")
print("unique", len(unique), out / "contact_sheet.png")
