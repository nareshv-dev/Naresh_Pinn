from pathlib import Path
from pypdf import PdfReader

source = Path(r"C:\Users\naresh v\Downloads\Physics_Informed_Neural_Networks_EV_Battery_Phase_I_Report.pdf")
reader = PdfReader(str(source))
out = Path(r"E:\Naresh_Projects\PINN_EV_battery\report_analysis\extracted_images")
out.mkdir(parents=True, exist_ok=True)
for page_index, page in enumerate(reader.pages):
    print("PAGE", page_index + 1)
    for image_index, image_file in enumerate(page.images):
        image = image_file.image
        print(" ", image_file.name, image.size, image.mode)
        if image.width >= 300 and image.height >= 150:
            extension = "png"
            image.convert("RGB").save(out / f"page_{page_index+1:02d}_image_{image_index:02d}.{extension}")
print(out)
