import re
from pathlib import Path
from pypdf import PdfReader

old = Path(r"E:\Naresh_Projects\PINN_EV_battery\output\naresh_v1_extracted.txt").read_text(encoding="utf-8", errors="ignore")
pdf = Path(r"E:\Naresh_Projects\PINN_EV_battery\output\originality_revision\battery_pinn_ieee_conference_originality_revision.pdf")
new = "\n".join((page.extract_text() or "") for page in PdfReader(str(pdf)).pages)
Path(r"E:\Naresh_Projects\PINN_EV_battery\output\originality_revision\revised_text.txt").write_text(new, encoding="utf-8")

def body_words(value: str):
    value = value.split("References", 1)[0]
    return re.findall(r"[a-z0-9]+", value.lower())

old_words = body_words(old[old.find("A Physics-Consistent"):])
new_words = body_words(new)
for n in (5, 8, 12):
    old_ngrams = {tuple(old_words[i:i+n]) for i in range(len(old_words)-n+1)}
    new_ngrams = [tuple(new_words[i:i+n]) for i in range(len(new_words)-n+1)]
    overlap = sum(gram in old_ngrams for gram in new_ngrams)
    print(f"{n}-gram overlap with submitted prose: {100*overlap/max(1,len(new_ngrams)):.2f}%")
print("new body words", len(new_words))
