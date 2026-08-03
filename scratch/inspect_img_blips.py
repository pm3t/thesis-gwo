import docx

doc = docx.Document(r"c:\GWO\Journal2.docx")

with open(r"c:\GWO\scratch\img_analysis.txt", "w", encoding="utf-8") as f:
    for i, p in enumerate(doc.paragraphs):
        blips = p._element.xpath('.//a:blip/@r:embed')
        if blips or "Gambar" in p.text:
            f.write(f"P{i}: blips={blips} | text='{p.text}'\n")
