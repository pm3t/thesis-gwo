import docx

doc = docx.Document(r"c:\GWO\Journal2.docx")

for i, p in enumerate(doc.paragraphs):
    images = []
    for r in p.runs:
        if 'drawing' in r._element.xml:
            images.append("DRAWING")
    if images or "Gambar" in p.text:
        print(f"P{i}: text='{p.text}' | drawings={images}")
