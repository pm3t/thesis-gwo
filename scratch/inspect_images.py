import docx

doc = docx.Document(r"c:\GWO\Journal2.docx")

print("Rel count:", len(doc.part.rels))
for rel_id, rel in doc.part.rels.items():
    if "image" in rel.target_ref:
        print(rel_id, rel.target_ref)
