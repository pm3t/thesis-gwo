import os
from PIL import Image

analysis_dir = r"c:\GWO\Analysis"
for f in os.listdir(analysis_dir):
    if f.endswith(".png"):
        path = os.path.join(analysis_dir, f)
        img = Image.open(path)
        print(f"{f}: size={img.size}, mode={img.mode}, filesize={os.path.getsize(path)} bytes")
