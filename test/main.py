import os
import cv2
from img2table.document import Image
from img2table.ocr import SuryaOCR

# --- Configure Tesseract ---
tess_path = r"C:\Users\121807\Documents"
os.environ["PATH"] = tess_path + os.pathsep + os.environ.get("PATH", "")

# Initialize OCR
ocr = SuryaOCR(langs=["en"])

# --- Preprocess image ---
img_path = "../images/input_images/coupling.png"
img = cv2.imread(img_path)

# Upscale
img = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

# Grayscale + threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Save preprocessed image
proc_path = "preprocessed_table.png"
cv2.imwrite(proc_path, thresh)

# --- Table extraction ---
doc = Image(src=proc_path)

tables = doc.extract_tables(
    ocr=ocr,
    implicit_rows=True,
    borderless_tables=True
)

if tables:
    df = tables[0].df
    print(df)
    df.to_csv("output_table.csv", index=False)
    print("✅ Table extracted and saved as output_table.csv")
else:
    print("⚠️ No tables detected. Try tweaking preprocessing or PSM mode.")
