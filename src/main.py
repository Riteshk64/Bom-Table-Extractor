import TableExtractor as te
import TableOCRExtractor as toe
import cv2

table_extractor = te.TableExtractor("./images/d1.jpg")
cropped_img = table_extractor.execute()

# Step 2: OCR
ocr_extractor = toe.TableOCRExtractor("temp_cropped_table.jpg", tesseract_path=r"C:\Users\121807\Documents\tesseract.exe")
df = ocr_extractor.execute()

print(df)