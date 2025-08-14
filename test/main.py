import TableExtractor as te
import TableOCRExtractor as toe
import cv2

table_extractor = te.TableExtractor("../images/input_images/d1.jpg")
cropped_img = table_extractor.execute()

cv2.imwrite("temp_cropped_table.jpg", cropped_img)

ocr_extractor = toe.TableOCRExtractor("temp_cropped_table.jpg", tesseract_path=r"C:\Users\121807\Documents\tesseract.exe")
df = ocr_extractor.execute()

print(df)