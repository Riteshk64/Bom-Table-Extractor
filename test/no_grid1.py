import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd =r"C:\Users\121807\Documents\tesseract.exe"

image = cv2.imread(r'..\images\input_images\coupling.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (55,2))
detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detect_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image, [c], -1, (255,255,255), 9)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,55))
detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
cnts = cv2.findContours(detect_vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(image, [c], -1, (255,255,255), 9)

data = pytesseract.image_to_string(image, lang='eng',config='--psm 6')
print(data)

cv2.imshow('image', image)
cv2.waitKey()