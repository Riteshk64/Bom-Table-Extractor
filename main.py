import cv2
import numpy as np
import os
import pytesseract

def preprocess_and_detect_tables(image_path, show_debug=True):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(~gray, 255,
                                   cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 15, -10)

    # Horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)

    # Vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    # Combine lines
    table_mask = cv2.add(horizontal_lines, vertical_lines)

    # Find contours
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    table_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w < 100 or h < 100:
            continue  # Skip tiny regions

        aspect_ratio = w / h
        area = w * h

        # Create mask for this region to count lines inside
        roi_mask = table_mask[y:y+h, x:x+w]
        line_pixels = cv2.countNonZero(roi_mask)

        line_density = line_pixels / area

        # Heuristics: reject diagrams and keep only table-looking things
        if 0.4 < aspect_ratio < 10 and line_density > 0.01:
            table_boxes.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    if show_debug:
        cv2.imshow("Filtered Tables", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return table_boxes

image_path = "images/coupling.jpg"

boxes = preprocess_and_detect_tables(image_path)
print("Detected", len(boxes), "tables.")