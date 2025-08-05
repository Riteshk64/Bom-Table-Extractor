import cv2
import numpy as np
import os
import easyocr  # NEW
import re

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
        debug_path = os.path.join("tables_output", "filtered_tables_debug.jpg")
        os.makedirs("tables_output", exist_ok=True)
        cv2.imwrite(debug_path, image)
        print(f"[INFO] Saved debug image with detected tables to: {debug_path}")

    return table_boxes

def preprocess_for_ocr(cropped_img):
    gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

    # Resize to boost clarity
    scaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Remove noise with bilateral filter
    denoised = cv2.bilateralFilter(scaled, 9, 75, 75)

    # Apply threshold to make characters solid
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to connect broken characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return closed

def apply_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

def clean_and_group_bom_text(raw_text):
    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line = re.sub(r'[^\x00-\x7F]+', '', line)
        line = re.sub(r'\s{2,}', ' ', line)
        line = re.sub(r'[^A-Za-z0-9 \-()&.,]', '', line)
        cleaned_lines.append(line)

    grouped_rows = []
    temp_row = []
    for line in cleaned_lines:
        temp_row.append(line)
        if len(temp_row) >= 2 and len(" ".join(temp_row)) > 25:
            grouped_rows.append(temp_row)
            temp_row = []

    if temp_row:
        grouped_rows.append(temp_row)

    return grouped_rows

def ocr_tables_easyocr(image_path, table_boxes, save_dir="tables_output"):
    os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread(image_path)
    reader = easyocr.Reader(['en'], model_storage_directory=r'C:\Users\121807\.EasyOCR', download_enabled=False)
    table_texts = []

    for i, (x, y, w, h) in enumerate(table_boxes):
        cropped = image[y:y+h, x:x+w]

        if cropped.shape[1] < 800:
            cropped = cv2.resize(cropped, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        enhanced = apply_clahe(cropped)
        results = reader.readtext(enhanced, detail=0, paragraph=False)
        text = "\n".join(results)
        table_texts.append((i, text))

        cv2.imwrite(os.path.join(save_dir, f"table_{i}.png"), cropped)
        with open(os.path.join(save_dir, f"table_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

        # Log cleaned rows to console
        print(f"\n[DEBUG] Cleaned rows for table {i}:")
        rows = clean_and_group_bom_text(text)
        for idx, row in enumerate(rows):
            print(f"Row {idx+1}: {' | '.join(row)}")

    return table_texts

# ======== MAIN ========
image_path = "images/d3.jpg"

boxes = preprocess_and_detect_tables(image_path)
print("Detected", len(boxes), "tables.")
ocr_results = ocr_tables_easyocr(image_path, boxes)