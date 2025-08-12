import cv2 
import numpy as np
import pytesseract
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\121807\Documents\tesseract.exe"

# ===== Load image =====
table = cv2.imread(r"process_images\table_extractor\9_cropped_table.jpg")
table = cv2.copyMakeBorder(table, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
table_c = cv2.GaussianBlur(cv2.cvtColor(table, cv2.COLOR_BGR2GRAY), (3,3), 0, 0)

_, thre = cv2.threshold(table_c, 200, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

# ===== Detect rows =====
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
morph = cv2.morphologyEx(thre, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(morph, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
rows = [cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours]
rows = sorted(rows, key=lambda b: b[1])

# ===== Detect columns =====
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
morph2 = cv2.morphologyEx(thre, cv2.MORPH_CLOSE, kernel2)
contours, _ = cv2.findContours(morph2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cols = [cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours]
cols = sorted(cols, key=lambda b: b[0])

# ===== Detect cell bounding boxes =====
_, thre2 = cv2.threshold(thre, 0, 255, cv2.THRESH_BINARY_INV)
no_table = cv2.bitwise_and(morph, thre2)
no_table = cv2.bitwise_and(morph2, no_table)

kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2))
mask = cv2.morphologyEx(no_table, cv2.MORPH_CLOSE, kernel2)
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundRect = [cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours]

# ===== OCR each cell (first code’s preprocessing) =====
text_position = []
offset = 2  # smaller padding

for rect in boundRect:
    if rect[2] > 1 and rect[3] > 5:
        # Crop cell
        cell_img = table[rect[1]-offset:rect[1]+rect[3]+offset, rect[0]-offset:rect[0]+rect[2]+offset]
        
        # === First code's preprocessing ===
        gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
        thresh_cell = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        # OCR with higher accuracy
        text = pytesseract.image_to_string(thresh_cell, lang='eng', config='--psm 7').strip()

        # Identify row and column index
        r = c = -1
        for i in range(len(rows)-1):
            if rows[i][1] < rect[1] < rows[i+1][1]:
                r = i
                break
        for i in range(len(cols)-1):
            if cols[i][0] < rect[0] < cols[i+1][0]:
                c = i
                break

        if r != -1 and c != -1:
            text_position.append({'Text': text.split("\n")[0], "row": r, 'col': c})
        
        
        cv2.rectangle(table, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[3]), (0, 255, 0), 1)
        cv2.putText(table, f"{rect[0]},{rect[1]}", (rect[0], rect[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)

cv2.imwrite("test_outputs/debug_cells.jpg", table)

# ===== Merge duplicates in same cell =====
merged = {}
for t in text_position:
    key = (t['row'], t['col'])
    if key not in merged:
        merged[key] = t['Text']
    else:
        merged[key] += " " + t['Text']

# ===== Build final table =====
max_row = max(r for r, _ in merged.keys())
max_col = max(c for _, c in merged.keys())
table_data = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

for (r, c), text in merged.items():
    table_data[r][c] = text

df = pd.DataFrame(table_data)

import re

def clean_id(val):
    """Fix OCR mistakes in part/item IDs only."""
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Common OCR misreads
    val = val.replace('O', '0').replace('o', '0')
    val = val.replace('Q', '0').replace('l', '1')
    return val

def clean_text(val):
    """General cleanup for non-ID fields."""
    if pd.isna(val):
        return val
    val = str(val).strip()
    val = re.sub(r'\s+', ' ', val)
    val = re.sub(r'[§|]', '', val)
    return val


def clean_quantity(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    val = val.replace('O', '0').replace('o', '0')
    val = val.replace('l', '1').replace('L', '1')
    val = val.replace('z', '2').replace('Z', '2')
    val = re.sub(r'[^\d]', '', val)  # remove non-digit characters
    return val if val else np.nan

# Convert all empty strings to NaN
df = df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)

# Drop rows that are fully empty
df = df.dropna(how='all')

# Drop rows that have less than 2 non-empty cells (likely junk rows)
df = df.dropna(thresh=2)

# Reset index after cleaning
df = df.reset_index(drop=True)

# If first row is empty or junk, drop it
if df.iloc[0].isnull().all() or all(str(x).strip() == '' for x in df.iloc[0]):
    df = df.iloc[1:].reset_index(drop=True)

# Apply cleaning
df.iloc[:, 0] = df.iloc[:, 0].map(clean_text)
df.iloc[:, 1] = df.iloc[:, 1].map(clean_id)
df.iloc[:, 2:] = df.iloc[:, 2:].map(clean_text)

# Drop columns that are fully NaN
df = df.dropna(axis=1, how='all')

# Use the first row as headers
df.columns = df.iloc[0]
df = df[1:].reset_index(drop=True)

df['QUANTITY'] = df['QUANTITY'].map(clean_quantity)

cols_to_clean = ['ITEM', 'QUANTITY', 'OK. Equivalent']
df[cols_to_clean] = df[cols_to_clean].apply(lambda col: col.str.replace('.', '', regex=False))
df[cols_to_clean] = df[cols_to_clean].apply(lambda col: col.str.replace(',', '', regex=False))

print(df)