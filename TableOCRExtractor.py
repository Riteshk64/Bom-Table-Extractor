import cv2
import numpy as np
import pytesseract
import pandas as pd
import re
import os

class TableOCRExtractor:
    def __init__(self, table_image, tesseract_path=None):
        # table_image is a NumPy array now
        self.table_image = table_image
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        self.df = None

    def execute(self):
        self.load_and_preprocess_table()
        self.detect_rows_and_columns()
        self.detect_cells_and_ocr()
        self.build_dataframe()
        self.clean_dataframe()
        return self.df

    def load_and_preprocess_table(self):
        self.table = cv2.copyMakeBorder(self.table_image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255,255,255])
        self.table_gray = cv2.GaussianBlur(cv2.cvtColor(self.table, cv2.COLOR_BGR2GRAY), (3,3), 0, 0)
        _, self.thre = cv2.threshold(self.table_gray, 200, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)

    # === Step 2: Detect rows & columns ===
    def detect_rows_and_columns(self):
        kernel_row = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
        morph_row = cv2.morphologyEx(self.thre, cv2.MORPH_CLOSE, kernel_row)
        contours, _ = cv2.findContours(morph_row, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.rows = sorted([cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours], key=lambda b: b[1])

        kernel_col = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
        morph_col = cv2.morphologyEx(self.thre, cv2.MORPH_CLOSE, kernel_col)
        contours, _ = cv2.findContours(morph_col, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.cols = sorted([cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours], key=lambda b: b[0])

    # === Step 3: Detect cells and OCR ===
    def detect_cells_and_ocr(self):
        _, thre2 = cv2.threshold(self.thre, 0, 255, cv2.THRESH_BINARY_INV)
        no_table = cv2.bitwise_and(cv2.morphologyEx(self.thre, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))), thre2)
        no_table = cv2.bitwise_and(cv2.morphologyEx(self.thre, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))), no_table)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,2))
        mask = cv2.morphologyEx(no_table, cv2.MORPH_CLOSE, kernel2)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        boundRect = [cv2.boundingRect(cv2.approxPolyDP(c, 3, True)) for c in contours]

        self.text_position = []
        offset = 2

        for rect in boundRect:
            if rect[2] > 1 and rect[3] > 5:
                cell_img = self.table[rect[1]-offset:rect[1]+rect[3]+offset, rect[0]-offset:rect[0]+rect[2]+offset]
                gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
                thresh_cell = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                text = pytesseract.image_to_string(thresh_cell, lang='eng', config='--psm 7').strip()

                r, c = -1, -1
                for i in range(len(self.rows)-1):
                    if self.rows[i][1] < rect[1] < self.rows[i+1][1]:
                        r = i
                        break
                for i in range(len(self.cols)-1):
                    if self.cols[i][0] < rect[0] < self.cols[i+1][0]:
                        c = i
                        break

                if r != -1 and c != -1:
                    self.text_position.append({'Text': text.split("\n")[0], "row": r, 'col': c})

        self.text_position = sorted(self.text_position, key=lambda x: (x['row'], x['col'], x['Text']))

    # === Step 4: Build raw dataframe ===
    def build_dataframe(self):
        merged = {}
        for t in self.text_position:
            key = (t['row'], t['col'])
            merged[key] = merged.get(key, "") + (" " if key in merged else "") + t['Text']

        max_row = max(r for r, _ in merged.keys())
        max_col = max(c for _, c in merged.keys())
        table_data = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]

        for (r, c), text in merged.items():
            table_data[r][c] = text

        self.df = pd.DataFrame(table_data)

    # === Step 5: Clean dataframe ===
    def clean_dataframe(self):
        def clean_id(val):
            if pd.isna(val): return val
            val = str(val).strip().replace('O', '0').replace('o', '0').replace('Q', '0').replace('l', '1')
            return val

        def clean_text(val):
            if pd.isna(val): return val
            val = str(val).strip()
            val = re.sub(r'\s+', ' ', val)
            val = re.sub(r'[§|]', '', val)
            return val

        def clean_quantity(val):
            if pd.isna(val): return val
            val = str(val).strip()
            val = val.replace('O', '0').replace('o', '0')
            val = val.replace('l', '1').replace('L', '1')
            val = val.replace('z', '2').replace('Z', '2')
            val = re.sub(r'[^\d]', '', val)
            return val if val else np.nan

        self.df = self.df.replace(r'^\s*$', np.nan, regex=True).infer_objects(copy=False)
        self.df = self.df.dropna(how='all').dropna(thresh=2).reset_index(drop=True)

        if self.df.iloc[0].isnull().all() or all(str(x).strip() == '' for x in self.df.iloc[0]):
            self.df = self.df.iloc[1:].reset_index(drop=True)

        self.df.iloc[:, 0] = self.df.iloc[:, 0].map(clean_text)
        self.df.iloc[:, 1] = self.df.iloc[:, 1].map(clean_id)
        self.df.iloc[:, 2:] = self.df.iloc[:, 2:].map(clean_text)

        self.df = self.df.dropna(axis=1, how='all')
        self.df.columns = self.df.iloc[0]
        self.df = self.df[1:].reset_index(drop=True)

        if 'QUANTITY' in self.df.columns:
            self.df['QUANTITY'] = self.df['QUANTITY'].map(clean_quantity)

        cols_to_clean = [c for c in ['ITEM', 'QUANTITY', 'Equivalent OK.'] if c in self.df.columns]
        self.df[cols_to_clean] = self.df[cols_to_clean].apply(lambda col: col.str.replace('.', '', regex=False))
        self.df[cols_to_clean] = self.df[cols_to_clean].apply(lambda col: col.str.replace(',', '', regex=False))
