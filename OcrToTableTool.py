import cv2
import numpy as np
import subprocess
import re

class OcrToTableTool:

    def __init__(self, image, original_image):
        self.thresholded_image = image
        self.original_image = original_image

    def execute(self):
        self.clean_noise()
        self.dilate_image()
        self.store_process_image('0_dilated_image.jpg', self.dilated_image)
        self.find_contours()
        self.store_process_image('1_contours.jpg', self.image_with_contours_drawn)
        self.convert_contours_to_bounding_boxes()
        self.store_process_image('2_bounding_boxes.jpg', self.image_with_all_bounding_boxes)
        self.mean_height = self.get_mean_height_of_bounding_boxes()
        self.sort_bounding_boxes_by_y_coordinate()
        self.club_all_bounding_boxes_by_similar_y_coordinates_into_rows()
        self.sort_all_rows_by_x_coordinate()
        self.crop_each_bounding_box_and_ocr()
        self.generate_csv_file()

    # def threshold_image(self):
    #     return cv2.threshold(self.grey_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    def threshold_image(self):
        return cv2.adaptiveThreshold(
        self.grey, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21, 5
    )

    def clean_noise(self):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        self.thresholded_image = cv2.morphologyEx(self.thresholded_image, cv2.MORPH_OPEN, kernel)

    def convert_image_to_grayscale(self):
        return cv2.cvtColor(self.image, self.dilated_image)

    def dilate_image(self):
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        self.dilated_image = cv2.dilate(self.thresholded_image, horizontal_kernel, iterations=1)
    
    def find_contours(self):
        result = cv2.findContours(self.dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = result[0]
        self.image_with_contours_drawn = self.original_image.copy()
        cv2.drawContours(self.image_with_contours_drawn, self.contours, -1, (0, 255, 0), 3)
    
    def approximate_contours(self):
        self.approximated_contours = []
        for contour in self.contours:
            approx = cv2.approxPolyDP(contour, 3, True)
            self.approximated_contours.append(approx)

    def draw_contours(self):
        self.image_with_contours = self.original_image.copy()
        cv2.drawContours(self.image_with_contours, self.approximated_contours, -1, (0, 255, 0), 5)

    def convert_contours_to_bounding_boxes(self):
        self.bounding_boxes = []
        self.image_with_all_bounding_boxes = self.original_image.copy()
        
        for contour in self.contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
            if w * h < 100:
                continue

            self.bounding_boxes.append((x, y, w, h))
            self.image_with_all_bounding_boxes = cv2.rectangle(
                self.image_with_all_bounding_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2
            )


    def get_mean_height_of_bounding_boxes(self):
        heights = []
        for bounding_box in self.bounding_boxes:
            x, y, w, h = bounding_box
            heights.append(h)
        return np.mean(heights)

    def sort_bounding_boxes_by_y_coordinate(self):
        self.bounding_boxes = sorted(self.bounding_boxes, key=lambda x: x[1])

    def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(self):
        self.rows = []
        half_of_mean_height = self.mean_height / 2
        current_row = [ self.bounding_boxes[0] ]
        for bounding_box in self.bounding_boxes[1:]:
            current_bounding_box_y = bounding_box[1]
            previous_bounding_box_y = current_row[-1][1]
            distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)
            if distance_between_bounding_boxes <= half_of_mean_height:
                current_row.append(bounding_box)
            else:
                self.rows.append(current_row)
                current_row = [ bounding_box ]
        self.rows.append(current_row)

    def sort_all_rows_by_x_coordinate(self):
        for row in self.rows:
            row.sort(key=lambda x: x[0])

    def crop_each_bounding_box_and_ocr(self):
        self.table = []
        current_row = []
        image_number = 0
        for row in self.rows:
            for bounding_box in row:
                x, y, w, h = bounding_box
                padding = 3
                x = max(x - padding, 0)
                y = max(y - padding, 0)
                w = w + 2 * padding
                h = h + 2 * padding

                # padding_left = 4
                # padding_top = 3
                # padding_right = 3
                # padding_bottom = 3

                # x = max(x - padding_left, 0)
                # y = max(y - padding_top, 0)
                # w = w + padding_left + padding_right
                # h = h + padding_top + padding_bottom

                cropped_image = self.original_image[y:y+h, x:x+w]

                gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (3, 3), 0)
                enhanced = cv2.adaptiveThreshold(
                    blurred, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    15, 2
                )

                image_slice_path = "./ocr_slices/img_" + str(image_number) + ".jpg"
                cv2.imwrite(image_slice_path, enhanced)
                results_from_ocr = self.get_result_from_tersseract(image_slice_path)
                results_from_ocr = self.clean_text(results_from_ocr)
                results_from_ocr = self.fix_spacing(results_from_ocr)

                current_row.append(results_from_ocr)
                image_number += 1
            self.table.append(current_row)
            current_row = []
    
    def clean_text(self, text):
        if text.count(',') > 2 and not any(c.isalpha() for c in text):
            text = text.replace(',', '.')
        else:
            text = text.replace(',', ';')
        return text.strip()

    def fix_spacing(self, text):
        return re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    def get_result_from_tersseract(self, image_path):
        tesseract_path = r'"C:\Users\121807\Documents\tesseract.exe"'
        command = f'{tesseract_path} "{image_path}" - -l eng --oem 3 --psm 6 --dpi 72 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()-_.*%/+:, "'
        output = subprocess.getoutput(command)
        return output.strip()

    def generate_csv_file(self):
        with open("output.csv", "w") as f:
            for row in self.table:
                f.write(",".join(row) + "\n")

    def store_process_image(self, file_name, image):
        path = "./process_images/ocr_table_tool/" + file_name
        cv2.imwrite(path, image)