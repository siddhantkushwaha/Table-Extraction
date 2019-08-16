# %%

import pytesseract

from PIL import Image
import numpy as np
import cv2 as cv

import utils
from table import Table

# %%


ext_img = Image.open('data/example.jpg')
ext_img.save("out/target.jpg", "JPEG")

image = cv.imread("out/target.jpg")

# convert to grayscale
grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# %%

# ADAPTIVE THRESHOLDING
# Thresholding changes pixels' color values to a specified pixel value if the current pixel value
# is less than a threshold value, which could be:
#
# 1. a specified global threshold value provided as an argument to the threshold function (simple thresholding),
# 2. the mean value of the pixels in the neighboring area (adaptive thresholding - mean method), 3. the weighted sum
# of neighborhood values where the weights are Gaussian windows (adaptive thresholding - Gaussian method).
#
# The last two parameters to the adaptiveThreshold function are the size of the neighboring area and
# the constant C which is subtracted from the mean or weighted mean calculated.

MAX_THRESHOLD_VALUE = 255
BLOCK_SIZE = 15
THRESHOLD_CONSTANT = 0

# Filter image
filtered = cv.adaptiveThreshold(~grayscale, MAX_THRESHOLD_VALUE, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                BLOCK_SIZE, THRESHOLD_CONSTANT)

cv.imwrite('out/filtered.jpg', filtered)

# %%

# HORIZONTAL AND VERTICAL LINE ISOLATION
# To isolate the vertical and horizontal lines,
#
# 1. Set a scale.
# 2. Create a structuring element.
# 3. Isolate the lines by eroding and then dilating the image.

SCALE = 15

# Isolate horizontal and vertical lines using morphological operations
horizontal = filtered.copy()
vertical = filtered.copy()

horizontal_size = int(horizontal.shape[1] / SCALE)
horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
utils.isolate_lines(horizontal, horizontal_structure)
#
vertical_size = int(vertical.shape[0] / SCALE)
vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
utils.isolate_lines(vertical, vertical_structure)

cv.imwrite('out/horizontal.jpg', horizontal)
cv.imwrite('out/vertical.jpg', vertical)

# %%

# TABLE EXTRACTION
# Create an image mask with just the horizontal
# and vertical lines in the image. Then find
# all contours in the mask.

mask = horizontal + vertical
cv.imwrite('out/mask.jpg', mask)

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Find intersections between the lines
# to determine if the intersections are table joints.
intersections = cv.bitwise_and(horizontal, vertical)
cv.imwrite('out/intersections.jpg', intersections)

# %%

tables = []
for i, contour in enumerate(contours):

    # verify that Region of Interest (ROI) is a table
    rect, table_joints = utils.verify_table(contour, intersections)
    if rect is None or table_joints is None:
        continue

    # create an object for table
    table = Table(rect[0], rect[1], rect[2], rect[3])

    # Get an n-dimensional array of the coordinates of the table joints
    joint_coords = []
    for i in range(len(table_joints)):
        joint_coords.append(table_joints[i][0][0])
    joint_coords = np.asarray(joint_coords)

    # Returns indices of coordinates in sorted order
    # Sorts based on parameters (keys) starting from the last parameter, then second-to-last, etc
    sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
    joint_coords = joint_coords[sorted_indices]

    # Store joint coordinates in the table instance
    table.set_joints(joint_coords)

    tables.append(table)

    # write images for detected tables
    cv.rectangle(image, (table.x, table.y), (table.x + table.w, table.y + table.h), (255, 0, 0), 4, 8, 0)
    cv.imwrite(f'out/table-{i}.jpg', image)

# %%

# ZOOM FACTOR
MULT = 3

# Process each table's ROI
for i, table in enumerate(tables):
    # crop the table
    table_roi = image[table.y:table.y + table.h, table.x:table.x + table.w]
    # resize/rescale
    table_roi = cv.resize(table_roi, (table.w * MULT, table.h * MULT))

    cv.imwrite(f'out/table-roi-{i}.jpg', table_roi)

    table_entries = table.get_table_entries()
    for r, row in enumerate(table_entries):
        for c, cell in enumerate(row):
            cell_cropped = table_roi[cell[1] * MULT: (cell[1] + cell[3]) * MULT,
                           cell[0] * MULT:(cell[0] + cell[2]) * MULT]

            cell_name = f'table-cell-{i}-{r}-{c}'

            # applying ocr, results not so great
            # text = pytesseract.image_to_string(cell_cropped, config='--psm 10', lang='eng')
            # print(cell_name, text.encode())

            cv.imwrite(f'out/{cell_name}.jpg', cell_cropped)
