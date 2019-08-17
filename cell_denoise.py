"""
Script for improving ocr accuracy on image
1. by removing borders by detecting contours
2. by removing noise via increasing contrast and decreasing brightness
"""

import cv2 as cv
import numpy as np


# The OCR gives correct result for cell3-enhanced (increased contrast and removed borders)
# We'll try to make cell3.jpg like cell3-enhanced.jpeg
def enhance(image, iterations=0):
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Remove borders
    inv = 255 - image
    horizontal_img = inv
    vertical_img = inv

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (100, 1))
    horizontal_img = cv.erode(horizontal_img, kernel, iterations=1)
    horizontal_img = cv.dilate(horizontal_img, kernel, iterations=50)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 100))
    vertical_img = cv.erode(vertical_img, kernel, iterations=1)
    vertical_img = cv.dilate(vertical_img, kernel, iterations=50)

    mask_img = np.bitwise_or(vertical_img, horizontal_img)

    no_border = np.bitwise_or(image, mask_img)
    for i in range(iterations):
        no_border = np.bitwise_or(no_border, mask_img)

    # increase contrast
    alpha = 3.0
    beta = -57
    cb = cv.convertScaleAbs(no_border, alpha=alpha, beta=beta)

    # apply gaussian adaptive thresholding to thicken the text
    adt = cv.adaptiveThreshold(cb, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 999, 2)

    return adt


if __name__ == '__main__':
    cell = cv.imread('data/table-cells/cell5.jpg')
    enhanced = enhance(cell)

    cv.imwrite('out/cell_enhanced.jpg', enhanced)

    import pytesseract

    text1 = pytesseract.image_to_string(cell)
    text2 = pytesseract.image_to_string(enhanced)

    print(text1, text2, sep=', ')
