from PIL import Image
import cv2 as cv
import pandas as pd

from extract import extract
from ocr import ocr

if __name__ == '__main__':
    ext_img = Image.open('data/example2.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)

    for table in tables:
        ocr_data = ocr(table.image)

        # TODO insert words into right cells based on coordinates
