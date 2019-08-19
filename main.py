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

        cells = table.get_cells()
        ocr_data = ocr(table.image)

        for i, data in ocr_data.iterrows():
            from utils import get_centroid

            centroid = get_centroid(data['left'], data['left'] + data['width'], data['top'],
                                    data['top'] + data['height'])
            print(centroid)

            # TODO insert words into right cells based on coordinates
