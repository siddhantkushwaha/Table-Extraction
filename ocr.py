import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 as cv

from cell_denoise import enhance
from extract import extract

import pytesseract as ts


def get_text(image):
    return ts.image_to_string(image, config='--psm 10')


def ocr(cell):
    enhanced = enhance(cell)

    intensity = np.sum(enhanced) / (enhanced.shape[0] * enhanced.shape[1])

    text = None
    if 2 < intensity < 253:
        text = get_text(enhanced)

    return enhanced, text


if __name__ == '__main__':
    ext_img = Image.open('data/example0.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)

    table = tables[0]
    for i, row in enumerate(table, 0):
        for j, cell in enumerate(row, 0):
            e, te = ocr(cell)

            fig = plt.figure()

            ax1 = plt.subplot(1, 2, 1)
            plt.imshow(cell)

            ax2 = plt.subplot(1, 2, 2)
            ax2.set_title(te)
            plt.imshow(e, cmap='gray')

            plt.savefig(f'out/cell-{i}-{j}.jpg')
            plt.close(fig)
