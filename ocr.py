import matplotlib.pyplot as plt
import pytesseract as ts
from PIL import Image
import numpy as np
import cv2 as cv

from cell_denoise import enhance
from extract import extract


def ocr(cell):
    enhanced = enhance(cell)

    intensity = np.sum(enhanced) / (enhanced.shape[0] * enhanced.shape[1])

    text = None
    if 2 < intensity < 253:
        text = ts.image_to_string(enhanced, config='--psm 10')

    return enhanced, text


if __name__ == '__main__':
    ext_img = Image.open('data/example.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)

    table = tables[0]
    for cell in table:
        c = cell['cell']
        e, te = ocr(c)

        fig = plt.figure()

        ax1 = plt.subplot(1, 2, 1)
        plt.imshow(c)

        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title(te)
        plt.imshow(e, cmap='gray')

        plt.savefig(f'out/{cell["row"]}-{cell["column"]}.jpg')
        plt.close(fig)
