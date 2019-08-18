from PIL import Image
import cv2 as cv
import pandas as pd

from extract import extract
from ocr import ocr

if __name__ == '__main__':
    ext_img = Image.open('data/example0.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)

    for i, table in enumerate(tables, 1):
        data = []
        for row in table:
            data.append([])
            for cell in row:
                _, text = ocr(cell)
                data[-1].append(text)
        df = pd.DataFrame(data)
        df.to_excel(f'out/table-{i}.xlsx', index=False)
