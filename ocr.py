from io import StringIO

import pytesseract as ts
import cv2 as cv

import pandas as pd


def get_image_data(image):
    ts_data = ts.image_to_data(image)
    df = pd.read_csv(StringIO(ts_data), sep='\t')
    return df


def ocr(image):
    df = get_image_data(image)

    out = []
    for i, row in df.iterrows():
        text = row['text']
        if type(text) is str:
            text = text.strip()
            if len(text) > 0:
                cell_info = {
                    'text': text,
                    'left': row['left'],
                    'top': row['top'],
                    'height': row['height'],
                    'width': row['width']
                }
                out.append(cell_info)
    return pd.DataFrame(out)


if __name__ == '__main__':
    img = cv.imread('out/t1.jpg')
    ocr_data = ocr(img)
