import pytesseract as ts
import cv2 as cv

import pandas as pd


def get_image_data(image):
    df = ts.image_to_data(image, output_type=ts.Output.DATAFRAME)
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
    img = cv.imread('data/table0.jpg')

    ocr_data = ocr(img)

    for i, data in ocr_data.iterrows():
        cv.rectangle(img=img, pt1=(data['left'], data['top']),
                     pt2=(data['left'] + data['width'], data['top'] + data['height']), color=(0, 255, 0), thickness=1)
        cv.putText(img=img, text=data['text'], color=(0, 0, 255), org=(data['left'], data['top']),
                   fontFace=cv.FONT_HERSHEY_PLAIN, fontScale=1)

        print(data['text'])

    cv.imwrite('out/ocr_results.jpg', img)
