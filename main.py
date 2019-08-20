import pandas as pd
import cv2 as cv

from extract import extract

if __name__ == '__main__':
    image = cv.imread("data/example1.jpg")

    tables = extract(image)
    for table in tables:
        df = pd.DataFrame(table.data)
        print(df)
