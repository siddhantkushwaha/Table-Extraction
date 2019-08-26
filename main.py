import pandas as pd
import cv2 as cv

from extract import extract


def main(image):
    tables = extract(image)
    for table in tables:
        df = pd.DataFrame(table.data)
        print(df)


if __name__ == '__main__':
    image = cv.imread("data/example1.jpg")
    main(image)
