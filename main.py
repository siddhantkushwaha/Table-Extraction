import pandas as pd
import cv2 as cv

from extract import extract


def main(image):
    tables = extract(image)
    for table in tables:
        df = pd.DataFrame(table.data)
        yield df


if __name__ == '__main__':
    path = 'data/example1.jpg'
    image = cv.imread(path)

    tables = extract(image)
    for table in tables:
        print(table)
