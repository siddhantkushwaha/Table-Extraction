import pandas as pd
import cv2 as cv

from extract import extract


def main(image):
    tables = extract(image)
    for table in tables:
        df = pd.DataFrame(table.data)
        yield df


def run(path='data/example1.jpg'):
    image = cv.imread(path)
    yield from main(image)


if __name__ == '__main__':
    for t in run():
        print(t)
