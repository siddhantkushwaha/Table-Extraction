from PIL import Image
import cv2 as cv
import pandas as pd

from extract import extract
from ocr import ocr
from utils import get_centroid


def find_cell(cells, point):
    for i, row in enumerate(cells):
        for j, cell in enumerate(row):
            if cell[0][0] <= point[0] <= cell[1][0] and cell[0][1] <= point[1] <= cell[1][1]:
                return i, j
    return None


# TODO - improve this algorithm
def create_table_df(table):
    cells = table.get_cells()
    ocr_data = ocr(table.image)

    table_data = []
    for row in cells:
        table_data.append([])
        for _ in row:
            table_data[-1].append([])

    for i, data in ocr_data.iterrows():
        centroid = get_centroid(data['left'], data['left'] + data['width'], data['top'],
                                data['top'] + data['height'])

        cell = find_cell(cells, centroid)
        if cell is not None:
            table_data[cell[0]][cell[1]].append(data['text'])

    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            if len(cell) == 0:
                table_data[i][j] = None
            else:
                table_data[i][j] = ' '.join(cell)

    table_df = pd.DataFrame(table_data)
    return table_df


if __name__ == '__main__':
    ext_img = Image.open('data/example1.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)

    for table in tables:
        df = create_table_df(table)
        print(df)
