from ocr import ocr
from utils import get_centroid


class Table:
    def __init__(self, image, intersection_points):
        self.h = image.shape[0]
        self.w = image.shape[1]

        self.image = image
        self.intersection_points = intersection_points
        self.cells = None

        self.data = None

    def find_cells(self):

        methods = [self.get_cells_v1, self.get_cells_v2]
        for method in methods:
            try:
                self.cells = method()
                break
            except Exception as e:
                print(f'{method} failed. {e}')

    # Based on assumption that there's no colspan or rowspan
    # and when all rows have equal number of columns
    def get_cells_v1(self):
        m = len(self.intersection_points)
        table_cells = []
        for i in range(m - 1):
            table_cells.append([])
            n = len(self.intersection_points[i])
            for j in range(n - 1):
                tl = self.intersection_points[i][j]
                br = self.intersection_points[i + 1][j + 1]

                table_cells[-1].append((tl, br))

        return table_cells

    # try this when method fails
    def get_cells_v2(self):
        m = len(self.intersection_points)
        table_cells = []
        for i in range(m - 1):
            table_cells.append([])
            n = len(self.intersection_points[i])
            for j in range(n - 1):
                tl = self.intersection_points[i][j]
                tr = self.intersection_points[i][j + 1]
                bl = (tl[0], self.intersection_points[i + 1][0][1])
                br = (tr[0], bl[1])

                table_cells[-1].append((tl, br))

        return table_cells

    def find_cell_for_point(self, point):
        for i, row in enumerate(self.cells):
            for j, cell in enumerate(row):
                if cell[0][0] <= point[0] <= cell[1][0] and cell[0][1] <= point[1] <= cell[1][1]:
                    return i, j
        return None

    # TODO - improve this algorithm
    def build(self):

        if self.cells is None:
            self.find_cells()

        ocr_data = ocr(self.image)

        table_data = []
        for row in self.cells:
            table_data.append([])
            for _ in row:
                table_data[-1].append([])

        for i, data in ocr_data.iterrows():
            centroid = get_centroid(data['left'], data['left'] + data['width'], data['top'],
                                    data['top'] + data['height'])

            cell = self.find_cell_for_point(centroid)
            if cell is not None:
                table_data[cell[0]][cell[1]].append(data['text'])

        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                if len(cell) == 0:
                    table_data[i][j] = None
                else:
                    table_data[i][j] = ' '.join(cell)

        self.data = table_data
