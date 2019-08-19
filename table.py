class Table:
    def __init__(self, image, intersection_points):
        self.h = image.shape[0]
        self.w = image.shape[1]

        self.image = image
        self.intersection_points = intersection_points

        self.cells = None

    def get_cells(self):

        if self.cells is not None:
            return self.cells

        methods = [self.get_cells_v1, self.get_cells_v2]
        for method in methods:
            try:
                self.cells = method()
                break
            except Exception as e:
                print(f'{method} failed. {e}')

        return self.cells

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
