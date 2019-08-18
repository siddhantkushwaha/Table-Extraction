class Table:
    def __init__(self, image, intersection_points):
        self.h = image.shape[0]
        self.w = image.shape[1]

        self.image = image
        self.intersection_points = intersection_points

    # Based on assumption that there's no colspan or rowspan
    def get_cells(self):

        m = len(self.intersection_points)
        # print(m)

        table_cells = []
        for i in range(m):
            table_cells.append([])
            n = len(self.intersection_points[i])
            # print(n)
            for j in range(n):
                tl = self.intersection_points[i][j]
                tr = self.intersection_points[i][j + 1]
                bl = self.intersection_points[i + 1][j]
                br = self.intersection_points[i + 1][j + 1]

                top_edge = min(tl[0], tr[0])
                bottom_edge = max(bl[0], br[0])

                left_edge = min(tl[1], bl[1])
                right_edge = max(tr[1], br[1])

                cell = self.image[top_edge: bottom_edge, left_edge: right_edge]
                table_cells[-1].append(cell)

        return table_cells
