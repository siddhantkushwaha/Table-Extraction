# in case of merged cells
def get_cells(image, intersection_points):
    m = len(intersection_points)
    # print(m)

    table_cells = []
    for i in range(m - 1):
        table_cells.append([])
        n = len(intersection_points[i])
        # print(n)
        for j in range(n - 1):
            tl = intersection_points[i][j]
            tr = intersection_points[i][j + 1]
            bl = (intersection_points[i + 1][0][0], tr[1])
            br = (bl[0], tr[1])

            top_edge = min(tl[0], tr[0])
            bottom_edge = max(bl[0], br[0])

            left_edge = min(tl[1], bl[1])
            right_edge = max(tr[1], br[1])

            cell = image[top_edge: bottom_edge, left_edge: right_edge]

            table_cells[-1].append(cell)

    return table_cells
