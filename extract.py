from PIL import Image
import cv2 as cv
import numpy as np

from table import Table
from utils import get_mask, find_corners_from_contour, crop_and_warp, verify_table, add_border_padding


def extract(image):
    mask, horizontal, vertical = get_mask(image)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find intersections between the lines to determine if the intersections are table joints.
    intersections = cv.bitwise_and(horizontal, vertical)

    tables = []
    for i, contour in enumerate(contours):

        # verify that Region of Interest (ROI) is a table
        rect = verify_table(contour, intersections)
        if rect is None:
            continue

        corners = find_corners_from_contour(contour)
        table_image = crop_and_warp(image, corners)

        # add outer borders artificially, some images may not have outer borders
        # this will lead to outer columns being omitted
        table_image = add_border_padding(table_image, w=(2, 2, 2, 4))
        # cv.imwrite(f'out/table{i}.jpg', table_image)

        # find table joints, intersections for the warped table
        _, h, v = get_mask(table_image)
        table_intersections = cv.bitwise_and(h, v)

        table_joints, _ = cv.findContours(table_intersections, cv.RETR_CCOMP,
                                          cv.CHAIN_APPROX_SIMPLE)

        if len(table_joints) < 5:
            continue

        # create an object for table
        table = Table(table_image)

        # Get an n-dimensional array of the coordinates of the table joints
        joint_coords = []
        for j in range(len(table_joints)):
            joint_coords.append(table_joints[j][0][0])
        joint_coords = np.asarray(joint_coords)

        # Returns indices of coordinates in sorted order
        # Sorts based on parameters (keys) starting from the last parameter, then second-to-last, etc
        sorted_indices = np.lexsort((joint_coords[:, 0], joint_coords[:, 1]))
        joint_coords = joint_coords[sorted_indices]

        # Store joint coordinates in the table instance
        table.set_joints(joint_coords)

        tables.append(table)

    # Scale factor
    mult = 3

    out_tables = []
    for i, table in enumerate(tables):
        out_tables.append([])

        table_image = cv.resize(table.image, (table.w * mult, table.h * mult))

        table_entries = table.get_table_entries()
        for r, row in enumerate(table_entries):
            out_tables[-1].append([])
            for c, cell in enumerate(row):
                cell_cropped = table_image[cell[1] * mult: (cell[1] + cell[3]) * mult,
                               cell[0] * mult:(cell[0] + cell[2]) * mult]

                out_tables[-1][-1].append({'row': r, 'column': c, 'cell': cell_cropped})
                # cv.imwrite(f'out/cell-{i}-{r}-{c}.jpg', cell_cropped)

    return out_tables


if __name__ == '__main__':
    ext_img = Image.open('data/example1.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)
