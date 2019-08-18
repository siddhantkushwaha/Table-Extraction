from PIL import Image
import cv2 as cv

from table import Table
from utils import get_grid_mask, find_corners_from_contour, crop_and_warp, verify_table, add_border_padding, \
    find_intersection_mean_cords


def extract(image):
    mask, horizontal, vertical = get_grid_mask(image)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find intersections between the lines to determine if the intersections are table joints.
    intersections = cv.bitwise_and(horizontal, vertical)

    table_cells = []
    for table_number, contour in enumerate(contours):

        # verify that Region of Interest (ROI) is a table
        rect = verify_table(contour, intersections)
        if rect is None:
            continue

        corners = find_corners_from_contour(contour)
        table_image = crop_and_warp(image, corners)

        # add outer borders artificially, some images may not have outer borders
        # this will lead to outer columns being omitted
        table_image = add_border_padding(table_image, w=(2, 2, 2, 4), color=(100, 100, 100))

        # find table joints, intersections for the warped table
        m, h, v = get_grid_mask(table_image)
        table_intersections = cv.bitwise_and(h, v)

        intersection_points = find_intersection_mean_cords(table_intersections)

        if len(intersection_points) < 5:
            continue

        table = Table(table_image, intersection_points)

        try:
            table_cells.append(table.get_cells())
        except Exception as e:
            print(e)
            for row in intersection_points:
                print(row)

    return table_cells


if __name__ == '__main__':
    ext_img = Image.open('data/example1.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    extracted_tables = extract(target_img)
    # print(extracted_tables)
