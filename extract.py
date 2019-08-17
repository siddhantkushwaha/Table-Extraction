from PIL import Image
import numpy as np
import cv2 as cv

from table import Table


# Apply morphology operations
def isolate_lines(src, structuring_element):
    cv.erode(src, structuring_element, src, (-1, -1))  # makes white spots smaller
    cv.dilate(src, structuring_element, src, (-1, -1))


# Verify if the region inside a contour is a table
# If it is a table, returns the bounding rect
# and the table joints. Else return None.
def verify_table(contour, intersections):
    min_table_area = 50  # min table area to be considered a table
    epsilon = 3  # epsilon value for contour approximation

    area = cv.contourArea(contour)

    if area < min_table_area:
        return None, None

    # approxPolyDP approximates a polygonal curve within the specified precision
    curve = cv.approxPolyDP(contour, epsilon, True)

    # boundingRect calculates the bounding rectangle of a point set (eg. a curve)
    rect = cv.boundingRect(curve)  # format of each rect: x, y, w, h

    # Finds the number of joints in each region of interest (ROI)
    # Format is in row-column order (as finding the ROI involves numpy arrays)
    # format: image_mat[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w]
    possible_table_region = intersections[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    possible_table_joints, _ = cv.findContours(possible_table_region, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Determines the number of table joints in the image
    # If less than 5 table joints, then the image
    # is likely not a table
    if len(possible_table_joints) < 5:
        return None, None

    return rect, possible_table_joints


def extract(image):
    # convert to grayscale
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # ADAPTIVE THRESHOLDING
    # Thresholding changes pixels' color values to a specified pixel value if the current pixel value
    # is less than a threshold value, which could be:
    #
    # 1. a specified global threshold value provided as an argument to the threshold function (simple thresholding),
    # 2. the mean value of the pixels in the neighboring area (adaptive thresholding - mean method), 3. the weighted sum
    # of neighborhood values where the weights are Gaussian windows (adaptive thresholding - Gaussian method).
    #
    # The last two parameters to the adaptiveThreshold function are the size of the neighboring area and
    # the constant C which is subtracted from the mean or weighted mean calculated.

    max_threshold_value = 255
    block_size = 15
    threshold_constant = 0

    # Filter image
    filtered = cv.adaptiveThreshold(~grayscale, max_threshold_value, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY,
                                    block_size, threshold_constant)

    # HORIZONTAL AND VERTICAL LINE ISOLATION
    # To isolate the vertical and horizontal lines,
    #
    # 1. Set a scale.
    # 2. Create a structuring element.
    # 3. Isolate the lines by eroding and then dilating the image.

    scale = 15

    # Isolate horizontal and vertical lines using morphological operations
    horizontal = filtered.copy()
    vertical = filtered.copy()

    horizontal_size = int(horizontal.shape[1] / scale)
    horizontal_structure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    isolate_lines(horizontal, horizontal_structure)

    vertical_size = int(vertical.shape[0] / scale)
    vertical_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_size))
    isolate_lines(vertical, vertical_structure)

    # Create an image mask with just the horizontal
    # and vertical lines in the image. Then find
    # all contours in the mask.

    mask = horizontal + vertical
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Find intersections between the lines
    # to determine if the intersections are table joints.
    intersections = cv.bitwise_and(horizontal, vertical)

    tables = []
    for i, contour in enumerate(contours):

        # verify that Region of Interest (ROI) is a table
        rect, table_joints = verify_table(contour, intersections)
        if rect is None or table_joints is None:
            continue

        # create an object for table
        table = Table(rect[0], rect[1], rect[2], rect[3])

        # Get an n-dimensional array of the coordinates of the table joints
        joint_coords = []
        for i in range(len(table_joints)):
            joint_coords.append(table_joints[i][0][0])
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
    # Process each table's ROI
    for i, table in enumerate(tables):
        out_tables.append([])

        # crop the table
        table_roi = image[table.y:table.y + table.h, table.x:table.x + table.w]
        # resize/rescale
        table_roi = cv.resize(table_roi, (table.w * mult, table.h * mult))

        table_entries = table.get_table_entries()
        for r, row in enumerate(table_entries):
            out_tables[-1].append([])
            for c, cell in enumerate(row):
                cell_cropped = table_roi[cell[1] * mult: (cell[1] + cell[3]) * mult,
                               cell[0] * mult:(cell[0] + cell[2]) * mult]

                out_tables[-1][-1].append({'row': r, 'column': c, 'cell': cell_cropped})

    return out_tables


if __name__ == '__main__':
    ext_img = Image.open('data/example.jpg')
    ext_img.save("out/target.jpg", "JPEG")
    target_img = cv.imread("out/target.jpg")

    tables = extract(target_img)
    print(tables)
