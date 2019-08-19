import operator

import numpy as np
import cv2 as cv


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
        return None

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
        return None

    return rect


def find_corners_from_contour(polygon):
    """Finds the 4 extreme corners of the contour given."""

    # Use of `operator.itemgetter` with `max` and `min` allows us to get the index of the point
    # Each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively.

    # Bottom-right point has the largest (x + y) value
    # Top-left has point smallest (x + y) value
    # Bottom-left point has smallest (x - y) value
    # Top-right point has largest (x - y) value

    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1]
                                 for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1]
                                    for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1]
                                  for pt in polygon]), key=operator.itemgetter(1))

    # Return an array of all 4 points using the indices
    # Each point is in its own array of one coordinate
    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]


def distance_between(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def crop_and_warp(img, crop_rect):
    """Crops and warps a rectangular section from an image into a square of similar size."""

    # Rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[
                                                         0], crop_rect[1], crop_rect[2], crop_rect[3]

    # Explicitly set the data type to float32 or `getPerspectiveTransform` will throw an error
    src = np.array([top_left, top_right, bottom_right,
                    bottom_left], dtype='float32')

    b = distance_between(top_left, bottom_left)
    l = distance_between(bottom_left, bottom_right)

    # Describe a rect with side of the calculated length and breadth, this is the new perspective we want to warp to
    dst = np.array([[0, 0], [l - 1, 0], [l - 1, b - 1],
                    [0, b - 1]], dtype='float32')

    # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = cv.getPerspectiveTransform(src, dst)

    # Performs the transformation on the original image
    return cv.warpPerspective(img, m, (int(l), int(b)))


def get_grid_mask(image):
    # convert to grayscale
    grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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

    return mask, horizontal, vertical


# Add border to image after trimming, and add padding to it
def add_border_padding(image, w=(2, 2, 2, 2), color=(0, 0, 0)):
    x, y, _ = image.shape
    image = image[w[0]:x - w[1], w[2]:y - w[3]]
    image = cv.copyMakeBorder(image, 2, 2, 2, 2, cv.BORDER_CONSTANT, value=color)
    image = cv.copyMakeBorder(image, 16, 16, 16, 16, cv.BORDER_CONSTANT, value=(255, 255, 255))
    return image


# Given an image of intersection spots, pick one from each cluster
def find_intersection_mean_cords(intersections):
    cutoff = 5

    points = []
    _y = 0
    for y, r in enumerate(intersections, 0):
        for x, c in enumerate(r, 0):
            intensity = intersections[y][x]
            if intensity == 255:
                p = (x, y)
                if y - _y > cutoff:
                    points.append([p])
                else:
                    points[-1].append(p)
                _y = y

    coords = []
    for row in points:
        new_row = []
        old_row = sorted(row)
        _x = 0
        for point in old_row:
            if point[0] - _x > cutoff:
                new_row.append(point)
                _x = point[0]
        coords.append(new_row)

    return coords


# get centroid of a a rectangle
def get_centroid(left, right, top, bottom):
    return (left + right) / 2, (top + bottom) / 2
