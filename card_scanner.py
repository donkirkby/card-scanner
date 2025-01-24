import math
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 143116200
DISPLAY_WIDTH = 620

def main():
    original = cv2.imread('images/page4.png')
    full_width, full_height, channels = original.shape
    scale = DISPLAY_WIDTH / full_width
    image = cv2.resize(original, None, fx=scale, fy=scale)
    height, width, channels = image.shape
    end_margin = round(height * 0.06)
    side_margin = round(width * 0.03)
    intersect_margin = round(width * 0.03)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([140, 10, 0])
    brown_hi = np.array([160, 255, 255])

    # Mask image to only select browns
    mask = cv2.inRange(hsv, brown_lo, brown_hi)

    # Change image to red where we found brown
    hsv[mask > 0] = (255, 255, 255)
    hsv[mask == 0] = (0, 0, 0)

    monochrome = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    aperture_size = 7  # round(width * 0.005) + 1
    edges = cv2.Canny(monochrome, width * 0.075, width * 0.3, None, aperture_size)

    anchor_size = round(width / 650)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (2*anchor_size + 1, 2*anchor_size + 1),
                                       (anchor_size, anchor_size))
    edges = cv2.dilate(edges, kernel)
    # show_cv(edges)
    display = image
    min_line_length = width * 0.03
    max_line_gap = width * 0.015
    lines = cv2.HoughLinesP(edges,
                            1,
                            np.pi / 180,
                            50,
                            None,
                            20,
                            10)
    vertical_lines = []
    horizontal_lines = []
    if lines is not None:
        for line, in lines:
            x1, y1, x2, y2 = line
            theta = math.atan2(y1 - y2, x1 - x2)
            off_vertical = abs(math.pi / 2 - abs(theta))
            colour = (0, 0, 255)
            is_vertical = False
            if off_vertical < math.pi / 8:
                is_vertical = True
                colour = (255, 0, 0)
            elif off_vertical > math.pi * 3 / 8:
                is_vertical = False
                colour = (0, 255, 0)
            if is_vertical and x1 < side_margin or width - side_margin < x1:
                continue
            if not is_vertical and y1 < end_margin or height - end_margin < y1:
                continue
            if is_vertical:
                vertical_lines.append(line)
            else:
                horizontal_lines.append(line)
            # cv2.line(display, (x1, y1), (x2, y2), colour, 2)

    # converted = cv2.cvtColor(display, cv2.COLOR_HSV2BGR)
    # show_cv(display)
    # return

    # show_cv(display)
    # return
    # vertical_lines.sort(key=lambda x: (x[0], x[1]))
    # for i, line in enumerate(vertical_lines):
    #     if i == 0:
    #         cv2.line(display, line[:2], line[2:], (255, 0, 0), 2)
    # for i, line in enumerate(horizontal_lines):
    #     if i == 0:
    #         cv2.line(display, line[:2], line[2:], (0, 255, 0), 2)
    print(f'Finding intersections in {len(horizontal_lines)} horizontal '
          f'and {len(vertical_lines)} vertical lines.')
    intersections = []
    for v in vertical_lines:
        intersecting_lines = find_intersecting_lines(v,
                                                     horizontal_lines,
                                                     intersect_margin)

        p1 = v[:2]
        p2 = v[2:]
        for p in (p1, p2):
            closest_line = find_closest_line(intersecting_lines, p)
            # cv2.line(display, closest_line[:2], closest_line[2:], (0, 255, 0), 2)
            x_i, y_i = find_intersection(v, closest_line)
            intersections.append((x_i, y_i))
            if len(intersections) % 100 == 0:
                print(f'Found {len(intersections)} intersections.')

    card_rects = []
    intersections_array = np.array(intersections, dtype=np.float32)
    centres = cluster_points(intersections_array, 16)
    for p_corner, y_sign in (((0, 0), 1),
                             ((width, 0), -1),
                             ((0, height), -1),
                             ((width, height), 1)):
        card_rect = []
        card_corner = find_closest_point(p_corner, centres)
        add_corner(card_corner, card_rect)
        # cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

        short_corner = find_closest_point(card_corner, centres, exclude_gap=0)
        add_corner(short_corner, card_rect)
        # cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

        short_vector = (short_corner - card_corner)
        dx, dy = short_vector * 1.5
        x_i, y_i = card_corner
        target = (x_i + dy, y_i + dx*y_sign)
        long_corner = find_closest_point(target, centres)
        add_corner(long_corner, card_rect)
        # cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

        long_vector = (long_corner - card_corner)
        far_target = card_corner + short_vector + long_vector
        far_corner = find_closest_point(far_target, centres)
        add_corner(far_corner, card_rect)
        # cv2.circle(display, (x, y), 2, (0, 0, 255), -1)

        card_rects.append(card_rect)

    card_rect = card_rects[3]
    top_left = find_closest_point((0, 0), card_rect)
    top_right = find_closest_point((width, 0), card_rect)
    bottom_left = find_closest_point((0, height), card_rect)
    bottom_right = find_closest_point((width, height), card_rect)
    left_dir = np.arctan2(bottom_left[1] - top_left[1],
                          top_left[0] - bottom_left[0])
    top_dir = np.arctan2(top_left[1] - top_right[1],
                         top_right[0] - top_left[0])
    right_dir = np.arctan2(bottom_right[1] - top_right[1],
                           top_right[0] - bottom_right[0])
    bottom_dir = np.arctan2(bottom_left[1] - bottom_right[1],
                            bottom_right[0] - bottom_left[0])
    rotation = (left_dir + right_dir + top_dir + bottom_dir - np.pi) / 4
    card_width = round(np.linalg.norm(top_right - top_left))
    card_height = round(np.linalg.norm(top_left - bottom_left))

    rot_mat = cv2.getRotationMatrix2D(top_left.astype(np.float32)/scale,
                                      -rotation*180/np.pi,
                                      1)
    display = cv2.warpAffine(original, rot_mat, (full_height, full_width))
    # cv2.line(display, v[:2], v[2:], (255, 0, 0), 2)

    cropped = display[
              round(top_left[1]/scale):round((top_left[1]+card_height)/scale),
              round(top_left[0]/scale):round((top_left[0]+card_width)/scale)]

    show_cv(cropped)
    # display = image.convert("RGB")
    # display_bytes = BytesIO()
    # display.save(display_bytes, "PNG")
    # display_array = np.asarray(bytearray(display_bytes.getvalue()), dtype="uint8")
    # display_cv = cv2.imdecode(display_array, 0)

    # # Detecting Edges on the Image using the argument ImageFilter.FIND_EDGES
    # edges = image.filter(ImageFilter.FIND_EDGES)

    # edges_bytes = BytesIO()
    # image.save(edges_bytes, "PNG")
    # bytes_array = np.asarray(bytearray(edges_bytes.getvalue()), dtype=np.uint8)
    # edges_cv = cv2.imdecode(bytes_array, 0)
    # cv2.ximgproc.EdgeDrawing.detectEdges(edges_cv)
    # # edges = cv2.xim
    # contours = cv2.findContours(edges_cv,
    #                             cv2.RETR_EXTERNAL,
    #                             cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(display_cv, contours[0], 0, 0)

    # show_cv(image)


def add_corner(card_corner, card_rect):
    x_i, y_i = card_corner
    x = round(x_i)
    y = round(y_i)
    card_rect.append(np.array((x, y)))
    return x, y


def find_closest_point(target, points, exclude_gap = -1):
    min_distance = np.inf
    closest_point = target
    corner_array = np.array(target)
    for intersection in points:
        distance = np.linalg.norm(corner_array - np.array(intersection))
        if exclude_gap < distance < min_distance:
            closest_point = intersection
            min_distance = distance
    return closest_point


def find_intersecting_lines(v, orthogonal_lines, intersect_margin):
    p1 = v[:2]
    p2 = v[2:]
    shortest_distance = np.inf
    for line in orthogonal_lines:
        for p3 in (line[:2], line[2:]):
            distance = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
            shortest_distance = min(shortest_distance, distance)
    intersection_size = shortest_distance + intersect_margin
    intersecting_lines = []
    for line in orthogonal_lines:
        for p3 in (line[:2], line[2:]):
            distance = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
            if distance <= intersection_size:
                intersecting_lines.append(line)
    return intersecting_lines


def find_closest_line(lines, start):
    closest_line = lines[0]
    shortest_distance = np.inf
    for line in lines:
        for end in (line[:2], line[2:]):
            distance = np.linalg.norm(start - end)
            if distance < shortest_distance:
                shortest_distance = distance
                closest_line = line
    return closest_line


def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # compute determinant
    # noinspection DuplicatedCode
    p_x = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    # noinspection DuplicatedCode
    p_y = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    return p_x, p_y


def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    x = points.dtype
    _, _, centers = cv2.kmeans(points,
                               nclusters,
                               None,
                               criteria,
                               10,
                               cv2.KMEANS_PP_CENTERS)
    return centers


def show_cv(cv_image):
    width = cv_image.shape[1]  # height, width, channels
    if width > DISPLAY_WIDTH:
        scale = DISPLAY_WIDTH / width
        cv_image = cv2.resize(cv_image, None, fx=scale, fy=scale)

    success, result_array = cv2.imencode('.png', cv_image)
    result_bytes = BytesIO(result_array)
    result = Image.open(result_bytes)
    # Saving the Image Under the name Edge_Sample.png
    result.show()


if __name__ in ("__live_coding__", "__main__"):
    main()
