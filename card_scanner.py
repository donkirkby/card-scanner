import math
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 143116200
DISPLAY_WIDTH = 620

# card-suit: (left, top, right, bottom) margins are fraction of width or height
MARGIN_OVERRIDES = {'k-w': (0.05, 0, -0.04, 0),
                    'j-e': (0, 0, 0, 0.0175),
                    'j-a': (0, 0, 0.02, 0),
                    'j-w': (0, 0, 0.02, 0),
                    't-e': (0, 0, 0.02, 0),  # min 0.01
                    'a-a': (0, 0, 0.02, 0)}  # min 0.01

def main() -> None:
    in_folder = Path('images_in')
    image_path: Path
    for image_path in sorted(in_folder.glob('*.png')):
        scan_image(image_path)
        if __name__ == '__live_coding__':
            break
        print(image_path.name)


def scan_image(image_path: Path) -> None:
    out_folder = Path('images_out')
    template = cv2.imread("bridge-card.png")
    original = cv2.imread(str(image_path))
    blank = cv2.imread('images/blank-big.png')
    card_name = image_path.stem

    full_height, full_width, channels = original.shape
    scale = DISPLAY_WIDTH / full_width
    image = cv2.resize(original, None, fx=scale, fy=scale)
    height, width, channels = image.shape
    end_margin = round(height * 0.06)
    side_margin = round(width * 0.03)
    intersect_margin = round(width * 0.03)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

    # Define lower and uppper limits of what we call "brown"
    brown_lo = np.array([145, 50, 150])
    brown_hi = np.array([150, 70, 200])

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
    # return

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
            assert colour is not None
            # cv2.line(image, (x1, y1), (x2, y2), colour, 2)

    # converted = cv2.cvtColor(display, cv2.COLOR_HSV2BGR)
    # show_cv(image)
    # return

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
        # cv2.circle(image,
        #            [round(x) for x in card_corner],
        #            2,
        #            (0, 0, 255),
        #            -1)

        short_corner = find_closest_point(card_corner, centres, exclude_gap=0)
        add_corner(short_corner, card_rect)
        # cv2.circle(image,
        #            [round(x) for x in short_corner],
        #            2,
        #            (0, 0, 255),
        #            -1)

        short_vector = (short_corner - card_corner)
        dx, dy = short_vector * 1.5
        x_i, y_i = card_corner
        target = (x_i + dy, y_i + dx*y_sign)
        long_corner = find_closest_point(target, centres)
        add_corner(long_corner, card_rect)
        # cv2.circle(image,
        #            [round(x) for x in long_corner],
        #            2,
        #            (0, 0, 255),
        #            -1)

        long_vector = (long_corner - card_corner)
        far_target = card_corner + short_vector + long_vector
        far_corner = find_closest_point(far_target, centres)
        add_corner(far_corner, card_rect)
        # cv2.circle(image,
        #            [round(x) for x in far_corner],
        #            2,
        #            (0, 0, 255),
        #            -1)

        card_rects.append(card_rect)

    for card_index, (card_rect, suit_name) in enumerate(zip(card_rects, 'aefw')):
        is_back = card_name == 'x' and card_index >= 2

        margin_override = MARGIN_OVERRIDES.get(f'{card_name}-{suit_name}',
                                               (0, 0, 0, 0))


        top_left = find_closest_point((0, 0), card_rect)
        top_right = find_closest_point((width, 0), card_rect)
        bottom_left = find_closest_point((0, height), card_rect)
        bottom_right = find_closest_point((width, height), card_rect)
        (left_override,
         top_override,
         right_override,
         bottom_override) = margin_override
        start_width = round(np.linalg.norm(top_right - top_left))
        start_height = round(np.linalg.norm(top_left - bottom_left))
        if left_override:
            top_left[0] -= start_width * left_override
            bottom_left[0] -= start_width * left_override
        if right_override:
            top_right[0] += start_width * right_override
            bottom_right[0] += start_width * right_override
        if top_override:
            top_left[1] -= start_height * top_override
            top_right[1] -= start_height * top_override
        if bottom_override:
            bottom_left[1] += start_height * bottom_override
            bottom_right[1] += start_height * bottom_override
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
        display = cv2.warpAffine(original, rot_mat, (full_width, full_height))

        cropped = display[
                  round(top_left[1]/scale):round((top_left[1]+card_height)/scale),
                  round(top_left[0]/scale):round((top_left[0]+card_width)/scale)]
        full_card_height, full_card_width, channels = cropped.shape
        final_height, final_width, channels = template.shape
        if is_back:
            card_scale = min(final_width / full_card_width,
                             final_height / full_card_height) * 1.04  # back
        else:
            card_scale = min(final_width / full_card_width,
                             final_height / full_card_height) * 0.9  # front
        scaled_card = cv2.resize(cropped, None, fx=card_scale, fy=card_scale)
        scaled_height, scaled_width, channels = scaled_card.shape

        blank_cropped = blank[
                  100:100+round(final_height/card_scale),
                  :round(final_width/card_scale)]
        padded_card = cv2.resize(blank_cropped,
                                 None,
                                 fx=card_scale,
                                 fy=card_scale)

        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        threshold = cv2.threshold(gray_template,
                                  140,
                                  255,
                                  cv2.THRESH_BINARY)[1]
        inverted_threshold = 255 - threshold
        masked_template = cv2.bitwise_and(template, template, mask=inverted_threshold)
        top_padding = (final_height - scaled_height) // 2
        side_padding = (final_width - scaled_width) // 2
        if top_padding > 0:
            top_trimmed = 0
        else:
            top_trimmed = -top_padding
            top_padding = 0
        if side_padding > 0:
            side_trimmed = 0
        else:
            side_trimmed = -side_padding
            side_padding = 0

        scaled_source = scaled_card[
                        top_trimmed:top_trimmed + final_height,
                        side_trimmed:side_trimmed + final_width]
        if is_back:
            padded_card[
            top_padding:top_padding + scaled_height,
            side_padding:side_padding + scaled_width] = scaled_source
        else:
            mask = np.zeros(shape=(scaled_height, scaled_width), dtype=scaled_card.dtype)
            centre = (final_width//2, final_height//2)
            mask_margin = 0.03
            mask[
                round(scaled_height*mask_margin):round(scaled_height*(1-mask_margin)),
                round(scaled_width*mask_margin):round(scaled_width*(1-mask_margin))] = 255
            padded_card = cv2.seamlessClone(scaled_source,
                                            padded_card,
                                            mask,
                                            centre,
                                            cv2.MIXED_CLONE)
        masked_card = cv2.bitwise_and(padded_card, padded_card, mask=threshold)
        combined_card = masked_card + masked_template

        card_out_path = out_folder / f'card-{card_name}-{suit_name}.png'
        cv2.imwrite(str(card_out_path), padded_card)
        template_out_path = out_folder / f'template-{card_name}-{suit_name}.png'
        cv2.imwrite(str(template_out_path), combined_card)

        if card_index == 0 and __name__ == '__live_coding__':
            show_cv(padded_card)
            return


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
    # noinspection PyTypeChecker
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
