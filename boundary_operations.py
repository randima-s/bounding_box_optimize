import math
import warnings
import numpy as np
from image_processing_error import BoundaryError


class BoundaryOperations:
    @staticmethod
    def get_min_bounding_rect(convexhull):
        if len(convexhull) < 4:
            raise BoundaryError("convexhull should have at least 4 points")
        else:
            convexhull = np.reshape(convexhull, (convexhull.shape[0], 2))

            # compute edges
            edges = np.subtract(convexhull[1:], convexhull[0:-1])

            # calculate angles
            angles = np.zeros(len(edges))
            for i in range(len(angles)):
                angles[i] = math.atan2(edges[i, 1], edges[i, 0])

            # get first quadrant angles
            angles = np.mod(angles, math.pi / 2)

            # get unique angles
            angles = np.unique(angles)

            rotations = np.vstack(
                [np.cos(angles), np.cos(angles - math.pi / 2), np.cos(angles + math.pi / 2), np.cos(angles)]).T
            rotations = rotations.reshape(-1, 2, 2)

            # rotate all points
            rot_points = np.dot(rotations, convexhull.T)

            # find border size and compute areas
            max_x = np.max(rot_points[:, 0], axis=1)
            min_x = np.min(rot_points[:, 0], axis=1)
            max_y = np.max(rot_points[:, 1], axis=1)
            min_y = np.min(rot_points[:, 1], axis=1)
            area = np.multiply(max_x - min_x, max_y - min_y)

            # get minimum area index
            min_area_index = np.argmin(area)

            x1 = max_x[min_area_index]
            x2 = min_x[min_area_index]
            y1 = max_y[min_area_index]
            y2 = min_y[min_area_index]
            r = rotations[min_area_index]

            bounding_box = np.zeros((4, 2))
            bounding_box[0] = np.dot([x1, y2], r)  # 2 1 0 3
            bounding_box[1] = np.dot([x2, y2], r)
            bounding_box[2] = np.dot([x2, y1], r)
            bounding_box[3] = np.dot([x1, y1], r)

            bounding_box = bounding_box.reshape((-1, 1, 2))
            return bounding_box

    @staticmethod
    def check_intersection(boundary, mask):
        # temp_mask_img = np.zeros(mask.shape).astype(np.uint8)  # for debugging
        # temp_mask_img[mask == 1] = 100

        if len(boundary) != 4:
            raise BoundaryError("must have four line segments")
        else:
            rows, cols = mask.shape
            # check and limit edges
            for i in range(len(boundary)):
                if boundary[i][0][0] > cols - 1:
                    boundary[i][0][0] = cols - 1
                elif boundary[i][0][0] < 0:
                    boundary[i][0][0] = 0

                if boundary[i][0][1] > rows - 1:
                    boundary[i][0][1] = rows - 1
                elif boundary[i][0][1] < 0:
                    boundary[i][0][1] = 0

            # cv2.polylines(temp_mask_img, boundary.astype(int), True, 255, 10, cv2.LINE_4)
            # cv2.imshow("intersection", temp_mask_img)

            line_equations = []
            for i in range(len(boundary)):
                j = i + 1
                if j == 4:
                    j = 0

                x0, y0 = boundary[i][0]
                x1, y1 = boundary[j][0]

                # check if vertical or horizontal line
                if abs(y1 - y0) < abs(x1 - x0):
                    if x0 > x1:  # line horizontal top, move down until intersection >= 0.7
                        ret = BoundaryOperations.adjust_horizontal_line(x1, y1, x0, y0, mask, False)
                    else:  # line horizontal bottom, move up until intersection >= 0.7
                        ret = BoundaryOperations.adjust_horizontal_line(x0, y0, x1, y1, mask, True)
                else:
                    if y0 > y1:  # line vertical right, move left until intersection >= 0.7
                        ret = BoundaryOperations.adjust_vertical_line(x1, y1, x0, y0, mask, False)
                    else:  # line vertical left, move right until intersection >= 0.7
                        ret = BoundaryOperations.adjust_vertical_line(x0, y0, x1, y1, mask, True)

                # cv2.line(temp_mask_img, ret[0].astype(int), ret[1].astype(int), 255, 4)
                # cv2.imshow("intersection", temp_mask_img)
                # cv2.waitKey(0)

                line_eq = BoundaryOperations.get_line_equation(ret[0], ret[1])  # returns m,c of y = mx + c
                line_equations.append(line_eq)

            new_boundary_points = []
            for i in range(len(line_equations)):
                j = i + 1
                if j == 4:
                    j = 0

                intersect = BoundaryOperations.get_line_intersection(line_equations[i], line_equations[j])
                if BoundaryOperations.is_point_inside(intersect, mask.shape):
                    new_boundary_points.append(intersect)

                    # cv2.circle(temp_mask_img, intersect.astype(int), 10, 255, 4)
                    # cv2.imshow("intersection", temp_mask_img)
                    # cv2.waitKey(0)

            return np.array(new_boundary_points).reshape((-1, 1, 2))

    @staticmethod
    def get_line_equation(point1, point2):
        # solve y_1 = m * x_1 + c,
        #       y_2 = m * x_2 + c
        # return m, c
        x_1 = point1[0]
        x_2 = point2[0]
        y_1 = point1[1]
        y_2 = point2[1]

        if x_1 == x_2:
            if y_1 == y_2:  # same point
                raise BoundaryError("Can't get line equation using same point")
            else:  # vertical line
                m, c = np.inf, x_1
        else:
            m = (y_1 - y_2) / (x_1 - x_2)
            c = (x_2 * y_1 - x_1 * y_2) / (x_2 - x_1)

        return m, c

    @staticmethod
    def is_point_inside(point, mask_shape):
        if (point[0] < 0) or (point[0] > mask_shape[1]):
            return False
        if (point[1] < 0) or (point[1] > mask_shape[0]):
            return False
        return True

    @staticmethod
    def get_line_intersection(line1, line2):
        # lines should be in [m, c] form of y = mx + c
        # solve y = m_1 * x + c_1
        #       y = m_2 * x + c_2
        # where m_1 != m_2 (parallel lines)
        # m = Inf for vertical line
        m_1 = line1[0]
        m_2 = line2[0]
        c_1 = line1[1]
        c_2 = line2[1]

        if np.isinf(m_1):
            if np.isinf(m_2):  # parallel
                raise BoundaryError("Can't find intersection of parallel lines")
            else:
                x_intersect = c_1
                y_intersect = c_2 + m_2 * c_1
        elif np.isinf(m_2):
            x_intersect = c_2
            y_intersect = c_1 + m_1 * c_2
        else:
            if m_1 == m_2:  # parallel
                raise BoundaryError("Can't find intersection of parallel lines")
            else:
                x_intersect = (c_2 - c_1) / (m_1 - m_2)
                y_intersect = (m_1 * c_2 - m_2 * c_1) / (m_1 - m_2)

        return np.array([x_intersect, y_intersect])

    @staticmethod
    def adjust_horizontal_line(x0, y0, x1, y1, mask, is_bottom_line):
        rows, cols = mask.shape
        input_cords = np.array([[x0, y0], [x1, y1]])

        row_count = 0
        num_set = 0

        while 1:
            dx = x1 - x0
            dy = y1 - y0
            yi = 1

            if dy < 0:
                yi = -1
                dy = -dy

            d = 2 * dy - dx
            y = y0

            x = x0
            while x <= x1:
                x += 1

                rx = round(x)
                ry = round(y)

                # Check limits
                if ry < 0:
                    ry = 0
                if ry > rows - 1:
                    ry = rows - 1
                if rx < 0:
                    rx = 0
                if rx > cols - 1:
                    rx = cols - 1

                num_set += mask[ry, rx]
                row_count += 1

                # increment y if line has a slope greater than a given value, initially y = 0.5x
                if d > 0:
                    y = y + yi
                    d = d - 2 * dx  # increase slope threshold, eg y = 1.5x

                d = d + 2 * dy  # decrease slope threshold, eg y = 0.25x

            pc = np.divide(num_set, row_count)
            if pc < 0.7:
                if is_bottom_line:
                    y0 = y0 - 1
                    y1 = y1 - 1
                    if (y0 < 0) or (y1 < 0):  # intersection criteria not met
                        warnings.warn("intersection criteria not met", RuntimeWarning)
                        return input_cords
                else:
                    y0 = y0 + 1
                    y1 = y1 + 1
                    if (y0 >= rows) or (y1 >= rows):  # intersection criteria not met
                        warnings.warn("intersection criteria not met", RuntimeWarning)
                        return input_cords

                num_set = 0
                row_count = 0
            else:
                break

        return np.array([[x0, y0], [x1, y1]])

    @staticmethod
    def adjust_vertical_line(x0, y0, x1, y1, mask, is_left_line):
        rows, cols = mask.shape
        input_cords = np.array([[x0, y0], [x1, y1]])

        col_count = 0
        num_set = 0

        while 1:
            dx = x1 - x0
            dy = y1 - y0
            xi = 1

            if dx < 0:
                xi = -1
                dx = -dx

            d = 2 * dx - dy
            x = x0

            y = y0
            while y <= y1:
                y += 1

                rx = round(x)
                ry = round(y)

                # Check limits
                if ry < 0:
                    ry = 0
                if ry > rows - 1:
                    ry = rows - 1
                if rx < 0:
                    rx = 0
                if rx > cols - 1:
                    rx = cols - 1

                num_set += mask[ry, rx]
                col_count += 1

                if d > 0:
                    x = x + xi
                    d = d - 2 * dy

                d = d + 2 * dx

            pc = np.divide(num_set, col_count)

            if pc < 0.7:
                if is_left_line:
                    x0 = x0 + 1
                    x1 = x1 + 1
                    if (x0 > cols) or (x1 > cols):
                        warnings.warn("intersection criteria not met", RuntimeWarning)
                        return input_cords
                else:
                    x0 = x0 - 1
                    x1 = x1 - 1
                    if (x0 < 0) or (x1 < 0):
                        warnings.warn("intersection criteria not met", RuntimeWarning)
                        return input_cords

                num_set = 0
                col_count = 0
            else:
                break

        return np.array([[x0, y0], [x1, y1]])
