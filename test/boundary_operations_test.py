from unittest.mock import Mock
import numpy as np
import pytest
import cv2
from src.boundary_operations import BoundaryOperations
from src.image_processing_error import BoundaryError


class TestBoundaryOperations:
    def test_get_min_bounding_rect(self):
        # from minBoundingBox.m
        convexhull = np.array([[-5.83555318139395, 3.83490071805846],
                               [-5.86354648572901, 3.81446120429713],
                               [-5.92267020598832, 3.72719556526233],
                               [-6.08489289178581, 3.48653281246975],
                               [-6.22582463739058, 3.27275037262288],
                               [-6.92158141757596, 2.21553718678317],
                               [-6.92911972188903, 2.16605231123762],
                               [-3.24158652326951, -0.260836267567583],
                               [-1.15369474296670, -1.63493488169220],
                               [-1.11373581201636, -1.66053689593738],
                               [2.45678111022476, -0.927013607326139],
                               [-5.83555318139395, 3.83490071805846]])

        # from minBoundingBox.m, order of points can be different from matlab
        bb_expected = np.array([[2.45678111022476, -0.927013607326138],
                                [1.25456124626881, -3.02054329365496],
                                [-7.15516236353907, 1.80878243318473],
                                [-5.95294249958312, 3.90231211951355]])
        bb_expected = bb_expected.reshape((4, 1, 2))

        bb = BoundaryOperations.get_min_bounding_rect(convexhull)
        np.testing.assert_allclose(bb, bb_expected, atol=1e-3)

        convexhull = np.array([[-5.83555318139395, 3.83490071805846],
                               [-5.86354648572901, 3.81446120429713]])
        with pytest.raises(BoundaryError) as exc_info:
            BoundaryOperations.get_min_bounding_rect(convexhull)
        assert "convexhull should have at least 4 points" in str(exc_info.value)

    def test_check_intersection(self, mocker):
        mock_obj_adjust_horizontal_line = Mock(return_value=[0, 0])
        mock_obj_adjust_vertical_line = Mock(return_value=[0, 0])
        mock_obj_get_line_equation = Mock(return_value=[0, 0])
        mock_obj_get_line_intersection = Mock(return_value=[10, 10])
        mocker.patch("src.boundary_operations.BoundaryOperations.adjust_horizontal_line",
                     mock_obj_adjust_horizontal_line)
        mocker.patch("src.boundary_operations.BoundaryOperations.adjust_vertical_line",
                     mock_obj_adjust_vertical_line)
        mocker.patch("src.boundary_operations.BoundaryOperations.get_line_equation",
                     mock_obj_get_line_equation)
        mocker.patch("src.boundary_operations.BoundaryOperations.get_line_intersection",
                     mock_obj_get_line_intersection)

        boundary = np.array([[-10, -10],
                             [90, 10],
                             [102, 204],
                             [20, 180]])
        boundary = np.reshape(boundary, (4, 1, 2))
        BoundaryOperations.check_intersection(boundary, np.zeros((200, 100)).astype(np.uint8))

        # must have two vertical lines and two horizontal lines
        assert mock_obj_adjust_horizontal_line.call_count == 2
        assert mock_obj_adjust_vertical_line.call_count == 2
        # 4 lines
        assert mock_obj_get_line_equation.call_count == 4
        assert mock_obj_get_line_intersection.call_count == 4

        boundary = np.array([[-10, -10],
                             [102, 204],
                             [20, 180]])
        boundary = np.reshape(boundary, (3, 1, 2))
        with pytest.raises(BoundaryError) as exc_info:
            BoundaryOperations.check_intersection(boundary, np.zeros((100, 200)).astype(np.uint8))
        assert "must have four line segments" in str(exc_info)

    def test_get_line_equation(self):
        # same point
        with pytest.raises(BoundaryError) as exc_info:
            BoundaryOperations.get_line_equation(np.array([1, 2]), np.array([1, 2]))
        assert "Can't get line equation using same point" in str(exc_info.value)

        # vertical line x = const
        m, c = BoundaryOperations.get_line_equation(np.array([1, 2]), np.array([1, 4]))
        assert np.isinf(m)
        assert c == 1

        # two points on line y = 2 * x + 5
        m, c = BoundaryOperations.get_line_equation(np.array([1, 7]), np.array([4, 13]))
        assert int(m) == 2
        assert int(c) == 5

    def test_is_point_inside(self):
        mask_shape = (10, 20)

        # condition 1 L
        assert not BoundaryOperations.is_point_inside((-1, 5), mask_shape)
        # condition 1 R
        assert not BoundaryOperations.is_point_inside((1, 11), mask_shape)
        # condition 2 L
        assert not BoundaryOperations.is_point_inside((1, -1), mask_shape)
        # condition 2 R
        assert not BoundaryOperations.is_point_inside((1, 21), mask_shape)
        # pass
        assert BoundaryOperations.is_point_inside((1, 2), mask_shape)

    def test_get_line_intersection(self):
        # two parallel vertical lines x = 1, x = 2
        with pytest.raises(BoundaryError) as exc_info:
            BoundaryOperations.get_line_intersection((np.inf, 1), (np.inf, 2))
        assert "Can't find intersection of parallel lines" in str(exc_info.value)

        # one vertical lines x = 1, y = 2 * x + 1
        intersect = BoundaryOperations.get_line_intersection((np.inf, 1), (2, 1))
        assert intersect[0] == 1
        assert intersect[1] == 3

        # one vertical lines y = 2 * x + 1, x = 1
        intersect = BoundaryOperations.get_line_intersection((2, 1), (np.inf, 1))
        assert intersect[0] == 1
        assert intersect[1] == 3

        # two parallel NOT vertical lines y = 2 * x + 1, y = 2 * x + 2
        with pytest.raises(BoundaryError) as exc_info:
            BoundaryOperations.get_line_intersection((2, 1), (2, 2))
        assert "Can't find intersection of parallel lines" in str(exc_info.value)

        # two intersecting lines y = 2 * x + 1, y = x + 2
        intersect = BoundaryOperations.get_line_intersection((2, 1), (1, 2))
        assert intersect[0] == 1
        assert intersect[1] == 3

    def test_adjust_horizontal_line(self):
        # bottom horizontal line move up
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 20], [80, 60], [20, 70]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(10, 60, 80, 90, mask, True)
        np.testing.assert_allclose(ret, np.array([[10, 37], [80, 67]]), atol=5)

        # bottom horizontal line move up
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 20], [80, 70], [20, 60]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(10, 90, 90, 60, mask, True)
        np.testing.assert_allclose(ret, np.array([[10, 66], [90, 36]]), atol=5)

        # top horizontal line move down
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 50], [80, 20], [80, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(10, 25, 90, 45, mask, False)
        np.testing.assert_allclose(ret, np.array([[10, 45], [90, 65]]), atol=5)

        # top horizontal line move down
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 50], [80, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(10, 25, 90, 45, mask, False)
        np.testing.assert_allclose(ret, np.array([[10, 32], [90, 52]]), atol=5)

        # line doesnt require moving
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 20], [80, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(18, 40, 82, 40, mask, False)
        np.testing.assert_allclose(ret, np.array([[18, 40], [82, 40]]), atol=5)

        # point outside boundary
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[2, 2], [98, 2], [98, 80], [2, 98]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(-2, 80, 102, 102, mask, True)
        np.testing.assert_allclose(ret, np.array([[-2, 68], [102, 90]]), atol=5)

        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[2, 2], [98, 20], [98, 98], [2, 98]])], 1)
        ret = BoundaryOperations.adjust_horizontal_line(10, 25, 90, -2, mask, False)
        np.testing.assert_allclose(ret, np.array([[10, 33], [90, 6]]), atol=5)

        # intersection criteria not met
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[2, 2], [98, 2], [98, 80], [2, 98]])], 1)
        with pytest.warns(RuntimeWarning) as record:
            ret = BoundaryOperations.adjust_horizontal_line(10, 60, 150, 102, mask, True)
        assert len(record) == 1
        assert record[0].message.args[0] == "intersection criteria not met"
        np.testing.assert_allclose(ret, np.array([[10, 60], [150, 102]]), atol=5)

        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[2, 2], [98, 2], [98, 80], [2, 98]])], 1)
        with pytest.warns(RuntimeWarning) as record:
            ret = BoundaryOperations.adjust_horizontal_line(10, 20, 150, 25, mask, False)
        assert len(record) == 1
        assert record[0].message.args[0] == "intersection criteria not met"
        np.testing.assert_allclose(ret, np.array([[10, 20], [150, 25]]), atol=5)

    def test_adjust_vertical_line(self):
        # left vertical line move right
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[50, 20], [80, 20], [80, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_vertical_line(10, 10, 40, 90, mask, True)
        np.testing.assert_allclose(ret, np.array([[43, 10], [73, 90]]), atol=5)

        # left vertical line move right
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 20], [80, 80], [50, 80]])], 1)
        ret = BoundaryOperations.adjust_vertical_line(40, 10, 10, 90, mask, True)
        np.testing.assert_allclose(ret, np.array([[72, 10], [42, 90]]), atol=5)

        # right vertical line move left
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 20], [60, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_vertical_line(70, 10, 90, 90, mask, False)
        np.testing.assert_allclose(ret, np.array([[45, 10], [65, 90]]), atol=5)

        # right vertical line move left
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [60, 20], [80, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_vertical_line(90, 10, 70, 90, mask, False)
        np.testing.assert_allclose(ret, np.array([[64, 10], [44, 90]]), atol=5)

        # line doesnt require moving
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[20, 20], [80, 20], [80, 80], [20, 80]])], 1)
        ret = BoundaryOperations.adjust_vertical_line(40, 10, 40, 90, mask, False)
        np.testing.assert_allclose(ret, np.array([[40, 10], [40, 90]]), atol=5)

        # point outside boundary
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[50, 2], [98, 2], [98, 98], [2, 98]])], 1)
        ret = BoundaryOperations.adjust_vertical_line(-1, -2, 20, 102, mask, True)
        np.testing.assert_allclose(ret, np.array([[33, -2], [54, 102]]), atol=5)

        # intersection criteria not met
        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[40, 40], [60, 40], [60, 60], [40, 60]])], 1)
        with pytest.warns(RuntimeWarning) as record:
            ret = BoundaryOperations.adjust_vertical_line(45, 2, 45, 98, mask, True)
        assert len(record) == 1
        assert record[0].message.args[0] == "intersection criteria not met"
        np.testing.assert_allclose(ret, np.array([[45, 2], [45, 98]]), atol=5)

        mask = np.zeros((100, 100)).astype(np.uint8)
        cv2.fillPoly(mask, [np.array([[40, 40], [60, 40], [60, 60], [40, 60]])], 1)
        with pytest.warns(RuntimeWarning) as record:
            ret = BoundaryOperations.adjust_vertical_line(45, 2, 45, 98, mask, False)
        assert len(record) == 1
        assert record[0].message.args[0] == "intersection criteria not met"
        np.testing.assert_allclose(ret, np.array([[45, 2], [45, 98]]), atol=5)
