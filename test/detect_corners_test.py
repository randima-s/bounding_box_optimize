from unittest.mock import Mock
import numpy as np
import pytest
from src.detect_corners import DetectCorners


class TestDetectCorners:
    def test_get_largest_segment_convexhull(self, mocker):
        mock_cv2_findContours = Mock(return_value=([20, 10, 30, 40], None))
        mock_cv2_contourArea = Mock(side_effect=lambda x: (x, None))
        mock_cv2_convexHull = Mock(side_effect=lambda x: x)

        mocker.patch("src.detect_corners.cv2.findContours", mock_cv2_findContours)
        mocker.patch("src.detect_corners.cv2.contourArea", mock_cv2_contourArea)
        mocker.patch("src.detect_corners.cv2.convexHull", mock_cv2_convexHull)

        mask = np.zeros((100, 200)).astype(np.uint8)
        hull = DetectCorners.get_largest_segment_convexhull(mask)
        assert hull == 40  # should be sorted in reverse order

        # If no contours are detected
        mock_cv2_findContours.return_value = ([], None)
        with pytest.raises(Exception) as exc_info:
            hull = DetectCorners.get_largest_segment_convexhull(mask)
        assert "No contours found" in str(exc_info.value)
