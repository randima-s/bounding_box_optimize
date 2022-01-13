import numpy as np
import cv2
from image_processing_error import CornerDetectError
from boundary_operations import BoundaryOperations
from constants import ImageProcessingConsts


class DetectCorners:
    def __init__(self, image_path):
        self.__image_path = image_path
        self.__image_gray = None
        self.__image_original = None
        self.__mask = None
        self.__read_image()

    def detect_plate_corners(self):
        if self.__image_original is not None and self.__mask is not None:
            img_edges = DetectCorners.detect_edges(self.__image_gray)  # get edges of image object
            convexhull = DetectCorners.get_largest_segment_convexhull(img_edges)
            bounding_box = BoundaryOperations.get_min_bounding_rect(convexhull)

            #  optimize bounding box
            bounding_box_adjusted = BoundaryOperations.check_intersection(bounding_box, self.__mask)

            # DO NOT DELETE
            # used for testing - showing image with bounding boxes
            b_box = np.reshape(bounding_box_adjusted, (4, 2))
            b_box = np.int32(b_box)
            cv2.circle(self.__image_original, b_box[0], ImageProcessingConsts.CORNER_MARKER_RADIUS,
                       ImageProcessingConsts.COLOR_RED, ImageProcessingConsts.LINE_WIDTH)
            cv2.circle(self.__image_original, b_box[1], ImageProcessingConsts.CORNER_MARKER_RADIUS,
                       ImageProcessingConsts.COLOR_RED, ImageProcessingConsts.LINE_WIDTH)
            cv2.circle(self.__image_original, b_box[2], ImageProcessingConsts.CORNER_MARKER_RADIUS,
                       ImageProcessingConsts.COLOR_RED, ImageProcessingConsts.LINE_WIDTH)
            cv2.circle(self.__image_original, b_box[3], ImageProcessingConsts.CORNER_MARKER_RADIUS,
                       ImageProcessingConsts.COLOR_RED, ImageProcessingConsts.LINE_WIDTH)
            cv2.line(self.__image_original, b_box[0], b_box[1], ImageProcessingConsts.COLOR_BLUE,
                     ImageProcessingConsts.LINE_WIDTH)
            cv2.line(self.__image_original, b_box[1], b_box[2], ImageProcessingConsts.COLOR_BLUE,
                     ImageProcessingConsts.LINE_WIDTH)
            cv2.line(self.__image_original, b_box[2], b_box[3], ImageProcessingConsts.COLOR_BLUE,
                     ImageProcessingConsts.LINE_WIDTH)
            cv2.line(self.__image_original, b_box[3], b_box[0], ImageProcessingConsts.COLOR_BLUE,
                     ImageProcessingConsts.LINE_WIDTH)

            cv2.imshow("bbox", self.__image_original)
            cv2.waitKey(0)

            return bounding_box_adjusted

    def __read_image(self):
        image = cv2.imread(self.__image_path)
        self.__image_original = image
        self.__image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.__mask = np.where(self.__image_gray > ImageProcessingConsts.MASK_THRESHOLD, 0, 1).astype('uint8')

    @staticmethod
    def detect_edges(image):
        edges = cv2.Canny(image, 100, 200)
        return edges

    @staticmethod
    def get_largest_segment_convexhull(mask):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            hull = cv2.convexHull(contours[0])
            return hull
        else:
            raise CornerDetectError("No contours found")


if __name__ == "__main__":
    img_path = "data/img4.jpg"
    corner_detector = DetectCorners(img_path)
    corner_detector.detect_plate_corners()
