import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


class Canny(object):
    def __init__(self, max_low_thresh_hold=100, ratio=3, kernel_size=3):
        self.max_low_threshold = max_low_thresh_hold
        self.ratio = ratio
        self.kernel_size = kernel_size

    def get_edge_map(self, image, val):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        img_blur = cv.blur(gray, (3, 3))
        low_threshold = val
        edges = cv.Canny(img_blur, low_threshold, 150, self.kernel_size)
        mask = edges != 0
        dst = image * (mask[:, :, None].astype(image.dtype))
        return edges, gray, dst
