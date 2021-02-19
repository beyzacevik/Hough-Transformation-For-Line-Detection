import numpy as np
import math
from .Hough import *
from .Canny import *
import os
class Visualize(object):

    def __init__(self, dataset_path, original_path, detection_path):
        self.dataset_path = dataset_path
        self.original_path = original_path
        self.detection_path = detection_path
        self.Hough = Hough(4)
        self.Canny = Canny()

    def execute(self):
        for filename in os.listdir(self.original_path):
            original_image = cv.imread(os.path.join(self.original_path, filename))
            detected_image = cv.imread(os.path.join(self.detection_path, filename))


            image, gray, orig = self.Canny.get_edge_map(original_image, 100)
            hough_space = self.Hough.vote(image)
            line_indices = self.Hough.find_maximas(hough_space, n_lines=20)

            f = plt.figure(figsize=(20, 10))
            f.add_subplot(1, 4, 1)
            plt.imshow(original_image, interpolation='nearest')
            plt.axis("off")
            f.add_subplot(1, 4, 2)
            plt.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
            plt.axis("off")
            # plt.savefig(fname="hello.png")
            orig_img, detected_img = self.Hough.draw_line( original_image, detected_image,line_indices,hough_space)
            f.add_subplot(1, 4, 3)
            plt.imshow(orig_img, interpolation='nearest')
            plt.axis("off")
            f.add_subplot(1, 4, 4)
            plt.imshow(detected_img, interpolation='nearest')
            plt.axis("off")
            plt.show()
