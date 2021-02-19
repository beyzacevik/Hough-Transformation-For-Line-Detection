import numpy as np
import math
import cv2 as cv

class Hough(object):
    def __init__(self, n_edges, theta_min=-90, theta_max=90,threshold=10):
        self.n_edges = n_edges
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.threshold = threshold
        self.thetas = [i for i in range(abs(self.theta_min) + abs(self.theta_max)+1)]

    def vote(self, image):
        image_shape = image.shape
        y = image_shape[0]
        x = image_shape[1]

        self.rho_max = int(math.hypot(x,y))
        num_rhos = 2*self.rho_max+1

        num_thetas = len(self.thetas)
        accumulator = np.zeros((num_rhos,num_thetas))

        for y in range(y):
            for x in range(x):

                if image[y, x] > self.threshold:
                    for i_theta in self.thetas:
                        theta = np.radians(i_theta)
                        cos_theta = math.cos(theta)
                        sin_theta = math.sin(theta)
                        rho = round(x*cos_theta + y*sin_theta) + self.rho_max
                        accumulator[rho, i_theta] += 1

        return accumulator

    def find_maximas(self, accumulator, n_lines):

        indices = np.argpartition(accumulator.ravel(), - n_lines)[-n_lines:]

        return np.column_stack(np.unravel_index(indices, accumulator.shape))


    def draw_line(self, orig_image,detected_image, line_indices,hough_space):

        for i_rho, i_theta in line_indices:
            theta = math.radians(i_theta)
            a = math.cos(theta)
            b = math.sin(theta)

            rho = i_rho - self.rho_max

            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))


            cv.line(orig_image,(x1,y1),(x2,y2),(0,0,255),2)
            cv.line(detected_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return orig_image, detected_image