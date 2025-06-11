import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.filters import gaussian, sobel
from skimage.transform import hough_line, hough_line_peaks
import numpy as np
from libtiff import TIFF


class Calibrator:
    def __init__(self,
                 left_filepath, # filepath for the left calibration image
                 right_filepath, # filepath for the right calibration image
                 left_dim_tuple, # (int, int) tuple of dimensions for the left calibration grid (squares)
                 right_dim_tuple, # (int, int) tuple of dimensions for the right calibration grid (squares)
                 binary_threshold, # max brightness in sobel image * binary_threshold sets binary threshold
                 orthog_error = 2*(np.pi) / 90, # maximum difference from pi/2 for grid edge lines
                 gaussian_sigma = 5
                )
        self.gaussian_sigma = gaussian_sigma
        self.binary_threshold = binary_threshold
        self.left_binary = tiff_to_binary_outline(left_filepath)
        self.right_binary = tiff_to_binary_outline(right_filepath)
        left_line_r0s, left_line_angles = get_lines(self.left_binary, left_dim_tuple[0]+left_dim_tuple[1]+2)
        self.left_lines = (left_line_r0s, left_line_angles) 
        right_line_r0s, right_line_angles = get_lines(self.right_binary, right_dim_tuple[0]+right_dim_tuple[1]+2)
        self.right_lines = (right_line_r0s,right_line_angles)

    def tiff_to_binary_outline(filepath)
        im = plt.imread(filepath)
        im_blur = gaussian(im,sigma=self.gaussian_sigma)
        im_sobel = sobel(im_blur)
        threshold = im_sobel.max() * self.binary_threshold
        thresholded = im_sobel > threshold
        return thresholded

    def get_lines(binary_outline, num_lines)
        tested_angles = np.linspace(-(np.pi / 2)+(np.pi/2), (np.pi / 2)+(np.pi/2), 360, endpoint=False)
        
        line_r_0s = []
        line_angles = []

        h, theta, d = hough_line(thresholded, theta=tested_angles)
        
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, num_peaks=num_lines, min_distance=100)):
            (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
            line_r_0s.append((x0,y0))
            line_angles.append(np.tan(angle + np.pi / 2))

        return line_r_0s, line_angles
