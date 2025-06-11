import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.filters import gaussian, sobel
from skimage.transform import hough_line, hough_line_peaks, hough_circle, hough_circle_peaks
import numpy as np
from libtiff import TIFF


class Calibrator:
    def __init__(self,
                 left_filepath, # filepath for the left calibration image
                 right_filepath, # filepath for the right calibration image
                 left_dim_tuple, # (int, int) tuple of dimensions for the left calibration grid (squares)
                 right_dim_tuple, # (int, int) tuple of dimensions for the right calibration grid (squares)
                 binary_threshold, # max brightness in sobel image * binary_threshold sets binary threshold
                 gaussian_sigma = 5,
                 min_distance = 100
                )
        self.gaussian_sigma = gaussian_sigma
        self.binary_threshold = binary_threshold
        self.min_distance = min_distance
        self.left_binary = tiff_to_binary_outline(left_filepath)
        self.right_binary = tiff_to_binary_outline(right_filepath)
        self.left_intersections = get_intersections(self.left_binary,left_dim_tuple[0]+left_dim_tuple[1]+2)
        self.right_intersections = get_intersections(self.right_binary,right_dim_tuple[0]+right_dim_tuple[1]+2)
        


    def tiff_to_binary_outline(filepath)
        im = plt.imread(filepath)
        im_blur = gaussian(im,sigma=self.gaussian_sigma)
        im_sobel = sobel(im_blur)
        threshold = im_sobel.max() * self.binary_threshold
        thresholded = im_sobel > threshold
        return thresholded

    def get_circles(binary_outline, grid_dim, num_circles = 3)
        shape = binary_outline.shape
        rad_min = int(min(shape) / (4*(min(GRID_DIMENSIONS)+2)))
        rad_max = int(max(shape) / (4*(max(GRID_DIMENSIONS) - 2)))
        hough_radii = np.arange(rad_min, rad_max, 10)
        hough_res = hough_circle(thresholded, hough_radii)
        
        accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=3, min_xdistance = self.min_distance, min_ydistance = self.min_distance)
    def get_intersections(binary_outline, num_lines)
        tested_angles = np.linspace(-(np.pi / 2)+(np.pi/2), (np.pi / 2)+(np.pi/2), 360, endpoint=False)
        
        line_r_0s = []
        line_angles = []

        h, theta, d = hough_line(thresholded, theta=tested_angles)
        peaks = hough_line_peaks(h, theta, d, num_peaks=num_lines, min_distance=self.min_distance)
        intersections = hough_intersections(peaks, thresholded.shape)

        return intersections
    def line_intersection(x1, y1, angle1, x2, y2, angle2):
        # Direction vectors
        dx1, dy1 = np.cos(angle1 + np.pi/2), np.sin(angle1 + np.pi/2)
        dx2, dy2 = np.cos(angle2 + np.pi/2), np.sin(angle2 + np.pi/2)
        
        # Check if lines are parallel
        det = dx1 * dy2 - dx2 * dy1
        if abs(det) < 1e-10:
            return None
        
        # Solve for intersection
        # Line 1: (x1, y1) + t1 * (dx1, dy1)
        # Line 2: (x2, y2) + t2 * (dx2, dy2)
        # Set equal and solve for t1
        
        t1 = ((x2 - x1) * dy2 - (y2 - y1) * dx2) / det
        
        x_intersect = x1 + t1 * dx1
        y_intersect = y1 + t1 * dy1
        
        return (x_intersect, y_intersect)

    def find_all_intersections(points_angles):
        intersections = []
        n = len(points_angles)
        
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1, angle1 = points_angles[i]
                x2, y2, angle2 = points_angles[j]
                
                intersection = line_intersection(x1, y1, angle1, x2, y2, angle2)
                
                if intersection is not None:
                    intersections.append({
                        'point': intersection,
                        'line1_idx': i,
                        'line2_idx': j
                    })
        
        return intersections

    def hough_intersections(hough_results, image_shape=None, filter_bounds=True):
        _, angles, distances = hough_results
        
        points_angles = []
        for angle, dist in zip(angles, distances):
            # Point on line closest to origin
            x0 = dist * np.cos(angle)
            y0 = dist * np.sin(angle)
            points_angles.append((x0, y0, angle))
        
        intersections = find_all_intersections(points_angles)
        
        if filter_bounds and image_shape is not None:
            height, width = image_shape
            margin = 10  # Allow small margin outside image
            filtered = []
            for inter in intersections:
                x, y = inter['point']
                if -margin <= x <= width + margin and -margin <= y <= height + margin:
                    filtered.append(inter)
            intersections = filtered
        
        return intersections
