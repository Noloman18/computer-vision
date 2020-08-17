import math

import numpy as np
import cv2

from shared_functions import load_image, show_img_side_by_side


def create_gaussian_filter(kernel_size: int = 11, sigma: float = None):
    if sigma is None:
        sigma = (kernel_size - 1) / 6

    constant_term = 1 / (2 * math.pi * math.pow(sigma, 2))
    g_filter = np.zeros((kernel_size, kernel_size), dtype=np.float)
    middle = int(kernel_size / 2)
    for y in range(-middle, middle + 1):
        for x in range(-middle, middle + 1):
            the_exponent = math.exp(-1 * ((math.pow(x, 2) + math.pow(y, 2)) / (2 * math.pow(sigma, 2))))
            g_filter[y + middle, x + middle] = constant_term * the_exponent
    return g_filter


def create_log_filter(kernel_size: int = 11, sigma: float = None):
    g_filter = np.zeros((kernel_size, kernel_size), dtype=np.float)
    middle = int(kernel_size / 2)

    if sigma is None:
        sigma = middle/2

    constant_term = -1 / (math.pi * math.pow(sigma, 4))
    sigma_constant_term = 2 * math.pow(sigma, 2)

    for y in range(-middle, middle + 1):
        for x in range(-middle, middle + 1):
            x2_y2 = math.pow(x, 2) + math.pow(y, 2)
            the_exponent = math.exp(-1 * x2_y2 / sigma_constant_term)
            g_filter[y + middle, x + middle] = \
                constant_term * (1 - x2_y2 / sigma_constant_term) * the_exponent

        return g_filter


def create_dog_filter(kernel_size: int = 11, sigma: float = 1.0, K:float = 1.0):
    g_filter = np.zeros((kernel_size, kernel_size), dtype=np.float)
    middle = int(kernel_size / 2)

    constant_term = 1 / (2 * math.pi * math.pow(sigma, 2))
    k_constant_term = 1 / math.pow(K, 2)
    sigma_constant_term = 1 / math.pow(sigma, 2)

    blah = k_constant_term * constant_term

    for y in range(-middle, middle + 1):
        for x in range(-middle, middle + 1):
            x2_y2 = math.pow(x, 2) + math.pow(y, 2)
            the_exponent = math.exp(-1 * x2_y2 / sigma_constant_term)
            g_filter[y + middle, x + middle] = \
                constant_term * (1 - x2_y2 / sigma_constant_term) * the_exponent

        return g_filter

if __name__ == '__main__':
    color_img = load_image('../../resources/puzzle-pieces/image-102.jpg')
    show_img_side_by_side(
        color_img,
        cv2.Laplacian(color_img, 3, 3),
        'Gaussian filter',
        '../../resources/puzzle-pieces/filters/image-102_k3_sigma_2_log.jpg')
