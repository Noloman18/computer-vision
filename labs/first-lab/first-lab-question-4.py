import cv2
import numpy as np

from shared_functions import get_name
from shared_functions import get_parent


def image_statistics(title, image):
    print(title)
    print(image.shape)
    print('Mean:', np.mean(image))
    print('Deviation:', np.std(image))
    print('Max:', np.max(image))
    print('Min:', np.min(image))


def perform_contrast_stretch(image_name):
    fname = get_name(image_name)
    parent = get_parent(image_name)
    original_img = cv2.imread(image_name)
    original_img = cv2.resize(original_img, (600, 480))
    modified_img = original_img.copy()

    for i in range(3):
        ith_channel = modified_img[:, :, i]
        min, max = np.min(ith_channel), np.max(ith_channel)
        multiplier = 255 / (max - min)
        modified_img[:, :, i] = multiplier * (ith_channel-min)

    cv2.imwrite('{}/CONTRAST_STRETCHED_{}.png'.format(parent, fname), modified_img)
    cv2.imshow('Original image', original_img)
    cv2.imshow('Contrast, stretched image', modified_img)
    image_statistics('Original', original_img)
    image_statistics('Stretched', modified_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perform_histogram_equalization(image_name):
    fname = get_name(image_name)
    parent = get_parent(image_name)
    original_img = cv2.imread(image_name)
    original_img = cv2.resize(original_img, (600, 480))
    modified_img = original_img.copy()

    for i in range(3):
        ith_channel = modified_img[:, :, i]
        modified_img[:, :, i] = cv2.equalizeHist(ith_channel)

    cv2.imwrite('{}/HISTOGRAM_EQUALIZED_{}.png'.format(parent, fname), modified_img)
    cv2.imshow('Original image', original_img)
    cv2.imshow('Equalized', modified_img)
    image_statistics('Original', original_img)
    image_statistics('Equalized', modified_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    perform_contrast_stretch('../../resources/puzzle-pieces/image-50.jpg')
    perform_histogram_equalization('../../resources/puzzle-pieces/image-50.jpg')
