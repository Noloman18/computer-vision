import math

import cv2
import numpy as np

from shared_functions import convolve
from shared_functions import horizontal_prewitt
from shared_functions import laplacian
from shared_functions import load_image
from shared_functions import vertical_prewitt
from shared_functions import SHOULD_RESIZE

SHOULD_FLIP = False

def validate(sub_img, img_tuple):
    invalid = np.sum(sub_img[:, :, :] != img_tuple[:, :, :])
    if invalid > 0:
        raise Exception("Stacking error occurred")


def stack_images(image_tuple: tuple):
    num_tuples = len(image_tuple)
    feature_img = np.zeros((image_tuple[0].shape[0], image_tuple[0].shape[1], image_tuple[0].shape[2] * num_tuples))

    for i in range(num_tuples):
        idx = 3 * i
        feature_img[:, :, idx:idx + 3] = image_tuple[i]
        validate(feature_img[:, :, idx:idx + 3], image_tuple[i])

    return feature_img


def compute_feature_matrix(img, alt_repr_img):
    img_1 = img if not SHOULD_FLIP else alt_repr_img
    img_2 = alt_repr_img if not SHOULD_FLIP else img
    h_prewitt = convolve(img_1, horizontal_prewitt())
    v_prewitt = convolve(img_1, vertical_prewitt())
    custom_laplacian = convolve(img_1, laplacian())
    return stack_images((img_1, img_2, h_prewitt, v_prewitt, custom_laplacian))


def compute_feature_matrix_and_reshape(img, alt_repr_img, fmask: str):
    mask_img = load_image(fmask)
    mask_img = mask_img[:, :, 0]
    features = compute_feature_matrix(img, alt_repr_img)
    number_of_background_pixels = np.sum(mask_img == 0)
    train_features = np.zeros((number_of_background_pixels, features.shape[2]), dtype=np.float)

    idx = 0
    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            if mask_img[y, x] == 0:
                train_features[idx] = features[y, x, :]
                idx += 1
    return train_features


def calculate_probablity(mean, covariance, ith_feature):
    comp1 = 1 / (math.pow(2 * math.pi, 7.5) * np.power(np.linalg.det(covariance), 0.5))
    covariance_inv = np.linalg.inv(covariance)
    first_term = -0.5 * (ith_feature - mean).transpose()
    second_term = np.matmul(first_term, covariance_inv)
    third_term = (ith_feature - mean)
    comp2 = np.matmul(second_term, third_term)
    return comp1 * math.exp(comp2)


def calculate_epsilon(mean, covariance, img, alt_repr_img, mask):
    features = compute_feature_matrix(img, alt_repr_img)
    mask_img = load_image(mask)
    mask_img = np.sum(mask_img, axis=2)

    comp1 = 1 / (math.pow(2 * math.pi, 7.5) * np.power(np.linalg.det(covariance), 0.5))
    covariance_inv = np.linalg.inv(covariance)

    average_background = []
    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            if mask_img[y, x] == 0:
                ith_feature = features[y, x, :]
                first_term = -0.5 * (ith_feature - mean).transpose()
                second_term = np.matmul(first_term, covariance_inv)
                third_term = (ith_feature - mean)
                comp2 = np.matmul(second_term, third_term)
                result = comp1 * math.exp(comp2)
                average_background.append(result)

    return np.average(average_background)


def evaluate_features(mean, covariance, epsilon, img, alt_repr_img, mask):
    features = compute_feature_matrix(img, alt_repr_img)
    mask_img = load_image(mask)
    mask_img = np.sum(mask_img, axis=2)

    comp1 = 1 / (math.pow(2 * math.pi, 7.5) * np.power(np.linalg.det(covariance), 0.5))
    covariance_inv = np.linalg.inv(covariance)

    tp = np.sum(mask_img == 0)
    tn = np.sum(mask_img != 0)
    fp = 0
    fn = 0

    for y in range(features.shape[0]):
        for x in range(features.shape[1]):
            ith_feature = features[y, x, :]
            first_term = -0.5 * (ith_feature - mean).transpose()
            second_term = np.matmul(first_term, covariance_inv)
            third_term = (ith_feature - mean)
            comp2 = np.matmul(second_term, third_term)
            result = comp1 * math.exp(comp2)
            if mask_img[y, x] == 0 and result < epsilon:
                fn += 1
            elif mask_img[y, x] != 0 and result >= epsilon:
                fp += 1

    return tp / (tp + fp), tp / (tp + fn), (tp + tn) / (tp + tn + fp + fn)


if __name__ == '__main__':
    SHOULD_RESIZE = False
    rgb_train = load_image('../../resources/puzzle-pieces/image-50.jpg')
    hsv_train = cv2.cvtColor(rgb_train, cv2.COLOR_BGR2HSV)
    reshaped_train_features = compute_feature_matrix_and_reshape(rgb_train, hsv_train,
                                                                 '../../resources/puzzle-pieces/image-50.png')
    train_mean = np.mean(reshaped_train_features, axis=0)
    feature_covariance = np.cov(reshaped_train_features.transpose())
    train_epsilon = calculate_epsilon(train_mean, feature_covariance, rgb_train, hsv_train,
                                      '../../resources/puzzle-pieces/image-50.png')

    # Testing...
    test_image = '../../resources/puzzle-pieces/image-102.png'
    rgb_test = load_image('../../resources/puzzle-pieces/image-102.jpg')
    hsv_test = cv2.cvtColor(rgb_test, cv2.COLOR_BGR2HSV)

    precision, recall, accuracy = evaluate_features(train_mean, feature_covariance, train_epsilon/10000000000, rgb_test, hsv_test,
                                                    test_image)

    print('Precision', precision, 'Recall', recall, 'Accuracy', accuracy)
