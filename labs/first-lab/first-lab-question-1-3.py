import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from shared_functions import get_name
from shared_functions import get_parent
from shared_functions import extract_channel
from shared_functions import convert_bgr_to_int
from shared_functions import extract_background
from shared_functions import extract_foreground

def show_cv2_rgb_image(color_fname: str):
    file_name = get_name(color_fname)
    parent = get_parent(color_fname)
    color_fname.rindex('/')
    cv2_img = cv2.imread(color_fname)
    cv2_img = cv2.resize(cv2_img, (600, 480))

    cv2.imshow('Original image', cv2_img)
    cv2.imwrite('{}/channel_{}_rgb.png'.format(parent, file_name), cv2_img)
    cv2.imshow('Gray scale', cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY))
    cv2.imwrite('{}/channel_{}_rgb_grey.png'.format(parent, file_name), cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY))
    cv2.imshow('Red channel', extract_channel('RED', cv2_img))
    cv2.imwrite('{}/channel_{}_red.png'.format(parent, file_name), extract_channel('RED', cv2_img))
    cv2.imshow('Green channel', extract_channel('GREEN', cv2_img))
    cv2.imwrite('{}/channel_{}_green.png'.format(parent, file_name), extract_channel('GREEN', cv2_img))
    cv2.imshow('Blue channel', extract_channel('BLUE', cv2_img))
    cv2.imwrite('{}/channel_{}_blue.png'.format(parent, file_name), extract_channel('BLUE', cv2_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_cv2_hsv_image(color_fname: str):
    file_name = get_name(color_fname)
    parent = get_parent(color_fname)
    cv2_img = cv2.imread(color_fname)
    cv2_img = cv2.resize(cv2_img, (600, 480))
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2HSV)
    cv2.imshow('Original image', cv2_img)
    cv2.imwrite('{}/channel_{}_hsv.png'.format(parent, file_name), cv2_img)
    cv2.imshow('Hue channel', extract_channel('HUE', cv2_img))
    cv2.imwrite('{}/channel_{}_hue.png'.format(parent, file_name), extract_channel('HUE', cv2_img))
    cv2.imshow('Saturation channel', extract_channel('SATURATION', cv2_img))
    cv2.imwrite('{}/channel_{}_saturation.png'.format(parent, file_name), extract_channel('SATURATION', cv2_img))
    cv2.imshow('Value channel', extract_channel('VALUE', cv2_img))
    cv2.imwrite('{}/channel_{}_value.png'.format(parent, file_name), extract_channel('VALUE', cv2_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def minimum_maximum_intensities(gray_img):
    return np.min(gray_img[:, :]), np.max(gray_img[:, :]), np.mean(gray_img), np.var(gray_img)


def descriptive_statistics(color_fname: str, mask_fname: str):
    img = mpimg.imread(color_fname)
    mask_img = mpimg.imread(mask_fname)
    black_pixel_indexes = mask_img[:, :, 0] == 0
    white_pixel_indexes = mask_img[:, :, 0] == 1
    black_pixel_count = np.sum(black_pixel_indexes)
    white_pixel_count = np.sum(white_pixel_indexes)

    cv2_img = cv2.imread(color_fname)
    cv2_gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

    whole_img_minimum, whole_img_maximum, whole_mean, whole_variance = minimum_maximum_intensities(cv2_gray)
    bg_img_minimum, bg_img_maximum, bg_mean, bg_variance = minimum_maximum_intensities(
        np.multiply(cv2_gray, black_pixel_indexes))
    fg_img_minimum, fg_img_maximum, fg_mean, fg_variance = minimum_maximum_intensities(
        np.multiply(cv2_gray, white_pixel_indexes))

    print('1. What is the width of the image?', img.shape[1])
    print('2. What is the height of the image?', img.shape[0])
    print('3. How many pixels are in the image in total', img.shape[0] * img.shape[1])
    print('4. How many black pixels are in the mask?', black_pixel_count)
    print('5. How many white pixels are in the mask?', white_pixel_count)
    print('6. Minimum pixel value in the image?', whole_img_minimum)
    print('7. Maximum pixel value in the image?', whole_img_maximum)
    print('8. What are the minimum and maximum pixel values of the puzzle pixels? [min={} max={}]'.format(
        fg_img_minimum, fg_img_maximum))
    print('9. What are the minimum and maximum pixel values of the background pixels? [min={} max={}]'.format(
        bg_img_minimum, bg_img_maximum))
    print('10. What is the mean pixel intensity in the image? {:.2f}'.format(whole_mean))
    print('11. What is the mean brightness of the puzzle pixels? {:.2f}'.format(fg_mean))
    print('12. What is the mean brightness of the background pixels? {:.2f}'.format(bg_mean))
    print('13. What is the variance in the greyscale intensities for puzzle pixels? {:.2f}'.format(fg_variance))
    print('14. What is the variance in the greyscale intensities for background pixels? {:.2f}'.format(bg_variance))


def display_foreground(color_fname: str, mask_fname: str, channel: str = None):
    cv2_img = extract_foreground(color_fname, mask_fname)
    cv2_img = cv2.resize(cv2_img, (600, 480))
    if channel is not None:
        cv2_img = extract_channel(channel, cv2_img)
    cv2.imshow('Foreground', cv2_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_background(color_fname: str, mask_fname: str, channel: str = None):
    cv2_img = extract_background(color_fname, mask_fname)
    cv2_img = cv2.resize(cv2_img, (600, 480))
    if channel is not None:
        cv2_img = extract_channel(channel, cv2_img)
    cv2.imshow('Background', cv2_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def display_channel_histogram(output_name: str, color_image, channel: str = None, kde: bool = False):
    image_values = np.sum(extract_channel(channel, color_image), axis=2).ravel().tolist() if channel is not None \
        else convert_bgr_to_int(color_image)
    new_array = []
    for i in image_values:
        if i > 0:
            new_array.append(i)
    save_figure(output_name, new_array, kde, channel)


def save_figure(output_name, new_array, kde: bool = False, channel=None):
    sns.set()
    plt.figure()
    sns.distplot(new_array, hist=True, norm_hist=True, kde=kde)
    output_file = '../../resources/puzzle-pieces/{}_{}_{}_kde.png'.format(output_name, channel, 'with' if kde else 'no')
    plt.savefig(output_file)
    print('Finished saving', output_file)


if __name__ == '__main__':
    show_cv2_rgb_image('../../resources/puzzle-pieces/image-50.jpg')
    show_cv2_hsv_image('../../resources/puzzle-pieces/image-50.jpg')
    display_channel_histogram('HISTOGRAM_IMAGE_50', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), 'RED')
    display_channel_histogram('HISTOGRAM_IMAGE_50', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), 'GREEN')
    display_channel_histogram('HISTOGRAM_IMAGE_50', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), 'BLUE')
    display_channel_histogram('HISTOGRAM_IMAGE_50',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV),
                              'HUE')
    display_channel_histogram('HISTOGRAM_IMAGE_50',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV),
                              'SATURATION')
    display_channel_histogram('HISTOGRAM_IMAGE_50',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV),
                              'VALUE')
    display_channel_histogram('HISTOGRAM_IMAGE_50_RGB', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'))
    display_channel_histogram('HISTOGRAM_IMAGE_50_HSV',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV))
    save_figure('HISTOGRAM_IMAGE_50_GRAY',
                cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2GRAY).ravel())
    display_channel_histogram('MASK_50', extract_foreground('../../resources/puzzle-pieces/image-50.jpg',
                                                            '../../resources/puzzle-pieces/image-50.png'), 'RED')
    display_channel_histogram('MASK_50', extract_foreground('../../resources/puzzle-pieces/image-50.jpg',
                                                            '../../resources/puzzle-pieces/image-50.png'), 'GREEN')
    display_channel_histogram('MASK_50', extract_foreground('../../resources/puzzle-pieces/image-50.jpg',
                                                            '../../resources/puzzle-pieces/image-50.png'), 'BLUE')

    display_channel_histogram('HISTOGRAM_IMAGE_50', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), 'RED', kde=True)
    display_channel_histogram('HISTOGRAM_IMAGE_50', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), 'GREEN', kde=True)
    display_channel_histogram('HISTOGRAM_IMAGE_50', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), 'BLUE', kde=True)
    display_channel_histogram('HISTOGRAM_IMAGE_50_RGB', cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), kde=True)
    display_channel_histogram('HISTOGRAM_IMAGE_50',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV),
                              'HUE', kde=True)
    display_channel_histogram('HISTOGRAM_IMAGE_50',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV),
                              'SATURATION', kde=True)
    display_channel_histogram('HISTOGRAM_IMAGE_50',
                              cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2HSV),
                              'VALUE', kde=True)
    save_figure('HISTOGRAM_IMAGE_50_GRAY',
                cv2.cvtColor(cv2.imread('../../resources/puzzle-pieces/image-50.jpg'), cv2.COLOR_BGR2GRAY).ravel(),
                kde=True)
    display_channel_histogram('HISTOGRAM_MASK_50', extract_foreground('../../resources/puzzle-pieces/image-50.jpg',
                                                            '../../resources/puzzle-pieces/image-50.png'), 'RED',
                              kde=True)
    display_channel_histogram('HISTOGRAM_MASK_50', extract_foreground('../../resources/puzzle-pieces/image-50.jpg',
                                                            '../../resources/puzzle-pieces/image-50.png'), 'GREEN',
                              kde=True)
    display_channel_histogram('HISTOGRAM_MASK_50', extract_foreground('../../resources/puzzle-pieces/image-50.jpg',
                                                            '../../resources/puzzle-pieces/image-50.png'), 'BLUE',
                              kde=True)

    display_foreground('../../resources/puzzle-pieces/image-50.jpg', '../../resources/puzzle-pieces/image-50.png')

    descriptive_statistics('../../resources/puzzle-pieces/image-50.jpg', '../../resources/puzzle-pieces/image-50.png')
