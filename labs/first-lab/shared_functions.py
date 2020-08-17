import cv2
import matplotlib.image as mpimg
import numpy as np

SHOULD_RESIZE = True

def get_name(file_name: str):
    components = file_name.split('/')
    full_name = components[len(components) - 1]
    return full_name[0:-4]


def get_parent(file_name: str):
    idx = file_name.rindex('/')
    return file_name[0:idx]


def extract_channel(channel: str, color_image):
    new_image = color_image.copy()
    if channel == 'RED':
        new_image[:, :, 0] = 0
        new_image[:, :, 1] = 0
    elif channel == 'GREEN':
        new_image[:, :, 0] = 0
        new_image[:, :, 2] = 0
    elif channel == 'BLUE':
        new_image[:, :, 1] = 0
        new_image[:, :, 2] = 0
    elif channel == 'HUE':
        new_image[:, :, 0] = 0
        new_image[:, :, 1] = 0
    elif channel == 'SATURATION':
        new_image[:, :, 0] = 0
        new_image[:, :, 2] = 0
    elif channel == 'VALUE':
        new_image[:, :, 1] = 0
        new_image[:, :, 2] = 0

    flattened_array = np.sum(new_image, axis=2)
    print('Channel={}, Min={}, Max={}'.format(channel, np.min(flattened_array), np.max(flattened_array)))
    return new_image


def convert_bgr_to_int(color_image):
    _x_y_z = color_image.shape
    c_offset = _x_y_z[0] * _x_y_z[1]
    y_offset = _x_y_z[0]

    flattened_image = color_image.ravel().tolist()

    pixel_list = []
    for y in range(color_image.shape[0]):
        for x in range(color_image.shape[1]):
            r = flattened_image[2 * c_offset + y * y_offset + x]
            g = flattened_image[1 * c_offset + y * y_offset + x]
            b = flattened_image[0 * c_offset + y * y_offset + x]
            pixel_list.append((r << 16) + (g << 8) + b)
    return pixel_list


def horizontal_prewitt():
    return np.array([[1, 0, -1],
                     [1, 0, -1],
                     [1, 0, -1]])


def vertical_prewitt():
    return np.array([[1, 1, 1],
                     [0, 0, 0],
                     [-1, -1, -1]])


def laplacian():
    return np.array([[0, 1, 0],
                     [1, -4, 1],
                     [0, 1, 0]])


def extract_foreground(color_fname: str, mask_fname: str):
    mask_img = mpimg.imread(mask_fname)
    white_pixel_indexes = mask_img[:, :, 0] == 1
    cv2_img = cv2.imread(color_fname)
    cv2_img[:, :, 0] = np.multiply(cv2_img[:, :, 0], white_pixel_indexes)
    cv2_img[:, :, 1] = np.multiply(cv2_img[:, :, 1], white_pixel_indexes)
    cv2_img[:, :, 2] = np.multiply(cv2_img[:, :, 2], white_pixel_indexes)
    return cv2_img


def extract_background(color_fname: str, mask_fname: str):
    mask_img = mpimg.imread(mask_fname)
    black_pixel_indexes = mask_img[:, :, 0] == 0
    cv2_img = cv2.imread(color_fname)
    cv2_img[:, :, 0] = np.multiply(cv2_img[:, :, 0], black_pixel_indexes)
    cv2_img[:, :, 1] = np.multiply(cv2_img[:, :, 1], black_pixel_indexes)
    cv2_img[:, :, 2] = np.multiply(cv2_img[:, :, 2], black_pixel_indexes)
    return cv2_img


def resize_image(color_img):
    if SHOULD_RESIZE:
        return cv2.resize(color_img, (640, 480))
    else:
        return color_img


def load_image(color_image_name: str):
    color_image = cv2.imread(color_image_name)
    color_image = resize_image(color_image)
    return color_image


def show_img_side_by_side(cv2_img_1, cv2_img_2, title: str = 'Not specified', output_path: str = None):
    stacked = np.hstack((cv2_img_1, cv2_img_2))
    cv2.imshow(title, stacked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output_path is not None:
        cv2.imwrite(output_path, stacked)
        print('Finished saving image to path', output_path)


def convolve(img, kernel, should_clip: bool = True):
    result = cv2.filter2D(img, -1, kernel)
    if should_clip:
        result[result < 0] = 0
        result[result > 255] = 255
    return result
