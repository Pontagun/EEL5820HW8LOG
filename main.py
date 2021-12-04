import cv2
import numpy as np
from PIL import Image
import math


def pixel_log255(unorm_image):
    pxmin = unorm_image.min()
    pxmax = unorm_image.max()

    for i in range(unorm_image.shape[0]):
        for j in range(unorm_image.shape[1]):
            # unorm_image[i, j] = (255 / math.log10(256)) * math.log10(1 + (255 / pxmax) * unorm_image[i, j])
            unorm_image[i, j] = ((unorm_image[i, j] - pxmin) / (pxmax - pxmin)) * 255

    norm_image = unorm_image
    return norm_image


def iLoG(shape, std):
    s = (shape, shape)
    l_o_g = np.zeros(s)

    a = 1 / (2 * math.pi * (std ** 4))

    lim = int(math.floor(shape / 2))

    for row in range(-lim, lim + 1):
        for col in range(-lim, lim + 1):
            b = ((row ** 2) + (col ** 2) - (2 * (std ** 2))) / (std ** 2)
            c = np.exp((-1 / (2 * (std ** 2))) * (row ** 2 + col ** 2))
            l_o_g[row + lim][col + lim] = a * b * c

    return l_o_g


if __name__ == '__main__':

    image = cv2.imread('bank 256x256.jpg', 0)

    h = iLoG(17, 2)

    grad = cv2.filter2D(image, -1, h)

    grad = pixel_log255(grad)
    grad = grad.flatten()

    for i, val in enumerate(grad):
        if val > 30:
            grad[i] = 255
        else:
            grad[i] = 0

    grad = grad.reshape((image.shape[0], image.shape[1]))

    im = Image.fromarray(grad)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("grad.jpg")


