import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


def plotHistogram(img, title):
    gray_hist = cv.calcHist([img], [0], None, [256], [0, 256])
    # list of images, channel, mask, bins, range
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist)
    plt.xlim([0, 256])
    plt.show()

def display(title, img, sf=1):
    img = rescale(img, sf)
    cv.imshow(title, img)

def blur(img, kernel_size):
    return cv.GaussianBlur(img, (kernel_size,kernel_size), 0)


def imreconstruct(marker: np.ndarray, mask: np.ndarray, radius: int = 1):
    # Source: https://gist.github.com/Semnodime/ddf1e63d4405084f886204e73ecfabcd
    """Iteratively expand the markers white keeping them limited by the mask during each iteration.
    :param marker: Grayscale image where initial seed is white on black background.
    :param mask: Grayscale mask where the valid area is white on black background.
    :param radius Can be increased to improve expansion speed while causing decreased isolation from nearby areas.
    :returns A copy of the last expansion.
    Written By Semnodime.
    """
    kernel = np.ones(shape=(radius * 2 + 1,) * 2, dtype=np.uint8)
    while True:
        expanded = cv.dilate(src=marker, kernel=kernel)
        cv.bitwise_and(src1=expanded, src2=mask, dst=expanded)

        # Termination criterion: Expansion didn't change the image at all
        if (marker == expanded).all():
            return expanded
        marker = expanded


def imreconstruct_ftc(marker, mask):
    '''

    :param marker:
    :param mask:
    :return:
    Written by Franz Tapia Chaca
    '''


def imcomplement(img):
    return 255 - img