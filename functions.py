import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu

from numba import jit

def rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)



def mask(original, binary):
    '''
    :param original: The input microscopy image
    :param binary: The binary image as a result of segmentation
    :return: The regions of interest on the original image defined by segmentation
    '''
    return cv.bitwise_and(original, original, mask=binary)


def plotHistogram(img1, label1, img2, label2, title):
    gray_hist_1 = cv.calcHist([img1], [0], None, [256], [0, 256])
    gray_hist_2 = cv.calcHist([img2], [0], None, [256], [0, 256])
    # list of images, channel, mask, bins, range
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('# of pixels')
    plt.plot(gray_hist_1, label=label1)
    plt.plot(gray_hist_2, label=label2)
    plt.xlim([0, 256])
    plt.ylim([0, 300000])
    plt.legend()
    plt.show()


def display(title, img, sf=1):
    img = rescale(img, sf)
    cv.imshow(title, img)


def blur(img, kernel_size):
    return cv.GaussianBlur(img, (kernel_size,kernel_size), 0)


def imcomplement(img):
    return 255 - img


def mat2gray(img):
    return (img-np.amin(img))/(np.amax(img)-np.amin(img))

def adaptiveThreshold(img, windowSize, fudgeFactor, imDataType, mode='mean'):
    print("\n")
    print(windowSize)
    tic = time.process_time()
    # 1 ) already a grayscale image
    # print(img)
    img_double = mat2gray(img)
    toc = time.process_time() - tic
    print("mat2gray: " + str(toc))
    # print(img_double)
    # print(np.shape(img_double))

    # 2 ) imfilter
    tic = time.process_time()
    if mode == 'mean':
        kernel = np.ones((windowSize, windowSize), dtype=np.float32)
        kernel /= windowSize**2
        convolved = cv.filter2D(img_double, -1, kernel)  # execute correlation. Returns np.float64
    toc = time.process_time() - tic
    print("filter: " + str(toc))
    # print(convolved)
    # print(np.shape(convolved))

        # convolved = cv.blur(img, (windowSize, windowSize))
    # elif mode == 'median':
    #     convolved = cv.medianBlur(img, windowSize)
    # elif mode == 'Gaussian':
    #     convolved = cv.GaussianBlur(img, (windowSize, windowSize), 0)

    # 3 )           - correct element wise subtraction
    tic = time.process_time()
    subtract = np.subtract(convolved, img_double)
    toc = time.process_time() - tic
    print("sub: " + str(toc))
    #
    # 4 ) Calculates Otsu's threshold for each element
    tic = time.process_time()
    otsu = threshold_otsu(subtract)
    toc = time.process_time() - tic
    print("otsu: " + str(toc))
    # # _, otsu = cv.threshold(substract, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # print(otsu)
    #
    # 5 ) thresholding  with otsu
    tic = time.process_time()
    imDataInfo = np.iinfo(imDataType)
    final = ((subtract > otsu*fudgeFactor) * imDataInfo.max).astype(np.uint8)  # typecast to uint8 to save memory
    # OrganoTrack, _ = cv.threshold(substract, otsu*fudgeFactor, 255, cv.THRESH_BINARY)
    # print(np.shape(OrganoTrack))
    # print(type(OrganoTrack[0][0]))
    toc = time.process_time() - tic
    print("OrganoTrack: " + str(toc))
    return final
