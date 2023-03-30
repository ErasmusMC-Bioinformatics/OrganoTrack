import time
import os
from datetime import datetime
import cv2 as cv
import numpy as np
from skimage.morphology import reconstruction
from skimage.filters import threshold_otsu


def imcomplement(img):
    return 255 - img


def mat2gray(img):
    return (img-np.amin(img))/(np.amax(img)-np.amin(img))

def adaptiveThreshold(img, windowSize, fudgeFactor, imDataType, mode='mean'):
    # print("\n")
    # print(windowSize)
    tic = time.process_time()
    # 1 ) already a grayscale image
    # print(img)
    img_double = mat2gray(img)
    toc = time.process_time() - tic
    # print("mat2gray: " + str(toc))
    # print(img_double)
    # print(np.shape(img_double))

    # 2 ) imfilter
    tic = time.process_time()
    if mode == 'mean':
        kernel = np.ones((windowSize, windowSize), dtype=np.float32)
        kernel /= windowSize**2
        convolved = cv.filter2D(img_double, -1, kernel)  # execute correlation. Returns np.float64
    toc = time.process_time() - tic
    # print("filter: " + str(toc))
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
    # print("sub: " + str(toc))
    #
    # 4 ) Calculates Otsu's threshold for each element
    tic = time.process_time()
    otsu = threshold_otsu(subtract)
    toc = time.process_time() - tic
    # print("otsu: " + str(toc))
    # # _, otsu = cv.threshold(substract, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # print(otsu)
    #
    # 5 ) thresholding  with otsu
    tic = time.process_time()
    imDataInfo = np.iinfo(imDataType)
    final = ((subtract > otsu*fudgeFactor) * imDataInfo.max).astype(np.uint8)  # typecast to uint8 to save memory
    # final, _ = cv.threshold(substract, otsu*fudgeFactor, 255, cv.THRESH_BINARY)
    # print(np.shape(final))
    # print(type(final[0][0]))
    toc = time.process_time() - tic
    # print("final: " + str(toc))
    return final



def imreconstruct(marker, mask, imDataType):

    # > Clip values to the level of mask before reconstruction
    if np.any(marker > mask):

        above = np.where(marker > mask)     # tuple of x-coordinates and y-coordinates, e.g. (555, 555) and (1000,1000)
        #                                     thus, (array([ 555, 1000]), array([ 555, 1000]))
        #                                     dim = 2 x n, n = number of px satisfying condition

        for px in range(len(above[0])):  # for each px in marker > mask, clip to mask
            marker[above[0][px], above[1][px]] = mask[above[0][px], above[1][px]]  # for arrays, [,] faster than [][]

    return (reconstruction(marker, mask)).astype(imDataType)  # reconstruction returns float64. Convert to e.g. uint8


def SegmentWithOrganoSegPy(img):
    '''
    segmentWithOrganoSegPy: segments
    :param img:
    :return:
    '''

    imageDisplayScale = 0.25
    imgDataType = img.dtype  # image.dtype = the bitsize of the image, e.g. uint8 (0 - 255) or uint16 (0 65535)

    # Converting to grayscale
    if img.shape == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    times = []  # Stores duration of each operation

    '''
        Smoothening
    '''
    # Sharpening
    tic = time.process_time()
    kernel_size = 5     # as in OrganoSeg
    std_dev = 1         # as in OrganoSeg
    img_gauss = cv.GaussianBlur(img, (kernel_size, kernel_size), std_dev)
    weight = 0.8        # as in OrganoSeg
    img_sharp = cv.addWeighted(img, 1+weight, img_gauss, -weight, 0)  # Unsharp masking method
    toc = time.process_time() - tic
    times.append(toc)
    # display('01', img_sharp, imageDisplayScale)

    # Median blurring 0.014, 0.017, 0.007 s
    tic = time.process_time()
    median_kernel = 3       # as in OrganoSeg
    img_blur = cv.medianBlur(img_sharp, median_kernel)
    toc = time.process_time() - tic
    times.append(toc)
    # display('02', img_blur, imageDisplayScale)

    '''
        Opening
    '''

    # Erosion 0.032, 0.032, 0.021 s
    tic = time.process_time()
    se_size = 5                     # Structuring element size, as in OrganoSeg
    se_shape = cv.MORPH_ELLIPSE     # Structuring element shape, as in OrganoSeg
    structuring_element = cv.getStructuringElement(se_shape, (2 * se_size - 1, 2 * se_size - 1))
    img_eroded = cv.erode(img_blur, structuring_element)
    toc = time.process_time() - tic
    times.append(toc)
    # display('03', img_eroded, imageDisplayScale)

    # Reconstruct 3.022, 3.453, 2.159 s
    tic = time.process_time()
    img_reconstructed_1 = imreconstruct(img_eroded, img_blur, imgDataType)
    toc = time.process_time() - tic
    times.append(toc)
    # display('04', img_reconstructed_1, imageDisplayScale)   # reconstructed


    '''
        Closing
    '''
    # Dilation 2.307, 2.419, 1.871 s
    tic = time.process_time()
    img_dilated = cv.dilate(img_reconstructed_1, structuring_element)
    # display('05', img_dilated, imageDisplayScale)
    img_reconstructed_2 = imreconstruct(np.invert(img_dilated), np.invert(img_reconstructed_1), imgDataType)
    # display('06 again', img_reconstructed_2, imageDisplayScale)
    img_smoothed = np.invert(img_reconstructed_2)
    toc = time.process_time() - tic
    times.append(toc)
    # display('05', img_smoothed, imageDisplayScale)

    '''
        Adaptive thresholding
    '''
    # # OpenCV method
    # adaptiveSum = np.zeros((np.shape(img)[0], np.shape(img)[1]))
    # for j in range(20, 251, 10):
    #     adaptiveIter = cv.adaptiveThreshold(img_smoothed, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, j-1, 2)
    #     adaptiveSum += adaptiveIter

    # adaptive thresholding 16.824, 16.448, 14.636 s
    tic = time.process_time()
    adaptiveSum = np.zeros(img.shape, dtype=imgDataType)
    fudgeFactor = 0.5
    # windowSize = 250
    # img_double = mat2gray(img_smoothed)
    # kernel = np.ones((windowSize, windowSize), dtype=np.float32)
    # kernel /= windowSize ** 2
    # corr = correlate(img_double, kernel)
    # print(type(corr))
    # print(corr.shape)
    # print(type(corr[0,0]))
    # convolved = cv.filter2D(img_double, -1, kernel)
    # print(type(convolved))
    # print(convolved.shape)
    # print(type(convolved[0,0]))

    for windowSize in range(20, 251, 10):
        adaptiveIter = adaptiveThreshold(img_smoothed, windowSize, fudgeFactor, imgDataType)
        adaptiveSum = np.add(adaptiveSum, adaptiveIter)

    toc = time.process_time() - tic
    times.append(toc)
    # display('06', adaptiveSum, imageDisplayScale)

    '''
        Removing small noise
    '''
    # 0.370, 0.243, 0.438 s
    tic = time.process_time()
    # code from https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv

    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(adaptiveSum)
    # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information.
    # here, we're interested only in the size of the blobs, contained in the last column of stats.
    sizes = stats[:, -1]
    # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
    # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below.
    sizes = sizes[1:]
    nb_blobs -= 1

    # minimum size of particles we want to keep (number of pixels).
    # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
    min_size = 150

    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs, dtype=np.uint8)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255

    toc = time.process_time() - tic
    times.append(toc)
    # display('07', im_result, imageDisplayScale)

    '''
        Smoothen
    '''
    # 0.0139, 0.0163, 0.004 s
    tic = time.process_time()
    kernel = np.ones((3, 3), np.uint16)
    binary = cv.morphologyEx(im_result, cv.MORPH_CLOSE, kernel)
    toc = time.process_time() - tic
    times.append(toc)
    # display('08 smoothed', binary, imageDisplayScale)

    # '''
    #     Removing boundary objects
    # '''
    # # 0.0276, 0.0149, 0.016 s
    # tic = time.process_time()
    # # Find contours in the binary image
    # contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #
    # # # This code visualises the contours found
    # # h, w = binary.shape[:2]
    # # blank = np.zeros((h, w), np.uint8)
    # # maxLength = len(contours[0])
    # # maxIndex = 0
    # # for j in range(len(contours)):
    # #     if len(contours[j]) > maxLength:
    # #         maxLength = len(contours[j])
    # #         maxIndex = j
    # #     for i in range(len(contours[j])):
    # #         blank[contours[j][i][0][1]][contours[j][i][0][0]] = 255
    # # print(maxIndex)  # 632
    # # display('removed', blank, 0.5)
    #
    # # Iterate over the contours and remove the ones that are partially in the image
    # for contour in contours:
    #     x, y, w, h = cv.boundingRect(contour)
    #     # openCV documentation: "Calculates and returns the minimal up-right bounding rectangle for the specified point set"
    #
    #     if x == 0 or y == 0 or x+w == img.shape[1] or y+h == img.shape[0]:
    #         # Contour is partially in the image, remove it
    #         cv.drawContours(binary, [contour], contourIdx=-1, color=0, thickness=-1)
    #         # all contours in the list, because contourIdx = -1, are filled with colour 0, because thickness < 0
    #
    # toc = time.process_time() - tic
    # times.append(toc)
    # # display('09 removed boundary', binary, imageDisplayScale)

    '''
        Filling holes
    '''
    # 8.575, 9.611, 8.633 s
    tic = time.process_time()
    inversed = np.invert(binary)
    # display('inversed', inversed, 0.5)

    # Find the contours of the binary image
    contours, _ = cv.findContours(inversed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)  # 1 contour found
    h, w = inversed.shape[:2]
    marker_image = np.zeros((h, w), np.uint8)
    for j in range(len(contours)):
        for i in range(len(contours[j])):
            marker_image[contours[j][i][0][1], contours[j][i][0][0]] = 255
    # display('seed image', marker_image, 0.5)

    equal = False
    se_size = 3                     # Structuring element size
    se_shape = cv.MORPH_RECT     # Structuring element shape
    structuring_element = cv.getStructuringElement(se_shape, (se_size, se_size))
    oldImage = marker_image
    blank = np.zeros((h, w), np.uint8)
    while not equal:

        newImage = cv.bitwise_and(cv.dilate(oldImage, structuring_element), inversed)
        difference = np.subtract(newImage, oldImage)
        if np.array_equal(difference, blank):
            equal = True
            print('True')
        else:
            oldImage = newImage

    filled = np.invert(newImage)
    _,filled_binary = cv.threshold(filled,50,255,cv.THRESH_BINARY)
    toc = time.process_time() - tic
    times.append(toc)
    # print(times)
    # display('10 filled', filled_binary, imageDisplayScale)  # rethreshold this one



    return filled_binary

