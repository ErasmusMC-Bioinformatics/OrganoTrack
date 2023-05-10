import time
import os
from datetime import datetime
import cv2 as cv
import numpy as np
from skimage.morphology import reconstruction
from skimage.filters import threshold_otsu
from Exporting import SaveData
from Displaying import Display

# temporary
from pathlib import Path
from Importing import ReadImages


def Smoothen(image):
    # Sharpening
    kernel_size = 5  # as in OrganoSeg
    std_dev = 1  # as in OrganoSeg
    img_gauss = cv.GaussianBlur(image, (kernel_size, kernel_size), std_dev)
    weight = 0.8  # as in OrganoSeg
    img_sharp = cv.addWeighted(image, 1 + weight, img_gauss, -weight, 0)  # Unsharp masking method

    # Median blurring 0.014, 0.017, 0.007 s

    median_kernel = 3  # as in OrganoSeg
    img_blur = cv.medianBlur(img_sharp, median_kernel)

    return img_blur

def OpenAndClose(image, imgDataType):
    # Erosion 0.032, 0.032, 0.021 s

    se_size = 5  # Structuring element size, as in OrganoSeg
    se_shape = cv.MORPH_ELLIPSE  # Structuring element shape, as in OrganoSeg
    structuring_element = cv.getStructuringElement(se_shape, (2 * se_size - 1, 2 * se_size - 1))
    img_eroded = cv.erode(image, structuring_element)

    # Reconstruct 3.022, 3.453, 2.159 s

    img_reconstructed_1 = imreconstruct(img_eroded, image, imgDataType)

    # Dilation 2.307, 2.419, 1.871 s

    img_dilated = cv.dilate(img_reconstructed_1, structuring_element)

    img_reconstructed_2 = imreconstruct(np.invert(img_dilated), np.invert(img_reconstructed_1), imgDataType)

    img_smoothed = np.invert(img_reconstructed_2)
    return img_smoothed

def imreconstruct(marker, mask, imDataType):

    # > Clip values to the level of mask before reconstruction
    if np.any(marker > mask):

        above = np.where(marker > mask)     # tuple of x-coordinates and y-coordinates, e.g. (555, 555) and (1000,1000)
        #                                     thus, (array([ 555, 1000]), array([ 555, 1000]))
        #                                     dim = 2 x n, n = number of px satisfying condition

        for px in range(len(above[0])):  # for each px in marker > mask, clip to mask
            marker[above[0][px], above[1][px]] = mask[above[0][px], above[1][px]]  # for arrays, [,] faster than [][]

    return (reconstruction(marker, mask)).astype(imDataType)  # reconstruction returns float64. Convert to e.g. uint8

def CallAdaptiveThreshold(image, imgDataType):
    # # OpenCV method
    # adaptiveSum = np.zeros((np.shape(img)[0], np.shape(img)[1]))
    # for j in range(20, 251, 10):
    #     adaptiveIter = cv.adaptiveThreshold(img_smoothed, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, j-1, 2)
    #     adaptiveSum += adaptiveIter

    # adaptive thresholding 16.824, 16.448, 14.636 s
    adaptiveSum = np.zeros(image.shape, dtype=imgDataType)
    fudgeFactor = 0.5

    for windowSize in range(20, 251, 10):
        adaptiveIter = adaptiveThreshold(image, windowSize, fudgeFactor, imgDataType)
        adaptiveSum = np.add(adaptiveSum, adaptiveIter)

    return adaptiveSum


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


def mat2gray(img):
    return (img-np.amin(img))/(np.amax(img)-np.amin(img))


def RemoveSmallNoise(image):
    # 0.370, 0.243, 0.438 s
    # code from https://stackoverflow.com/questions/42798659/how-to-remove-small-connected-objects-using-opencv

    # find all of the connected components (white blobs in your image).
    # im_with_separated_blobs is an image where each detected blob has a different pixel value ranging from 1 to nb_blobs - 1.
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(image)
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

    return im_result

def RemoveBoundaryObjects(image):
    # 0.0276, 0.0149, 0.016 s
    # Find contours in the binary image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # # This code visualises the contours found
    # h, w = binary.shape[:2]
    # blank = np.zeros((h, w), np.uint8)
    # maxLength = len(contours[0])
    # maxIndex = 0
    # for j in range(len(contours)):
    #     if len(contours[j]) > maxLength:
    #         maxLength = len(contours[j])
    #         maxIndex = j
    #     for i in range(len(contours[j])):
    #         blank[contours[j][i][0][1]][contours[j][i][0][0]] = 255
    # print(maxIndex)  # 632
    # display('removed', blank, 0.5)

    # Iterate over the contours and remove the ones that are partially in the image
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # openCV documentation: "Calculates and returns the minimal up-right bounding rectangle for the specified point set"

        if x == 0 or y == 0 or x + w == image.shape[1] or y + h == image.shape[0]:
            # Contour is partially in the image, remove it
            cv.drawContours(image, [contour], contourIdx=-1, color=0, thickness=-1)
            # all contours in the list, because contourIdx = -1, are filled with colour 0, because thickness < 0
    return image


def FillHoles(image):
    # 8.575, 9.611, 8.633 s

    inversed = np.invert(image)
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
    se_size = 3  # Structuring element size
    se_shape = cv.MORPH_RECT  # Structuring element shape
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
    _, filled_binary = cv.threshold(filled, 50, 255, cv.THRESH_BINARY)\

    return filled_binary


def SegmentWithOrganoSegPy(images, saveSegmentation, exportPath, imagePaths):


    segmentedImages = []

    if saveSegmentation:
        # Make segmentation export path
        dateTimeNow = datetime.now()
        segmentedExportPath = exportPath / ('segmented-' + dateTimeNow.strftime('%d.%m.%Y-%H_%M_%S'))
        os.mkdir(segmentedExportPath)

    for count, img in enumerate(images):

        imgDataType = img.dtype  # image.dtype = the bitsize of the image, e.g. uint8 (0 - 255) or uint16 (0 65535)

        # Converting to grayscale
        if img.shape == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img_blur = Smoothen(img)

        img_smoothed = OpenAndClose(img_blur, imgDataType)

        adaptiveSum = CallAdaptiveThreshold(img_smoothed, imgDataType)

        im_result = RemoveSmallNoise(adaptiveSum)

        # Smoothen, 0.0139, 0.0163, 0.004 s
        kernel = np.ones((3, 3), np.uint16)
        binary = cv.morphologyEx(im_result, cv.MORPH_CLOSE, kernel)

        binary = RemoveBoundaryObjects(binary)

        filled_binary = FillHoles(binary)

        segmentedImages.append(filled_binary)

        if saveSegmentation:
            cv.imwrite(str(segmentedExportPath / imagePaths[count].name), filled_binary)

    return segmentedImages

if __name__ == '__main__':
    # Import
    dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/improving-segmentation/input')
    exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/improving-segmentation/output')

    # Segmentation
    saveSegmentations = True
    segmentedPaths = Path(
        '/home/franz/Documents/mep/data/for-creating-OrganoTrack/improving-segmentation/output/segmented')

    # Executing segmentation
    inputImages, imageNames = ReadImages(dataPath)
    imagesInAnalysis = SegmentWithOrganoSegPy(inputImages, saveSegmentations, exportPath, imageNames)
    cv.waitKey(0)
    print('segmentation')