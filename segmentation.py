import cv2 as cv
from functions import display, blur, imcomplement, adaptiveThreshold
import numpy as np
from skimage.morphology import reconstruction
from skimage.filters import threshold_otsu, threshold_local

def imreconstruct(marker, mask, imDataType):
    if np.any(marker > mask):

        above = np.where(marker > mask)  # 2 x n, n = number of px satisfying condition

        for px in range(len(above[0])):  # for each px in marker > mask, clip to mask
            marker[above[0][px]][above[1][px]] = mask[above[0][px]][above[1][px]]
    return (reconstruction(marker, mask)).astype(imDataType)  # return grayscale values


def fill_holes(image):
    # Copy the input image to prevent modifying the original
    filled = image.copy()

    # Invert the image (so that the holes become white)
    filled = cv.bitwise_not(filled)

    # Find contours of the holes
    contours, hierarchy = cv.findContours(filled, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    # Fill the holes
    for i, contour in enumerate(contours):
        cv.drawContours(filled, contours, i, (65535, 65535, 65535), cv.FILLED, hierarchy=hierarchy)

    # Invert the image back to the original format
    filled = cv.bitwise_not(filled)

    return filled

scale = 0.5

'''
    Reading image
'''
dir = '/home/franz/Documents/mep/data/preliminary-gt-dataset'

img = cv.imread(dir+'/d0r1t0.tiff', cv.IMREAD_ANYDEPTH)
imgDataType = img.dtype  # uint16 for 16-bit images

'''
    Smoothing
'''

# Sharpening
kernel_size = 5
std_dev = 2
img_gauss = cv.GaussianBlur(img, (kernel_size, kernel_size), std_dev)
weight = 0.8
img_sharp = cv.addWeighted(img, 1+weight, img_gauss, -weight, 0)
# display('01', img_sharp, scale)

# Median blurring
median_kernel = 3
img_blur = cv.medianBlur(img_sharp, median_kernel)
# display('02', img_blur, scale)

'''
    Opening
'''

# Erosion
se_size = 5                     # Structuring element size
se_shape = cv.MORPH_ELLIPSE     # Structuring element shape
structuring_element = cv.getStructuringElement(se_shape, (2 * se_size + 1, 2 * se_size + 1))
img_eroded = cv.erode(img_blur, structuring_element)
# display('03', img_eroded, scale)

# Reconstruct
second_se_size = 10
second_se = cv.getStructuringElement(se_shape, (2 * second_se_size + 1, 2 * second_se_size + 1))
img_reconstructed_1 = imreconstruct(img_eroded, img_blur, imgDataType)
# display('04', img_reconstructed_1, scale)   # reconstructed


'''
    Closing
'''
# Dilation
img_dilated = cv.dilate(img_reconstructed_1, structuring_element)
# display('05', img_dilated, scale)
img_reconstructed_2 = imreconstruct(cv.bitwise_not(img_dilated), cv.bitwise_not(img_reconstructed_1), imgDataType)
# display('06 again', img_reconstructed_2, scale)
img_smoothed = cv.bitwise_not(img_reconstructed_2)
# display('07', img_smoothed, scale)

'''
    Adaptive thresholding
'''
# # OpenCV method
# adaptiveSum = np.zeros((np.shape(img)[0], np.shape(img)[1]))
# for j in range(20, 251, 10):
#     adaptiveIter = cv.adaptiveThreshold(img_smoothed, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, j-1, 2)
#     adaptiveSum += adaptiveIter

# MATLAB copy
adaptiveSum = (np.zeros((np.shape(img)[0], np.shape(img)[1]))).astype(imgDataType)
fudgeFactor = 0.5
adaptiveMethod = 'OrganoSeg'

if adaptiveMethod == 'OrganoSeg':

    for windowSize in range(20, 251, 10):
        adaptiveIter = adaptiveThreshold(img_smoothed, windowSize, fudgeFactor, imgDataType)
        adaptiveSum = np.add(adaptiveSum, adaptiveIter)

elif adaptiveMethod == 'skimage':

    for windowSize in range(13, 22, 2):
        print(windowSize)
        convolved = cv.blur(img, (windowSize, windowSize))  # uint8
        # display('blur', convolved, 0.5)
        substract = cv.subtract(convolved, img)  # uint8
        # display('subtract', substract, 0.5)
        otsu = threshold_otsu(substract)
        adaptiveIter = (threshold_local(substract, windowSize, 'mean', otsu*fudgeFactor)).astype(imgDataType)
        adaptiveIter = cv.bitwise_not(adaptiveIter)
        # display(str(windowSize), adaptiveIter, 0.5)
        adaptiveSum = cv.add(adaptiveSum, adaptiveIter)

# display('adaptive final', adaptiveSum, 0.75)

kernel = np.ones((3,3),np.uint16)
closing = cv.morphologyEx(adaptiveSum, cv.MORPH_CLOSE, kernel)

display('closed', closing, 0.5)



# filled = fill_holes(closing)
#
# display('filled', filled, 0.75)

cv.waitKey(0)