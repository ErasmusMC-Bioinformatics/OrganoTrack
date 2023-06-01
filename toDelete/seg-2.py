import cv2 as cv
from functions import display, blur, imreconstruct, imcomplement
import numpy as np

'''
    Reading image
'''
dir = '/home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/data/preliminary-gt-dataset'

img = cv.imread(dir+"/d0r1t0.tiff")  # (1080, 1080, 3)
# print(np.shape(img))
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # (1080, 1080)
# print(np.shape(img))
display('00', img, 0.9)
'''
    Smoothing
'''

# Sharpening
kernel_size = 5
std_dev = 2
img_gauss = cv.GaussianBlur(img, (kernel_size, kernel_size), std_dev)
weight = 0.8
img_sharp = cv.addWeighted(img, 1+weight, img_gauss, -weight, 0)
display('01', img_sharp, 0.9)

# Median blurring
median_kernel = 3
img_blur = cv.medianBlur(img_sharp, median_kernel)
display('02', img_blur, 0.9)

'''
    Opening
'''

# Erosion
struct_element_size = 5
struct_element_shape = cv.MORPH_ELLIPSE
struct_element = cv.getStructuringElement(struct_element_shape, (2 * struct_element_size + 1, 2 * struct_element_size + 1))
img_eroded = cv.erode(img_blur, struct_element)
display('03', img_eroded, 0.9)

# Reconstruct
img_reconstructed_1 = imreconstruct(img_eroded, img_blur)
display('04', img_reconstructed_1, 0.9)  # work on

'''
    Closing
'''
# Dilation=
struct_element_size = 5
struct_element_shape = cv.MORPH_ELLIPSE
struct_element = cv.getStructuringElement(struct_element_shape, (2 * struct_element_size + 1, 2 * struct_element_size + 1))
img_dilated = cv.dilate(img_reconstructed_1, struct_element)
display('05', img_dilated, 0.9)

img_reconstructed_2 = imreconstruct(~img_dilated, ~img_reconstructed_1)
display('06', img_reconstructed_2, 0.9)
img_smoothed = ~img_reconstructed_2
display('07', img_smoothed, 0.9)



'''
    Adaptive thresholding
'''
adaptiveSum = np.zeros((np.shape(img)[0], np.shape(img)[1]))
for j in range(20, 251, 10):
    adaptiveIter = cv.adaptiveThreshold(img_smoothed, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, j-1, 2)
    adaptiveSum += adaptiveIter

display('08', adaptiveSum, 0.9)
# inversed = ~adaptiveSum
# display('09', inversed, 0.9)

'''
    Closing
'''


cv.waitKey(0)