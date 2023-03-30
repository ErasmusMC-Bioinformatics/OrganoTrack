import cv2 as cv
import numpy as np


def Rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def Display(title, img, sf=1):
    '''
    :param title: title of image display
    :param img: image to display
    :param sf: scaling factor, 0 to 1
    '''
    img = Rescale(img, sf)
    cv.imshow(title, img)


def DisplayImages(collectiveTitle, imageSet, displayScale):
    '''
    :param collectiveTitle: title descriving the imageSet
    :param imageSet: a list of images
    :param displayScale: scaling factor, 0 to 1
    '''
    for i in range(len(imageSet)):
        Display(collectiveTitle + ' ' + str(i), imageSet[i], displayScale)


def ConvertLabelledImageToBinary(labelledImage):
    '''
    :param labelledImage: skimage-labeled image
    :return: a binary imaeg suitable for OpenCV displaying
    '''
    binaryImage = labelledImage.astype(np.uint8)
    binaryImage[binaryImage != 0] = 255
    return binaryImage


def Mask(original, binary):
    '''
    :param original: the grayscale image
    :param binary: the binary detection
    :return: the masked original with the binary regions
    '''
    return cv.bitwise_and(original, original, mask=binary)


def displayingTrackedSet(collectiveTitle, trackedSet, displayScale):
    '''
    :param collectiveTitle: title descriving the imageSet
    :param trackedSet: tracked set of Images, where the tracked objects have the same label
    :param displayScale: scaling factor, 0 to 1
    :return:
    '''
    for i in range(trackedSet.shape[0]):
        Display(collectiveTitle + ' ' + str(i), ConvertLabelledImageToBinary(trackedSet[i]), displayScale)