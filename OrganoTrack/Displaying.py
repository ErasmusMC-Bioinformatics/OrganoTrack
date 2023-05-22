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

def ExportImageWithContours(ori, pred): #, imagePath, exportPath):

    # Convert image to colour
    img = cv.cvtColor(ori, cv.COLOR_GRAY2BGR)

    # Get contours
    contours, _ = cv.findContours(pred, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the image
    back = img.copy()
    cv.drawContours(image=back, contours=contours, contourIdx=-1,
                    color=(46, 204, 113), thickness=5, lineType=cv.LINE_AA)
    alpha = 1

    # Combine the images
    return cv.addWeighted(img, 1-alpha, back, alpha, 0)
    #
    # exportDir = str(exportPath / 'contours-roundness-filtered' / (imagePath.stem + '.png'))
    # exportDirUpdated = exportDir.replace("\\", "/")
    # cv.imwrite(exportDirUpdated, result)


if __name__ == '__main__':

    rawImg = cv.imread("C:/Users/franz/Documents/OrganoTrackl/d1r1t0.tiff", cv.IMREAD_GRAYSCALE)
    predImg = cv.imread("C:/Users/franz/Documents/OrganoTrackl/d1r1t0.png", cv.IMREAD_GRAYSCALE)
    # # predImg = cv.imread("", cv.IMREAD_GRAYSCALE)
    # # _, predImg = cv.threshold(HarmonyImg, 50, 255, cv.THRESH_BINARY)
    # DisplayWithContours('hello', rawImg, predImg, 0.5)
    # cv.waitKey(0)
