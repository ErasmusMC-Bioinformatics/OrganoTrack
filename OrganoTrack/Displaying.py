import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def Rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def Display(window_title, image, scaling_factor=1):
    image = Rescale(image, scaling_factor)
    cv.imshow(window_title, image)


def DisplayImages(collective_title_for_image_set, image_set, scaling_factor):
    for i in range(len(image_set)):
        Display(collective_title_for_image_set + ' ' + str(i), image_set[i], scaling_factor)


def ConvertLabelledImageToBinary(skimage_labelled_img):
    binaryImage = skimage_labelled_img.astype(np.uint8)
    binaryImage[binaryImage != 0] = 255
    return binaryImage


def Mask(original, binary):
    return cv.bitwise_and(original, original, mask=binary)


def displayingTrackedSet(collective_title_for_image_set, tracked_set, scaling_factor):
    for i in range(tracked_set.shape[0]):
        Display(collective_title_for_image_set + ' ' + str(i), ConvertLabelledImageToBinary(tracked_set[i]), scaling_factor)

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

if __name__ == '__main__':

    rawImg = cv.imread("C:/Users/franz/Documents/OrganoTrackl/d1r1t0.tiff", cv.IMREAD_GRAYSCALE)
    predImg = cv.imread("C:/Users/franz/Documents/OrganoTrackl/d1r1t0.png", cv.IMREAD_GRAYSCALE)
    # # predImg = cv.imread("", cv.IMREAD_GRAYSCALE)
    # # _, predImg = cv.threshold(HarmonyImg, 50, 255, cv.THRESH_BINARY)
    # DisplayWithContours('hello', rawImg, predImg, 0.5)
    # cv.waitKey(0)
