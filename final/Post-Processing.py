from Displaying import Display
import cv2 as cv
import numpy as np
import random as rng

rng.seed(12345)

def SeparateOverlappingObjets(currentImage):

    displayState = False
    displayScale = 0.5

    # Perform the distance transform algorithm
    dist = cv.distanceTransform(currentImage, cv.DIST_L2, 3)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)

    if displayState:
        Display('distance transform', dist, displayScale)

    # Threshold to obtain the peaks
    # This will be the markers for the foreground objects
    _, dist = cv.threshold(dist, 0.4, 1.0, cv.THRESH_BINARY)
    if displayState:
        Display('peaks', dist, displayScale)

    # Dilate a bit the peaks image
    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dist = cv.dilate(dist, kernel1)
    if displayState:
        Display('dilated', dist, displayScale)

    # Create the CV_8U version of the distance image
    # It is needed for findContours()
    dist_8u = dist.astype('uint8')

    # Find total markers
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Create the marker image for the watershed algorithm
    markers = np.zeros(dist.shape, dtype=np.int32)

    # Draw the foreground markers
    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)

    # Draw the background marker
    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    markers_8u = (markers * 10).astype('uint8')
    if displayState:
        Display('Markers', markers_8u, displayScale)

    #
    backtorgb = cv.cvtColor(currentImage, cv.COLOR_GRAY2RGB)

    # Perform the watershed algorithm
    cv.watershed(backtorgb, markers)

    # mark = np.zeros(markers.shape, dtype=np.uint8)
    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    if displayState:
        Display('Markers_v2', mark, displayScale)


    # Generate random colors
    colors = []
    for contour in contours:
        colors.append((rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)))

    # Create the result image
    dst = np.zeros((markers.shape[0], markers.shape[1], 3), dtype=np.uint8)

    # Fill labeled objects with random colors
    for i in range(markers.shape[0]):
        for j in range(markers.shape[1]):
            index = markers[i, j]
            if index > 0 and index <= len(contours):
                dst[i, j, :] = colors[index - 1]

    # Visualize the final image
    if displayState:
        Display('Final Result', dst, displayScale)

    # convert the input image to grayscale
    gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)


    # apply thresholding to convert grayscale to binary image
    _, thresh = cv.threshold(gray, 10, 255, 0)
    if displayState:
        Display('Final binary', thresh, displayScale)
