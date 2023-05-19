from Displaying import ConvertLabelledImageToBinary
from Measuring import CalculateRoundness
from skimage.measure import regionprops_table, label
import pandas as pd
from skimage.util import map_array
import numpy as np
import math
import cv2 as cv

# temporary imports
import time

# Based on a measure of skimage
# https://stackoverflow.com/questions/66619685/how-do-i-filter-by-area-or-eccentricity-using-skimage-measure-regionprops-on-a-b

def FilterByFeature(binaryImages, filterFeature, filterThreshold):
    '''
    :param binaryImages: list of numpy array binary images
    :param filterFeature: feature to filter image by
    :param filterThreshold: threshold value below which objects are removed
    :return: binary image of filtered input image
    '''

    print('\nFiltering by ' + filterFeature + ', allowing objects with ' + filterFeature + ' >' + str(filterThreshold))

    filteredImages = []

    for binaryImage in binaryImages:
        # > Convert image to labelled
        labelledImage = label(binaryImage)

        # > Define properties to measure
        selectedProperties = ['label', filterFeature]
        if filterFeature == 'roundness':
            selectedProperties = ['label', 'area', 'perimeter']  # label is the integer label of the object

        # > Measure the features
        measurementsDict = regionprops_table(  # Compute image properties and return them as a pandas-compatible table
            labelledImage,
            properties=selectedProperties,
        )

        # > Convert feature data structure to df
        measurementsDf = pd.DataFrame(measurementsDict)

        # > If roundness, calculate and keep that only
        if filterFeature == 'roundness':
            measurementsDf[filterFeature] = CalculateRoundness(measurementsDf['area'], measurementsDf['perimeter'])
            measurementsDf.drop(columns=['area', 'perimeter'])


        # > Remove labels according to the filtering feature and threshold
        keptObjectLabels = measurementsDf['label'] * (measurementsDf[filterFeature] > filterThreshold)  # if condition False, label becomes 0
        selectedObjects = map_array(  # Map values from input array from input_vals to output_vals.
            labelledImage,
            np.asarray(measurementsDf['label']),
            np.asarray(keptObjectLabels),
        )

        filteredImages.append(ConvertLabelledImageToBinary(selectedObjects))

    return filteredImages

def RemoveBoundaryObjects(images):

    filteredImages = []

    for image in images:
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
        filteredImages.append(image)

    return filteredImages