from Displaying import ConvertLabelledImageToBinary
from Measuring import CalculateRoundness
from skimage.measure import regionprops_table, label
import pandas as pd
from skimage.util import map_array
import numpy as np
import math

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