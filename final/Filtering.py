from Displaying import ConvertLabelledImageToBinary

from skimage.measure import regionprops_table, label
import pandas as pd
from skimage.util import map_array
import numpy as np
import math

# temporary imports
import time

# Based on a measure of skimage
# https://stackoverflow.com/questions/66619685/how-do-i-filter-by-area-or-eccentricity-using-skimage-measure-regionprops-on-a-b

def FilterByFeature(binaryImage, filterFeature, filterThreshold):
    '''
    :param binaryImage: numpy array binary image
    :param filterFeature: feature to filter image by
    :param filterThreshold: threshold value below which objects are removed
    :return: binary image of filtered input image
    '''

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
        measurementsDf['roundness'] = 4 * math.pi * measurementsDf['area'] / (measurementsDf['perimeter'] ** 2)
        measurementsDf.drop(columns=['area', 'perimeter'])


    # > Remove labels according to the filtering feature and threshold
    keptObjectLabels = measurementsDf['label'] * (measurementsDf[filterFeature] > filterThreshold)  # if condition False, label becomes 0
    selectedObjects = map_array(  # Map values from input array from input_vals to output_vals.
        labelledImage,
        np.asarray(measurementsDf['label']),
        np.asarray(keptObjectLabels),
    )

    return ConvertLabelledImageToBinary(selectedObjects)