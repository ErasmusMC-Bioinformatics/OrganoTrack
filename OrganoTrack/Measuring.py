from pathlib import Path
import numpy as np
import skimage.measure
import pandas as pd
import cv2 as cv
import math
from Displaying import Display

def CalculateRoundness(area, perimeter):
    return 4 * math.pi * area / (perimeter ** 2)

def MeasureMorphometry(unlabeledImage):
    # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
    labeledImage = skimage.measure.label(unlabeledImage)
    regions = skimage.measure.regionprops(labeledImage)  # a list of skimage.measure._regionprops.RegionProperties objs
    #                                                      for as many labelled regions as there are.
    #                                                      Each object has morphometry attributes. e.g. obj1.area

    # Create circularity
    return regions

def Test_MeasureMorphometry():
    dataDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/measuring-size'
    image = cv.imread(dataDir+'/squares.png', cv.IMREAD_GRAYSCALE)
    labeled = skimage.measure.label(image)
    props = MeasureMorphometry(labeled)
    print('hello')

def AnalyzeAndExport(images: np.ndarray, path: Path):
    '''

    :param images: Labelled image
    :param path:
    :return:
    '''
    with pd.ExcelWriter(str(path.absolute())) as writer:
        # path.absolute() contains the entire path to the file or directory that you need to access
        # https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops
        propertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                         'eccentricity', 'equivalent_diameter_area', 'euler_number',
                         'extent', 'feret_diameter_max', 'orientation',
                         'perimeter', 'perimeter_crofton', 'solidity']

        size = (np.max(images)+1, images.shape[0])
        data = {propertyName: pd.DataFrame(np.ndarray(size, dtype=str)) for propertyName in propertyNames}

        for t in range(images.shape[0]):                            # for each image
            regions = skimage.measure.regionprops(images[t])            # calculate regions
            for propertyName in propertyNames:                          # for each property
                for region in regions:                                      # for each region
                    value = getattr(region, propertyName)
                    label = region.label
                    data[propertyName].iloc[label, t] = str(value)

        for propertyName in propertyNames:
            data[propertyName].to_excel(writer, sheet_name=propertyName)

if __name__ == '__main__':
    Test_MeasureMorphometry()

    # display('OrganoTrack', image, 0.25)
    # cv.waitKey(0)
