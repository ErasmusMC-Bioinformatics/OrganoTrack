from stitch import read_images
from segmentation import segment
from functions import display
import numpy as np
import time
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import math
import pandas as pd

'''
    If there is any code to change / look at, search for ?
'''

def plotHistogram(data, numBins, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.hist(data, bins=numBins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.show()


'''
    Welcome
'''
print("Welcome to OrganoTrack")

'''
    Read data
'''

# print("Enter naming structure of data")
# < Naming structure
# Determine appropriate data structure

# dataDirectory = input("Enter data directory. The directory should not have spaces: ")
dataDirectory = '/home/franz/Documents/mep/data/2023-02-24-Cis-Tos-dataset-mathijs/AZh5/Day-12/renamed'
fields = 1
wells = [('04', '04'), ('04', '07'), ('05', '04'), ('05', '09'), ('05', '10'), ('05', '11')]
experiment = 'Azh5-reseeding'
positions = 1
channels = 1
timePoints = 1

# Reading data
# imData = read_images(dataDirectory, wells, positions, channels, timePoints, fields)
# images = [imData[i][0][0][0][0] for i in range(len(imData))]


# Read images according to data structure
    # If image not grayscale, read as is and convert to grayscale when begin to segment

'''
    Segmentation
'''

# print("Choose from options")
# < OrganoSeg / OrganoID / Optimise parameters and segment

# If OrganoSeg / OrganoID:
    # segmentation = segment(images)

# segmentation_times = []
#
# segmented_images = []
# for i in range(len(imData)):
#     print('Segmenting image ' + str(i+1))
#     tic = time.process_time()
#     segmented_images.append(segment(images[i]))
#     toc = time.process_time() - tic
#     segmentation_times.append(toc)
# print('segmentation times in seconds: ')
# print(segmentation_times)

store_seg_dir = "/home/franz/Documents/mep/data/2023-02-24-Cis-Tos-dataset-mathijs/AZh5/Day-12/renamed/segmented"
input_images = glob.glob(store_seg_dir + "/*.png")

segmentedImages = []
for image_path in input_images:
    segmentedImages.append(cv.imread(image_path, cv.IMREAD_ANYDEPTH))

# for i in range(len(segmentedImages)):
#     display('segmented ' + str(i), segmentedImages[i], 0.25)

# image = segmented[0]

# display('first', image, 0.25)


# Diplay images
# for i in range(len(segmented)):
#     display(str(i), segmented[i], 0.25)
# cv.waitKey(0)



# for i in range(len(segmented_images)):
#     filename = "/segmentation-" + str(i+1) + ".png"
#     cv.imwrite(store_seg_dir+filename, segmented_images[i])

# Elif optimise parameters:
    # for each parameter combination:
        # calculate segmentation performance
            # (user can input % of dataset to test this on)
            # Does decreasing to 8-bit make the operation run faster? Does it affect seg performance?
        # store segmenation performance for each combination

    # return combination with highest performance

    # segmentation = segment_with_optimal_params(images)

    # further processing:
        # ID every object
        # x remove noise
        # x smoothen
        # x include hole closing of everything (but keep the identity of out of focus organoids)
        # x # Remove border objects
        # Remove out of focus organoids


# Report segmentation metrics (using GT dataset)


'''
    User selection / Filtering / Measuring features
'''


# Select unmerged organoids (only if there is time in the data)
# If one object in timepoint t+1 overlaps with more than 1 object in timepoint t:
    # remove that object from the image of timepoint t+1


'''
    Measuring sizes of organoids
'''
segmentedObjectDFs = []  # the feature dataframes of all objects in each segmented image

for image in segmentedImages:

    # > Find all the object contours in the binary image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours
    #   a list of Numpy arrays. Each array has the (x,y) oordinates of points that make up a contour

    # cv.RETR_EXTERNAL
    #   If there are objects within an object (e.g. a hole in a binary object), cv.RETR_EXTERNAL returns only the
    #   outer contour (the binary object) and not the inner (the hole) contour.

    # cv.CHAIN_APPROX_SIMPLE
    #   A line can be represented as all the points that makeit, or by the two end point. cv.CHAIN_APPROX_SIMPLE
    #   only returns the two endpoints of the line, saving memory.

    # > For each contour found, get
    areas = [cv.contourArea(contour) for contour in contours]  # in pixels^2?
    # plotHistogram(areas, numBins=20, title='Areas', xlabel='Object area', ylabel='Count')

    perimeters = [cv.arcLength(contour, closed=True) for contour in contours]  # in pixels?
    # closed
    #   as the objects are binary objects, they have closed curves/contours
    # plotHistogram(perimeters, numBins=20, title='Perimeters', xlabel='Object perimeter', ylabel='Count')

    circularities = [4 * math.pi * areas[i] / perimeters[i]**2 for i in range(len(areas))]  # dimensionless
    # plotHistogram(circularities, numBins=20, title='Circularities', xlabel='Object circularity', ylabel='Count')

    # > Convert feature data into dataframes
    dictOfObjectFeatures = {'Contour': contours, 'Area': areas, 'Perimeter': perimeters, 'Circularity': circularities}
    allObjectFeatures = pd.DataFrame(dictOfObjectFeatures, index=range(1, len(contours)+1))
    # index goes from 1 to the total number of objects
    segmentedObjectDFs.append(allObjectFeatures)

# > Get exclusion criteria
exclFeature = 'Circularity'
exclValue = 0.4  # exclude all those under this value

# > Copy image file to edit and create a space to store the included objects and their features
segmentedImagesCopy = segmentedImages[:]
includedObjectDFs = []

# > For each image, filter out objects to exclude from image and keep the data of objects to include
for i in range(len(segmentedImagesCopy)):
    # > Define element-specific exclusion condition
    exclusionCondition = segmentedObjectDFs[i][exclFeature] < exclValue  # update so that code works when iteritems is deprecated

    # > Get objects to exclude
    excludedObjects = segmentedObjectDFs[i][exclusionCondition]  # one dataframe of objects to exclude for htat image

    # > Remove the excluded objects from the image
    contoursToExclude = excludedObjects['Contour'].tolist()
    cv.drawContours(segmentedImagesCopy[i], contoursToExclude, contourIdx=-1, color=0, thickness=-1)

    # > Keep the objects desired
    includedObjects = segmentedObjectDFs[i][~exclusionCondition]
    includedObjectDFs.append(includedObjects)

for i in range(len(segmentedImagesCopy)):
    display('filtered ' + str(i), segmentedImages[i], 0.25)

cv.waitKey(0)



# Remaining: organoids of interest, with their measurements available (export data)


'''
    Data exporting
'''
# Export to CSV

# Plotting
