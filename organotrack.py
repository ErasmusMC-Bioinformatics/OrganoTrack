from stitch import readImages
from segmentation import segmentWithOrganoSegPy
from functions import display
import numpy as np
import time
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import math
import pandas as pd
import os
from datetime import datetime

'''
    If there is any code to change / look at, search for ?
'''

def saveData(parentDataDir, images, imageNames):
    '''
    :param inputDataDir: parent directory where image data will be stored
    :param images: image data
    :param imageNames: image names for storage
    '''

    # > Create a unique daughter path for storage
    dateTimeNow = datetime.now()
    storagePath = parentDataDir + '/segmented-' + dateTimeNow.strftime('%d.%m.%Y-%H_%M_%S')
    os.mkdir(storagePath)

    # > Store
    for i in range(len(images)):
        cv.imwrite(storagePath + '/' + imageNames[i], images[i])


def plotHistogram(data, numBins, title, xlabel, ylabel):
    fig, ax = plt.subplots()
    ax.hist(data, bins=numBins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.show()


def organotrack():
    '''
        Welcome
    '''
    print("Welcome to OrganoTrack!")
    loadingChoice = input("\nChoose an option: "
                          "\n1) Load new image data for segmentation "
                          "\n2) Load segmented (binary) images")
    if loadingChoice == '1':
        dataDir = input("\nEnter image data directory. Note that the directory should only have image files: ")
        inputImages, inputImagesNames = readImages(dataDir)

        '''
            Segmentation
        '''
        segChoice = input("\nHow do you want to analyse your data? "
                          "\n1) Optimise segmentation parameters before segmentation "
                          "\n2) Segment with OrganoSeg (Python version) \n")  # print("Choose from options")

        segmentedImages = []

        if segChoice == '1':  # Optimise segmentation parameters
            print("SegChoice 1")

        elif segChoice == '2':  # Segment with OrganoSegPy
            print("\nSegmenting with OrganoSeg...")
            segmentedImages = []
            for i in range(len(inputImages)):
                print("\nSegmenting image " + str(i + 1) + "...")
                segmentedImages.append(segmentWithOrganoSegPy(inputImages[i]))
                print("Image " + str(i + 1) + " segmented.")
            print("\nSegmentation completed.")


            # > Store segmentation results
            storeChoice = input("\nDo you want to store the segmentation results? Y/N")

            if storeChoice == 'Y' or 'y':
                saveData(dataDir, segmentedImages, inputImagesNames)

    elif loadingChoice == '2':
        print("Enter loading data directory")
    # ? check if user gave a valid directory

    '''
        Read data
    '''

    # experimentInfoDir = input("Enter experiment information: ")
    # < Naming structure
    # Determine appropriate data structure
    # dataDirectory = input("Enter data directory. The directory should not have spaces: ")

    # dataDirectory = experimentInfo[0]
    # fields = experimentInfo[1]
    # wells = experimentInfo[2]
    # experimentName = experimentInfo[3]
    # positions = experimentInfo[4]
    # channels = experimentInfo[5]
    # timePoints = experimentInfo[6]

    # > Read data according to data structure desired





    # images = [imData[i][0][0][0][0] for i in range(len(imData))]  # making a single list of images




    # store_seg_dir = "/home/franz/Documents/mep/data/2023-02-24-Cis-Tos-dataset-mathijs/AZh5/Day-12/renamed/segmented"
    # input_images = glob.glob(store_seg_dir + "/*.png")
    #
    # for image_path in input_images:
    #     segmentedImages.append(cv.imread(image_path, cv.IMREAD_ANYDEPTH))

    # for i in range(len(segmentedImages)):
    #     display('segmented ' + str(i), segmentedImages[i], 0.25)

    # cv.waitKey(0)

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
    # measureChoice = input("Do you want to measure the objects by size and circularity? Y/N")
    # filterChoice = input("Do you want to filter the objects by size and circularity? Y/N ")
    # filterChoice = 'Y'
    # if filterChoice == 'Y':

    # '''
    #     Measuring features of objects
    # '''
    #
    # # > Measure features of each object in the image
    # segmentedObjectDFs = []  # the feature dataframes of all objects in each segmented image
    #
    # for image in segmentedImages:
    #
    #
    # # > Find all the object contours in the binary image
    #     contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    #     # contours
    #     #   a list of Numpy arrays. Each array has the (x,y) oordinates of points that make up a contour
    #
    #     # cv.RETR_EXTERNAL
    #     #   If there are objects within an object (e.g. a hole in a binary object), cv.RETR_EXTERNAL returns only the
    #     #   outer contour (the binary object) and not the inner (the hole) contour.
    #
    #     # cv.CHAIN_APPROX_SIMPLE
    #     #   A line can be represented as all the points that makeit, or by the two end point. cv.CHAIN_APPROX_SIMPLE
    #     #   only returns the two endpoints of the line, saving memory.
    #
    #     # > For each contour found, get
    #     areas = [cv.contourArea(contour) for contour in contours]  # in pixels^2?
    #     # plotHistogram(areas, numBins=20, title='Areas', xlabel='Object area', ylabel='Count')
    #
    #     perimeters = [cv.arcLength(contour, closed=True) for contour in contours]  # in pixels?
    #     # closed
    #     #   as the objects are binary objects, they have closed curves/contours
    #     # plotHistogram(perimeters, numBins=20, title='Perimeters', xlabel='Object perimeter', ylabel='Count')
    #
    #     circularities = [4 * math.pi * areas[i] / perimeters[i] ** 2 for i in range(len(areas))]  # dimensionless
    #     # plotHistogram(circularities, numBins=20, title='Circularities', xlabel='Object circularity', ylabel='Count')
    #
    #     # > Convert feature data into dataframes
    #     dictOfObjectFeatures = {'Contour': contours, 'Area': areas, 'Perimeter': perimeters,
    #                             'Circularity': circularities}
    #     allObjectFeatures = pd.DataFrame(dictOfObjectFeatures, index=range(1, len(contours) + 1))
    #     # index goes from 1 to the total number of objects
    #     segmentedObjectDFs.append(allObjectFeatures)
    #
    # # for i in range(len(segmentedImages)):
    # # # display('ctr', segmentedImages[0], 0.25)
    # # h, w = segmentedImages[0].shape[:2]
    # # blank = np.zeros((h, w), np.uint8)
    # # contours = segmentedObjectDFs[0]['Contour'].tolist()
    # # for i in range(len(segmentedObjectDFs[0])):
    # #     cv.drawContours(blank, [contours[i]], contourIdx=-1, color=1, thickness=2)
    # #     display(str(i), blank, 0.25)
    # #     blank = np.zeros((h, w), np.uint8)
    # # cv.waitKey(0)
    #
    # # > Get exclusion criteria
    # # exclFeature = input("By what feature do you want to filter objects? \n Area \n Circularity")
    # # exclValue = float(input("Under which value of " + exclFeature + " should objects be excluded?"))
    # exclFeature = 'Circularity'
    # exclValue = 0.4
    # # > Copy image file to edit and create a space to store the included objects and their features
    # segmentedImagesCopy = segmentedImages
    # includedObjectDFs = []
    #
    # # > For each image, filter out objects to exclude from image and keep the data of objects to include
    # for i in range(len(segmentedImagesCopy)):
    #     # > Define element-specific exclusion condition
    #     exclusionCondition = segmentedObjectDFs[i][exclFeature] < exclValue  # update so that code works when iteritems is deprecated
    #
    #     # > Get objects to exclude
    #     excludedObjects = segmentedObjectDFs[i][exclusionCondition]  # one dataframe of objects to exclude for htat image
    #
    #     # > Remove the excluded objects from the image
    #     contoursToExclude = excludedObjects['Contour'].tolist()
    #     cv.drawContours(segmentedImagesCopy[i], contoursToExclude, contourIdx=-1, color=0, thickness=-1)
    #
    #     # > Keep the objects desired
    #     includedObjects = segmentedObjectDFs[i][~exclusionCondition]
    #     includedObjectDFs.append(includedObjects)
    #
    # # for i in range(len(segmentedImagesCopy)):
    # #     display('filtered ' + str(i), segmentedImages[i], 0.25)
    #
    # # cv.waitKey(0)
    #
    # return segmentedImages, segmentedObjectDFs, segmentedImagesCopy, includedObjectDFs
    # exportChoice = input("Do you want to export your data as a .csv? Y/N")

    # if exportChoice == 'Y':




    # Select unmerged organoids (only if there is time in the data)
    # If one object in timepoint t+1 overlaps with more than 1 object in timepoint t:
        # remove that object from the image of timepoint t+1







    # Remaining: organoids of interest, with their measurements available (export data)


    # '''
    #     Data exporting
    # '''
    # Export to CSV

    # Plotting

