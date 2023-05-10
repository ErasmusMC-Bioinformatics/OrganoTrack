import os
from datetime import datetime
import cv2 as cv
import pandas as pd
import numpy as np
from skimage.measure import regionprops
from Measuring import CalculateRoundness

def SaveData(exportPath, images, imagePaths):
    '''
    :param exportPath: parent directory for analysis export
    :param images: list of images
    :param imageNames: list of input image paths
    '''
    # > Create a unique daughter path for storage
    dateTimeNow = datetime.now()
    storagePath = exportPath / ('segmented-' + dateTimeNow.strftime('%d.%m.%Y-%H_%M_%S'))
    os.mkdir(storagePath)

    # > Store
    for i in range(len(images)):
        cv.imwrite(str(storagePath / imagePaths[i].name), images[i])

def Export(outputPath, propertiesToMeasure, analysedDataset, plateLayout):
    # analysedDataset = entire data analysed

    dataDict = {}  # dictionary with property names as keys and dictionaries as values.
    # The lower level dictionaries have well names as keys and property measure df's as values.
    with pd.ExcelWriter(str(outputPath.absolute())) as writer:
        # path.absolute() contains the entire path to .xlsx file, i.e. /home/... in Linux or C:/... in Windows

        for propertyName in propertiesToMeasure:  # if 'roundness'
            for i in range(len(imageStacks)):       # for each stack

                # > Create the dataframe
                size = (np.max(imageStacks[i]) + 1,     # the highest label in the whole stack + 1
                        imageStacks[i].shape[0])        # num of images in stack (i.e. num of time points)
                # the first element has +1 so that index numbers in the df/spreadsheet are equal to the label numbers
                data = pd.DataFrame(np.ndarray(size, dtype=str))

                # > Fill the dataframe
                for t in range(imageStacks[i].shape[0]):        # for each image in the stack
                    regions = regionprops(imageStacks[i][t])        # get RegionProperties objects
                    for region in regions:                          # for each RP object
                        if propertyName == 'roundness':
                            value = CalculateRoundness(getattr(region, 'area'), getattr(region, 'perimeter'))
                        else:
                            value = getattr(region, propertyName)           # get the property value of that region
                        label = region.label                            # get the label
                        data.iloc[label, t] = str(value)                # store the property value by its label in df

                # > Load dataframe into spreadsheet

                data.to_excel(writer, sheet_name=propertyName, startrow=1, startcol=i * (size[1] + 2))
                # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other

                writer.sheets[propertyName].cell(row=1, column=i * (size[1] + 2) + 1).value = imageConditions[i]
                # Within .cell(), row and column are 1-indexed


# Function adapted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def ExportImageStackMeasurements(outputPath, propertiesToMeasure, imageStacks, imageConditions):
    '''
    :param outputPath: Complete path to the .xlsx file
    :param propertiesToMeasure: List of names of morphological properties that skimage.measure.regionprops returns
    :param imageStacks: List of lists, each of which contains n 3D arrays (time dim x 2D numpy array images).
            n is the number of fields of a well
            Image regions of interest are labelled
    :param imageConditions: List of string-based experimental conditions of the corresponding images
    '''

    dataDict = {}  # dictionary with property names as keys and dictionaries as values.
    # The lower level dictionaries have well names as keys and property measure df's as values.
    with pd.ExcelWriter(str(outputPath.absolute())) as writer:
        # path.absolute() contains the entire path to .xlsx file, i.e. /home/... in Linux or C:/... in Windows

        for propertyName in propertiesToMeasure:  # if 'roundness'
            for i in range(len(imageStacks)):       # for each stack

                # > Create the dataframe
                size = (np.max(imageStacks[i]) + 1,     # the highest label in the whole stack + 1
                        imageStacks[i].shape[0])        # num of images in stack (i.e. num of time points)
                # the first element has +1 so that index numbers in the df/spreadsheet are equal to the label numbers
                data = pd.DataFrame(np.ndarray(size, dtype=str))

                # > Fill the dataframe
                for t in range(imageStacks[i].shape[0]):        # for each image in the stack
                    regions = regionprops(imageStacks[i][t])        # get RegionProperties objects
                    for region in regions:                          # for each RP object
                        if propertyName == 'roundness':
                            value = CalculateRoundness(getattr(region, 'area'), getattr(region, 'perimeter'))
                        else:
                            value = getattr(region, propertyName)           # get the property value of that region
                        label = region.label                            # get the label
                        data.iloc[label, t] = str(value)                # store the property value by its label in df

                # > Load dataframe into spreadsheet

                data.to_excel(writer, sheet_name=propertyName, startrow=1, startcol=i * (size[1] + 2))
                # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other

                writer.sheets[propertyName].cell(row=1, column=i * (size[1] + 2) + 1).value = imageConditions[i]
                # Within .cell(), row and column are 1-indexed


def ExportSingleImageMeasurements(outputPath, propertiesToMeasure, singleImages, imageConditions):
    with pd.ExcelWriter(str(outputPath.absolute())) as writer:

        for propertyName in propertiesToMeasure:
            for i in range(len(singleImages)):

                size = (np.max(singleImages[i])+1, 1)
                data = pd.DataFrame(np.ndarray(size, dtype=str))

                regions = regionprops(singleImages[i])
                for region in regions:
                    value = getattr(region, propertyName)
                    label = region.label
                    data.iloc[label, 0] = str(value)

                data.to_excel(writer, sheet_name=propertyName, startcol=i*(size[1]+2), startrow=1)
                writer.sheets[propertyName].cell(row=1, column=i*(size[1]+2)+1).value = imageConditions[i]
