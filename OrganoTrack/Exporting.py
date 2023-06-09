import os
from datetime import datetime
import cv2 as cv
import pandas as pd
import numpy as np
from skimage.measure import regionprops
from OrganoTrack.Measuring import CalculateRoundness


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

# def Export(outputPath, propertiesToMeasure, analysedDataset, plateLayout):
#     # analysedDataset = entire data analysed
#
#     dataDict = {}  # dictionary with property names as keys and dictionaries as values.
#     # The lower level dictionaries have well names as keys and property measure df's as values.
#
#     with pd.ExcelWriter(str(outputPath.absolute())) as writer:
#         # path.absolute() contains the entire path to .xlsx file, i.e. /home/... in Linux or C:/... in Windows
#
#         for propertyName in propertiesToMeasure:  # if 'roundness'
#             # startrow = 1
#             # for each well
#             for i in range(len(imageStacks)):       # for each field stack in a well
#
#                 # > Create the dataframe
#                 size = (np.max(imageStacks[i]) + 1,     # the highest label in the whole stack + 1
#                         imageStacks[i].shape[0])        # num of images in stack (i.e. num of time points)
#                 # the first element has +1 so that index numbers in the df/spreadsheet are equal to the label numbers
#                 data = pd.DataFrame(np.ndarray(size, dtype=str))
#
#                 # > Fill the dataframe
#                 for t in range(imageStacks[i].shape[0]):        # for each image in the stack
#                     regions = regionprops(imageStacks[i][t])        # get RegionProperties objects
#                     for region in regions:                          # for each RP object
#                         if propertyName == 'roundness':
#                             value = CalculateRoundness(getattr(region, 'area'), getattr(region, 'perimeter'))
#                         else:
#                             value = getattr(region, propertyName)           # get the property value of that region
#                         label = region.label                            # get the label
#                         data.iloc[label, t] = str(value)                # store the property value by its label in df
#
#                 # > Load dataframe into spreadsheet
#
#                 data.to_excel(writer, sheet_name=propertyName, startrow=1, startcol=i * (size[1] + 2))
#                 # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other
#                 # update start row for second field stack in well
#
#                 writer.sheets[propertyName].cell(row=1, column=i * (size[1] + 2) + 1).value = imageConditions[i]
#                 # Within .cell(), row and column are 1-indexed


def CreateDfForExport(imageStack):
    stackNonZeroTrackNumbers = imageStack[imageStack != 0]
    minTrackNumber = np.min(stackNonZeroTrackNumbers)
    maxTrackNumber = np.max(stackNonZeroTrackNumbers)
    numberOfTracks = maxTrackNumber - minTrackNumber + 1

    numberOfTimePoints = imageStack.shape[0]

    dfSize = (numberOfTracks, numberOfTimePoints)
    timePointLabels = ['t{}'.format(i) for i in range(numberOfTimePoints)]
    data = pd.DataFrame(np.ndarray(dfSize, dtype=str),
                        index=range(minTrackNumber, maxTrackNumber + 1),  # the index will follow the track numbers
                        columns=timePointLabels)
    return data

def FillInDfForExport(imageStack, propertyName, propertyMeasurementsForTracks, well, field):

    for timePoint in range(imageStack.shape[0]):  # for each image in the stack
        regions = regionprops(imageStack[timePoint])  # get RegionProperties objects
        for region in regions:  # for each RP object
            print(f'well {well}, field {field}, timepoint {timePoint}, region label: {region.label}')
            if propertyName == 'roundness':
                value = CalculateRoundness(getattr(region, 'area'), getattr(region, 'perimeter'))
            else:
                value = getattr(region, propertyName)  # get the property value of that region
            label = region.label        # the label is the integer skimage label of the object
            timePointLabel = 't{}'.format(timePoint)
            propertyMeasurementsForTracks.loc[label, timePointLabel] = str(value)
        print('f')
    return propertyMeasurementsForTracks


def GetWellConditionText(plateLayout, well):
    wellCondition = plateLayout[well[0] - 1][well[1] - 1]  # get plateLayout list of condition, concentration, and unit
    wellConditionStrings = [str(element) for element in wellCondition]  # convert the float concentration to string
    wellCoordinates = ['well', str(well)]  # get well coordinates to add
    return ' '.join(wellCoordinates + wellConditionStrings)  # join all into one string


def MeasureAndExport(outputPath, propertiesToMeasure, imageStacks, plateLayout):

    with pd.ExcelWriter(str(outputPath.absolute())) as writer:
    # path.absolute() contains the entire path to .xlsx file, i.e. /home/... in Linux or C:/... in Windows

        for propertyName in propertiesToMeasure:
            print(f'measuring {propertyName}')
            latestExportColumnForWell = 0  # 0 for the first well
            for wellIndex, (well, wellFieldImages) in enumerate(imageStacks.items()):
                print(f'Well {well}')
                sortedFields = sorted(wellFieldImages, key=int)
                latestExportRowForWell = 1

                for field in sortedFields:
                    print(f'Field {field}')
                    imageStack = imageStacks[well][field]

                    propertyMeasurementsForTracks = CreateDfForExport(imageStack)
                    propertyMeasurementsForTracks = FillInDfForExport(imageStack, propertyName, propertyMeasurementsForTracks, well, field)

                    # > Load dataframe into spreadsheet
                    wellCondition = GetWellConditionText(plateLayout, well)

                    if field != 1:  # assumes that the first field is always numbered 1
                        propertyMeasurementsForTracks.to_excel(writer, sheet_name=propertyName,
                                                               startrow=latestExportRowForWell,
                                                               startcol=latestExportColumnForWell,
                                                               header=False)
                        latestExportRowForWell += propertyMeasurementsForTracks.shape[0]
                    else:
                        propertyMeasurementsForTracks.to_excel(writer, sheet_name=propertyName,
                                                               startrow=latestExportRowForWell,
                                                               startcol=latestExportColumnForWell)
                        # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other
                        latestExportRowForWell += propertyMeasurementsForTracks.shape[0] + 1

                        writer.sheets[propertyName].cell(row=1, column=1+latestExportColumnForWell).value = wellCondition
                        # Within .cell(), row and column are 1-indexed
                latestExportColumnForWell += propertyMeasurementsForTracks.shape[1] + 2






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
