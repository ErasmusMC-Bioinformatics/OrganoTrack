from OrganoTrack.Importing import ReadImages, ReadImages2, ReadPlateLayout, UpdatePlateLayoutWithImageNames
from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Exporting import SaveData, MeasureAndExport, ExportSingleImageMeasurements
from OrganoTrack.Filtering import FilterByFeature, RemoveBoundaryObjects
from OrganoTrack.Displaying import DisplayImages, Display, ConvertLabelledImageToBinary, displayingTrackedSet, ExportImageWithContours, Mask
from OrganoTrack.Tracking import track, SaveImages, MakeDirectory, stack, LabelAndStack, Label, UpdateTrackedStack

# temporary imports
import cv2 as cv
from pathlib import Path
from PIL import Image
from datetime import datetime
from OrganoTrack.ImageHandling import DrawRegionsOnImages
import numpy as np
import pandas as pd
import skimage.measure
import matplotlib.pyplot as plt
import time
from itertools import chain


def RunOrganoTrack(importPath: Path, identifiers, exportPath: Path,
                   segmentOrgs = False, segParams = None, saveSegParams = None, segmentedImagesPath = None,
                   filterBoundary=False, filterOrgs = False, filterCriteria = None,
                   trackOrgs = False, timePoints = None, overlayTrack = False,
                   exportOrgMeasures = False, numberOfWellFields = None, morphPropsToMeasure = None,
                   plotData = False, loadDataForPlotting = False, pathDataForPlotting = None):


    if segmentOrgs:
        inputImages, imageNames = ReadImages(importPath, identifiers)
        for well, wellFieldImages in inputImages.items():
            for field, fieldTimeImages in wellFieldImages.items():
                inputImages[well][field] = SegmentWithOrganoSegPy(fieldTimeImages, segParams, saveSegParams)

    else:  # Load saved segmentations
        inputImages, imageNames = ReadImages(segmentedImagesPath, identifiers)

    plateLayout = ReadPlateLayout(importPath)
    plateLayout = UpdatePlateLayoutWithImageNames(plateLayout, imageNames)


    if filterBoundary:
        for well, wellFieldImages in inputImages.items():
            for field, fieldTimeImages in wellFieldImages.items():
                inputImages[well][field] = RemoveBoundaryObjects(fieldTimeImages)


    if filterOrgs:

        filterOpsIndeces = [filterCriteria.index(filterOp) for filterOp in filterCriteria if(type(filterOp) is str)]
        morphologicalPropertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                                      'eccentricity', 'equivalent_diameter_area', 'euler_number',
                                      'extent', 'feret_diameter_max', 'orientation',
                                      'perimeter', 'perimeter_crofton', 'roundness', 'solidity']
        # the index of the strings
        for i in filterOpsIndeces:  # for each filterOp
            if filterCriteria[i] in morphologicalPropertyNames:
                for well, wellFieldImages in inputImages.items():
                    for field, fieldTimeImages in wellFieldImages.items():
                        inputImages[well][field] = FilterByFeature(fieldTimeImages, filterCriteria[i], filterCriteria[i+1])


    if trackOrgs:
        for well, wellFieldImages in inputImages.items():
            highestTrackIDnum = 0
            sortedFields = sorted(wellFieldImages, key=int)

            for field in sortedFields:
                print(f'Tracking well {well}, field = {field}')
                timeLapseSet = inputImages[well][field]
                trackedTimeLapseSet = track(timeLapseSet)

                # non_zero_values = trackedTimeLapseSet[trackedTimeLapseSet != 0]
                # min_value = np.min(non_zero_values)
                # max_value = np.max(non_zero_values)
                # print(f'Before update: min = {min_value}, max = {max_value}')
                # print(f'Number to add = {highestTrackIDnum}')
                trackedTimeLapseSet2 = np.where(trackedTimeLapseSet != 0, trackedTimeLapseSet + highestTrackIDnum, trackedTimeLapseSet)
                highestTrackIDnum = np.max(trackedTimeLapseSet2)

                # non_zero_values2 = trackedTimeLapseSet2[trackedTimeLapseSet2 != 0]
                # min_value2 = np.min(non_zero_values2)
                # max_value2 = np.max(non_zero_values2)
                # print(f'After update: min = {min_value2}, max = {max_value2}')

                inputImages[well][field] = trackedTimeLapseSet2

    print('h')

    if exportOrgMeasures:
        measuresFileName = 'trackedMeasures.xlsx'
        trackedMeasurementsPerWell = MeasureAndExport(exportPath / measuresFileName, morphPropsToMeasure, inputImages, plateLayout)
        dateTimeNow2 = datetime.now()
        print(dateTimeNow2)
        print('h')


        # if overlayTrack:
        #     # Create masks
        #     maskedImages = [Mask(ori, pred) for ori, pred in zip(inputImages, binaryTrackedList)]
        #     # maskedImages = [ExportImageWithContours(ori, pred) for ori, pred in zip(inputImages, binaryTrackedList)]
        #
        #     # Regather timelapse masked images
        #     maskedImages = [maskedImages[i * timePoints:(i + 1) * timePoints]
        #                      for i in range((len(maskedImages) + timePoints - 1) // timePoints )]
        #     # maskedImages = [[maskedImages[0], maskedImages[1], maskedImages[2], maskedImages[3]]]
        #     # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]
        #
        #     # Convert images to PIL format to use OrganoID functions
        #
        #     maskedImagesPIL = [[Image.fromarray(img) for img in maskedSet] for maskedSet in maskedImages]
        #     # lsit of list of PIL images
        #     print('h')
        #
        #     # Storage function
        #     def Output(name: str, data, count):
        #         if exportPath is not None:
        #             MakeDirectory(exportPath)
        #             SaveImages(data, "_" + name.lower(), maskedImagesPIL[count], exportPath, imageNamesCollected[count])  # pilImages is a list of PIL Image.Image objects
        #             # imagePathsForExport = [imagePaths + '/' + imageName for imageName in rawImageNames]
        #
        #     # Create an overlay and output it
        #     for i in range(len(trackedSets)):  # for each timelapse set
        #         overlayImages = DrawRegionsOnImages(trackedSets[i], stack(maskedImages[i]), (255, 255, 255), 50, (0, 255, 0))  # np.array, likely 3D
        #         Output('Overlay', overlayImages, i)
        #         print('tracking')


if __name__ == '__main__':
    print('h')