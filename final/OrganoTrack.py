from Importing import ReadImages, ReadPlateLayout, ReadImage, UpdatePlateLayoutWithImageNames
from Detecting import SegmentWithOrganoSegPy
from Exporting import SaveData, ExportImageStackMeasurements, ExportSingleImageMeasurements
from Filtering import FilterByFeature
from Displaying import DisplayImages, Display, ConvertLabelledImageToBinary, displayingTrackedSet
from Tracking import track, SaveImages, MakeDirectory, stack, LabelAndStack, Label
from Measuring import MeasureMorphometry

# temporary imports
import cv2 as cv
from Displaying import Mask
from pathlib import Path
from PIL import Image
from ImageHandling import DrawRegionsOnImages
import numpy as np
import pandas as pd
import skimage
import time
from itertools import chain



def RunOrganoTrack(importPath = None, exportPath = None, livePreview = False,
                   saveSeg = False, segmentOrgs = True, segmentedImagesPath = None,
                   filterOrgs = False, filterCriteria = None,
                   trackOrgs = False, timePoints = None, overlayTrack = False,
                   exportOrgMeasures = False, morphPropsToMeasure = None):

    # times = []
    # tic = time.process_time()
    inputImages, inputImagesPaths = ReadImages(importPath)
    # toc = time.process_time() - tic
    # times.append(toc)

    plateLayout = ReadPlateLayout(importPath)
    plateLayout = UpdatePlateLayoutWithImageNames(plateLayout, inputImagesPaths)

    if segmentOrgs:
        # Segment
        # tic = time.process_time()
        imagesInAnalysis = SegmentWithOrganoSegPy(inputImages, saveSeg, exportPath, inputImagesPaths)
        # toc = time.process_time() - tic
        # times.append(toc)
    else:
        # Load segmentations
        imagesInAnalysis, imageNames = ReadImages(segmentedImagesPath)

    if livePreview:
        print('live preview')
        # Display('segmented example', imagesInAnalysis[0], 0.5)
        # cv.waitKey(0)
        # continueViewing = True
        # while continueViewing:
        #     viewWell = input('What other well do you want to see? ')
        #     well =

    if filterOrgs:

        # allow the creation of a set of filters that work on all the images

        filterOpsIndeces = [filterCriteria.index(filterOp) for filterOp in filterCriteria if(type(filterOp) is str)]
        morphologicalPropertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                                      'eccentricity', 'equivalent_diameter_area', 'euler_number',
                                      'extent', 'feret_diameter_max', 'orientation',
                                      'perimeter', 'perimeter_crofton', 'roundness', 'solidity']
        # the number of strings

        # the index of the strings
        for i in filterOpsIndeces:  # for each filterOp
            if filterCriteria[i] in morphologicalPropertyNames:
                imagesInAnalysis = FilterByFeature(imagesInAnalysis, filterCriteria[i], filterCriteria[i+1])

            if livePreview:
                print('h')
                # Display('Filtered by ' + filterCriteria[i], imagesInAnalysis[0], 0.5)
                # cv.waitKey(0)
                # SegmentWithOrganoSegPy(inputImages, saveSeg, exportPath, inputImagesPaths)

    if trackOrgs:
        # Tracking
        timelapseSets = [imagesInAnalysis[i * timePoints:(i + 1) * timePoints]
                         for i in range((len(imagesInAnalysis) + timePoints - 1) // timePoints )]
        # timelapseSets = [[imagesInAnalysis[0], imagesInAnalysis[1], imagesInAnalysis[2], imagesInAnalysis[3]]]
        # line above from https://www.geeksforgeeks.org/break-list-chunks-size-n-python/. Compare other methods
        # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]

        imageNamesCollected = [imageNames[i * timePoints:(i + 1) * timePoints]
                               for i in range((len(imageNames) + timePoints - 1) // timePoints )]
        # imageNamesCollected = [[imageNames[0], imageNames[1], imageNames[2], imageNames[3]]]

        trackedSets = [track(timelapseSet) for timelapseSet in timelapseSets]  # track expects a list timelapse images
        # [ tracked stack 1, tracked stack 2, ..., tracked 3D array n ]

        binaryTrackedSets = [ConvertLabelledImageToBinary(trackedSet) for trackedSet in trackedSets]
        # [ tracked 3D array 1, tracked 3D array 2, ..., tracked 3D array n ]

        binaryTrackedSetsList = [[binaryTrackedSet[i] for i in range(len(binaryTrackedSet))] for binaryTrackedSet in binaryTrackedSets]
        # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]

        binaryTrackedList = list(chain(*binaryTrackedSetsList))
        # [ time lapse set 1, t0, t1, t2, t3, ..., timelapse set n ]

        if overlayTrack:
            # Create masks
            maskedImages = [Mask(ori, pred) for ori, pred in zip(inputImages, binaryTrackedList)]

            # Regather timelapse masked images
            maskedImages = [maskedImages[i * timePoints:(i + 1) * timePoints]
                             for i in range((len(maskedImages) + timePoints - 1) // timePoints )]
            # maskedImages = [[maskedImages[0], maskedImages[1], maskedImages[2], maskedImages[3]]]
            # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]

            # Convert images to PIL format to use OrganoID functions

            maskedImagesPIL = [[Image.fromarray(img) for img in maskedSet] for maskedSet in maskedImages]
            # lsit of list of PIL images
            print('h')

            # Storage function
            def Output(name: str, data, count):
                if exportPath is not None:
                    MakeDirectory(exportPath)
                    SaveImages(data, "_" + name.lower(), maskedImagesPIL[count], exportPath, imageNamesCollected[count])  # pilImages is a list of PIL Image.Image objects
                    # imagePathsForExport = [imagePaths + '/' + imageName for imageName in rawImageNames]

            # Create an overlay and output it
            for i in range(len(trackedSets)):  # for each timelapse set
                overlayImages = DrawRegionsOnImages(trackedSets[i], stack(maskedImages[i]), (255, 255, 255), 16, (0, 255, 0))  # np.array, likely 3D
                Output('Overlay', overlayImages, i)
                print('tracking')

    if exportOrgMeasures:
        if trackOrgs:
            measuresFileName = 'trackedMeasures.xlsx'
            conditions = [" ".join(str(item) for item in alist) for alist in plateLayout[1][1:8]]
            ExportImageStackMeasurements(exportPath / measuresFileName, morphPropsToMeasure, trackedSets, conditions)
            print('h')
    print('curre')
    # return times


if __name__ == '__main__':
    segmented = True

    # Reading
    imagePaths = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1'
    rawImages, rawImageNames = ReadImages(imagePaths)

    imagePathsForExport = [imagePaths + '/' + imageName for imageName in rawImageNames]

    if segmented:
        segmentedImages = [SegmentWithOrganoSegPy(image) for image in rawImages]
        # Load segmentations
        ReadImages()


    else:
        # Segment
        segmentedImages = [SegmentWithOrganoSegPy(image) for image in rawImages]

        # Save segmentations
        outputPath = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1-segmented'
        SaveData(outputPath, segmentedImages, rawImageNames)



    filterOp = False
    measureOp = False
    trackingCode = False

    if filterOp:
        # Preparing for filtering
        filterFeature = 'area'
        filterThreshold = 450

        # Filtering
        filteredImages = [FilterByFeature(image, filterFeature, filterThreshold) for image in segmentedImages]


    if measureOp:
        # Measuring
        outPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/buildingExport/data.xlsx')

        labeledImages = [Label(image) for image in filteredImages]
        images1 = [LabelAndStack(segmentedImages)]
        images2 = [LabelAndStack(filteredImages)]
        # images = images1 + images2
        images = labeledImages
        conditions = ['t1', 't2', 't3', 't4']
        # images = labeledImages

        propertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                         'eccentricity', 'equivalent_diameter_area', 'euler_number',
                         'extent', 'feret_diameter_max', 'orientation',
                         'perimeter', 'perimeter_crofton', 'solidity']


        if images[0].ndim == 3:  # stacks
            ExportImageStackMeasurements(outPath, propertyNames, images, conditions)

        else:
            ExportSingleImageMeasurements(outPath, propertyNames, images, conditions)


    if trackingCode:
        # Tracking
        trackedSet = track(filteredImages)
        binaryTrackedSet = ConvertLabelledImageToBinary(trackedSet)
        binaryTrackedSetList = [binaryTrackedSet[i] for i in range(len(binaryTrackedSet))]

        # Create masks
        maskedImages = [Mask(ori, pred) for ori, pred in zip(rawImages, binaryTrackedSetList)]
        maskedPath = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1-segmented/masked_segmented-31.03.2023-11_43_56'
        SaveData(maskedPath, maskedImages, rawImageNames)

        # Convert images to PIL format to use OrganoID functions
        outputPath = Path(maskedPath)
        maskedImagesPIL = [Image.fromarray(img) for img in maskedImages]


        # Storage function
        def Output(name: str, data):
            if outputPath is not None:
                MakeDirectory(outputPath)
                SaveImages(data, "_" + name.lower(), maskedImagesPIL, outputPath, imagePathsForExport)  # pilImages is a list of PIL Image.Image objects


        # Create an overlay and output it
        overlayImages = DrawRegionsOnImages(trackedSet, stack(maskedImages), (255, 255, 255), 16, (0, 255, 0))  # np.array, likely 3D
        Output('Overlay', overlayImages)


        # displayingTrackedSet('tracked', trackedSet, 0.5)


    cv.waitKey(0)