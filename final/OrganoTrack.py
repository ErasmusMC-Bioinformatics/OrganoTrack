from Importing import ReadImages
from Detecting import SegmentWithOrganoSegPy
from Exporting import SaveData, ExportImageStackMeasurements, ExportSingleImageMeasurements
from Filtering import FilterByFeature
from Displaying import DisplayImages, Display, ConvertLabelledImageToBinary, displayingTrackedSet
from Measuring import MeasureMorphometry
from OrganoTracking import track

# temporary imports
import cv2 as cv
from Displaying import Mask
from pathlib import Path
from PIL import Image
from OrganoTracking import track, SaveImages, MakeDirectory
from ImageHandling import DrawRegionsOnImages
import numpy as np
import pandas as pd
from skimage.measure import label
import skimage

def stack(images):
    return np.stack(images[:])

def LabelAndStack(images):
    return label(stack(images))

def RunOrganoTrack(inputImagesPaths, exportPath, segment: False, loadSeg: False, segmentedImagePaths):
    inputImages, inputImagesNames = ReadImages(inputImagesPaths)

    if segment:
        segmentedImages = [SegmentWithOrganoSegPy(image) for image in inputImages]
        SaveData(exportPath, segmentedImages, inputImagesNames)

    if loadSeg:
        segmentedImages, inputImagesNames = ReadImages(segmentedImagePaths)



if __name__ == '__main__':
    segmented = True

    # Reading
    imagePaths = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1'
    rawImages, rawImageNames = ReadImages(imagePaths)

    imagePathsForExport = [imagePaths + '/' + imageName for imageName in rawImageNames]

    if segmented:

        # Load segmentations
        segmentedImagePaths = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1-segmented/segmented-31.03.2023-11_43_56'
        segmentedImages, imageNames = ReadImages(segmentedImagePaths)

    else:
        # Segment
        segmentedImages = [SegmentWithOrganoSegPy(image) for image in rawImages]

        # Save segmentations
        outputPath = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1-segmented'
        SaveData(outputPath, segmentedImages, rawImageNames)

    # DisplayImages('pre-filter', segmentedImages, 0.5)

    # Preparing for filtering
    filterFeature = 'area'
    filterThreshold = 450

    # Filtering
    filteredImages = [FilterByFeature(image, filterFeature, filterThreshold) for image in segmentedImages]

    # Measuring
    outPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/buildingExport/data.xlsx')

    labeledImages = [label(image) for image in filteredImages]
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

    # DisplayImages('post-filter', filteredImages, 0.5)
    trackingCode = False

    if trackingCode:
        # Tracking
        trackedSet = track(filteredImages)


        # Create masks
        maskedImages = [Mask(ori, pred) for ori, pred in zip(rawImages, segmentedImages)]
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


        displayingTrackedSet('tracked', trackedSet, 0.5)


    cv.waitKey(0)