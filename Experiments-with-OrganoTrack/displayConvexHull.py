from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import DisplayImages, ExportImageWithContours
from pathlib import Path
import cv2 as cv
import random as rng
import numpy as np
import os


def convexHull():
    rng.seed(12345)

    imagesDir = Path('G:/My Drive/mep/image-analysis-pipelines/organoID/convexHull/input')
    exportPath = Path('G:/My Drive/mep/image-analysis-pipelines/organoID/convexHull')
    segmentedImagesPath = exportPath / 'OrganoTrack-segmented'
    images, imagesPaths = ReadImages(imagesDir)

    if not os.path.exists(segmentedImagesPath):
        extraBlur = False
        blurSize = 3
        displaySegStep = False
        segParams = [0.5, 250, 150, extraBlur, blurSize, displaySegStep]
        saveSegParams = [True, exportPath, imagesPaths]
        segmentedImages = SegmentWithOrganoSegPy(images, segParams, saveSegParams)
    else:
        segmentedImages, imagesPaths = ReadImages(segmentedImagesPath)

    oneImage = segmentedImages[2]

    # displayScale = 1
    # DisplayImages('convexHull', segmentedImages, displayScale)
    # cv.waitKey(0)


    contours, _ = cv.findContours(oneImage, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Find the convex hull object for each contour
    hullsForOneImage = []
    for contour in contours:
        hull = cv.convexHull(contour)
        hullsForOneImage.append(hull)



    # Draw contours + hull results
    drawing = np.zeros((oneImage.shape[0], oneImage.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        cv.drawContours(drawing, contours, i, (46, 204, 113), 5)
        cv.drawContours(drawing, hullsForOneImage, i, (182, 38, 155), 5)


    # Show in a window
    # cv.drawContours(originalImage, drawing, -1, (0, 255, 0), 2)
    DisplayImages('h', [drawing], 0.5)
    # cv.imwrite(str(exportPath / 'convexHull.png'), drawing)
    cv.waitKey(0)
