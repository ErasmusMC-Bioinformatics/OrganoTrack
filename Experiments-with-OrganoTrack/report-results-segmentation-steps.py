from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import ExportImageWithContours, Display
from pathlib import Path
import cv2 as cv

imagesDir = Path('/home/franz/Documents/mep/report/results/segmentation-steps-of-organosegpy/input')
exportPath = Path('/home/franz/Documents/mep/report/results/segmentation-steps-of-organosegpy/output')

images, imagesPaths = ReadImages(imagesDir)

extraBlur = False
blurSize = 3
segParams = [0.5, 250, 150, extraBlur, blurSize]
saveSegParams = [False, exportPath, imagesPaths]
segmentedImages = SegmentWithOrganoSegPy(images, segParams, saveSegParams)

for original, prediction in zip(images,segmentedImages):
    overlayed = ExportImageWithContours(original, prediction)
    Display('overlayed', overlayed, 0.5)
cv.waitKey(0)
