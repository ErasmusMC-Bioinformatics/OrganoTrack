from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import DisplayImages, ExportImageWithContours, Display
from pathlib import Path
import cv2 as cv

imagesDir = Path('/home/franz/Documents/mep/report/results/import-segment-diff-images/input')
exportPath = Path('/home/franz/Documents/mep/report/results/import-segment-diff-images/output')

images, imagesPaths = ReadImages(imagesDir)

extraBlur = False
blurSize = 3
segParams = [0.5, 250, 150, extraBlur, blurSize]
saveSegParams = [True, exportPath, imagesPaths]
segmentedImages = SegmentWithOrganoSegPy(images, segParams, saveSegParams)

for i, (original, prediction) in enumerate(zip(images,segmentedImages)):
    overlayed = ExportImageWithContours(original, prediction)
    Display(str(i), overlayed, 0.5)
cv.waitKey(0)
