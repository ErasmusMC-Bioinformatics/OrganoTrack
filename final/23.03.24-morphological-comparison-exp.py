from Importing import readImages
from Detecting import segmentWithOrganoSegPy, saveData
from Displaying import display

import cv2 as cv

dataDir = '/home/franz/Documents/mep/data/experiments/organoid-morphology'

images, imageNames = readImages(dataDir)

segmentedImages = [segmentWithOrganoSegPy(img) for img in images]
saveData(dataDir, segmentedImages, imageNames)

displayScale = 0.25
for i in range(len(segmentedImages)):
    display(imageNames[i], segmentedImages[i], displayScale)

cv.waitKey(0)