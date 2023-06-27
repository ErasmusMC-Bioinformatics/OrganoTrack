from pathlib import Path
from OrganoTrack.Importing import ReadImages
import cv2 as cv
import numpy as np
import tifffile as tf
import os

# loadimages
# > Get the names and extensions of the image files in the directory
imagesDir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/sam-segment-part1-cis-data/output/SAM-segmented/images')
inputImagesNames = sorted(os.listdir(imagesDir))

# > Create directory paths for each image file
inputImagesPaths = [imagesDir / imageName for imageName in inputImagesNames]

# > Read images
inputImages = [tf.imread(str(imagePath), cv.IMREAD_GRAYSCALE) for imagePath in inputImagesPaths]

exportDir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/sam-segment-part1-cis-data/newImages')

# convert all to numpy and export
for i, image in enumerate(inputImages):
    fixed = image.astype(np.uint8)
    _, fixed = cv.threshold(fixed, 0, 255, cv.THRESH_BINARY)  # anything more than 0 becomes full
    cv.imwrite(str(exportDir / inputImagesPaths[i].name), fixed)