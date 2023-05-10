from pathlib import Path
from Detecting import Evaluate
import numpy as np
from Importing import ReadImages
from Detecting import SegmentWithOrganoSegPy
from Displaying import DisplayImages
import cv2 as cv

# Paths: GT, ori, export
set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions')

# GT images
groundTruthImages, _ = ReadImages(set1_GT_Dir)
# DisplayImages('ground truth', groundTruthImages, 0.5)


# Prediction images
oriImages, oriImageNames = ReadImages(set1_ori_Dir)
segParams = [0.5, 250, 150]
saveSegParams = [True, exportPath, oriImageNames]
predictionImages = SegmentWithOrganoSegPy(oriImages, segParams, saveSegParams)
# DisplayImages('prediction', predictionImages, 0.5)
# cv.waitKey(0)

# Image Overlay
saveImgOverlay = [True, exportPath, None]

segmentationScores = np.zeros((len(predictionImages), 3))

for i, (prediction, groundTruth) in enumerate(zip(predictionImages, groundTruthImages)):
    saveImgOverlay[2] = oriImageNames[i]
    segmentationScores[i] = np.asarray(Evaluate(prediction, groundTruth, saveImgOverlay))

#has both GT and ori



# Ground truth figures, set 2: OrganoID
