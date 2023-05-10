from pathlib import Path
from Detecting import Evaluate
import numpy as np
from Importing import ReadImages
from Detecting import SegmentWithOrganoSegPy
from Displaying import DisplayImages
import cv2 as cv
from datetime import datetime
import os
import pandas as pd

def SaveOverlay(overlay, exportPath, imagePath):
    cv.imwrite(str(exportPath / imagePath.name), overlay)

# EMC dataset
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
# exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions')
# set1_pred_Dir = exportPath / 'segmented'
# datasetName = 'EMC'

# OrganoID Gemcitabine dataset
set1_GT_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/groundTruth')
set1_ori_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/original')
exportPath = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/export')
set1_pred_Dir = exportPath / 'segmented'
datasetName = 'OrganoID OriginalData'


# GT images
groundTruthImages, _ = ReadImages(set1_GT_Dir)


if not os.path.exists(set1_pred_Dir):
    oriImages, imageNames = ReadImages(set1_ori_Dir)
    segParams = [0.5, 250, 150]
    saveSegParams = [True, exportPath, imageNames]
    predictionImages = SegmentWithOrganoSegPy(oriImages, segParams, saveSegParams)
else:
    predictionImages, imageNames = ReadImages(set1_pred_Dir)


segmentationScores = np.zeros((len(predictionImages), 3))

overlayExportPath = exportPath / 'overlay'
if not os.path.exists(overlayExportPath):
    os.mkdir(overlayExportPath)


for i, (prediction, groundTruth) in enumerate(zip(predictionImages, groundTruthImages)):
    segmentationScores[i], overlay = Evaluate(prediction, groundTruth)
    SaveOverlay(overlay, overlayExportPath, imageNames[i])

indexNames = []
for i in range(len(imageNames)):
    indexNames.append(imageNames[i].name)

df = pd.DataFrame(segmentationScores, index=indexNames, columns=['f1', 'iou', 'dice'])

outputPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictionsdata.xlsx')

with pd.ExcelWriter(str(outputPath.absolute())) as writer:
    df.to_excel(writer, sheet_name=datasetName, startrow=1)
    # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other


print(segmentationScores)

