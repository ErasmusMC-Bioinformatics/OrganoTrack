from pathlib import Path
from OrganoTrack.Detecting import Evaluate, SegmentWithOrganoSegPy
import numpy as np
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import DisplayImages
import cv2 as cv
from datetime import datetime
import os
import pandas as pd

def SaveOverlay(overlay, exportPath, imagePath):
    cv.imwrite(str(exportPath / imagePath.name), overlay)

# EMC dataset - segmented by OrganoTrack
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
# exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions')
# set1_pred_Dir = exportPath / 'segmented'
# segmenter = 'OrganoTrack'
# datasetName = 'EMC'

# EMC dataset - segmented by OrganoID
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
# exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions')
# set1_pred_Dir = exportPath / 'OrganoID_segmented'
# segmenter = 'OrganoID'
# datasetName = 'EMC'

# # EMC dataset - segmented by SAM
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
# exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions')
# set1_pred_Dir = exportPath / 'OrganoID_segmented'
# segmenter = 'OrganoID'
# datasetName = 'EMC'

# EMC dataset - segmented by Harmony
set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
exportPath = Path('/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/images-for-OrganoTrack-Harmony-comparison/export')
set1_pred_Dir = exportPath / 'OrganoID_segmented'
segmenter = 'OrganoID'
datasetName = 'EMC'

# OrganoID Gemcitabine dataset - segmented by OrganoTrack
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/groundTruth')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/original')
# exportPath = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/export')
# set1_pred_Dir = exportPath / 'segmented'
# segmenter = 'OrganoTrack'
# datasetName = 'OrganoID-OriginalData'

# OrganoID Gemcitabine dataset - segmented by OrganoID
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/groundTruth')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/original')
# exportPath = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/export')
# set1_pred_Dir = exportPath / 'OrganoID-segmented'
# segmenter = 'OrganoID'
# datasetName = 'OrganoID-OriginalData'

# # OrganoID MouseOrganoids dataset - segmented by OrganoTrack
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/GroundTruth')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Original')
# exportPath = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Export')
# set1_pred_Dir = exportPath / 'segmented'
# segmenter = 'OrganoTrack'
# datasetName = 'OrganoID-MouseOrganoids'

# # OrganoID MouseOrganoids dataset - segmented by OrganoID
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/GroundTruth')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Original')
# exportPath = Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Export')
# set1_pred_Dir = exportPath / 'OrganoID-segmented'
# segmenter = 'OrganoID'
# datasetName = 'OrganoID-MouseOrganoids'


# Load GT images
groundTruthImages, gtImageNames = ReadImages(set1_GT_Dir)

# Get prediction images
if not os.path.exists(set1_pred_Dir) and segmenter == 'OrganoTrack':
    oriImages, imageNames = ReadImages(set1_ori_Dir)
    segParams = [0.5, 250, 150]
    saveSegParams = [True, exportPath, imageNames]
    predictionImages = SegmentWithOrganoSegPy(oriImages, segParams, saveSegParams)
else:
    predictionImages, imageNames = ReadImages(set1_pred_Dir)




# Create directory to store overlays
overlayExportPath = exportPath / (segmenter+'-overlay')
if not os.path.exists(overlayExportPath):
    os.mkdir(overlayExportPath)

segmentationScores = np.zeros((len(predictionImages), 3))
# Evaluating prediction against ground truth
for i, (prediction, groundTruth) in enumerate(zip(predictionImages, groundTruthImages)):
    segmentationScores[i], overlay = Evaluate(prediction, groundTruth)
    SaveOverlay(overlay, overlayExportPath, imageNames[i])


# Exportnig segmentation accuracies
indexNames = []
for i in range(len(imageNames)):
    indexNames.append(imageNames[i].name)

df = pd.DataFrame(segmentationScores, index=indexNames, columns=['f1', 'iou', 'dice'])

outputPath = exportPath / (segmenter+'-seg-scores-'+datasetName+'.xlsx')

with pd.ExcelWriter(str(outputPath.absolute())) as writer:
    df.to_excel(writer, sheet_name=datasetName, startrow=1)
    # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other


print(segmentationScores)

