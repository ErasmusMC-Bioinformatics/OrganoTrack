from pathlib import Path
from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Evaluating import EvaluateSegmentationAccuracy
import numpy as np
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import DisplayImages
import cv2 as cv
from datetime import datetime
import os
import pandas as pd

def SaveOverlay(overlay, exportPath, imagePath):
    cv.imwrite(str(exportPath / imagePath.name), overlay)


def GetDatasetsDirs(datasets):
    datasetDirs = {'EMC-prelim': Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset'),
                   'OrganoID-Mouse': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids'),
                   'OrganoID-Original': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData')}
    return datasetDirs[datasets]

def LoadImages(datasetsDirs, predictionMethods):

    datasetsGroundTruthsAndPredictions = dict()

    for dataset in list(datasetsDirs.keys()):
        oneDatasetGroundTruthsAndPredictions = dict()
        datasetDir = datasetsDirs[dataset]

        groundTruthDir = datasetDir / 'groundTruth'
        groundTruthImages, groundTruthImagesNames = ReadImages(groundTruthDir)
        oneDatasetGroundTruthsAndPredictions['groundTruth'] = [groundTruthImages, groundTruthImagesNames]

        for method in predictionMethods:
            predictionDir = datasetDir / 'predictions' / (method + '-segmented')
            if method == 'OrganoTrack' and not os.path.exists(predictionDir):
                originalImagesDir = datasetDir / 'original'
                originalImages, predictionImagesNames = ReadImages(originalImagesDir)
                segParams = [0.5, 250, 150]
                exportPath = datasetDir / 'predictions'
                saveSegParams = [True, exportPath, predictionImagesNames]
                predictionImages = SegmentWithOrganoSegPy(originalImages, segParams, saveSegParams)
            else:
                predictionImages, predictionImagesNames = ReadImages(predictionDir)

            oneDatasetGroundTruthsAndPredictions[method] = [predictionImages, predictionImagesNames]

        datasetsGroundTruthsAndPredictions[dataset] = oneDatasetGroundTruthsAndPredictions

    return datasetsGroundTruthsAndPredictions

def CalculatePredictionScores(datasetsGtAndPreds, datasetsDirs, predictionMethods):

    datasetsSegScoresWithDiffMethods = dict()

    for dataset in list(datasetsDirs.keys()):
        segScoresWithDiffMethods = dict()
        datasetDir = datasetsDirs[dataset]
        groundTruthImages = datasetsGtAndPreds[dataset]['groundTruth'][0]

        for method in predictionMethods:
            predictedImages = datasetsGtAndPreds[dataset][method][0]
            predictedImagesNames = datasetsGtAndPreds[dataset][method][1]

            # Create directory to store overlays
            exportPath = datasetDir / 'predictions'
            overlayExportPath = exportPath / (method + '-overlay')
            if not os.path.exists(overlayExportPath):
                os.mkdir(overlayExportPath)

            segmentationScores = np.zeros((len(groundTruthImages), 3))

            # Evaluating prediction against ground truth
            for i, (prediction, groundTruth) in enumerate(zip(predictedImages, groundTruthImages)):
                segmentationScores[i], overlay = EvaluateSegmentationAccuracy(prediction, groundTruth)
                SaveOverlay(overlay, overlayExportPath, predictedImagesNames[i])

            segScoresWithDiffMethods[method] = segmentationScores
            segScoresWithDiffMethods['imageNames'] = predictedImagesNames

        datasetsSegScoresWithDiffMethods[dataset] = segScoresWithDiffMethods

def ExportPredictionScores(datasetsPredictionScores, analysisFileName, predictionMethods):
    # Exporting segmentation accuracies

    datasets = list(datasetsPredictionScores.keys())

    with pd.ExcelWriter(str(analysisFileName.absolute())) as writer:

        for dataset in datasets:

            for j, method in enumerate(predictionMethods):
                segmentationScores = datasetsPredictionScores[dataset][method]
                indexNames = []
                imageNames = datasetsPredictionScores[dataset]['imageNames']
                for i in range(len(imageNames)):
                    indexNames.append(imageNames[i].name)

                df = pd.DataFrame(segmentationScores, index=indexNames, columns=['f1', 'iou', 'dice'])
                df.to_excel(writer, sheet_name=dataset, startrow=1, startcol=j * (3 + 2))
                # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other

def LoadPredictionScoreAnalysis():
    pass


def PlotPredictionAccuracies():
    pass

def OrganoTrackVsHarmony():  # one dataset
    datasets = ['EMC-preliminary']
    predictors = ['Harmony', 'OrganoTrack']
    analysisDir = Path('/home/franz/Documents/mep/results/segmentation-analysis')

    analysisFile = analysisDir / (datasets[0] + '-' + predictors[0] + '-' + predictors[1] + '.xlsx')

    if not os.path.exists(analysisFile):
        datasetsDirs = GetDatasetsDirs(datasets)
        datasetsGtAndPreds = LoadImages(datasetsDirs, predictors)
        datasetsPredictionScores = CalculatePredictionScores(datasetsGtAndPreds, datasetsDirs, predictors)
        ExportPredictionScores(datasetsPredictionScores, analysisFile, predictors)
    else:
        ExperimentPredictionScores = LoadPredictionScoreAnalysis(analysisFile)

    PlotPredictionAccuracies()


if __name__ == '__main__':
    OrganoTrackVsHarmony()





'''
    Previous directories - ignore
'''
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

# # EMC dataset - segmented by Harmony
# set1_GT_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/annotated/annotations')
# set1_ori_Dir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating')
# exportPath = Path('/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/images-for-OrganoTrack-Harmony-comparison/export')
# set1_pred_Dir = exportPath / 'OrganoID_segmented'
# segmenter = 'OrganoID'
# datasetName = 'EMC'

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