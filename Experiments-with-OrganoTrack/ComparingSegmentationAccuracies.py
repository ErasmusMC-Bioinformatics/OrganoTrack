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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import openpyxl
from scipy.stats import ttest_ind

'''
    With this file, boxplots can be generated, comparing the segmentation accuracies for different methods on different
    datasets. It is required, however, that each dataset directory is divided into "original", "groundTruth", and 
    "predicted".
'''


def SaveOverlay(overlay, exportPath, imagePath):
    cv.imwrite(str(exportPath / imagePath.name), overlay)


# source: https://stackoverflow.com/questions/11517986/indicating-the-statistically-significant-difference-in-bar-graph
def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= 0.2*(ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)

def GetDatasetsDirs(datasets):
    datasetDirs = {'Bladder cancer': Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset'),
                   'Mouse': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids'),
                   'Salivary ACC': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/ACCOrganoids'),
                   'Colon': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/ColonOrganoids'),
                   'Lung': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/LungOrganoids'),
                   'PDAC': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/PDACOrganoids')}

    datasetsDirs = dict()
    for dataset in datasets:
        datasetsDirs[dataset] = datasetDirs[dataset]
    return datasetsDirs

def LoadImages(datasetsDirs, predictionMethods, extraOrganoTrackBlur=False, blurSize=3):

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
                segParams = [0.5, 250, 150, extraOrganoTrackBlur, blurSize]
                exportPath = datasetDir / 'predictions'
                saveSegParams = [True, exportPath, predictionImagesNames]
                predictionImages = SegmentWithOrganoSegPy(originalImages, segParams, saveSegParams)
            else:
                predictionImages, predictionImagesNames = ReadImages(predictionDir)

            oneDatasetGroundTruthsAndPredictions[method] = [predictionImages, predictionImagesNames]

        datasetsGroundTruthsAndPredictions[dataset] = oneDatasetGroundTruthsAndPredictions

    return datasetsGroundTruthsAndPredictions

def ViewLoadedImages(datasetsGtAndPreds):
    datasets = list(datasetsGtAndPreds.keys())
    methods = list(datasetsGtAndPreds[datasets[0]].keys())
    for dataset in datasets:
        for method in methods:
            images = datasetsGtAndPreds[dataset][method][0]
            for i, image in enumerate(images):
                cv.imshow(f'{dataset}, {method},{i}', image)
    cv.waitKey(0)

def CreateDirectoryToStoreOverlays(datasetDirectory, predictionMethod):
    exportPath = datasetDirectory / 'predictions'
    overlayExportPath = exportPath / (predictionMethod + '-overlay')
    if not os.path.exists(overlayExportPath):
        os.mkdir(overlayExportPath)
    return overlayExportPath

def CalculatePredictionScores(datasetsGtAndPreds, datasetsDirectories, predictionMethods):

    datasetsSegmentationScoresWithDiffMethods = dict()

    for dataset in list(datasetsDirectories.keys()):
        print(f'Evaluating the {dataset} dataset.')
        oneDatasetSegmentationScoresWithDiffMethods = dict()
        datasetDirectory = datasetsDirectories[dataset]
        groundTruthImages = datasetsGtAndPreds[dataset]['groundTruth'][0]

        for method in predictionMethods:
            predictedImages = datasetsGtAndPreds[dataset][method][0]
            predictedImagesNames = datasetsGtAndPreds[dataset][method][1]

            # Create directory to store overlays
            overlayExportPath = CreateDirectoryToStoreOverlays(datasetDirectory, method)

            segmentationScores = np.zeros((len(groundTruthImages), 3))

            # Evaluating prediction against ground truth
            for i, (prediction, groundTruth) in enumerate(zip(predictedImages, groundTruthImages)):
                segmentationScores[i], overlay = EvaluateSegmentationAccuracy(prediction, groundTruth)
                SaveOverlay(overlay, overlayExportPath, predictedImagesNames[i])

            oneDatasetSegmentationScoresWithDiffMethods[method] = segmentationScores
        oneDatasetSegmentationScoresWithDiffMethods['imageNames'] = predictedImagesNames

        datasetsSegmentationScoresWithDiffMethods[dataset] = oneDatasetSegmentationScoresWithDiffMethods

    return datasetsSegmentationScoresWithDiffMethods

def ExportPredictionScores(datasetsPredictionScores, analysisFileName, predictionMethods):
    # Exporting segmentation accuracies
    print(f'Exporting prediction scores to {analysisFileName}.')
    measureCount = 3
    datasets = list(datasetsPredictionScores.keys())

    with pd.ExcelWriter(str(analysisFileName.absolute())) as writer:

        for dataset in datasets:

            for j, method in enumerate(predictionMethods):
                methodSegmentationScores = datasetsPredictionScores[dataset][method]

                indexNames = []
                imagePaths = datasetsPredictionScores[dataset]['imageNames']
                for i in range(len(imagePaths)):
                    indexNames.append(imagePaths[i].name)

                df = pd.DataFrame(methodSegmentationScores, index=indexNames, columns=['f1', 'iou', 'dice'])
                df.to_excel(writer, sheet_name=dataset, startrow=1, startcol=j * (measureCount + 2))
                # Within .to_excel(), startrow/col are 0-indexed. Startcol calculated to fit df's next to each other

                writer.sheets[dataset].cell(row=1, column=j*(measureCount+2) + 1).value = str(method)
                # Within .cell(), row and column are 1-indexed

def LoadPredictionScoreAnalysis(analysisFilePath):
    xls = openpyxl.load_workbook(analysisFilePath)
    datasets = xls.sheetnames

    datasetsPredictionScores = dict()
    for dataset in datasets:
        oneDatasetMethodsPredictionsDf = pd.read_excel(analysisFilePath, sheet_name=str(dataset), header=None)

        oneDatasetMethodsPredictions = dict()

        firstRow = oneDatasetMethodsPredictionsDf.iloc[0]
        methods = firstRow.dropna().tolist()
        methodIndeces = list(firstRow.notna()[firstRow.notna() == True].index)

        for i, method in enumerate(methods):
            methodPredictions = np.array(oneDatasetMethodsPredictionsDf.iloc[2:, methodIndeces[i]+1:methodIndeces[i]+4], dtype=np.float64)
            oneDatasetMethodsPredictions[method] = methodPredictions

        datasetsPredictionScores[dataset] = oneDatasetMethodsPredictions

    return datasetsPredictionScores

def PlotPredictionAccuracies(datasetsPredictionScores, predictionMethods):
    datasets = list(datasetsPredictionScores.keys())
    # https: // matplotlib.org / stable / gallery / statistics / boxplot_demo.html
    measures = ['F1', 'IOU', 'Dice']
    measureIndex = [0, 1, 2]

    plt.rcParams.update({'font.size': 20})

    # Plotting colours
    colours = list(mcolors.CSS4_COLORS.keys())
    # chosenPoolOfColours = colours['blue']


    for segAccuracyMeasure, index in zip(measures, measureIndex):
        plotTicks, boxplotPositionsPerMethod = ComputeBoxplotTicksAndPositions(predictionMethods, len(datasets))
        fig, ax = plt.subplots()

        dataAcrossDatasetsAndMethods = []

        for method in predictionMethods:
            for dataset in datasets:
                dataAcrossDatasetsAndMethods.append(datasetsPredictionScores[dataset][method][:, index])

        # dataAcrossDatasetsAndMethods = np.array(dataAcrossDatasetsAndMethods).T
        ax.boxplot(dataAcrossDatasetsAndMethods, widths=0.5)  # one box plot corresponds to one method, not one dataset

        ax.set_ylabel(f'{segAccuracyMeasure} score')
        ax.set_xlabel('Datasets')
        ax.set_ylim(0, 100)
        plt.tight_layout()
        fig.show()


def ComputeBoxplotTicksAndPositions(methods, numDatasets):

    boxplotPositionsPerMethod = dict()
    numMethods = len(methods)

    plotTicks = ComputeBoxplotTicks(numMethods, numDatasets)

    for method in methods:
        positionsArray = ComputeMethodBoxplotPositions(plotTicks, numMethods)
        boxplotPositionsPerMethod[method] = positionsArray

    return plotTicks, boxplotPositionsPerMethod


def ComputeBoxplotTicks(numMethods, numDatasets):
    startingLength = 0.5
    boxWidth = 0.5
    spacesBetweenBoxes = 0.1

    startingTick = startingLength + (numMethods/2)*boxWidth + ((numMethods-1)/2)*spacesBetweenBoxes
    additionTick = startingLength + numMethods*boxWidth + (numMethods-1)*spacesBetweenBoxes

    plotTicks = np.zeros(numDatasets)
    plotTicks[0] = startingTick
    for i in range(1, len(plotTicks)):
        plotTicks[i] = plotTicks[i-1] + additionTick

    return plotTicks

def TestComputePlotTicks():
    numMethods = 3
    numDatasets = 4
    plotTicks = ComputeBoxplotTicks(numMethods, numDatasets)

def ComputeMethodBoxplotPositions(plotTicks, numDatasets, methods, w, s):
    boxWidth = 0.5
    spacesBetweenBoxes = 0.1
    numMethods = len(methods)
    result = dict()
    difference = np.zeros(numMethods)
    difference[0] = -((numMethods-1)/2)*(boxWidth+spacesBetweenBoxes)
    for i in range(1,len(difference)):
        difference[i]=difference[i-1]+(boxWidth+spacesBetweenBoxes)
    for i, method in enumerate(methods):
        methodPositions = np.array([difference[i] + plotTicks[j] for j in range(numDatasets)])
        result[method] = methodPositions
    return result

def OrganoTrackVsHarmony():  # one dataset
    datasets = ['EMC-cisplatin']
    predictors = ['Harmony', 'OrganoTrack']
    analysisDir = Path('/home/franz/Documents/mep/report/results/segmentation-analysis')
    analysisFile = analysisDir / (datasets[0] + '-' + predictors[0] + '-' + predictors[1] + '.xlsx')
    analysisExists = os.path.exists(analysisFile)
    toBlur = False

    if not analysisExists:
        datasetsDirs = GetDatasetsDirs(datasets)
        datasetsGtAndPreds = LoadImages(datasetsDirs, predictors, toBlur, 3)
        # ViewLoadedImages(datasetsGtAndPreds)
        datasetsPredictionScores = CalculatePredictionScores(datasetsGtAndPreds, datasetsDirs, predictors)
        ExportPredictionScores(datasetsPredictionScores, analysisFile, predictors)
    else:
        datasetsPredictionScores = LoadPredictionScoreAnalysis(analysisFile)

    # PlotPredictionAccuracies(datasetsPredictionScores, predictors)

    # Plotting
    x = np.array([1000])
    measures = ['F1', 'IOU', 'Dice']
    measureIndex = [0, 1, 2]
    boxWidth = 100
    plt.rcParams.update({'font.size': 20})

    measureSigPlot = {'F1': [87, 80], 'IOU': [80, 75], 'Dice': [87, 80]}
    for measure, index in zip(measures, measureIndex):
        data = []
        fig, ax = plt.subplots()
        for dataset in datasets:
            data1 = np.array(datasetsPredictionScores[dataset]['Harmony'][:, index]).T
            data.append(data1)
            ax.boxplot(data[0], positions=x-100, showfliers=False, widths=boxWidth) # one box plot corresponds to one method, not one dataset
            data2 = np.array(datasetsPredictionScores[dataset]['OrganoTrack'][:, index]).T
            data.append(data2)
            ax.boxplot(data[1], positions=x+100, showfliers=False, widths=boxWidth)
            # one box plot corresponds to one method, not one dataset
            # fill with colors, https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
            # colors = ['lightblue', 'lightgreen']
            # for bplot in bplot1:
            #     for patch, color in zip(bplot['boxes'], colors):
            #         patch.set_facecolor(color)

        ax.set_ylabel(f'{measure} score')
        ax.set_ylim(0, 100)
        labels = [item.get_text() for item in ax.get_xticklabels()]
        labels[0] = 'Baseline'  # change to OrganoTrack and baseline
        labels[1] = 'OrganoTrack'
        ax.set_xticklabels(labels)
        ax.set_xlim(800, 1200)
        valuesJitter = [np.random.normal(900, 10, 5), np.random.normal(1100, 10, 5)]
        palette = ['b', 'g', 'r', 'c', 'm', 'k']
        for jitter, val, c in zip(valuesJitter, data, palette):
            ax.scatter(jitter, val, alpha=0.4, color=c)
        plt.tight_layout()
        barplot_annotate_brackets(0, 1, .25, [900, 1100], measureSigPlot[measure])
        fig.show()
        dataA = np.asarray([float(i) for i in data[0]])
        dataB = np.asarray([float(i) for i in data[1]])
        print(ttest_ind(dataA, dataB))  # How likely is it that we would see two sets of samples like this if they were drawn from the same (but unknown) probability distribution?

def OrganoTrackVsOrganoID():
    datasets = ['EMC-cisplatin', 'OrganoID-Mouse', 'OrganoID-Original']
    predictors = ['OrganoTrack', 'OrganoID']
    analysisDir = Path('/home/franz/Documents/mep/results/segmentation-analysis')

    analysisFile = analysisDir / (datasets[0] + '-' + predictors[0] + '-' + predictors[1] + '.xlsx')
    analysisExists = os.path.exists(analysisFile)

    # if not os.path.exists(analysisFile):

    if not analysisExists:
        datasetsDirs = GetDatasetsDirs(datasets)
        datasetsGtAndPreds = LoadImages(datasetsDirs, predictors, False, 3)
        # ViewLoadedImages(datasetsGtAndPreds)
        datasetsPredictionScores = CalculatePredictionScores(datasetsGtAndPreds, datasetsDirs, predictors)
        ExportPredictionScores(datasetsPredictionScores, analysisFile, predictors)
    else:
        datasetsPredictionScores = LoadPredictionScoreAnalysis(analysisFile)

    PlotPredictionAccuracies(datasetsPredictionScores, predictors)

    # # Plotting
    # x = np.array([1000, 2000, 3000])  # 3 datasets
    # measures = ['F1', 'IOU', 'Dice']
    # measureIndex = [0, 1, 2]
    # boxWidth = 100
    # plt.rcParams.update({'font.size': 15})
    #
    # # https://stackoverflow.com/questions/14952401/creating-double-boxplots-i-e-two-boxes-for-each-x-value
    # for measure, index in zip(measures, measureIndex):
    #     fig, ax = plt.subplots()
    #     data1 = []
    #     data2 = []
    #     for dataset in datasets:
    #         data1.append(datasetsPredictionScores[dataset]['OrganoTrack'][:, index])
    #         data2.append(datasetsPredictionScores[dataset]['OrganoID'][:, index])
    #     ax.boxplot(data1, positions=x-100, showfliers=False, widths=boxWidth) # one box plot corresponds to one method, not one dataset
    #     ax.boxplot(data2, positions=x+100,
    #                    showfliers=False, widths=boxWidth)   # one box plot corresponds to one method, not one dataset
    #         # fill with colors, https://matplotlib.org/stable/gallery/statistics/boxplot_color.html
    #         # colors = ['lightblue', 'lightgreen']
    #         # for bplot in bplot1:
    #         #     for patch, color in zip(bplot['boxes'], colors):
    #         #         patch.set_facecolor(color)
    #
    #     ax.set_ylabel(f'{measure} score')
    #     ax.set_ylim(0, 100)
    #     plt.xticks(x)
    #     labels = [item.get_text() for item in ax.get_xticklabels()]
    #     labels[0] = 'EMC\npreliminary\ndataset'
    #     labels[1] = 'OrganoID\nmouse organoids\ndataset'
    #     labels[2] = 'OrganoID\noriginal organoids\ndataset'
    #
    #     ax.set_xticklabels(labels)
    #     ax.set_xlim(800, 3200)
    #
    #     ax.set_title(f'{measure} score') #  for OrganoTrack and the baseline on a sample of the EMC dataset
    #     palette = ['b', 'g', 'r', 'c', 'm', 'k']
    #     # for i in range(len(conditionsNewOrder)):
    #     #     xs.append(np.random.normal(i + 1, 0.04, areaDFsSorted[i]['Norm Frac Growth'].values.shape[0]))
    #     #
    #     # for x, val, c in zip(xs, normFracGrowthValues, palette):
    #     #     ax.scatter(x, val, alpha=0.4, color=c)
    #     plt.tight_layout()
    #     fig.show()
    #     print('h')


def OrganoTrackBlurring():
    datasets = ['EMC-cisplatin']
    predictors = ['OrganoTrack']
    analysisDir = Path('/home/franz/Documents/mep/results/segmentation-analysis')
    analysisFile = analysisDir / (datasets[0] + '-' + predictors[0] + '-blurring.xlsx')
    analysisExists = os.path.exists(analysisFile)
    blurSizes = [3, 5, 7, 9, 11, 13, 15]
    datasetsDirs = GetDatasetsDirs(datasets)
    for blurSize in blurSizes:
        datasetsGtAndPreds = LoadImages(datasetsDirs, predictors, True, blurSize)
        # gets GT and predictions ^
        datasetsPredictionScores = CalculatePredictionScores(datasetsGtAndPreds, datasetsDirs, predictors)

    # ViewLoadedImages(datasetsGtAndPreds)

def CreateAnalysisFileName(datasetsNames, predictorsNames, analysisDir):

    modifieddatasetsNames = [name.replace(' ', '-') if ' ' in name else name for name in datasetsNames]
    datasetsCombinedString = '_'.join(modifieddatasetsNames) + '-datasets'
    predictorsCombinedString = '_'.join(predictorsNames) + '-methods'
    analysisFilePath = analysisDir / (datasetsCombinedString + '-' + predictorsCombinedString + '.xlsx')

    return analysisFilePath

def OrganoTrackVsOrganoIDvsFarhan():
    # To compare fairly with OrganoSeg, you'll need to remove all border objects from all other images.
    datasets = ['Bladder cancer', 'Mouse', 'Salivary ACC', 'Colon', 'Lung', 'PDAC']
    predictors = ['OrganoTrack', 'OrganoID', 'Farhan1', 'Farhan2']
    analysisDir = Path('/home/franz/Documents/mep/report/results/segmentation-analysis')

    analysisFilePath = CreateAnalysisFileName(datasets, predictors, analysisDir)

    analysisExists = os.path.exists(analysisFilePath)

    if not analysisExists:
        datasetsDirs = GetDatasetsDirs(datasets)
        datasetsGtAndPreds = LoadImages(datasetsDirs, predictors, extraOrganoTrackBlur=False, blurSize=3)
        datasetsPredictionScores = CalculatePredictionScores(datasetsGtAndPreds, datasetsDirs, predictors)
        ExportPredictionScores(datasetsPredictionScores, analysisFilePath, predictors)
    else:
        datasetsPredictionScores = LoadPredictionScoreAnalysis(analysisFilePath)

    PlotPredictionAccuracies(datasetsPredictionScores, predictors)


if __name__ == '__main__':
    OrganoTrackVsOrganoIDvsFarhan()


