import matplotlib.ticker

from OrganoTrack.Importing import ReadImages
from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Displaying import ExportImageWithContours
from OrganoTrack.Filtering import RemoveBoundaryObjects, FilterByFeature
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pathlib import Path
import os
import math
import pandas as pd

# Directories
imagesDir = Path('/home/franz/Documents/mep/report/results/application-tosedostat/input')
exportDir = Path('/home/franz/Documents/mep/report/results/application-tosedostat/output')

predictionDir = exportDir / 'OrganoTrack-segmented'
images, imagesPaths = ReadImages(imagesDir)

if not os.path.exists(predictionDir):
    extraBlur = False
    blurSize = 3
    displaySegSteps = False
    segParams = [0.5, 250, 150, extraBlur, blurSize, displaySegSteps]
    saveSegParams = [True, exportDir, imagesPaths]
    imagesInAnalysis = SegmentWithOrganoSegPy(images, segParams, saveSegParams)
else:
    imagesInAnalysis, imagesPaths = ReadImages(predictionDir)

# DisplayImages('pre', imagesInAnalysis, 0.25)

imagesInAnalysis = RemoveBoundaryObjects(imagesInAnalysis)

# DisplayImages('post', imagesInAnalysis, 0.25)
# cv.waitKey(0)


# Export pre-filtering overlays
overlayExportPath = exportDir / 'preFilteringSegmentationOverlay'
if not os.path.exists(overlayExportPath):
    os.mkdir(overlayExportPath)
for i, (original, prediction) in enumerate(zip(images, imagesInAnalysis)):
    overlay = ExportImageWithContours(original, prediction)
    cv.imwrite(str(overlayExportPath / imagesPaths[i].name), overlay)

# Pre filtering data
preFilteringSegmentedObjectDFs = []
for image in imagesInAnalysis:

# > Find all the object contours in the binary image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours
    #   a list of Numpy arrays. Each array has the (x,y) oordinates of points that make up a contour

    # cv.RETR_EXTERNAL
    #   If there are objects within an object (e.g. a hole in a binary object), cv.RETR_EXTERNAL returns only the
    #   outer contour (the binary object) and not the inner (the hole) contour.

    # cv.CHAIN_APPROX_SIMPLE
    #   A line can be represented as all the points that makeit, or by the two end point. cv.CHAIN_APPROX_SIMPLE
    #   only returns the two endpoints of the line, saving memory.

    # > For each contour found, get
    areas = [cv.contourArea(contour) for contour in contours]  # in pixels^2?
    # plotHistogram(areas, numBins=20, title='Areas', xlabel='Object area', ylabel='Count')

    perimeters = [cv.arcLength(contour, closed=True) for contour in contours]  # in pixels?
    # closed
    #   as the objects are binary objects, they have closed curves/contours
    # plotHistogram(perimeters, numBins=20, title='Perimeters', xlabel='Object perimeter', ylabel='Count')

    circularities = [4 * math.pi * areas[i] / perimeters[i] ** 2 for i in range(len(areas))]  # dimensionless
    # plotHistogram(circularities, numBins=20, title='Circularities', xlabel='Object circularity', ylabel='Count')

    # > Convert feature data into dataframes
    dictOfObjectFeatures = {'Contour': contours, 'Area': areas, 'Perimeter': perimeters,
                            'Circularity': circularities}
    allObjectFeatures = pd.DataFrame(dictOfObjectFeatures, index=range(1, len(contours) + 1))
    # index goes from 1 to the total number of objects
    preFilteringSegmentedObjectDFs.append(allObjectFeatures)

# Filter by circularity and store images
postFilteringImagesInAnalysis = FilterByFeature(imagesInAnalysis, 'roundness', 0.4)

postFilteringOverlayExportPath = exportDir / 'postFilteringSegmentationOverlay'
if not os.path.exists(postFilteringOverlayExportPath):
    os.mkdir(postFilteringOverlayExportPath)
for i, (original, prediction) in enumerate(zip(images, postFilteringImagesInAnalysis)):
    overlay = ExportImageWithContours(original, prediction)
    cv.imwrite(str(postFilteringOverlayExportPath / imagesPaths[i].name), overlay)




# Post filtering data
segmentedObjectDFs = []
for image in postFilteringImagesInAnalysis:

# > Find all the object contours in the binary image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # contours
    #   a list of Numpy arrays. Each array has the (x,y) oordinates of points that make up a contour

    # cv.RETR_EXTERNAL
    #   If there are objects within an object (e.g. a hole in a binary object), cv.RETR_EXTERNAL returns only the
    #   outer contour (the binary object) and not the inner (the hole) contour.

    # cv.CHAIN_APPROX_SIMPLE
    #   A line can be represented as all the points that makeit, or by the two end point. cv.CHAIN_APPROX_SIMPLE
    #   only returns the two endpoints of the line, saving memory.

    # > For each contour found, get
    areas = [cv.contourArea(contour) for contour in contours]  # in pixels^2?
    # plotHistogram(areas, numBins=20, title='Areas', xlabel='Object area', ylabel='Count')

    perimeters = [cv.arcLength(contour, closed=True) for contour in contours]  # in pixels?
    # closed
    #   as the objects are binary objects, they have closed curves/contours
    # plotHistogram(perimeters, numBins=20, title='Perimeters', xlabel='Object perimeter', ylabel='Count')

    circularities = [4 * math.pi * areas[i] / perimeters[i] ** 2 for i in range(len(areas))]  # dimensionless
    # plotHistogram(circularities, numBins=20, title='Circularities', xlabel='Object circularity', ylabel='Count')

    # > Convert feature data into dataframes
    dictOfObjectFeatures = {'Contour': contours, 'Area': areas, 'Perimeter': perimeters,
                            'Circularity': circularities}
    allObjectFeatures = pd.DataFrame(dictOfObjectFeatures, index=range(1, len(contours) + 1))
    # index goes from 1 to the total number of objects
    segmentedObjectDFs.append(allObjectFeatures)


imageCondition = ['Cis 5+\nTos .01',
                  'Ctrl',
                  'Cis 5 ',
                  'Cis 2',
                  'Cis 2+\nTos .01',
                  'Tos .01']
#
newOrder = [6, 1, 5, 3, 4, 2]
sortedImageCondition = [x for _, x in sorted(zip(newOrder, imageCondition))]
sortedAnalysedImages = [x for _, x in sorted(zip(newOrder, postFilteringImagesInAnalysis))]
sortedObjectFeatures = [x for _, x in sorted(zip(newOrder, segmentedObjectDFs))]
sortedPreFiltObjectFeatures = [x for _, x in sorted(zip(newOrder, preFilteringSegmentedObjectDFs))]

# Getting area values into a list of lists
areaMeasurements, xs, circularityMeasurements = [], [], []
for i in range(len(sortedObjectFeatures)):
    circularityMeasurements.append(sortedObjectFeatures[i]['Circularity'].tolist())
    areaMeasurements.append(sortedObjectFeatures[i]['Area'].tolist())
    xs.append(np.random.normal(i + 1, 0.04, sortedObjectFeatures[i]['Area'].values.shape[0]))

circMeasurePreFilt, jitters = [], []
for j in range(len(sortedPreFiltObjectFeatures)):
    circMeasurePreFilt.append(sortedPreFiltObjectFeatures[j]['Circularity'].tolist())
    jitters.append(np.random.normal(j + 1, 0.04, sortedPreFiltObjectFeatures[j]['Area'].values.shape[0]))
#
# Converting list of lists into array of arrays for summing
y = np.array([np.array(xi) for xi in areaMeasurements], dtype=object)
areaSums = [np.sum(i) for i in y]
plt.rcParams.update({'font.size': 15})

# Object areas per condition
fig, ax = plt.subplots()
ax.boxplot(areaMeasurements, labels=sortedImageCondition, showfliers=False)
ax.set_ylabel('Area (px)')
ax.set_xlabel(r'Condition concentration ($\mu$M)')
ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# ax.set_title('Object area per condition')
palette = ['b', 'g', 'r', 'c', 'm', 'k']
for x, val, c in zip(xs, areaMeasurements, palette):
    ax.scatter(x, val, alpha=0.4, color=c)
plt.tight_layout()
plt.savefig(str(exportDir / 'object-area-per-condition.png'), dpi=300)
fig.show()

#
# # Total area per condition
# fig1, ax1 = plt.subplots()
# ax1.bar(sortedImageCondition, areaSums)
# ax1.set_ylabel('Area (px)')
# ax1.set_xlabel(r'Condition concentration ($\mu$M)')
# # ax.ticklabel_format(axis='y', style='sci', useMathText=True)
# ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# # ax1.set_title('Total object area per condition')
# plt.tight_layout()
# plt.savefig(str(exportDir / 'totalObjectAreaPerCondition.png'), dpi=300)
# fig1.show()
#
# # Total count per condition
# objectCount = np.array([len(i) for i in areaMeasurements])
# fig2, ax2 = plt.subplots()
# ax2.bar(sortedImageCondition, objectCount)
# ax2bars = ax.bar(sortedImageCondition, objectCount)
# ax2.bar_label(ax2bars)
# ax2.set_ylabel('Object count')
# ax2.set_ylim([0, 25])
# ax2.set_xlabel(r'Condition concentration ($\mu$M)')
# ax.set_xticklabels(objectCount.astype(int))
# # ax2.set_title('Total object count per condition')
# plt.tight_layout()
# plt.savefig(str(exportDir / 'totalObjectCountPerCondition.png'), dpi=300)
# fig2.show()
# #
# # Circularity per condition post filtering
# fig3, ax3 = plt.subplots()
# ax3.boxplot(circularityMeasurements, labels=sortedImageCondition, showfliers=False)
# ax3.set_ylabel('Circularity (a.u.)')
# ax3.set_ylim([0, 1])
# ax3.set_xlabel(r'Condition concentration ($\mu$M)')
# # ax3.set_title('Object circularity per condition')
# palette = ['b', 'g', 'r', 'c', 'm', 'k']
# for x, val, c in zip(xs, circularityMeasurements, palette):
#     ax3.scatter(x, val, alpha=0.4, color=c)
# plt.tight_layout()
# plt.savefig(str(exportDir / 'ObjectCircPerCondition.png'), dpi=300)
# fig3.show()
#
# # Circularity per condition post filtering
# fig4, ax4 = plt.subplots()
# ax4.boxplot(circMeasurePreFilt, labels=sortedImageCondition, showfliers=False)
# ax4.set_ylabel('Circularity (a.u.)')
# ax4.set_ylim([0, 1])
# ax4.set_xlabel(r'Condition concentration ($\mu$M)')
# # ax4.set_title('Object circularity per condition before filtering')
# palette = ['b', 'g', 'r', 'c', 'm', 'k']
# for jitter, val, c in zip(jitters, circMeasurePreFilt, palette):
#     ax4.scatter(jitter, val, alpha=0.4, color=c)
# plt.tight_layout()
# plt.savefig(str(exportDir / 'ObjectCircPerConditionPreFilt.png'), dpi=300)
# fig4.show()
#
# # Average area per condition
# averageAreaMeasurements = [np.average(measurements) for measurements in areaMeasurements]
# stdDevAreaMeasurements = [stdev(measurements) if len(measurements) > 1 else 0 for measurements in areaMeasurements]
#
# fig5, ax5 = plt.subplots()
# ax5.bar(sortedImageCondition, averageAreaMeasurements, yerr=stdDevAreaMeasurements, capsize=5, color='royalblue')
# ax5.set_ylabel('Area (px)')
# ax5.set_xlabel(r'Condition concentration ($\mu$M)')
# # ax.ticklabel_format(axis='y', style='sci', useMathText=True)
# ax5.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# # ax5.set_title('Average object area per condition')
# plt.tight_layout()
# plt.savefig(str(exportDir / 'AvgObjAreaPerCondition.png'), dpi=300)
# fig5.show()

filePath = Path('/home/franz/Documents/mep/data/experiments/2023-02-24-Cis-Tos-dataset-mathijs/220123_Tosedostat&Cisplatin.ods')
reseedingData = pd.read_excel(filePath, sheet_name='Reseeding (Azh4+5)', header=None)
avg = reseedingData.iloc[265,4:10].to_numpy(dtype=np.float64)
stdDev = reseedingData.iloc[266,4:10].to_numpy(dtype=np.float64)
fig6, ax6 = plt.subplots()
ax6.bar(sortedImageCondition, avg, yerr=stdDev, capsize=5, color='salmon')
ax6.set_ylabel('Cell viability (% of control)')
ax6.set_xlabel(r'Condition concentration ($\mu$M)')
# ax.ticklabel_format(axis='y', style='sci', useMathText=True)
ax6.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# ax6.set_title('Cell viability per condition')
plt.tight_layout()
plt.savefig(str(exportDir / 'CellViabilityPerCondition.png'), dpi=300)
fig6.show()
print('h')

# call organotrack to analyse data
    # receive filtered images and included object DFs

# plot data
