from OrganoTrack import RunOrganoTrack
from pathlib import Path
import os

# Import
dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/input')
exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output')

# Check results
checkResults = True

# Segmentation
segment = False
extraBlur = False
blurSize = 3
segParams = [0.5, 250, 150, extraBlur, 3]
saveSegParams = [True, exportPath]
segmentedPaths = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/segmented')
# fudgeFactor, maxWindowSize, minObjectSize, extraBlur, blurSize
# def RunOrganoTrack(importPath = None, exportPath = None, livePreview = False,
#                    segmentOrgs = True, segParams = None, saveSegParams = None, segmentedImagesPath = None,
#                    filterOrgs = False, filterCriteria = None, filterBoundary=False,
#                    trackOrgs = False, timePoints = None, overlayTrack = False,
#                    exportOrgMeasures = False, numberOfWellFields = None, morphPropsToMeasure = None,
#                    plotData = False, loadDataForPlotting = False, pathDataForPlotting = None):
# Selection of organoids
filterBoundary = True
filterOrganoids = True
filterBy = ['area', 450, 'roundness', 0.5]  # minimum filter


# Track organoids
trackOrgs = True
timePoints = 4
overlayTracking = False

# Measure organoids
exportOrgMeasures = True
morphPropertiesToMeasure = ['area', 'roundness', 'eccentricity', 'solidity']

# Handle plots
handlePlotting = False
loadExportedData = False
exportedDataPath = Path('G:/My Drive/mep/image-analysis-pipelines/OrganoTrack/trackedMeasures.xlsx')


RunOrganoTrack(dataPath, exportPath, checkResults,
               segment, segParams, saveSegParams, segmentedPaths,
               filterBoundary, filterOrganoids, filterBy,
               trackOrgs, timePoints, overlayTracking,
               exportOrgMeasures, morphPropertiesToMeasure,
               handlePlotting, loadExportedData, exportedDataPath)


# # RunOrganoTrack(plotData=handlePlotting, loadDataForPlotting=loadExportedData, pathDataForPlotting=exportedDataPath)
#
# import2 = True
# importPath2 = Path('C:/Users/franz/Documents/OrganoTrackl/stackofd1r1t0/input')
# exportPath2 = Path('C:/Users/franz/Documents/OrganoTrackl/stackofd1r1t0/output')
#
# # Check results
# checkResults2 = False
#
# # Segmentation
# saveSegmentations2 = False
# segment2 = False
# segmentedPaths2 = Path('C:/Users/franz/Documents/OrganoTrackl/stackofd1r1t0/output/segmented')
#
# # Selection of organoids
# filterOrganoids2 = True
# filterBy2 = ['area', 450, 'roundness', 0.5]  # minimum filter
#
# # Track organoids
# trackOrgs2 = True
# timePoints2 = 4
# overlayTracking2 = True
#
#
#
# RunOrganoTrack(import2, importPath2, exportPath2, checkResults2,
#                        saveSegmentations2, segment2, segmentedPaths2,
#                        filterOrganoids2, filterBy2,
#                        trackOrgs2, timePoints2, overlayTracking2)
#                        # exportOrgMeasures, morphPropertiesToMeasure,
#                        # handlePlotting, loadExportedData, exportedDataPath)
