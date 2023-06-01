from OrganoTrack.OrganoTrack import RunOrganoTrack
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

# Selection of organoids
filterBoundary = False
filterOrganoids = False
filterBy = ['area', 450, 'roundness', 0.5]  # minimum filter


# Track organoids
trackOrgs = False
timePoints = 4
overlayTracking = True

# Measure organoids
exportOrgMeasures = False
numberOfWellFields = 25
morphPropertiesToMeasure = ['area', 'roundness', 'eccentricity', 'solidity']

# Handle plots
handlePlotting = True
loadExportedData = True
exportedDataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/trackedMeasures.xlsx')


RunOrganoTrack(dataPath, exportPath, checkResults,
               segment, segParams, saveSegParams, segmentedPaths,
               filterBoundary, filterOrganoids, filterBy,
               trackOrgs, timePoints, overlayTracking,
               exportOrgMeasures, numberOfWellFields, morphPropertiesToMeasure,
               handlePlotting, loadExportedData, exportedDataPath)

