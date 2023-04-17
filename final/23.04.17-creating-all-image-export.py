from OrganoTrack import RunOrganoTrack
from pathlib import Path
import os

# Import
dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-all-cis-data/input')
exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-all-cis-data/output')

# Check results
checkResults = False

# Segmentation
saveSegmentations = False
segment = False
segmentedPaths = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-all-cis-data/output/segmented')

# Selection of organoids
filterOrganoids = True
filterBy = ['area', 150, 'roundness', 0.5]  # minimum filter

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
                       saveSegmentations, segment, segmentedPaths,
                       filterOrganoids, filterBy,
                       trackOrgs, timePoints, overlayTracking,
                       exportOrgMeasures, morphPropertiesToMeasure,
                       handlePlotting, loadExportedData, exportedDataPath)

