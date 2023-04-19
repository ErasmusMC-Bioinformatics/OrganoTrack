from OrganoTrack import RunOrganoTrack
from pathlib import Path
import os

# Import
dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/input')
exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output')

# Check results
checkResults = False

# Segmentation
saveSegmentations = False
segment = False
segmentedPaths = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/segmented')

# Selection of organoids
filterOrganoids = True
filterBy = ['area', 150, 'roundness', 0.5]  # minimum filter

# Track organoids
trackOrgs = True
timePoints = 4
overlayTracking = False

# Export measurements
exportOrgMeasures = True
numFields = 1
morphPropertiesToMeasure = ['area', 'roundness', 'eccentricity', 'solidity']

# Export


# Handle plots
handlePlotting = False
loadExportedData = False
exportedDataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/trackedMeasures.xlsx')


RunOrganoTrack(dataPath, exportPath, checkResults,
               saveSegmentations, segment, segmentedPaths,
               filterOrganoids, filterBy,
               trackOrgs, timePoints, overlayTracking,
               exportOrgMeasures, numFields, morphPropertiesToMeasure,
               handlePlotting, loadExportedData, exportedDataPath)

