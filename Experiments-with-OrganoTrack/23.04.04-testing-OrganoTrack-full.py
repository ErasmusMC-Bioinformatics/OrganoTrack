from OrganoTrack.OrganoTrack import RunOrganoTrack
from pathlib import Path
import os

# Import
dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/import')
identifiers = {'row': 'R',
               'column': 'C',
               'field': 'F',
               'position': 'P',
               'timePoint': 'T'}
exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/export')

# Segmentation
segment = False
extraBlur = False
blurSize = 3
segParams = [0.5, 250, 150, extraBlur, 3]
saveSegParams = [False, exportPath]
segmentedPaths = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/export/segmented')

# Selection of organoids
filterBoundary = True
filterOrganoids = True
filterBy = ['area', 150] #, 'roundness', 0.5]  # minimum filter


# Track organoids
trackOrgs = True
timePoints = 4
overlayTracking = False

# Measure organoids
exportOrgMeasures = True
numberOfWellFields = 25
morphPropertiesToMeasure = ['area', 'roundness', 'eccentricity', 'solidity']

# Handle plots
handlePlotting = True
loadExportedData = True
exportedDataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/trackedMeasures.xlsx')


RunOrganoTrack(dataPath, identifiers, exportPath,
               segment, segParams, saveSegParams, segmentedPaths,
               filterBoundary, filterOrganoids, filterBy,
               trackOrgs, timePoints, overlayTracking,
               exportOrgMeasures, numberOfWellFields, morphPropertiesToMeasure,
               handlePlotting, loadExportedData, exportedDataPath)

