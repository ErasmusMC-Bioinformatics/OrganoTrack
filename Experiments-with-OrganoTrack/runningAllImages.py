from OrganoTrack.OrganoTrack import RunOrganoTrack
from pathlib import Path
import os

# Import
# dataPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/partOfDataset/input')
# dataPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/cisplatinDataset/input')
# exportPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/partOfDataset/output')
# exportPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/cisplatinDataset/output')

# Masks
# dataPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/masks/input')
# exportPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/masks/output')

# All Masks
dataPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/masks/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4')
exportPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/masks/output')

# Check results
checkResults = False

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
overlayTracking = False

# Measure organoids
exportOrgMeasures = False
numberOfWellFields = 25
morphPropertiesToMeasure = ['area', 'roundness', 'eccentricity', 'solidity']

# Handle plots
handlePlotting = False
loadExportedData = False
exportedDataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/trackedMeasures.xlsx')


RunOrganoTrack(dataPath, exportPath, checkResults,
               segment, segParams, saveSegParams, segmentedPaths,
               filterBoundary, filterOrganoids, filterBy,
               trackOrgs, timePoints, overlayTracking,
               exportOrgMeasures, numberOfWellFields, morphPropertiesToMeasure,
               handlePlotting, loadExportedData, exportedDataPath)

