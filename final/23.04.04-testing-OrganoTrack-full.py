from OrganoTrack import RunOrganoTrack
from pathlib import Path
import os

# Import
dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/input')
exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output')

# Check results
checkResults = True

# Segmentation
saveSegmentations = True
segment = False
segmentedPaths = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/segmented')

# Selection of organoids
filterOrganoids = True
filterBy = ['area', 450, 'roundness', 0.5]  # minimum filter

# Track organoids
trackOrgs = True
timePoints = 4
overlayTracking = False

# Measure organoids
exportOrgMeasures = True
morphPropertiesToMeasure = ['area', 'roundness', 'eccentricity', 'solidity']


times = RunOrganoTrack(dataPath, exportPath, checkResults,
                       saveSegmentations, segment, segmentedPaths,
                       filterOrganoids, filterBy,
                       trackOrgs, timePoints, overlayTracking,
                       exportOrgMeasures, morphPropertiesToMeasure)
print(times)