from OrganoTrack.OrganoTrack import run_OrganoTrack
from pathlib import Path
import os
from datetime import datetime


def testing_OrganoTrack_full():
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

    run_OrganoTrack(dataPath, identifiers, exportPath, segment, segParams, saveSegParams, segmentedPaths,
                    filterBoundary, filterOrganoids, filterBy, trackOrgs, timePoints, overlayTracking)


def creating_all_image_export():
    # Import
    dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/input')
    exportPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output')

    # Check results
    checkResults = False

    # Segmentation
    saveSegmentations = False
    segment = False
    segmentedPaths = Path(
        '/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/segmented')

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
    exportedDataPath = Path(
        '/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/trackedMeasures.xlsx')

    run_OrganoTrack(dataPath, exportPath, checkResults, saveSegmentations, segment, segmentedPaths, filterOrganoids,
                    filterBy, trackOrgs, timePoints, overlayTracking, exportOrgMeasures, numFields)

def cleaning_OrganoTrack():
    experimentPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full')
    automateExecution = True
    identifiers = {'row': 'r',
                   'column': 'c',
                   'field': 'f',
                   'position': 'p',
                   'timePoint': 'sk'}

    run_OrganoTrack(experimentPath, identifiers, automateExecution)


def harmony_segmented_all_cis_data():
    # Import
    import_path = Path(
        '/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/import')
    identifiers = {'row': 'R',
                   'column': 'C',
                   'field': 'F',
                   'position': 'P',
                   'timePoint': 'T'}
    export_path = Path(
        '/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/export')

    # Segmentation
    segment = False
    extra_blur = False
    blur_size = 3
    seg_parameters = [0.5, 250, 150, extra_blur, blur_size]
    params_to_save_segmentations = [False, export_path]
    path_to_segmented_imgs = export_path / 'Harmony-segmented'

    # Selection of organoids
    remove_boundary_objects = True
    filter_organoids_by_morphology = True
    morph_filter_criteria = ['area', 150]

    # Track organoids
    track_organoids = True

    # Measure organoids
    export_tracked_organoid_measurements = True
    morph_properties_to_measure = ['area', 'roundness', 'eccentricity', 'solidity']

    run_OrganoTrack(import_path, identifiers, export_path, segment, seg_parameters, params_to_save_segmentations,
                    path_to_segmented_imgs, remove_boundary_objects, filter_organoids_by_morphology,
                    morph_filter_criteria, track_organoids, export_tracked_organoid_measurements,
                    morph_properties_to_measure)

def running_all_images():
    # All Masks
    dataPath = Path(
        'C:/Users/franz/OneDrive/Documents/work-Franz/mep/masks/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4')
    exportPath = Path('C:/Users/franz/OneDrive/Documents/work-Franz/mep/masks/output')

    # Check results
    checkResults = False

    # Segmentation
    segment = False
    extraBlur = False
    blurSize = 3
    segParams = [0.5, 250, 150, extraBlur, 3]
    saveSegParams = [True, exportPath]
    segmentedPaths = Path(
        '/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/segmented')

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
    exportedDataPath = Path(
        '/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/output/trackedMeasures.xlsx')

    run_OrganoTrack(dataPath, exportPath, checkResults, segment, segParams, saveSegParams, segmentedPaths,
                    filterBoundary, filterOrganoids, filterBy, trackOrgs, timePoints, overlayTracking)


if __name__ == '__main__':
    harmony_segmented_all_cis_data()
