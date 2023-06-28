from OrganoTrack.Importing import ReadImages, load_plate_layout
from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Exporting import MeasureAndExport
from OrganoTrack.Filtering import FilterByFeature, RemoveBoundaryObjects

from OrganoTrack.Tracking import track, SaveImages, MakeDirectory, stack
from OrganoTrack.ImageHandling import DrawRegionsOnImages
from OrganoTrack.Displaying import Mask
from pathlib import Path
import numpy as np
from PIL import Image




def load_segmentations(segmentOrgs, importPath, identifiers, segParams, saveSegParams, segmentedImagesPath):

    if segmentOrgs:
        inputImages, imageNames = ReadImages(importPath, identifiers)
        for well, wellFieldImages in inputImages.items():
            for field, fieldTimeImages in wellFieldImages.items():
                inputImages[well][field] = SegmentWithOrganoSegPy(fieldTimeImages, segParams, saveSegParams)

    else:  # Load saved segmentations
        inputImages, imageNames = ReadImages(segmentedImagesPath, identifiers)

    return inputImages


def filter_boundary_objects(images_in_analysis):
    for well, wellFieldImages in images_in_analysis.items():
        for field, fieldTimeImages in wellFieldImages.items():
            images_in_analysis[well][field] = RemoveBoundaryObjects(fieldTimeImages)
    return images_in_analysis


def filter_organoids_by_morphology(filter_criteria, images_in_analysis):

    filterOpsIndeces = [filter_criteria.index(filterOp) for filterOp in filter_criteria if (type(filterOp) is str)]
    morphologicalPropertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                                  'eccentricity', 'equivalent_diameter_area', 'euler_number',
                                  'extent', 'feret_diameter_max', 'orientation',
                                  'perimeter', 'perimeter_crofton', 'roundness', 'solidity']
    # the index of the strings
    for i in filterOpsIndeces:  # for each filterOp
        if filter_criteria[i] in morphologicalPropertyNames:
            for well, wellFieldImages in images_in_analysis.items():
                for field, fieldTimeImages in wellFieldImages.items():
                    images_in_analysis[well][field] = FilterByFeature(fieldTimeImages, filter_criteria[i],
                                                                      filter_criteria[i + 1])
    return images_in_analysis


def track_orgs(images_in_analysis):

    for well, wellFieldImages in images_in_analysis.items():
        highestTrackIDnum = 0
        sortedFields = sorted(wellFieldImages, key=int)

        for field in sortedFields:
            print(f'Tracking well {well}, field = {field}')
            timeLapseSet = images_in_analysis[well][field]
            trackedTimeLapseSet = track(timeLapseSet)

            trackedTimeLapseSet2 = np.where(trackedTimeLapseSet != 0, trackedTimeLapseSet + highestTrackIDnum,
                                            trackedTimeLapseSet)
            highestTrackIDnum = np.max(trackedTimeLapseSet2)

            images_in_analysis[well][field] = trackedTimeLapseSet2
    return images_in_analysis

def export_org_measurements(exportPath, morphPropsToMeasure, images_in_analysis, plate_layout):
    measuresFileName = 'trackedMeasures.xlsx'
    trackedMeasurementsPerWell = MeasureAndExport(exportPath / measuresFileName, morphPropsToMeasure,
                                                  images_in_analysis, plate_layout)
    return trackedMeasurementsPerWell


def export_tracked_image_overlays():
    pass
    # if overlayTrack:
    #     # Create masks
    #     maskedImages = [Mask(ori, pred) for ori, pred in zip(inputImages, binaryTrackedList)]
    #     # maskedImages = [ExportImageWithContours(ori, pred) for ori, pred in zip(inputImages, binaryTrackedList)]
    #
    #     # Regather timelapse masked images
    #     maskedImages = [maskedImages[i * timePoints:(i + 1) * timePoints]
    #                      for i in range((len(maskedImages) + timePoints - 1) // timePoints )]
    #     # maskedImages = [[maskedImages[0], maskedImages[1], maskedImages[2], maskedImages[3]]]
    #     # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]
    #
    #     # Convert images to PIL format to use OrganoID functions
    #
    #     maskedImagesPIL = [[Image.fromarray(img) for img in maskedSet] for maskedSet in maskedImages]
    #     # lsit of list of PIL images
    #     print('h')
    #
    #     # Storage function
    #     def Output(name: str, data, count):
    #         if exportPath is not None:
    #             MakeDirectory(exportPath)
    #             SaveImages(data, "_" + name.lower(), maskedImagesPIL[count], exportPath, imageNamesCollected[count])  # pilImages is a list of PIL Image.Image objects
    #             # imagePathsForExport = [imagePaths + '/' + imageName for imageName in rawImageNames]
    #
    #     # Create an overlay and output it
    #     for i in range(len(trackedSets)):  # for each timelapse set
    #         overlayImages = DrawRegionsOnImages(trackedSets[i], stack(maskedImages[i]), (255, 255, 255), 50, (0, 255, 0))  # np.array, likely 3D
    #         Output('Overlay', overlayImages, i)
    #         print('tracking')

def run_OrganoTrack(import_path: Path, identifiers: dict, export_path: Path,
                    segment_organoids: bool, seg_parameters: list, params_to_save_segs: list, path_to_seg_imgs: Path,
                    remove_boundary_objects: bool, filter_by_morphology: bool, filter_criteria: list,
                    track_organoids: bool,
                    export_organoid_measurements: bool, morph_props_to_measure: list):

    images_in_analysis = load_segmentations(segment_organoids, import_path, identifiers, seg_parameters, params_to_save_segs, path_to_seg_imgs)
    plate_layout = load_plate_layout(import_path)

    if remove_boundary_objects:
        images_in_analysis = filter_boundary_objects(images_in_analysis)

    if filter_by_morphology:
        images_in_analysis = filter_organoids_by_morphology(filter_criteria, images_in_analysis)

    if track_organoids:
        images_in_analysis = track_orgs(images_in_analysis)

    if export_organoid_measurements:
        tracked_org_measurements_per_well = export_org_measurements(export_path, morph_props_to_measure, images_in_analysis, plate_layout)


if __name__ == '__main__':
    pass