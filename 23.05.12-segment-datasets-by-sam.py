from pathlib import Path
from final.Importing import ReadImages
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import cv2 as cv
from SamMeasureModelDuration import ExportBinaryMask, SegmentBySAM, ExportDurations
import time

def SegmentDatasetsbySam():
    datasetDirs = {'EMC': Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/sam-segment-part-cis-data/input'),
                   'OrganoID Original': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/original'),
                   'OrganoID Mouse': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Original')}

    exportDirs = {'EMC': Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/sam-segment-part-cis-data/output'),
                  'OrganoID Original': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/export'),
                  'OrganoID Mouse': Path('/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Export')}

    model = 'vit_b'
    modelCheckpoint = 'sam_vit_b_01ec64.pth'

    for dataset in list(datasetDirs.keys()):
        print(f'Dataset {dataset} started.')
        images, imagePaths = ReadImages(datasetDirs[dataset])
        images = [cv.cvtColor(img, cv.COLOR_GRAY2RGB) for img in images]
        datasetDurations = dict()
        datasetTimes = []

        for i, image in enumerate(images):
            print(f'Image {i + 1} started.')
            tic = time.time()
            samImage = SegmentBySAM(image, model, modelCheckpoint)
            toc = time.time() - tic
            datasetTimes.append(toc)
            ExportBinaryMask(samImage, model, exportDirs[dataset] / 'SAM-segmented', imagePaths[i])
            print(f'Image {i + 1} finished.')

        datasetDurations['durations'] = datasetTimes
        datasetDurations['imageNames'] = [imagePath.name for imagePath in imagePaths]
        ExportDurations(datasetDurations, exportDirs[dataset])
        print(f'Dataset {dataset} finished.')

if __name__ == '__main__':
    SegmentDatasetsbySam()