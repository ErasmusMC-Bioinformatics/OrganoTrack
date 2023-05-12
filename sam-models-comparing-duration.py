import cv2 as cv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
from pathlib import Path
from final.Importing import ReadImages
import time
import pandas as pd


def MeasureSegmentationDurationOfModels():
    imagesPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/sam-measuring-model-durations/original')
    images, imagesNames = ReadImages(imagesPath)
    images = [cv.cvtColor(img, cv.COLOR_GRAY2RGB)for img in images]

    exportDir = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/sam-measuring-model-durations/export')

    modelCheckpoints = {'vit_b': 'sam_vit_b_01ec64.pth',
                        'vit_h': 'sam_vit_h_4b8939.pth',
                        'vit_l': 'sam_vit_l_0b3195.pth'}

    modelsSegDurations = {'vit_b': None,
                          'vit_h': None,
                          'vit_l': None}


    for model in list(modelCheckpoints.keys()):
        print(f'Model {model} started.')
        modelDuration = []
        for i, image in enumerate(images):
            print(f'Image {i + 1} started.')
            tic = time.time()
            mask = SegmentBySAM(image, model, modelCheckpoints[model])
            toc = time.time() - tic
            modelDuration.append(toc)
            print(f'Segmentation duration: {toc}')
            ExportBinaryMask(mask, model, exportDir, imagesNames[i])
            print(f'Image {i + 1} finished.')
        modelsSegDurations[model] = modelDuration
        print(f'Model {model} finished.')
    modelsSegDurations['imageNames'] = [imagePath.name for imagePath in imagesNames]
    ExportDurations(modelsSegDurations, exportDir)

    print('f')


def SegmentBySAM(image, model, modelCheckpoint):
    sam = sam_model_registry[model](checkpoint=modelCheckpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image)


def ExportBinaryMask(mask, model, exportDir, fileName):
    booleanImageRegions = [mask[x]['segmentation'] for x in range(len(mask))]
    sumOfImageRegions = sum(booleanImageRegions)
    summedBooleanToBinaryMapping = np.zeros(np.max(sumOfImageRegions) + 1)
    if model == 'vit_h':
        summedBooleanToBinaryMapping[1:] = 1
    else:
        summedBooleanToBinaryMapping[2:] = 1
    binarySamMask = summedBooleanToBinaryMapping[sumOfImageRegions]
    cv.imwrite(str(exportDir / fileName.name), binarySamMask)


def ExportDurations(modelSegTimes, exportDir):
    modelSegTimesDF = pd.DataFrame.from_dict(modelSegTimes)
    modelSegTimesDF.to_excel(exportDir / 'SAM_model_segmentation_durations.xlsx')

def Plot():
    pass

if __name__ == '__main__':
    MeasureSegmentationDurationOfModels()