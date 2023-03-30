import cv2 as cv
from functions import mask
import skimage.measure
from stitch import readImages
import numpy as np
from code.final.Tracking import Track, Inverse, Overlap
from code.final.ImageHandling import DrawRegionsOnImages
from pathlib import Path
from PIL import Image


def Label(image):
    labeled = skimage.measure.label(image)
    return labeled

def stack(images):
    return np.stack(images[:])

def SaveImages(data, suffix, pilImages, outputPath, fileNames):
    from code.final.ImageHandling import ConvertImagesToPILImageStacks, SavePILImageStack
    stacks = ConvertImagesToPILImageStacks(data, pilImages)
    # stacks is a List of 3D np arrays

    for stack, fileName in zip(stacks, fileNames):
        p = Path(fileName)
        SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))

def MakeDirectory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise Exception("Could not find or create directory '" + str(path.absolute()) + "'.")

def track(segmentedTimelapseImages):
    '''
    :param segmentedTimelapseImages: a list of images (numpy arrays) that belong to one timelapse set
    :return: a 3D numpy array (stacked 2D arrays) with tracked objects labelled with their unique ID number
    '''

    # Label images
    labelledImages = [Label(image) for image in segmentedTimelapseImages]  # a List of 2D arrays with objects labelled with a number


    # Stack the labelled images into a 3d array
    timelapseStack = stack(labelledImages)  # a 3D array of 2D images


    # Track objects across timelapse images
    trackedTimelapseStack = Track(timelapseStack, 1, Inverse(Overlap))

    return trackedTimelapseStack


    # stacks is a list of sets of timelapse images
    # stacks has len = 1 if batch False or =4 if batch True, element type = np.ndarray of shape 4 x 512 x 512
    # for stack in stacks:

    # a stack is a single set of timelapse images
    # stack is np.ndarray, len = 4, shape = 4 x 512 x 512

        # jump to the next set of stacks in cleanedImages


if __name__ == '__main__':

    displayScale = 0.5

    # Load brightfield images
    dataDir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/only-control-and-cis-images-field-1'
    imageNames = ['r02c02f01p01-ch1sk1fk1fl1.tiff',
                  'r02c02f01p01-ch1sk2fk1fl1.tiff',
                  'r02c02f01p01-ch1sk3fk1fl1.tiff',
                  'r02c02f01p01-ch1sk4fk1fl1.tiff']

    imagePaths = [dataDir + '/' + imageName for imageName in imageNames]

    inputImages = [cv.imread(imagePath, cv.IMREAD_GRAYSCALE) for imagePath in imagePaths]


    # Load segmentation images
    dataDir2 = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/only-control-and-cis-images-field-1/segmented-06.03.2023-12_59_33'
    segmentedImages, imageNames = readImages(dataDir2)

    # Track objects over time
    trackedTimelapseStack = track(segmentedImages)


    # Create masks
    maskedImages = [mask(ori, pred) for ori, pred in zip(inputImages, segmentedImages)]


    # Convert images to PIL format to use OrganoID functions
    outputPath = Path('/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/only-control-and-cis-images-field-1/segmented-16.03.2023-15_41_00')
    maskedImagesPIL = [Image.fromarray(img) for img in maskedImages]


    # Storage function
    def Output(name: str, data):
        if outputPath is not None:
            MakeDirectory(outputPath)
            SaveImages(data, "_" + name.lower(), maskedImagesPIL, outputPath, imagePaths)  # pilImages is a list of PIL Image.Image objects


    # Create an overlay and output it
    overlayImages = DrawRegionsOnImages(trackedTimelapseStack, stack(maskedImages), (255, 255, 255), 16, (0, 255, 0))  # np.array, likely 3D
    Output('Overlay', overlayImages)

