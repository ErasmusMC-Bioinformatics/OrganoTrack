import cv2 as cv
import os
import pandas as pd
from pathlib import Path
# needed to install xlrd


def ReadPlateLayout(inputDataPath):
    '''
    :param plateLayoutDir: directory of plate layout file (.xlsx, .csv, or .tsv). The file must (!) follow directives.
    :return: A List of Lists, with elements as tuples (condition, concentration, concentration unit)
    '''

    print("\nReading plate layout...")

    plateLayoutFile = sorted(os.listdir(inputDataPath))[1]  # the second element should be the 'plate_layout.*' file
    plateLayoutDir = inputDataPath / plateLayoutFile

    extension = plateLayoutDir.suffix

    if extension == '.xlsx':
        plateLayoutInput = pd.read_excel(plateLayoutDir, header=None)

    elif extension == '.csv':
        plateLayoutInput = pd.read_csv(plateLayoutDir, header=None)

    elif extension == '.tsv':
        plateLayoutInput = pd.read_csv(plateLayoutDir, header=None, delimiter='\t')

    # > Get the index (row) number where 'conditions' is
    conditionsLoc = plateLayoutInput.index[plateLayoutInput[0] == 'conditions'].tolist()[0]
    concentrationsLoc = plateLayoutInput.index[plateLayoutInput[0] == 'concentrations'].tolist()[0]

    # > Split the input dataframe into the conditions and concentrations (reset index, dropping the added index column)
    conditions = plateLayoutInput.iloc[conditionsLoc+1:concentrationsLoc-1, :]
    concentrations = plateLayoutInput.iloc[concentrationsLoc+1:, :].reset_index(drop=True)

    concentrations.loc[2:9, 1:] = concentrations.loc[2:9, 1:].applymap(lambda x: round(float(x), 6))

    plateLayout = []

    unit = concentrations.iloc[0][1]
    rowNum = len(conditions.index) - 1
    colNum = len(conditions.columns) - 1

    for i in range(1, rowNum + 1):  # i = 1 to row num
        row = []
        conditionsRow = conditions.iloc[i]
        concentrationsRow = concentrations.iloc[i + 1]

        for j in range(1, colNum + 1):  # j = 1 to col num
            row.append([conditionsRow[j], concentrationsRow[j], unit])
        plateLayout.append(row)

    print("Plate layout read and created.")

    return plateLayout


def Test_ReadPlateLayout():
    dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/input')

    plateDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/buildingPipeline_Input/plate_layout.xlsx'

    plateDirCSV = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/buildingPipeline_Input/plate_layout.csv'

    plateDirTSV = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/buildingPipeline_Input/plate_layout.tsv'
    ReadPlateLayout(dataPath)

def ReadImages(inputDataPath):
    '''
    Read Images requires that the images are within a folder called '/images', that inputDataPath is the parent folder
    of '/images', and that '/images' is teh first item in a list of files in the parent folder (hence line 77 calls
    for the first element)
    '''
    print("\nReading data...")

    imagesFolderName = sorted(os.listdir(inputDataPath))[0]  # the first element should be the 'images' folder
    imagesFolderDir = inputDataPath / imagesFolderName

    # > Get the names and extensions of the image files in the directory
    inputImagesNames = sorted(os.listdir(imagesFolderDir))

    # > Create directory paths for each image file
    inputImagesPaths = [imagesFolderDir / imageName for imageName in inputImagesNames]

    # > Read images
    inputImages = [cv.imread(str(imagePath), cv.IMREAD_GRAYSCALE) for imagePath in inputImagesPaths]

    print("Finished reading data.")

    print("There is/are in total " + str(len(inputImages)) + " image(s).")

    return inputImages, inputImagesPaths

def ReadImages2(inputDataPath):
    '''
    Read Images requires that the images are within a folder called '/images', that inputDataPath is the parent folder
    of '/images', and that '/images' is teh first item in a list of files in the parent folder (hence line 77 calls
    for the first element)
    '''
    print("\nReading data...")
    images = dict()
    imageCount = 0
    imagesFolderName = sorted(os.listdir(inputDataPath))[0]  # the first element should be the 'images' folder
    imagesFolderDir = inputDataPath / imagesFolderName

    # > Get the names and extensions of the image files in the directory
    inputImagesNames = sorted(os.listdir(imagesFolderDir))

    # > Create directory paths for each image file
    inputImagesPaths = [imagesFolderDir / imageName for imageName in inputImagesNames]

    # > Read images
    for imagePath in inputImagesPaths:
        images[imagePath.stem] = cv.imread(str(imagePath), cv.IMREAD_GRAYSCALE)
        imageCount+=1
        print(f'Images read: {imageCount}')

    print("Finished reading data.")

    print("There is/are in total " + str(len(images)) + " image(s).")

    return images, inputImagesPaths


def ReadImage(imagePath):
    return cv.imread(str(imagePath), cv.IMREAD_GRAYSCALE)


def Test_ReadImages():
    dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/input')
    inputImages, inputImagesNames = ReadImages(dataPath)
    print('test')

def UpdatePlateLayoutWithImageNames(plateLayout, inputImagesPaths):
    # to the right well in plate layout
    # append a list of all the fields
    # with timelapse images

    print('')
    return plateLayout



if __name__ == '__main__':
    Test_ReadPlateLayout()
