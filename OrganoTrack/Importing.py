import cv2 as cv
import os
import pandas as pd
from pathlib import Path
from scipy import stats
import re


def load_plate_layout(inputDataPath: Path):
    '''
    :param plateLayoutDir: directory of plate layout file (.xlsx, .csv, or .tsv). The file must (!) follow directives.
    :return: A List of Lists, with elements as tuples (condition, concentration, concentration unit)
    '''

    print("\nReading plate layout...")
    plateLayoutFile = next(os.walk(inputDataPath))[2][0]
    plateLayoutDir = inputDataPath / plateLayoutFile

    extension = plateLayoutDir.suffix

    if extension == '.csv':
        plateLayoutInput = pd.read_csv(plateLayoutDir, header=None)

    elif extension == '.tsv':
        plateLayoutInput = pd.read_csv(plateLayoutDir, header=None, delimiter='\t')

    else:
        plateLayoutInput = pd.read_excel(plateLayoutDir, header=None)

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
    load_plate_layout(dataPath)

def GetIdentifierValue(imageName, identifier):
    pattern = r"{0}(\d+)".format(identifier)
    match = re.search(pattern, imageName)
    # re.search does not cut the imageName by the pattern found. If you do so, searching may be faster?
    return int(match.group(1))  # the identifier value

def GetIdentifierInfo(imageName, identifiers):
    identifierValues = dict()

    for identifierName, identifierCharacter in identifiers.items():
        identiferValue = GetIdentifierValue(imageName, identifierCharacter)
        identifierValues[identifierName] = identiferValue

    return identifierValues


def ReadImages(importPath: Path, identifiers: dict):
    print("Reading data...")
    imagesDirectoryPath = importPath / next(os.walk(importPath))[1][0]  # nxt()->tuple(imPath, directories, other files)
    imagesFileNamesWithExtensions = sorted(os.listdir(imagesDirectoryPath))

    imagesByWellsFieldsAndTimepoints = dict()
    imagesPaths = []

    for imageName in imagesFileNamesWithExtensions:
        identifierValues = GetIdentifierInfo(imageName, identifiers)
        row, column, field, position, timePoint = identifierValues['row'], identifierValues['column'], \
                                                  identifierValues['field'], identifierValues['position'], \
                                                  identifierValues['time_point']
        well = (row, column)
        if well not in imagesByWellsFieldsAndTimepoints:
            print(f'Reading well {well}...')
            imagesByWellsFieldsAndTimepoints[well] = dict()
        if field not in imagesByWellsFieldsAndTimepoints[well]:
            imagesByWellsFieldsAndTimepoints[well][field] = []
        image = cv.imread(str(imagesDirectoryPath / imageName), cv.IMREAD_GRAYSCALE)
        # Using cv.IMREAD_GRAYSCALE to convert any image to single channel, 8-bit grayscale image
        # See more reading flags here: https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga61d9b0126a3e57d9277ac48327799c80
        imagesByWellsFieldsAndTimepoints[well][field].append(image)
        imagesPaths.append(imagesDirectoryPath / imageName) # still a list

    return imagesByWellsFieldsAndTimepoints, imagesPaths


def Test_ReadImages():
    dataPath = Path('/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-all-cis-data/input')
    identifiers = {'row': 'r',  # identifiers should be an input by the user indeed, how to allow A3 input?
                   'column': 'c',
                   'field': 'f',
                   'position': 'p',
                   'time_point': 'sk'}
    images, imagesPaths = ReadImages(dataPath, identifiers)
    print('s')

def Test_ReadImagesWithHarmonyExport():
    dataPath = Path('/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/all-images')
    identifiers = {'row': 'R',  # identifiers should be an input by the user indeed, how to allow A3 input?
                   'column': 'C',
                   'field': 'F',
                   'position': 'P',
                   'time_point': 'T'}
    images, imagesPaths = ReadImages(dataPath, identifiers)


def UpdatePlateLayoutWithImageNames(plateLayout, inputImagesPaths):
    # to the right well in plate layout
    # append a list of all the fields
    # with timelapse images

    print('')
    return plateLayout



if __name__ == '__main__':
    Test_ReadImagesWithHarmonyExport()
