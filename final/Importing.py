import cv2 as cv
import os

def ReadPlateLayout(plateLayoutDir):
    print("Read plate")

def Test_ReadPlateLayout():
    plateDir = '/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Assaylayout/plate_layout.xlsx'
    ReadPlateLayout(plateDir)

def ReadImages(dataDir):
    '''

    :param dataDir: directory of image files
    :return: list of images
    '''

    print("\nReading data...")

    # > Get the names and extensions of the image files in the directory
    inputImagesNames = sorted(os.listdir(dataDir))

    # > Create directory paths for each image file
    imagePaths = [dataDir + '/' + imageName for imageName in inputImagesNames]

    # > Read images
    inputImages = [cv.imread(imagePath, cv.IMREAD_GRAYSCALE) for imagePath in imagePaths]

    print("Finished reading data.")

    print("\nThere is/are in total " + str(len(inputImagesNames)) + " image(s).")

    return inputImages, inputImagesNames



if __name__ == '__main__':
    Test_ReadPlateLayout()
