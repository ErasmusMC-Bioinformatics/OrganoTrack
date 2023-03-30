from Importing import ReadImages
from Detecting import SegmentWithOrganoSegPy
from Exporting import SaveData
from Filtering import FilterByFeature
from Displaying import DisplayImages, Display, ConvertLabelledImageToBinary, displayingTrackedSet
from Measuring import MeasureMorphometry
from OrganoTracking import track

# temporary imports
import cv2 as cv

def RunOrganoTrack(inputImagesPaths, exportPath, segment: False, loadSeg: False, segmentedImagePaths):
    inputImages, inputImagesNames = ReadImages(inputImagesPaths)

    if segment:
        segmentedImages = [SegmentWithOrganoSegPy(image) for image in inputImages]
        SaveData(exportPath, segmentedImages, inputImagesNames)

    if loadSeg:
        segmentedImages, inputImagesNames = ReadImages(segmentedImagePaths)



if __name__ == '__main__':
    segmented = True

    if segmented:

        # Load segmentations
        segmentedImagePaths = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1-segmented/segmented-30.03.2023-12_01_28'
        segmentedImages, imageNames = ReadImages(segmentedImagePaths)

    else:
        # Reading
        imagePaths = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1'
        rawImages, imageNames = ReadImages(imagePaths)

        # Segment
        segmentedImages = [SegmentWithOrganoSegPy(image) for image in rawImages]

        # Save segmentations
        outputPath = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/03.30-building-pipeline1-segmented'
        SaveData(outputPath, segmentedImages, imageNames)

    # DisplayImages('pre-filter', segmentedImages, 0.5)

    # Preparing for filtering
    filterFeature = 'area'
    filterThreshold = 450

    # Filtering
    filteredImages = [FilterByFeature(image, filterFeature, filterThreshold) for image in segmentedImages]

    # DisplayImages('post-filter', filteredImages, 0.5)

    # Tracking
    trackedSet = track(filteredImages)

    displayingTrackedSet('tracked', trackedSet, 0.5)


    cv.waitKey(0)