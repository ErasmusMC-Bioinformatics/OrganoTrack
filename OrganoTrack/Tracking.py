import cv2 as cv
from OrganoTrack.Displaying import Mask
import skimage.measure
from OrganoTrack.Importing import ReadImages
import numpy as np
from OrganoTrack.ImageHandling import DrawRegionsOnImages
from pathlib import Path
from PIL import Image
from typing import List, Optional, Callable
from skimage.measure._regionprops import regionprops, RegionProperties
from scipy.optimize import linear_sum_assignment
from OrganoTrack.HelperFunctions import printRep


class OrganoidTrack:
    # The track of an organoid across timelapse images
    def __init__(self):
        self.id = 0
        self.regionPropsPerFrame: List[Optional[RegionProperties]] = []
        #   A list of organoid (RegionProps) objects or
        #   'None' objects if no organoid is identified in the corresponding time point


CostFunctionType = Callable[[RegionProperties, RegionProperties], float]
#   A variable outlining the inputs and outputs of a function (a Callable)
#   that receives two RegionProperties objects and returns a float


def Label(image):
    labeled = skimage.measure.label(image)
    return labeled


def stack(images):
    return np.stack(images[:])


def LabelAndStack(images):
    return Label(stack(images))


# Function adopted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def SaveImages(data, suffix, pilImages, outputPath, fileNames):
    from OrganoTrack.ImageHandling import ConvertImagesToPILImageStacks, SavePILImageStack
    stacks = ConvertImagesToPILImageStacks(data, pilImages)
    # stacks is a List of 3D np arrays

    for stack, fileName in zip(stacks, fileNames):
        p = Path(fileName)
        SavePILImageStack(stack, outputPath / (p.stem + suffix + p.suffix))


# Function adopted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def MakeDirectory(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise Exception("Could not find or create directory '" + str(path.absolute()) + "'.")


def UpdateTrackedStack(stack: np.ndarray, highestTrackIDnum, field):
    if field != 1:
        stack = np.where(stack != 0, stack + highestTrackIDnum, stack)
        return stack
    else:
        return stack

def track(segmentedTimelapseImages):
    '''
    :param segmentedTimelapseImages: a list of images (numpy arrays) that belong to one timelapse set
    :return: a 3D numpy array (stacked 2D arrays) with tracked objects labelled with their unique ID number
    '''

    # Label images
    labelledImages = [Label(image) for image in segmentedTimelapseImages]
    # a List of 2D arrays with objects labelled with a number


    # Stack the labelled images into a 3d array
    timelapseStack = stack(labelledImages)  # a 3D array of 2D images


    # Track objects across timelapse images
    trackedTimelapseStack = Track(timelapseStack, 1, Inverse(Overlap))

    return trackedTimelapseStack


# Function adopted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
# Cost functions
def Overlap(a: RegionProperties, b: RegionProperties):
    return np.size(IntersectCoords(a.coords, b.coords))     # the number of overlapping pixels
    # a.coords gives the Coordinate list (row, col) of the region


def Inverse(f: CostFunctionType) -> CostFunctionType:
    def i_f(x, y):      # Define Inverse(Overlap)
        v = f(x, y)         # Copy the function Overlap
        if v == 0:          # If there's no overlap
            return np.inf       # Return infinity
        return 1 / v        # Otherwise return the inverse of the overlap

    return i_f           # Return the function defined above


# Function adopted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def IntersectCoords(a: np.ndarray, b: np.ndarray):
    imageWidth = max(np.max(a[:, 1]), np.max(b[:, 1]))
    aIndices = a[:, 0] * imageWidth + a[:, 1]
    bIndices = b[:, 0] * imageWidth + b[:, 1]
    return np.intersect1d(aIndices, bIndices)


# Function adopted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def LastDetection(track: OrganoidTrack):
    return next((x for x in reversed(track.regionPropsPerFrame) if x is not None), None)
    # Next: Retrieve the next item from the iterator
    # Reversed: Return a reverse iterator: thus, the last regionProps object is first


# Function adapted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def Track(images: np.ndarray, costOfNonAssignment: float, costFunction: CostFunctionType,
          trackLostCutoff=10):

    tracks = []                                         # the set of organoid tracks
    relabeledImages = np.zeros_like(images)
    print("Tracking images...", end="", flush=True)
    for i in range(images.shape[0]):                    # for each image
        printRep(str(i) + "/" + str(images.shape[0]))       #

        # mapping is a (n x 2) array that gives the new tracked index for the objects of images[i]
        # at the second iteration, tracks is filled with OrganoTrack objects created in the 1st iteration
        mapping = np.asarray(UpdateTracks(tracks, images[i], costOfNonAssignment, trackLostCutoff,
                                          costFunction))

        m2 = np.zeros(np.max(mapping[:, 0]) + 1)    # m2 = zeros (n+1) long. +1 to allow indexing by the non-0 label num
        m2[mapping[:, 0]] = mapping[:, 1]           # fill m2 with the labels of img2 by its indices mapped by img1
        mappedImage = m2[images[i]]                 # the elements of images[i] are indexed in m2 to make mappedImage
        relabeledImages[i] = mappedImage            # the last image in the tracked image set = mappedImage


    printRep("Done.")
    printRep(None)
    return relabeledImages                          # the tracked image set


# Function adopted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def IsTrackAvailable(track: OrganoidTrack, numFrames: int, trackLostCutoff: int):
    if trackLostCutoff is None or numFrames < trackLostCutoff:
        return True
    return any(track.regionPropsPerFrame[-trackLostCutoff:])
    # returns True if any of the elements of a given iterable are True else it returns False.


# Function adapted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def UpdateTracks(currentTracks: List[OrganoidTrack], nextImage: np.ndarray,
                 costOfNonAssignment, trackLostCutoff, costFunction):

    # At the 2nd iteration, currentTracks contains the tracks found in the 1st iteration
    mappingForThisImage = []

    nextID = max([x.id for x in currentTracks], default=0) + 1

    # Get the number of objects detected for all current tracks (all tracks have the same number of timepoints)
    numFrames = len(currentTracks[0].regionPropsPerFrame) if currentTracks else 0

    # Morphologically analyze labeled regions in the image
    detectedOrganoids = regionprops(nextImage)  # labelled imaged, 1-indexed

    # available tracks that can be continued further
    availableTracks = [t for t in currentTracks if IsTrackAvailable(t, numFrames, trackLostCutoff)]

    lastDetectedOrganoids = [LastDetection(availableTrack) for availableTrack in availableTracks]
    # a list of the last detected organoids in all the tracks

    # list of tuples of (detectedOrg objects, lastDetectedOrg objects)
    assignments = MatchOrganoidsInImages(detectedOrganoids,
                                         lastDetectedOrganoids,
                                         costFunction,              # Inverse(Overlap())
                                         costOfNonAssignment)

    for detectedOrganoid, lastDetectedOrganoid, merged in assignments:  # for each extracted tuple
        if not merged:
            if not lastDetectedOrganoid:  # if not None = True. Thus, if no organoids previously detected (t0 image)
                # Create new tracks for all firstly detected orgs
                track = OrganoidTrack()  # a track for each detection
                track.id = nextID        # starting with 1
                track.regionPropsPerFrame = [None] * numFrames   # e.g. [None] * 2 = [None, None], [None]* 0 = []
                # the above line fills a 'None' for the frames in which no object was found for that new track.
                currentTracks.append(track)
                nextID += 1

            else:  # if organoids previously detected
                track = next(t for t in availableTracks if LastDetection(t) is lastDetectedOrganoid)
                # go through all available Tracks for each detection match (long for-looping!)
                #   if the last detection of that Track is the match given here
                # get that track

            # Edit the regionPropsPerFrame attribute of the newly created track
            # by appending (the RegionProperties object of) the corresponding organoid
            track.regionPropsPerFrame.append(detectedOrganoid)

            # for each detected organoid, return a tuple of its organoid label and its track id
            if detectedOrganoid is not None:
                mappingForThisImage.append((detectedOrganoid.label, track.id))

        else:

            # find tracks and delete them
            if detectedOrganoid is not None:
                mappingForThisImage.append((detectedOrganoid.label, 0))

    return mappingForThisImage


# Function adapted from OrganoID (Matthews et al. 2022 PLOS Compt Biol)
def MatchOrganoidsInImages(organoidsA: List[RegionProperties], organoidsB: List[RegionProperties],
                           costFunction: CostFunctionType, costOfNonAssignment):
    fullSize = len(organoidsA) + len(organoidsB)
    costMatrix = np.zeros([fullSize, fullSize], dtype=float)
    # costMatrix is 0-indexed. Thus, label 1 of organoidsA is 0th row of costMatrix

    costNonA = np.full([len(organoidsA), len(organoidsA)], np.inf)
    costNonB = np.full([len(organoidsB), len(organoidsB)], np.inf)
    np.fill_diagonal(costNonA, costOfNonAssignment)  # fill diag with 1
    np.fill_diagonal(costNonB, costOfNonAssignment)  # fill diag with 1

    # the costMatrix is defined to have the two square regions be / within the large square, not \
    costMatrix[:len(organoidsA), len(organoidsB):] = costNonA
    costMatrix[len(organoidsA):, :len(organoidsB)] = costNonB

    for i, a in enumerate(organoidsA):  # returns a list of tuples, with the 1st element as an index from 0
        for j, b in enumerate(organoidsB):
            costMatrix[i, j] = costFunction(a, b)

    # Franz addition to identify merged organoids
    consideredMatrix = costMatrix[:len(organoidsA), :len(organoidsB)]   # get the overlap section of the costMatrix
    merged = np.sum((consideredMatrix != np.inf), axis=1) > 1           # if > 1 non-inf value in detection row, merged

    assignment = linear_sum_assignment(costMatrix)  # tuple of arrays, one of row indices, one of match column indices
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    # along the rows of the t_n+1 image,
    #   returns (row index of t_n+1, column index of t_n image with minimum cell value along the row)
    #           (row index of newly detected organoid, column index of previously detected organoid) with best overlap

    # the argument returned below is the same as the comment here:
    # alist = []
    # for i,j in zip(assignment[0], assignment[1]):
    #     if i < len(organoidsA) or j < len(organoidsB):
    #         if i < len(organoidsA):
    #             a = organoidsA[i]
    #             c = merged[i]
    #         else:
    #             a = None
    #             c = False
    #         if j < len(organoidsB):
    #             b = organoidsB[j]
    #         else:
    #             b = None
    #         alist.append((a, b, c))

    return [(organoidsA[i] if i < len(organoidsA) else None,
             organoidsB[j] if j < len(organoidsB) else None,
             merged[i] if i < len(organoidsA) else False)
            for i, j in zip(assignment[0], assignment[1])
            if i < len(organoidsA) or j < len(organoidsB)]


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
    segmentedImages, imageNames = ReadImages(dataDir2)

    # Track objects over time
    trackedTimelapseStack = track(segmentedImages)


    # Create masks
    maskedImages = [Mask(ori, pred) for ori, pred in zip(inputImages, segmentedImages)]


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
