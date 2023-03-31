# Tracker.py -- tracks organoids in sequences of labeled images

from typing import List, Optional, Callable
import numpy as np
from skimage.measure._regionprops import regionprops, RegionProperties
from scipy.optimize import linear_sum_assignment
from HelperFunctions import printRep


class OrganoidTrack:
    # Collection of data points for a single identified organoid
    def __init__(self):
        self.id = 0
        self.regionPropsPerFrame: List[Optional[RegionProperties]] = []  # a list of the RegionProps objects of
        #                                                                  the tracked org at each frame


CostFunctionType = Callable[[RegionProperties, RegionProperties], float]  # The type of a function that takes in
#                                                                           RegionProperties objects and returns
#                                                                           a float as output, e.g. Overlap or
#                                                                           PercentOverlap


# Cost functions
def Overlap(a: RegionProperties, b: RegionProperties):
    return np.size(IntersectCoords(a.coords, b.coords))     # the number of overlapping pixels
    # a.coords gives the Coordinate list (row, col) of the region


def Negative(f: CostFunctionType) -> CostFunctionType:
    return lambda x, y: -f(x, y)


def Inverse(f: CostFunctionType) -> CostFunctionType:
    def i_f(x, y):      # Define Inverse(Overlap)
        v = f(x, y)         # Copy the function Overlap
        if v == 0:          # If there's no overlap
            return np.inf       # Return infinity
        return 1 / v        # Otherwise return the inverse of the overlap

    return i_f           # Return the function defined above


def PercentOverlap(a: RegionProperties, b: RegionProperties):
    larger = max(a.area, b.area)
    return float(np.size(IntersectCoords(a.coords, b.coords))) / larger


def IntersectCoords(a: np.ndarray, b: np.ndarray):
    imageWidth = max(np.max(a[:, 1]), np.max(b[:, 1]))
    aIndices = a[:, 0] * imageWidth + a[:, 1]
    bIndices = b[:, 0] * imageWidth + b[:, 1]
    return np.intersect1d(aIndices, bIndices)


def LastDetection(track: OrganoidTrack):
    return next((x for x in reversed(track.regionPropsPerFrame) if x is not None), None)
    # Next: Retrieve the next item from the iterator
    # Reverse: Return a reverse iterator: thus, the last regionProps object is first


def Track(images: np.ndarray, costOfNonAssignment: float, costFunction: CostFunctionType,
          trackLostCutoff=10):
    tracks = []
    relabeledImages = np.zeros_like(images)
    print("Tracking images...", end="", flush=True)
    for i in range(images.shape[0]):
        printRep(str(i) + "/" + str(images.shape[0]))

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
    return relabeledImages                              # the tracked image set


def IsTrackAvailable(track: OrganoidTrack, numFrames: int, trackLostCutoff: int):
    if trackLostCutoff is None or numFrames < trackLostCutoff:
        return True
    return any(track.regionPropsPerFrame[-trackLostCutoff:])
    # returns True if any of the elements of a given iterable are True else it returns False.


def UpdateTracks(currentTracks: List[OrganoidTrack], nextImage: np.ndarray,
                 costOfNonAssignment, trackLostCutoff, costFunction):

    # At the 2nd iteration, currentTracks contains the tracks found in the 1st iteration
    mappingForThisImage = []

    nextID = max([x.id for x in currentTracks], default=0) + 1  # 2) 88 + 1 = 89

    # Get the number of frames already in the track of object 1 (but object 1 may be lost in the future)
    numFrames = len(currentTracks[0].regionPropsPerFrame) if currentTracks else 0

    # Morphologically analyze labeled regions in the image
    detectedOrganoids = regionprops(nextImage)

    availableTracks = [t for t in currentTracks if IsTrackAvailable(t, numFrames, trackLostCutoff)]

    lastDetectedOrganoids = [LastDetection(availableTrack) for availableTrack in availableTracks]

    # 1) list of tuples of (detectedOrg objects, tupled with lastDetectedOrg objects), (t0 org, None)
    assignments = MatchOrganoidsInImages(detectedOrganoids,         # at first 88 RegionProperties objects
                                         lastDetectedOrganoids,     # at first []
                                         costFunction,              # Inverse(Overlap(RegionProps object, RegionProps object))
                                         costOfNonAssignment)       # 1
    if currentTracks:
        print("h")
    for detectedOrganoid, lastDetectedOrganoid in assignments:  # for each extracted tuple
        if not lastDetectedOrganoid:  # if no organoids previously detected, thus this is t0 image
            # 1) Create new tracks for all firstly detected orgs
            track = OrganoidTrack()  # a track for each detection
            track.id = nextID        # starting with 1
            track.regionPropsPerFrame = [None] * numFrames   # e.g. [None] * 2 = [None, None]
            currentTracks.append(track)
            nextID += 1
        else:  # if organoids previously detected
            track = next(t for t in availableTracks if LastDetection(t) is lastDetectedOrganoid)

        # Edit the regionPropsPerFrame attribute of the newly created track
        # with the RegionProperties object of the corresponding organoid
        track.regionPropsPerFrame.append(detectedOrganoid)

        # for each detected organoid, return a tuple of its organoid label and its track id
        if detectedOrganoid is not None:
            mappingForThisImage.append((detectedOrganoid.label, track.id))
    print('ghe')
    return mappingForThisImage


def MatchOrganoidsInImages(organoidsA: List[RegionProperties], organoidsB: List[RegionProperties],
                           costFunction: CostFunctionType, costOfNonAssignment):
    fullSize = len(organoidsA) + len(organoidsB)        # 1) 88 + 0, 2) 88 + 123
    costMatrix = np.zeros([fullSize, fullSize], dtype=float)  # 1) 88 x 88, 2) 211 x 211

    costNonA = np.full([len(organoidsA), len(organoidsA)], np.inf)  # 1) 88 x 88 array full of inf, 2) 123 x 123 array full of inf
    costNonB = np.full([len(organoidsB), len(organoidsB)], np.inf)  # 1) 0 x 0 array full of inf, 2) 88 x 88 array full of inf
    np.fill_diagonal(costNonA, costOfNonAssignment)  # fill diag with 1
    np.fill_diagonal(costNonB, costOfNonAssignment)  # fill diag with 1

    # the costMatrix is defined to have the two square regions be / within the large square, not \
    costMatrix[:len(organoidsA), len(organoidsB):] = costNonA
    costMatrix[len(organoidsA):, :len(organoidsB)] = costNonB
    if organoidsB:
        print("h")
    for i, a in enumerate(organoidsA):  # returns a list of tuples, with the 1st element as an index from 0
        for j, b in enumerate(organoidsB):
            costMatrix[i, j] = costFunction(a, b)

    # Removing merged
    for i in range(len(organoidsA)):
        loc = np.where(costMatrix[i, :len(organoidsB)] != np.inf)
        if len(loc[0]) > 1:
            costMatrix[i, loc] = np.inf

    assignment = linear_sum_assignment(costMatrix)  # tuple of arrays, one of row indices, one of match column indices
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    # along rows (tn+1), returns row index with the column (tn) index that has the minimum value along the row

    returnFinal = [(organoidsA[i] if i < len(organoidsA) else None,
             organoidsB[j] if j < len(organoidsB) else None)
            for i, j in zip(assignment[0], assignment[1])
            if i < len(organoidsA) or j < len(organoidsB)]

    return returnFinal
