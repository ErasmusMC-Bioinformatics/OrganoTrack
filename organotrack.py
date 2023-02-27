from stitch import read_images
from segmentation import segment
from functions import display
import numpy as np
import time
import cv2 as cv
import glob
import matplotlib.pyplot as plt

'''
    Welcome
'''
print("Welcome to OrganoTrack")

'''
    Read data
'''

# print("Enter naming structure of data")
# < Naming structure
# Determine appropriate data structure

# dataDirectory = input("Enter data directory. The directory should not have spaces: ")
dataDirectory = '/home/franz/Documents/mep/data/2023-02-24-Cis-Tos-dataset-mathijs/AZh5/Day-12/renamed'
fields = 1
wells = [('04', '04'), ('04', '07'), ('05', '04'), ('05', '09'), ('05', '10'), ('05', '11')]
experiment = 'Azh5-reseeding'
positions = 1
channels = 1
timePoints = 1

# Reading data
# imData = read_images(dataDirectory, wells, positions, channels, timePoints, fields)
# images = [imData[i][0][0][0][0] for i in range(len(imData))]


# Read images according to data structure
    # If image not grayscale, read as is and convert to grayscale when begin to segment

'''
    Segmentation
'''

# print("Choose from options")
# < OrganoSeg / OrganoID / Optimise parameters and segment

# If OrganoSeg / OrganoID:
    # segmentation = segment(images)

# segmentation_times = []
#
# segmented_images = []
# for i in range(len(imData)):
#     print('Segmenting image ' + str(i+1))
#     tic = time.process_time()
#     segmented_images.append(segment(images[i]))
#     toc = time.process_time() - tic
#     segmentation_times.append(toc)
# print('segmentation times in seconds: ')
# print(segmentation_times)

store_seg_dir = "/home/franz/Documents/mep/data/2023-02-24-Cis-Tos-dataset-mathijs/AZh5/Day-12/renamed/segmented"
input_images = glob.glob(store_seg_dir + "/*.png")

segmented = []
for image_path in input_images:
    segmented.append(cv.imread(image_path, cv.IMREAD_ANYDEPTH))

image = segmented[0]

display('first', image, 0.25)


# Diplay images
# for i in range(len(segmented)):
#     display(str(i), segmented[i], 0.25)
# cv.waitKey(0)



# for i in range(len(segmented_images)):
#     filename = "/segmentation-" + str(i+1) + ".png"
#     cv.imwrite(store_seg_dir+filename, segmented_images[i])

# Elif optimise parameters:
    # for each parameter combination:
        # calculate segmentation performance
            # (user can input % of dataset to test this on)
            # Does decreasing to 8-bit make the operation run faster? Does it affect seg performance?
        # store segmenation performance for each combination

    # return combination with highest performance

    # segmentation = segment_with_optimal_params(images)

    # further processing:
        # ID every object
        # x remove noise
        # x smoothen
        # x include hole closing of everything (but keep the identity of out of focus organoids)
        # x # Remove border objects
        # Remove out of focus organoids


# Report segmentation metrics (using GT dataset)


'''
    User selection / Filtering / Measuring features
'''


# Select unmerged organoids (only if there is time in the data)
# If one object in timepoint t+1 overlaps with more than 1 object in timepoint t:
    # remove that object from the image of timepoint t+1


# Measuring sizes of organoids

method = 'blobDetect'

if method == 'contour':
    # find all the contours from the B&W image
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # needed to filter only our contours of interest

    # for each contour found
    areas = [cv.contourArea(cnt) for cnt in contours]

    # counts, bins = np.histogram(areas, 20)
    # plt.hist(bins[:-1], bins, weights=counts)
    plt.hist(areas, bins=20)
    plt.show()

elif method == 'connectedComp':
    # cv.waitKey(0)
    # Filter out organoids that do not meet a range in: size, circularity, STAR, SER, etc.
    # removing <10K pixels organoids
    nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(image)
    sizes = stats[:,-1]
    sizes = sizes[1:]
    nb_blobs -= 1
    plt.hist(sizes, bins=20)
    plt.show()

    min_size = 10000

    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs, dtype=np.uint8)
    # for every component in the image, keep it only if it's above min_size
    for blob in range(nb_blobs):
        if sizes[blob] >= min_size:
            # see description of im_with_separated_blobs above
            im_result[im_with_separated_blobs == blob + 1] = 255

    display('filtered', im_result, 0.25)
    nb_blobs2, im_with_separated_blobs2, stats2, _ = cv.connectedComponentsWithStats(im_result)
    sizes2 = stats2[:,-1]
    sizes2 = sizes2[1:]
    plt.hist(sizes2, bins=20)
    plt.show()

elif method == 'blobDetect':


    # Set up the detector with default parameters.
    # Setup SimpleBlobDetector parameters.
    params = cv.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 10000

    detector = cv.SimpleBlobDetector_create(params)
    print(type(detector))

    # Detect blobs.
    keypoints = detector.detect(image)
    print(type(keypoints))  # tuple
    print(len(keypoints))  # 0
    print(keypoints)  # ()

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
                                          cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    display("Keypoints", im_with_keypoints, 0.25)

cv.waitKey(0)



# Remaining: organoids of interest, with their measurements available (export data)


'''
    Data exporting
'''
# Export to CSV

# Plotting
