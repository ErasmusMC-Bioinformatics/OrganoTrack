# Fri 24 Feb 2023
# # Extract the boundary pixels from the contour as an array
# boundary_pixels = []
# i = 0
# for contour in contours:  # runs  only once
#     print(i)
#     boundary_pixels.append(contour.squeeze())  # squeeze reduces the dimensions of the tuple by one
#     i += 1
#
# # Display the boundary pixels
# print("Boundary pixels:")
# boundaries = np.array(boundary_pixels).squeeze(axis=0)
# print(boundaries)


'''
    Other options to measure object area
'''
# elif method == 'connectedComp':
#     # cv.waitKey(0)
#     # Filter out organoids that do not meet a range in: size, circularity, STAR, SER, etc.
#     # removing <10K pixels organoids
#     nb_blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(image)
#     sizes = stats[:, -1]
#     sizes = sizes[1:]
#     nb_blobs -= 1
#     plt.hist(sizes, bins=20)
#     plt.show()
#
#     min_size = 10000
#
#     # output image with only the kept components
#     im_result = np.zeros_like(im_with_separated_blobs, dtype=np.uint8)
#     # for every component in the image, keep it only if it's above min_size
#     for blob in range(nb_blobs):
#         if sizes[blob] >= min_size:
#             # see description of im_with_separated_blobs above
#             im_result[im_with_separated_blobs == blob + 1] = 255
#
#     display('filtered', im_result, 0.25)
#     nb_blobs2, im_with_separated_blobs2, stats2, _ = cv.connectedComponentsWithStats(im_result)
#     sizes2 = stats2[:,-1]
#     sizes2 = sizes2[1:]
#     plt.hist(sizes2, bins=20)
#     plt.show()
#
# elif method == 'blobDetect':
#
#
#     # Set up the detector with default parameters.
#     # Setup SimpleBlobDetector parameters.
#     params = cv.SimpleBlobDetector_Params()
#     # Change thresholds
#     params.minThreshold = 10
#     params.maxThreshold = 200
#
#     # Filter by Area.
#     params.filterByArea = True
#     params.minArea = 10000
#
#     detector = cv.SimpleBlobDetector_create(params)
#     print(type(detector))
#
#     # Detect blobs.
#     keypoints = detector.detect(image)
#     print(type(keypoints))  # tuple
#     print(len(keypoints))  # 0
#     print(keypoints)  # ()
#
#     # Draw detected blobs as red circles.
#     # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
#     # the size of the circle corresponds to the size of blob
#
#     im_with_keypoints = cv.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255),
#                                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
#     # Show blobs
#     display("Keypoints", im_with_keypoints, 0.25)

'''
    Segmenting with skimage
'''
# elif adaptiveMethod == 'skimage':
#
    # for windowSize in range(13, 22, 2):
    #     print(windowSize)
    #     convolved = cv.blur(img, (windowSize, windowSize))  # uint8
    #     # display('blur', convolved, 0.5)
    #     substract = cv.subtract(convolved, img)  # uint8
    #     # display('subtract', substract, 0.5)
    #     otsu = threshold_otsu(substract)
    #     adaptiveIter = (threshold_local(substract, windowSize, 'mean', otsu * fudgeFactor)).astype(imgDataType)
    #     adaptiveIter = cv.bitwise_not(adaptiveIter)
    #     # display(str(windowSize), adaptiveIter, 0.5)
    #     adaptiveSum = cv.add(adaptiveSum, adaptiveIter)

'''
    Old reading method
'''


def ReadImages(dataDir, wells, positions, channels, timePoints, fields):


    # edit:
    # - take image file format as input. File formats change.
    # - take well layout as e.g. (04, 04)
    # - take the number of positions and create a list with names to iterate through
    # - take the number of channels and create a list with names to iterate through
    # - take the number of timePoints and create a list with the names to iterate through
    # - make it so general that the user can add whatever variables they desire (positions, channels, timePoints)
    #   and the program would create the list for reading.
    # - or consider if all files should be read, irrespective of the name and organised as they come.
    # the core of the last 2 points is: does the image data need a data structure, or is a list with indexes of the contents?
    # it depends on how the user wants to process the data

    '''
        Purpose: To read all files of the imaging experiment

        Input:
            - wells = a list of tuples with the well's row and column numbers in text format,
            - positions = an integer quantity of the z positions that each well was imaged at,
            - channels = the number of imaging channels used for the imaging experiment,
            - fields = an integer quantity of the subfields imaged per well.

        Output:
            - imgs_all_wells = a nested list of images of each well subdivided by fields, captured at different z
            positions and with different channels (shape = wells x positions x channels x fields)
    '''

    '''
        1. Read images in stitching order

    '''

    # # Stitching order
    # fields_25_stitching_order = ['02', '03', '04', '05', '06',
    #                              '11', '10', '09', '08', '07',
    #                              '12', '13', '01', '14', '15',
    #                              '20', '19', '18', '17', '16',
    #                              '21', '22', '23', '24', '25' ]
    # fields_25_rows = 5
    #
    # fields_9_stitching_order = ['02', '03', '04',
    #                             '06', '01', '05',
    #                             '07', '08', '09']
    #
    # # using positions
    # fields_6_stitching_order = ['02', '01', '03',
    #                             '06', '05', '04']
    #
    # fields_9_rows = 3
    #
    # if fields == 6:
    #     stitching_order = fields_6_stitching_order
    # elif fields == 9:
    #     stitching_order = fields_9_stitching_order
    # elif fields == 25:
    #     stitching_order = fields_25_stitching_order
    # elif fields == 1:
    #     stitching_order = ['01']
    #
    # channel_names = ['1', '2', '3', '4']
    # position_names = ['01', '02', '03', '04', '05',
    #                   '06', '07', '08', '09', '10',
    #                   '11', '12', '13', '14', '15']
    # timePoint_names = ['1', '2', '3', '4']  # for however long there are timepoints
    #
    # # Reading images
    # imgs_all_wells = []  # dimensions: wells x positions x channels x time x fields
    #
    # for well in range(len(wells)):
    #     # imgs_one_well_all_channels = []  # channels x fields
    #     imgs_one_well_all_positions_channels_and_times = []  # dimensions: positions x channels x time x fields
    #
    #     for position in range(positions):
    #         imgs_one_well_position_all_channels_and_times = []  # dimensions: channels x time x fields
    #
    #         for channel in range(channels):
    #             imgs_one_channel_all_times_and_fields = []  # dimensions: time x fields
    #
    #             for timePoint in range(timePoints):
    #                 imgs_one_timePoint_all_fields = []  # dimensions: fields
    #
    #                 for field in range(len(stitching_order)):
    #                     # cv.IMREAD_GRAYSCALE allows a 16-bit image to remain as 16-bit
    #                     imgs_one_timePoint_all_fields.append(cv.imread(dir +
    #                                                                  '/r' + wells[well][0] +
    #                                                                  'c' + wells[well][1] +
    #                                                                  'f' + stitching_order[field] +
    #                                                                  'p' + position_names[position] +
    #                                                                  '-ch' + channel_names[channel] +
    #                                                                  'sk' + timePoint_names[timePoint] +
    #                                                                  'fk1fl1.tif', cv.IMREAD_ANYDEPTH))
    #                 # at the end of each time point
    #                 imgs_one_channel_all_times_and_fields.append(imgs_one_timePoint_all_fields)
    #
    #             # at the end of each channel
    #             imgs_one_well_position_all_channels_and_times.append(imgs_one_channel_all_times_and_fields)
    #
    #         # at the end of each well position
    #         imgs_one_well_all_positions_channels_and_times.append(imgs_one_well_position_all_channels_and_times)
    #
    #     # at the end of each well
    #     imgs_all_wells.append(imgs_one_well_all_positions_channels_and_times)
