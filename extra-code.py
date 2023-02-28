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

