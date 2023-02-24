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
