import numpy as np
import cv2 as cv
from stitch import read_images
from functions import plotHistogram, display
'''
    Normaliser
'''
ref_field = 12
dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/Images'
# def read_images(dir, fields, wells, channels, positions):
fields = 25
wells = [('02', '02')]
channels = 1  # focus on brightfield channel for now
positions = 1  # only 1 position for brightfield image
img_set = read_images(dir, wells, positions, channels, fields)

ref = img_set[0][0][0][ref_field]  # type array with elements type np.uint8
side = img_set[0][0][0][13]


plotHistogram(ref, 'ref')
plotHistogram(side, 'side')
display('ref', ref, 0.5)
display('side', side, 0.5)

# Mean and maxes
ref_mean = np.mean(ref)
side_mean = np.mean(side)
ref_max = np.max(ref)
scale_mean = ref_mean/side_mean
print(ref_mean)
print(ref_max)
print(scale_mean)

# Scaled image
side_scaled = side * scale_mean
print(type(side_scaled[100][100]))
side_scaled = side_scaled.astype(np.uint8)
print(type(side_scaled[100][100]))
plotHistogram(side_scaled, 'side scaled')
display('side scaled', side_scaled, 0.5)


cv.waitKey(0)


"""
    Building the stitcher
"""

# # input
# dir = '/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-1__2022-12-23T10_46_22-Measurement 1/hoechst-c4/r04c09'
# fields = 9
# wells = [('04', '09')]
# channels = 4
#
# """
#     Read
# """
#
#
# # Stitching order
# fields_25_stitching_order = ['02', '03', '04', '05', '06',
#                              '11', '10', '09', '08', '07',
#                              '12', '13', '01', '14', '15',
#                              '20', '19', '18', '17', '16',
#                              '21', '22', '23', '24', '25']
# fields_25_rows = 5
#
# fields_9_stitching_order = ['02', '03', '04',
#                             '06', '01', '05',
#                             '07', '08', '09']
# fields_9_rows = 3
#
# if fields == 9:
#     stitching_order = fields_9_stitching_order
# elif fields == 25:
#     stitching_order = fields_25_stitching_order
#
# print(stitching_order)
#
# # imgs_one_channel_all_fields = []  # fields
# #
# # for field in range(fields):
# #     imgs_one_channel_all_fields.append(cv.imread(dir +
# #                                                  '/r' + wells[0][0] +
# #                                                  'c' + wells[0][1] +
# #                                                  'f' + stitching_order[field] +
# #                                                  'p01-ch' + '2' +
# #                                                  'sk1fk1fl1.tiff'))
#
#
# # Reading images
# imgs_all_wells = []  # wells x channels x fields
# #
# for well in range(len(wells)):
#     imgs_one_well_all_channels = []  # channels x fields
#
#     for channel in range(channels):
#         imgs_one_channel_all_fields = []  # fields
#         channel_names = ['1', '2', '3', '4']
#
#         for field in range(len(stitching_order)):
#             imgs_one_channel_all_fields.append(cv.imread(dir +
#                                                          '/r' + wells[well][0] +
#                                                          'c' + wells[well][1] +
#                                                          'f' + stitching_order[field] +
#                                                          'p01-ch' + channel_names[channel] +
#                                                          'sk1fk1fl1.tiff'))
#
#         # at the end of each channel
#         imgs_one_well_all_channels.append(imgs_one_channel_all_fields)
#
#     # at the end of each well
#     imgs_all_wells.append(imgs_one_well_all_channels)
#
#
# """
#     Enhance contrast
# """
#
# for well in range(len(wells)):
#     for channel in range(channels):
#         for field in range(fields):
#             # Increase brightness
#             imgs_all_wells[well][channel][field], alpha, beta = automatic_brightness_and_contrast(imgs_all_wells[well][channel][field], 0.2)
#
#
# """
#     Stitch
# """
# stitched_imgs_all_wells = []  # wells x channels
#
# for well in range(len(wells)):
#     stitched_imgs_one_well_all_channels = []  # channels x fields
#
#     for channel in range(channels):
#         # Stitch columns, then stitch rows
#         if fields == 9:
#             first_row = np.concatenate((imgs_all_wells[well][channel][0], imgs_all_wells[well][channel][1], imgs_all_wells[well][channel][2]), axis=1)
#             second_row = np.concatenate((imgs_all_wells[well][channel][3], imgs_all_wells[well][channel][4], imgs_all_wells[well][channel][5]), axis=1)
#             third_row = np.concatenate((imgs_all_wells[well][channel][6], imgs_all_wells[well][channel][7], imgs_all_wells[well][channel][8]), axis=1)
#
#             # at the end of each channel, store
#             stitched_imgs_one_well_all_channels.append(np.concatenate((first_row, second_row, third_row), axis=0))
#
#         elif fields == 25:
#             first_row = np.concatenate((imgs_all_wells[well][channel][0], imgs_all_wells[well][channel][1], imgs_all_wells[well][channel][2], imgs_all_wells[well][channel][3], imgs_all_wells[well][channel][4]), axis=1)
#             second_row = np.concatenate((imgs_all_wells[well][channel][5], imgs_all_wells[well][channel][6], imgs_all_wells[well][channel][7], imgs_all_wells[well][channel][8], imgs_all_wells[well][channel][9]), axis=1)
#             third_row = np.concatenate((imgs_all_wells[well][channel][10], imgs_all_wells[well][channel][11], imgs_all_wells[well][channel][12], imgs_all_wells[well][channel][13], imgs_all_wells[well][channel][14]), axis=1)
#             fourth_row = np.concatenate((imgs_all_wells[well][channel][15], imgs_all_wells[well][channel][16], imgs_all_wells[well][channel][17], imgs_all_wells[well][channel][18], imgs_all_wells[well][channel][19]), axis=1)
#             fifth_row = np.concatenate((imgs_all_wells[well][channel][20], imgs_all_wells[well][channel][21], imgs_all_wells[well][channel][22], imgs_all_wells[well][channel][23], imgs_all_wells[well][channel][24]), axis=1)
#
#             # at the end of each channel, store
#             stitched_imgs_one_well_all_channels.append(np.concatenate((first_row, second_row, third_row, fourth_row, fifth_row), axis=0))
#
#     # at the end of each well, store
#     stitched_imgs_all_wells.append(stitched_imgs_one_well_all_channels)
#
#
# """
#     Display & Export
# """
# for well in range(len(wells)):
#     for channel in range(channels):
#
#         # Export
#         file_name = '/well-' + str(wells[well]) + '-channel-' + str(channel + 1) + '.png'
#         cv.imwrite(dir+file_name, stitched_imgs_all_wells[well][channel])
#
#         # Display
#         display('well: ' + str(wells[well]) + ', channel ' + str(channel + 1), stitched_imgs_all_wells[well][channel], 0.25)
#
# cv.waitKey(0)
