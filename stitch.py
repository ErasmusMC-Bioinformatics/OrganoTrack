'''
Stitch
- Combine all field images into one image
'''

import numpy as np
import cv2 as cv
from functions import rescale, display

def read_images(dir, wells, positions, channels, fields):
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

    # Stitching order
    fields_25_stitching_order = ['02', '03', '04', '05', '06',
                                 '11', '10', '09', '08', '07',
                                 '12', '13', '01', '14', '15',
                                 '20', '19', '18', '17', '16',
                                 '21', '22', '23', '24', '25' ]
    fields_25_rows = 5

    fields_9_stitching_order = ['02', '03', '04',
                                '06', '01', '05',
                                '07', '08', '09']

    # using positions
    fields_6_stitching_order = ['02', '01', '03',
                                '06', '05', '04']

    fields_9_rows = 3

    if fields == 6:
        stitching_order = fields_6_stitching_order
    elif fields == 9:
        stitching_order = fields_9_stitching_order
    elif fields == 25:
        stitching_order = fields_25_stitching_order

    channel_names = ['1', '2', '3', '4']
    position_names = ['01', '02', '03', '04', '05',
                      '06', '07', '08', '09', '10',
                      '11', '12', '13', '14', '15']

    # Reading images
    # imgs_all_wells = []  # wells x channels x fields
    imgs_all_wells = []  # wells x positions x channels x fields

    for well in range(len(wells)):
        # imgs_one_well_all_channels = []  # channels x fields
        imgs_one_well_all_positions_and_channels = []  # positions x channels x fields

        for position in range(positions):
            imgs_one_well_position_all_channels = []  # channels x fields

            for channel in range(channels):
                imgs_one_channel_all_fields = []  # fields


                for field in range(len(stitching_order)):
                    # Each image (fluorescent or brightfield) appears as 1080 x 1080 x 3,
                    # but is grayscale with one channel. Thus, these 3 pseudo 'channels'
                    # are compacted to one with cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                    imgs_one_channel_all_fields.append(cv.imread(dir +
                                                                 '/r' + wells[well][0] +
                                                                 'c' + wells[well][1] +
                                                                 'f' + stitching_order[field] +
                                                                 'p' + position_names[position] +
                                                                 '-ch' + channel_names[channel] +
                                                                'sk1fk1fl1.tiff', cv.IMREAD_GRAYSCALE))

                # at the end of each channel
                imgs_one_well_position_all_channels.append(imgs_one_channel_all_fields)

            # at the end of each well position
            imgs_one_well_all_positions_and_channels.append(imgs_one_well_position_all_channels)

        # at the end of each well
        imgs_all_wells.append(imgs_one_well_all_positions_and_channels)

    return imgs_all_wells


def stitch(unstitched_images, wells, positions, channels, fields):
    '''
        Function: To merge the different subfield images of one well into one image

        Input:
            - images_of_all_wells = a nested list of images of each well subdivided by fields, captured at different z
            positions and with different channels (shape = wells x positions x channels x fields),
            - wells = a list of tuples with the well's row and column numbers in text format,
            - positions = an integer quantity of the z positions that each well was imaged at,
            - channels = the number of imaging channels used for the imaging experiment,
            - fields = an integer quantity of the subfields imaged per well.

        Output:
            - normalised fields of a well
    '''

    """
    25 fields List order = [0,  1,  2,  3,  4,
                            5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24 ]
    """

    """
    9 fields List order = [ 0,  1,  2,
                            3,  4,  5,
                            6,  7,  8]
    """

    stitched_imgs = []  # wells x positions x channels

    for well in range(len(wells)):
        stitched_imgs_one_well_all_z_positions_and_channels = []  # positions x channels

        for position in range(positions):
            stitched_imgs_at_one_well_position_for_all_channels = []  # channels

            for channel in range(channels):

                # Stitch columns, then stitch rows
                if fields == 6:
                    first_row = np.concatenate((unstitched_images[well][position][channel][0],
                                                unstitched_images[well][position][channel][1],
                                                unstitched_images[well][position][channel][2]), axis=1)
                    second_row = np.concatenate((unstitched_images[well][position][channel][3],
                                                 unstitched_images[well][position][channel][4],
                                                 unstitched_images[well][position][channel][5]), axis=1)

                    # at the end of each channel, store the stitched image
                    stitched_imgs_at_one_well_position_for_all_channels.append(np.concatenate((first_row, second_row), axis=0))

                elif fields == 25:
                    first_row = np.concatenate((unstitched_images[well][position][channel][0],
                                                unstitched_images[well][position][channel][1],
                                                unstitched_images[well][position][channel][2],
                                                unstitched_images[well][position][channel][3],
                                                unstitched_images[well][position][channel][4]), axis=1)
                    second_row = np.concatenate((unstitched_images[well][position][channel][5],
                                                 unstitched_images[well][position][channel][6],
                                                 unstitched_images[well][position][channel][7],
                                                 unstitched_images[well][position][channel][8],
                                                 unstitched_images[well][position][channel][9]), axis=1)
                    third_row = np.concatenate((unstitched_images[well][position][channel][10],
                                                unstitched_images[well][position][channel][11],
                                                unstitched_images[well][position][channel][12],
                                                unstitched_images[well][position][channel][13],
                                                unstitched_images[well][position][channel][14]), axis=1)
                    fourth_row = np.concatenate((unstitched_images[well][position][channel][15],
                                                 unstitched_images[well][position][channel][16],
                                                 unstitched_images[well][position][channel][17],
                                                 unstitched_images[well][position][channel][18],
                                                 unstitched_images[well][position][channel][19]), axis=1)
                    fifth_row = np.concatenate((unstitched_images[well][position][channel][20],
                                                unstitched_images[well][position][channel][21],
                                                unstitched_images[well][position][channel][22],
                                                unstitched_images[well][position][channel][23],
                                                unstitched_images[well][position][channel][24]), axis=1)

                    # at the end of each channel, store
                    stitched_imgs_at_one_well_position_for_all_channels.append(
                        np.concatenate((first_row, second_row, third_row, fourth_row, fifth_row), axis=0))

            # at the end of each well position, store the 4 channel stitched images
            stitched_imgs_one_well_all_z_positions_and_channels.\
                append(stitched_imgs_at_one_well_position_for_all_channels)

        # at the end of each well
        stitched_imgs.append(stitched_imgs_one_well_all_z_positions_and_channels)

    return stitched_imgs


def export_imgs(imgs, dir, wells, channels, positions):

    position_names = ['01', '02', '03', '04', '05',
                      '06', '07', '08', '09', '10',
                      '11', '12', '13', '14', '15']

    for well in range(len(wells)):
        for position in range(positions):
            for channel in range(channels):
                # Export
                file_name = '/well-' + str(wells[well]) + '-p' + position_names[position] + '-channel-' + str(channel + 1) + '.png'
                cv.imwrite(dir + file_name, imgs[well][position][channel])


if __name__ == '__main__':
    # plates_input_dir = ['/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-1__2022-12-23T10_46_22-Measurement-1/export-ij',
    #                     '/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-2__2022-12-23T09_41_33-Measurement-1/export-ij']
    # plates_export_dir = ['/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-1__2022-12-23T10_46_22-Measurement-1/stitched_enhanced',
    #                      '/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-2__2022-12-23T09_41_33-Measurement-1/stitched_enhanced']
    import_dir = '/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-2-zstack__2022-12-23T10_01_24-Measurement-1/auto-enhanced-fiji'
    export_dir = '/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-2-zstack__2022-12-23T10_01_24-Measurement-1/z-stack-stitched-enhanced'
    fields = 6
    wells = [('03', '09'), ('03', '10'),
             ('04', '09'), ('04', '10')] #,
             # ('05', '09'), ('06', '09'), ('07', '09'), ('08', '09'), ('09', '09'), ('10', '09'),
             # ('11', '09'), ('12', '09'), ('13', '09'), ('14', '09')]
    channels = 4
    positions = 15
    # for plate in range(len(import_dir)):
    all_imgs = read_images(import_dir, fields, wells, channels, positions)
    stiched_imgs = stitch(all_imgs, fields, wells, channels, positions)
    export_imgs(stiched_imgs, export_dir, wells, channels, positions)
