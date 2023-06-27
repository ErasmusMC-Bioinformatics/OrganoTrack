# take middle patch as reference
# calculate avg pixel
# based on avg px, compute a scale
# compute avg for each patch
# scale the avg of each patch (and the whole patch) according to the reference

import cv2 as cv
import numpy as np
from stitch import read_images, stitch
from toDelete.functions import display, plotHistogram


def normalise(img_set, wells, positions, channels, fields, norm_method):
    '''
        Function: To normalise the greyscale values between different subfield images of one well

        Input:
        - different fields of a well

        Output:
        - normalised fields of a well
    '''

    ref_field = 12
    # plotHistogram(ref_field, 'ref')
    # plotHistogram(side, 'side')

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

    """
    25 fields List order = [0,  1,  2,  3,  4,
                            5,  6,  7,  8,  9,
                            10, 11, 12, 13, 14,
                            15, 16, 17, 18, 19,
                            20, 21, 22, 23, 24 ]
    """

    # Reading mask images
    mask_dir = '/home/franz/Documents/mep/data/25-field-outer-field-masks'

    # a dictionary, with key as the field number and data as the field image
    masks = {field: cv.cvtColor(cv.imread(mask_dir+'/field-'+str(field)+'-mask.tiff'), cv.COLOR_BGR2GRAY)
             for field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]}

    norm_img_set = []

    for well in range(len(wells)):
        norm_imgs_one_well_all_positions_and_channels = []  # dimensions: positions x channels x fields

        for position in range(positions):
            norm_imgs_one_well_position_all_channels = []  # dimensions: channels x fields

            for channel in range(channels):
                norm_imgs_one_channel_all_fields = []  # dimensions: fields

                ref_img = img_set[well][position][channel][ref_field]
                ref_mean = np.mean(ref_img)
                ref_median = np.median(ref_img)
                ref_max = np.max(ref_img)
                blur_size = 5
                ref_blur = cv.GaussianBlur(ref_img, (blur_size, blur_size), 0)

                # the ref field will also be normalised according to ref. Either leave, or add 'if' that will execute
                #   for all fields
                for field in range(len(stitching_order)):

                    if norm_method == 'mean shift':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(masks[field])]]
                            field_mean = np.mean(field_nonzero_values)
                        else:
                            field_mean = np.mean(img_set[well][position][channel][field])

                        diff_mean = ref_mean - field_mean

                        norm_imgs_one_channel_all_fields.\
                            append(img_set[well][position][channel][field] + int(diff_mean))

                    elif norm_method == 'convolution':
                        response = np.divide(ref_blur, ref_img)
                        norm_imgs_one_channel_all_fields. \
                            append((np.multiply(img_set[well][position][channel][field], response)).astype(np.uint8))

                    elif norm_method == 'mean shift self':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(img_set[well][position][channel][field])]]
                            field_mean = np.mean(field_nonzero_values)
                        else:
                            field_mean = np.mean(img_set[well][position][channel][field])

                        diff_mean = ref_mean - field_mean

                        norm_imgs_one_channel_all_fields. \
                            append(img_set[well][position][channel][field] + int(diff_mean))

                    elif norm_method == 'median shift':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(masks[field])]]
                            field_median = np.median(field_nonzero_values)
                        else:
                            field_median = np.median(img_set[well][position][channel][field])
                        diff_median = ref_median - field_median

                        norm_imgs_one_channel_all_fields. \
                            append(img_set[well][position][channel][field] + int(diff_median))

                    elif norm_method == 'median shift self':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(img_set[well][position][channel][field])]]
                            field_median = np.median(field_nonzero_values)
                        else:
                            field_median = np.median(img_set[well][position][channel][field])
                        diff_median = ref_median - field_median

                        norm_imgs_one_channel_all_fields. \
                            append(img_set[well][position][channel][field] + int(diff_median))

                    elif norm_method == 'mean scale':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(masks[field])]]
                            field_mean = np.mean(field_nonzero_values)
                        else:
                            field_mean = np.mean(img_set[well][position][channel][field])
                        scale_mean = ref_mean/field_mean

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field]*scale_mean).astype(np.uint8))

                    elif norm_method == 'mean scale self':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(img_set[well][position][channel][field])]]
                            field_mean = np.mean(field_nonzero_values)
                        else:
                            field_mean = np.mean(img_set[well][position][channel][field])
                        scale_mean = ref_mean/field_mean

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_mean).astype(np.uint8))

                    elif norm_method == 'median scale':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(masks[field])]]
                            field_median = np.median(field_nonzero_values)
                        else:
                            field_median = np.median(img_set[well][position][channel][field])
                        scale_median = ref_median/field_median

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_median).astype(np.uint8))

                    elif norm_method == 'median scale self':
                        if field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]:
                            field_nonzero_values = [img_set[well][position][channel][field][np.nonzero(img_set[well][position][channel][field])]]
                            field_median = np.median(field_nonzero_values)
                        else:
                            field_median = np.median(img_set[well][position][channel][field])
                        scale_median = ref_median/field_median

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_median).astype(np.uint8))

                    elif norm_method == 'ref mean factor':
                        scale_mean = ref_mean/ref_max

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_mean).astype(np.uint8))

                    elif norm_method == 'ref median factor':
                        scale_median = ref_median/ref_max

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_median).astype(np.uint8))

                    elif norm_method == 'ref mean factor plus one':
                        scale_mean = 1 + ref_mean/ref_max

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_mean).astype(np.uint8))

                    elif norm_method == 'ref median factor plus one':
                        scale_median = 1 + ref_median/ref_max

                        norm_imgs_one_channel_all_fields. \
                            append((img_set[well][position][channel][field] * scale_median).astype(np.uint8))


                norm_imgs_one_well_position_all_channels.append(norm_imgs_one_channel_all_fields)
            norm_imgs_one_well_all_positions_and_channels.append(norm_imgs_one_well_position_all_channels)
        norm_img_set.append(norm_imgs_one_well_all_positions_and_channels)
    return norm_img_set


if __name__ == '__main__':
    dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/Images'
    # def read_images(dir, fields, wells, channels, positions):
    fields = 25
    wells = [('02', '02')]
    channels = 1  # focus on brightfield channel for now
    positions = 1  # only 1 position for brightfield image
    img_set = read_images(dir, wells, positions, channels, fields)

    ref_field = 12
    other_field = 14

    plotHistogram(img_set[0][0][0][ref_field], 'Centre field', img_set[0][0][0][other_field], 'Field next to centre', 'Histogram pre-normalisation')
    display('centre field pre-normalisation', img_set[0][0][0][ref_field], 0.5)
    display('next field pre-normalisation', img_set[0][0][0][other_field], 0.5)
    pre_normalisation = stitch(img_set, wells, positions, channels, fields)

    norm_mode = 'convolution'
    img_set = normalise(img_set, wells, positions, channels, fields, norm_mode)
    plotHistogram(img_set[0][0][0][ref_field], 'Centre field', img_set[0][0][0][other_field], 'Field next to centre', 'Histogram post-normalisation')
    display('centre field post-normalisation', img_set[0][0][0][ref_field], 0.5)
    display('next field post-normalisation', img_set[0][0][0][other_field], 0.5)
    post_normalisation = stitch(img_set, wells, positions, channels, fields)

    display('pre-normalisation', pre_normalisation[0][0][0], 0.15)
    display('post-normalisation', post_normalisation[0][0][0], 0.15)

    cv.waitKey(0)



