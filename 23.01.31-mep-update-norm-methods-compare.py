# 30 January 2023
# Code to produce results for a slide for MEP Update 31 Jan 2023

import cv2 as cv
from stitch import read_images, stitch
from normalise import normalise
from functions import display

if __name__ == '__main__':
    # print ref image
    # print field image
    # post_normalisation of field for each normalisation method

    dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/Images'

    fields = 25
    wells = [('02', '02')]
    channels = 1  # focus on brightfield channel for now
    positions = 1  # only 1 position for brightfield image
    img_set = read_images(dir, wells, positions, channels, fields)

    ''' Slide 5'''
    # # reference image
    # ref_num = 12
    # ref_field = img_set[0][0][0][ref_num]
    # display('centre field pre-normalisation', ref_field, 0.5)
    #
    # other_field_num = 24
    # other_field = img_set[0][0][0][other_field_num]
    # display('other field pre-normalisation', other_field, 0.5)
    #
    # # pre-norm histogram
    # plotHistogram(ref_field, 'Centre field',
    #               other_field, 'Other field',
    #               'Pre-norm')
    #
    # norm_modes = ['mean shift', 'median shift',
    #               'mean scale', 'median scale',
    #               'ref mean factor', 'ref median factor',
    #               'ref mean factor plus one',  'ref median factor plus one']
    #
    # for i in range(len(norm_modes)):
    #     print(str(i+1)+") Norm mode "+norm_modes[i]+" begun.")
    #     norm_mode = norm_modes[i]
    #     norm_img_set = normalise(img_set, wells, positions, channels, fields, norm_mode)
    #     plotHistogram(norm_img_set[0][0][0][ref_num], 'Centre field',
    #                   norm_img_set[0][0][0][other_field_num], 'Other field',
    #                   'Post-norm with ' + norm_mode)
    #     display('centre field post-norm with ' + norm_mode, norm_img_set[0][0][0][ref_num], 0.5)
    #     display('other field post-norm with ' + norm_mode, norm_img_set[0][0][0][other_field_num], 0.5)
    #     print(str(i+1)+") Norm mode " + norm_modes[i] + " ended.")
    #
    # mask_dir = '/home/franz/Documents/mep/data/25-field-outer-field-masks'
    # masks = {field: cv.cvtColor(cv.imread(mask_dir + '/field-' + str(field) + '-mask.tiff'), cv.COLOR_BGR2GRAY)
    #          for field in [0, 1, 2, 3, 4, 5, 9, 10, 14, 15, 19, 20, 21, 22, 23, 24]}
    # display('mask', masks[24], 0.5)
    #
    # cv.waitKey(0)


    ''' Slide 6'''
    norm_modes = ['mean scale', 'median scale',
                  'ref mean factor plus one',  'ref median factor plus one']
    for norm_mode in norm_modes:
        norm_img_set = normalise(img_set, wells, positions, channels, fields, norm_mode)
        stitched_img_set = stitch(norm_img_set, wells, positions, channels, fields)
        display('Stitched, norm mode ' + norm_mode, stitched_img_set[0][0][0], 0.15)

    cv.waitKey(0)



