import cv2 as cv
from stitch import read_images, stitch
from normalise import normalise
from toDelete.functions import plotHistogram, display
import numpy as np

if __name__ == '__main__':
    # print ref image
    # print field image
    # post_normalisation of field for each normalisation method

    # Reading
    dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/Images'

    fields = 25
    wells = [('02', '02')]
    channels = 1  # focus on brightfield channel for now
    positions = 1  # only 1 position for brightfield image
    img_set = read_images(dir, wells, positions, channels, fields)

    # Reference field
    ref_field_num = 12
    ref_field = img_set[0][0][0][ref_field_num]
    display('ref field pre-norm', ref_field, 0.5)

    # Another example field
    other_field_num = 14
    other_field = img_set[0][0][0][other_field_num]
    display('other field pre-norm', other_field, 0.5)

    # Histogram and stitched
    plotHistogram(ref_field, 'Ref field', other_field, 'Other field', 'Pre-norm')
    pre_normalisation = stitch(img_set, wells, positions, channels, fields)
    display('pre-norm stitch', pre_normalisation[0][0][0], 0.15)

    # Building the normaliser
    blur_size = 5
    ref_blur = cv.GaussianBlur(ref_field, (blur_size, blur_size), 0)
    # kernel1 = 1/273 * np.array([[1, 4, 7, 4, 1],
    #                             [4, 16, 26, 16, 4],
    #                             [7, 26, 41, 26, 7],
    #                             [4, 16, 26, 16, 4],
    #                             [1, 4, 7, 4, 1]])
    # ref_blur = cv.filter2D(src=ref_field, ddepth=-1, kernel=kernel1)
    display('ref blurred', ref_blur, 0.5)
    # response = cv.divide(ref_blur, ref_img)
    response = np.divide(ref_blur, ref_field)
    print(response)
    display('response', response, 0.5)
    normalised = (np.multiply(other_field, response)).astype(np.uint8)

    # ref_blur = cv.GaussianBlur(ref_img, (blur_size, blur_size), 0)
    # response = np.divide(ref_blur, ref_field)
    #                         norm_imgs_one_channel_all_fields. \
    #                             append((np.multiply(img_set[well][position][channel][field], response)).astype(np.uint8))


    print(normalised)
    display('other post-norm', normalised, 0.5)


    # Normalising
    norm_mode = 'convolution'
    norm_img_set = normalise(img_set, wells, positions, channels, fields, norm_mode)

    # Displaying normalised fields and histogram
    norm_ref_field = norm_img_set[0][0][0][ref_field_num]
    display('ref field post-norm', norm_ref_field, 0.5)
    norm_other_field = norm_img_set[0][0][0][other_field_num]
    display('Other field post-norm', norm_other_field, 0.5)
    plotHistogram(norm_ref_field, 'Norm ref field', norm_other_field, 'Norm other field', 'Post-norm')

    # Displaying stitched normalised
    post_normalisation = stitch(norm_img_set, wells, positions, channels, fields)
    display('post-norm stitch', post_normalisation[0][0][0], 0.15)

    cv.waitKey(0)

