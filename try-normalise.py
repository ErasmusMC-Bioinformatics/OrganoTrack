import cv2 as cv
import numpy as np
from stitch import read_images, stitch
from functions import display

# reading image
dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/r2c2t0-all-fields'
fields = 25
wells = [('02', '02')]
channels = 1  # focus on brightfield channel for now
positions = 1  # only 1 position for brightfield image
img_set = read_images(dir, wells, positions, channels, fields)
ref = img_set[0][0][0][12]
display('ef', ref, 0.5)
corner = img_set[0][0][0][0]

mask_dir = '/home/franz/Documents/mep/data/25-field-outer-field-masks'
# reading mask
mask = cv.imread(mask_dir+"/field-0-mask.tiff")
mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)  # combine 3 channels into 1

# in mask, identify the indexes that are non-zero and
# get non-zero values from corner according to mask
corner_nonzero = [corner[np.nonzero(mask)]]
corner_nonzero_self = [corner[np.nonzero(corner)]]

# scaling by nonzero of mask
field_median = np.median(corner_nonzero)
ref_median = np.median(ref)
scale_median = ref_median/field_median
print(scale_median)
display('corner ', corner, 0.5)
corner = (corner*scale_median).astype(np.uint8)

# scaling by nonzero of corner itself
field_median_self = np.median(corner_nonzero_self)
ref_median = np.median(ref)
scale_median_self = ref_median/field_median_self
print(scale_median_self)
corner_self = (corner*scale_median_self).astype(np.uint8)


display('corner adjusted', corner, 0.5)
display('corner adjusted self', corner_self, 0.5)



cv.waitKey(0)

