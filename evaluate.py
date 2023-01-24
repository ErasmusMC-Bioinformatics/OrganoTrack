import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from functions import rescale, plotHistogram, display, blur

display_scale = 0.5

# Reading the image
gt_dir = '/home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/data/preliminary-gt-dataset/annotated'

gt_img = cv.imread(gt_dir + "/d0r1t0-unmerged.png", cv.IMREAD_GRAYSCALE)
_, gt_img = cv.threshold(gt_img, 50, 255, cv.THRESH_BINARY)
display('ground truth', gt_img, display_scale)
print(np.shape(gt_img))

img_dir = '/home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/image-analysis-pipelines/OrganoSeg/code/OrganoSeg'
img = cv.imread(img_dir + "/11.png", cv.IMREAD_GRAYSCALE)
display('organoseg', img, display_scale)
print(np.shape(img))

'''
    Evaluation: pseudo code

    Method_being_evaluated = 'OrganoTrack'
    Method_IOU = []
    Method_F1 = 0
    Method_Dice = []

    GT_image = Import GT image
    temp_segmented = segmented_image
    remove_border_objects(temp_segmented)

    GT_objects = []
    Parse through the GT image
        for each identified object in GT image
        create new organoid object
            store index of image file where object is found
            store pixel coordinates of object in that indexed image   
        store organoid object in list GT_objects

    number_of_GT_objects = len(GT_objects)

    for each object in GT_objects
        if there is an object in segmented image with any common px coordinates
            Get px coordinates of the Seg object that overlaps

            # F1 score
            TP_count++

            # IOU
            numerator = number of common px coordinates between GT object and Seg object
            denominator = len(non-doubled combine of px coordinates between GT object and Seg object)
            IOU = numerator / denominator
            Method_IOU.append(IOU)

            # Dice score
            dice_numerator = 2*numerator
            dice_denominator = number of px's in GT object + number of px's in Seg object
            dice = dice_numerator/dice_denominator
            Method_dice.append(dice)

            # Remove object from segmented image
            temp_segmented[px of segmented object] = 0

        else  # there is not an overlapping object in segmented image
            FN_count++

    FP_count = number of objects remaining in temp_segmented

    F1_score = 2*TP_count/(2*TP_count + FP_count + FN_count)
'''

cv.waitKey(0)
