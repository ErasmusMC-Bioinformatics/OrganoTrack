import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from functions import rescale, plotHistogram, display, blur

display_scale = 0.5

# Reading the image
gt_dir = '/home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/data/preliminary-gt-dataset/annotated'


gt_img = cv.imread(gt_dir+"/d0r1t0-unmerged.png", cv.IMREAD_GRAYSCALE)
_, gt_img = cv.threshold(gt_img, 50, 255,cv.THRESH_BINARY)
display('ground truth', gt_img, display_scale)
print(np.shape(gt_img))

img_dir = '/home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/image-analysis-pipelines/OrganoSeg/code/OrganoSeg'
img = cv.imread(img_dir+"/11.png", cv.IMREAD_GRAYSCALE)
display('organoseg', img, display_scale)
print(np.shape(img))

'''
    Evaluation: pseudo code
    
    GT_objects = number of objects in GT image
    
    for each object in range(GT_objects)
        if there is an overlapping object in segmented image
            1) TP_count++
            2) Calculate IOU. Store IOU for that (GT) object
            3) Calculate Dice score. Store for that (GT) object
            4) Remove that object from segmented image --> temp segmented
        else
            FN_count++
    
    FP_count = number of objects remaining in temp_segmented
    
    F1_score = 2*TP_count/(2*TP_count + FP_count + FN_count)
'''





cv.waitKey(0)
