import cv2 as cv
import numpy as np
from functions import rescale, plotHistogram, display, blur

display_scale = 0.5

# Reading the image
gt_dir = "G:/My Drive/mep/data/preliminary-gt-dataset/annotated/d0r1t0-unmerged.png"

img_gt = cv.imread(gt_dir, cv.IMREAD_GRAYSCALE)      # grayscale
_, img_gt = cv.threshold(img_gt, 50, 255, cv.THRESH_BINARY)                 # binary (0 or 255)
img_gt = img_gt / 255                                                         # binary (0 or 1)
display('ground truth', img_gt, 0.5)

pred_dir = "G:/My Drive/mep/data/preliminary-gt-dataset/skimage-new.png"
img_pred = cv.imread(pred_dir, cv.IMREAD_GRAYSCALE)
_, img_pred = cv.threshold(img_pred, 50, 255, cv.THRESH_BINARY)
img_pred = img_pred / 255
display('prediction', img_pred, 0.5)

img_sum = cv.add(img_pred, img_gt)                       # 0 or 1 or 2
display('sum', img_sum, 0.5)
tp_count = np.count_nonzero(img_sum == 2)
tn_count = np.count_nonzero(img_sum == 0)

img_or = cv.bitwise_or(img_pred, img_gt)             # 0 or 1, not 2
display('img or', img_or, 0.5)

img_fp = cv.subtract(img_or, img_gt)      # 0 or 1, not 2
fp_count = np.count_nonzero(img_fp == 1)
display('false positives', img_fp, 0.5)

img_fn = cv.subtract(img_or, img_pred)      # 0 or 1, not 2
fn_count = np.count_nonzero(img_fn == 1)
display('false negatives', img_fn, 0.5)


# F1 Score
F1_score = 2 * tp_count / (2 * tp_count + fp_count + fn_count)
print('F1 score: ' + str(np.ceil(F1_score*100)))

# IOU Score
iou_score = tp_count/np.count_nonzero(img_or == 1)
print('IOU score: ' + str(np.ceil(iou_score*100)))

# Dice score
dice_score = 2*tp_count/(np.count_nonzero(img_pred == 1) + np.count_nonzero(img_gt == 1))
print('Dice score: ' + str(np.ceil(dice_score*100)))

cv.waitKey(0)