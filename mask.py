import cv2 as cv
import numpy as np
from functions import rescale, display


dir = '/home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/data/mask'

organoid_mask = cv.imread(dir+'/d0r1t0_organoid.tiff')

organoid_mask = cv.cvtColor(organoid_mask, cv.COLOR_BGR2GRAY)
thresh, organoid_mask = cv.threshold(organoid_mask, 100, 255, cv.THRESH_BINARY)
print(np.shape(organoid_mask))

true_mask = cv.imread(dir+'/d0r1t0-true.png')
true_mask = cv.cvtColor(true_mask, cv.COLOR_BGR2GRAY)
thresh, true_mask = cv.threshold(true_mask, 10, 255, cv.THRESH_BINARY)
print(np.shape(organoid_mask))

img = cv.imread(dir+'/d0r1t0.tiff')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(np.shape(img))

organoid_masked = cv.bitwise_and(img,img,mask=organoid_mask)
display('OrganoID', organoid_masked, 0.5)

true_masked = cv.bitwise_and(img,img,mask=true_mask)
display('OGround trueh', true_masked, 0.5)

cv.waitKey(0)