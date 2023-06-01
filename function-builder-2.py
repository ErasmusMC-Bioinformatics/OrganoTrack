import cv2 as cv
from functions import display
from stitch import automatic_brightness_and_contrast

dir = '/home/franz/Documents/mep/data/organoid-images/221223-Staining-trial-OrganoTrack-BT-FT-MvR/221223-plate-1__2022-12-23T10_46_22-Measurement 1/hoechst-c4/r04c09'
fields = 9
wells = [('04', '09')]
channels = 4

img = cv.imread(dir +
                 '/r' + wells[0][0] +
                 'c' + wells[0][1] +
                 'f01' +
                 'p01-ch1' +
                 'sk1fk1fl1.tiff')

display('img', img, 0.25)

enhance_list = [0.25, 0.35]

for i in range(len(enhance_list)):
    auto_img, alpha, beta = automatic_brightness_and_contrast(img, enhance_list[i])
    display('auto img, enhance = ' + str(enhance_list[i]), auto_img, 0.75)
cv.waitKey(0)