import os
from datetime import datetime
import cv2 as cv


def SaveData(parentDataDir, images, imageNames):
    '''
    :param inputDataDir: parent directory where image data will be stored
    :param images: image data
    :param imageNames: image names for storage
    '''

    # > Create a unique daughter path for storage
    dateTimeNow = datetime.now()
    storagePath = parentDataDir + '/segmented-' + dateTimeNow.strftime('%d.%m.%Y-%H_%M_%S')
    os.mkdir(storagePath)

    # > Store
    for i in range(len(images)):
        cv.imwrite(storagePath + '/' + imageNames[i], images[i])