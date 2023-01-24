import cv2 as cv
from functions import display
from scipy import stats

dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/only-control-and-cis-images-field-1'

img = cv.imread(dir+"/r02c02f01p01-ch1sk1fk1fl1.tiff")  # (1080, 1080, 3)

m = stats.mode(img)
print(m[0])

# grayscale_values = []
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         grayscale_values.append(img[i, j])
#
# print(grayscale_values)
# display('hello', img, 0.5)
# running the code will take some time.
# example output: array([105, 105, 105], dtype=uint8), array([105, 105, 105], dtype=uint8),
# rray([104, 104, 104], dtype=uint8), array([105, 105, 105], dtype=uint8), array([105, 105, 105],
# dtype=uint8), array([106, 106, 106], dtype=uint8), array([108, 108, 108], dtype=uint8), array([107, 107, 107],
# dtype=uint8), array([107, 107, 107], dtype=uint8), array([106, 106, 106], dtype=uint8), array([106, 106, 106],
# dtype=uint8), array([107, 107, 107], dtype=uint8), array([106, 106, 106], dtype=uint8), array([106, 106, 106],
# dtype=uint8), array([105, 105, 105], dtype=uint8), array([104, 104, 104], dtype=uint8), array([104, 104, 104],
# dtype=uint8), array([106, 106, 106], dtype=uint8), array([107, 107, 107], dtype=uint8), array([108, 108, 108]