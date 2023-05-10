from final.Detecting import SegmentWithOrganoSegPy
from final.Displaying import Display
import cv2 as cv

dir = "/home/franz/Documents/mep/data/experiments/2023-02-24-Cis-Tos-dataset-mathijs/AZh5/Day-12"
imageName = "/D7 1.tif"

img = cv.imread(dir+imageName, cv.IMREAD_GRAYSCALE)
Display('ori', img, 0.25)
SegmentWithOrganoSegPy([img], False, None, None)
cv.waitKey(0)