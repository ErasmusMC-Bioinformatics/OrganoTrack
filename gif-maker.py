import imageio
import glob
import cv2 as cv

input_images = glob.glob("/home/franz/Documents/mep/data/preliminary-gt-dataset/d1*.tiff")

frames = []
for image_path in input_images:
    frames.append(cv.imread(image_path))

imageio.mimsave('/home/franz/Documents/mep/data/preliminary-gt-dataset/d1r1.gif', frames, format='GIF', duration=0.3)

