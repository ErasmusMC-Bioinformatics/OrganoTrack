import imageio
import glob
import cv2 as cv

dataDir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/making-gif'
input_images = glob.glob(dataDir + '/*.png')
input_images = sorted(input_images)
frames = []
for image_path in input_images:
    frames.append(cv.imread(image_path))


imageio.mimsave(dataDir + '/segmentation.gif', frames, format='GIF', duration=0.5)

print('hello')