import cv2 as cv
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import matplotlib.pyplot as plt
import time



def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    return img
    # ax.imshow(img)


# # 10 x image - it works with this
# image_dir = '/home/franz/Downloads/10x.jpg'
# img = cv.imread(image_dir)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Real organoid image
image_dir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating/images/d0r1t3.tiff'
img = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)  # uint8
img = cv.cvtColor(img,cv.COLOR_GRAY2RGB)  # makes grayscale image RGB - same values



# # Very small organoid image
# image_dir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/2.images-with-edited-names-finished-annotating/images/small1.png'
# img = cv.imread(image_dir)


# Show original image
tic = time.process_time()
plt.figure(figsize=(20,20))
plt.imshow(img)
plt.axis('off')
plt.show()
toc = time.process_time() - tic
print(toc)
print('Original image shown')


# '''
#     Generating masks with Model H
# '''
#
# sam_checkpoint_h = "sam_vit_h_4b8939.pth"
# model_type_h = "vit_h"
#
# print('generating mask - model H')
# tic_H = time.time()
# sam = sam_model_registry[model_type_h](checkpoint=sam_checkpoint_h)
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(img)
# toc_H = time.time() - tic_H
# print('Time elapsed with model H (s): ', toc_H)
#
# # Show segmented image
# plt.figure(figsize=(20,20))
# plt.imshow(img)
# show_anns(masks)
# plt.axis('off')
# plt.show()
# print('Model H figure shown')




'''
    Generating masks with Model B
'''
sam_checkpoint_b = "sam_vit_b_01ec64.pth"
model_type_b = "vit_b"

print('generating mask - model B')
tic_B = time.process_time()
sam = sam_model_registry[model_type_b](checkpoint=sam_checkpoint_b)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
toc_B = time.process_time() - tic_B
print('Time elapsed with model B (s): ', toc_B)

# Show segmented image
plt.figure(figsize=(20,20))
plt.imshow(img)
show_anns(masks)
plt.axis('off')
plt.show()
print('Model B figure shown')


'''
    Generating masks with Model L
'''
sam_checkpoint_l = "sam_vit_l_0b3195.pth"
model_type_l = "vit_l"

print('generating mask - model L')
tic_L = time.time()
sam = sam_model_registry[model_type_l](checkpoint=sam_checkpoint_l)
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)
toc_L = time.time() - tic_L
print('Time elapsed with model L (s): ', toc_L)

# Show segmented image
plt.figure(figsize=(20,20))
plt.imshow(img)
show_anns(masks)
plt.axis('off')
plt.show()
print('Model L figure shown')


sam_output = [masks[x]['segmentation'] for x in range(len(masks))]
sum_sam_output = sum(sam_output)
m2 = np.zeros(np.max(sum_sam_output) + 1)
m2[2:] = 1
binary_sam_output = m2[sum_sam_output]
cv.imshow('sam output', binary_sam_output)
cv.waitKey(0)


print('debug here')
