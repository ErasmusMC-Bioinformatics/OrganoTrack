import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

modelCheckpoints = {'vit_b': 'Experimental-building-of-functionalities/sam_vit_b_01ec64.pth',
                    'vit_h': 'sam_vit_h_4b8939.pth',
                    'vit_l': 'sam_vit_l_0b3195.pth'}



def show_anns(objectsSegmentedBySAM):
    if len(objectsSegmentedBySAM) == 0:
        return
    sortedSAMobjects = sorted(objectsSegmentedBySAM, key=(lambda x: x['area']), reverse=True)  # in decreasing area

    ax = plt.gca()
    ax.set_autoscale_on(False)

    imgHeight = sortedSAMobjects[0]['segmentation'].shape[0]  # look at just the first object
    imgWidth = sortedSAMobjects[0]['segmentation'].shape[1]
    colorObjectMasks = np.ones((imgHeight, imgWidth, 4))  # 3D array, height x width x 4. i.e. a stack of 4 images

    colorObjectMasks[:, :, 3] = 0              # the last image in the stack = 0
    for samObject in sortedSAMobjects:     #
        samObjectSegmentataionArray = samObject['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])  # [ 0-1 , 0-1 , 0-1 , 0.35 ]
        colorObjectMasks[samObjectSegmentataionArray] = color_mask
    return colorObjectMasks



def SegmentBySAM(image, model, modelCheckpoint):
    sam = sam_model_registry[model](checkpoint=modelCheckpoint)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image)


def main_loop():
    st.title("SAM Annotator")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff', 'tif'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    original_image = (original_image / 256).astype('uint8')
    model = list(modelCheckpoints.keys())[0]

    image_file = cv.cvtColor(original_image, cv.COLOR_GRAY2RGB)

    mask = SegmentBySAM(image_file, model, modelCheckpoints[model]) # SAM expects a 3 channel image
    # mask = list of dicts. Each dict element belongs to one segmented object
    # mask[0] = {'segmentation': array,
    #            'area': e.g. 1539
    #            'bbox': e.g. [0, 736, 51, 36]
    #            'predicted_iou': e.g. 1.0169870853424072
    #            'point_coords': e.g. [[16.875, 759.375]]
    #            'stability_score': e.g. 0.9896439909934998
    #            'crop_box': e.g. [0, 0, 1080, 1080]

    # mask[0]['segmentation'] = boolean array of object, array with dimensions of image

    plt.figure(figsize=(20, 20))
    plt.imshow(image_file)
    output = show_anns(mask)
    plt.axis('off')
    plt.show()

    st.text("Original Image vs Processed Image")
    st.image([original_image, output], width=600)


if __name__ == '__main__':
    main_loop()

