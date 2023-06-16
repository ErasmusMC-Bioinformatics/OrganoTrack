import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

model_checkpoints = {'vit_b': 'Experimental-building-of-functionalities/sam_vit_b_01ec64.pth'}


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
    ax.imshow(colorObjectMasks)


def segment_with_sam(image: np.ndarray, model: str, modelCheckpoint: str):
    sam = sam_model_registry[model](checkpoint=modelCheckpoint)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image)


def ensure_8_bit(image_16_bit):  # for now, assumes 16 bit and grayscale
    image_16_bit = image_16_bit.convert("I")
    image_16_bit = np.array(image_16_bit)
    image_8_bit = np.uint8(image_16_bit / np.max(image_16_bit) * 255)
    return Image.fromarray(image_8_bit)


def main_loop():
    st.title("SAM Annotator")

    # Importing image
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff', 'tif'])
    if not image_file:
        return
    image = Image.open(image_file)
    image = ensure_8_bit(image)
    image_array = np.array(image)

    # Segmenting with SAM
    image_bgr = cv.cvtColor(image_array, cv.COLOR_GRAY2RGB)
    model = 'vit_b'
    sam_segmentation = segment_with_sam(image_bgr, model, model_checkpoints[model])  # SAM expects a 3 channel image
    # mask = list of dicts. Each dict element belongs to one segmented object
    # mask[0] = {'segmentation': array,
    #            'area': e.g. 1539
    #            'bbox': e.g. [0, 736, 51, 36]
    #            'predicted_iou': e.g. 1.0169870853424072
    #            'point_coords': e.g. [[16.875, 759.375]]
    #            'stability_score': e.g. 0.9896439909934998
    #            'crop_box': e.g. [0, 0, 1080, 1080]

    # mask[0]['segmentation'] = boolean array of object, array with dimensions of image

    fig, ax = plt.subplots(figsize=(20, 20))
    plt.imshow(image_array)
    show_anns(sam_segmentation)
    st.pyplot(fig)


if __name__ == '__main__':
    main_loop()
