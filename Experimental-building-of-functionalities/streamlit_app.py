import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

modelCheckpoints = {'vit_b': '/home/franz/OrganoTrack/Experimental-building-of-functionalities/sam_vit_b_01ec64.pth',
                    'vit_h': 'sam_vit_h_4b8939.pth',
                    'vit_l': 'sam_vit_l_0b3195.pth'}



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

def SegmentBySAM(image, model, modelCheckpoint):
    sam = sam_model_registry[model](checkpoint=modelCheckpoint)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image)


def main_loop():
    st.title("Sam Annotator")
    st.subheader("This app allows you to segment an image and generate a binary image with SAM")
    st.text("We use OpenCV, the Segment Anythin Model, and Streamlit for this demo")

    # blur_rate = st.sidebar.slider("Blurring", min_value=0.5, max_value=3.5)
    # brightness_amount = st.sidebar.slider("Brightness", min_value=-50, max_value=50, value=0)
    # apply_enhancement_filter = st.sidebar.checkbox('Enhance Details')

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff', 'tif'])
    if not image_file:
        return None


    original_image = Image.open(image_file)
    original_image = np.array(original_image)
    st.markdown('Type of image before uint8 cnoversion: ' + str(type(original_image)))
    st.markdown('Printing the array: '+ str((original_image)))
    st.markdown('Getting the image shape: ' + str(np.shape(original_image)))

    original_image = (original_image / 256).astype('uint8')
    st.markdown('Type of image after uint8 cnoversion: '  + str(type(original_image)))
    st.markdown('Printing the array: '+ str((original_image)))
    st.markdown('Getting the image shape: ' + str(np.shape(original_image)))
    model = list(modelCheckpoints.keys())[0]

    image_file = cv.cvtColor(original_image, cv.COLOR_GRAY2RGB)
    mask = SegmentBySAM(image_file, model, modelCheckpoints[model]) # SAM expects a 3 channel image

    plt.figure(figsize=(20, 20))
    plt.imshow(image_file)
    output = show_anns(mask)
    plt.axis('off')
    plt.show()

    # processed_image = blur_image(original_image, blur_rate)
    # processed_image = brighten_image(processed_image, brightness_amount)
    #
    # if apply_enhancement_filter:
    #     processed_image = enhance_details(processed_image)

    st.text("Original Image vs Processed Image")
    st.image([original_image, output])


if __name__ == '__main__':
    main_loop()
