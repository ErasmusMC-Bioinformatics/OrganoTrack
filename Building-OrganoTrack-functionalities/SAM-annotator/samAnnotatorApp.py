import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from streamlit_image_coordinates import streamlit_image_coordinates
import base64
from io import BytesIO


model_checkpoints = {'vit_b': '/home/franz/OrganoTrack/Building-OrganoTrack-functionalities/sam_vit_b_01ec64.pth'}

st.set_page_config(
    page_title="SAM Annotator",
    layout="wide",
)

def create_rgba_array(objectsSegmentedBySAM):
    if len(objectsSegmentedBySAM) == 0:
        return
    sortedSAMobjects = sorted(objectsSegmentedBySAM, key=(lambda x: x['area']), reverse=True)  # in decreasing area

    ax = plt.gca()
    ax.set_autoscale_on(False)

    imgHeight = sortedSAMobjects[0]['segmentation'].shape[0]  # look at just the first object
    imgWidth = sortedSAMobjects[0]['segmentation'].shape[1]
    colorObjectMasks = np.ones((imgHeight, imgWidth, 4), dtype=np.uint8)  # 3D array, height x width x 4. i.e. a stack of 4 images

    colorObjectMasks[:, :, 3] = 0              # the last image in the stack = 0
    for samObject in sortedSAMobjects:     #
        samObjectSegmentataionArray = samObject['segmentation']
        color_mask = np.concatenate([np.random.randint(255, size=3), [90]])
        colorObjectMasks[samObjectSegmentataionArray] = color_mask
    return colorObjectMasks


def segment_with_sam(image: np.ndarray):
    image_bgr = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    model = 'vit_b'
    modelCheckpoint = model_checkpoints[model]
    sam = sam_model_registry[model](checkpoint=modelCheckpoint)
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator.generate(image_bgr)


def ensure_8_bit(image_16_bit):  # for now, assumes 16 bit and grayscale
    image_16_bit = image_16_bit.convert("I")
    image_16_bit = np.array(image_16_bit)
    image_8_bit = np.uint8(image_16_bit / np.max(image_16_bit) * 255)
    return Image.fromarray(image_8_bit)


def remove_clicked_object(sam_output, coordinates):
    x = coordinates['y']
    y = coordinates['x']
    for i, sam_object in enumerate(sam_output):
        object_array = sam_object['segmentation']
        if object_array[x][y]:
            del sam_output[i]  # this removes the first object that hits, though the clicked object may overlap with
            break              # the hit object, and you remove the unintended hit object
    return sam_output


def binarise_sam_output(sam_output):
    object_list_boolean = [sam_output[seg_object]['segmentation'] for seg_object in range(len(sam_output))]
    sumOfImageRegions = sum(object_list_boolean)
    summedBooleanToBinaryMapping = np.zeros(np.max(sumOfImageRegions) + 1)

    summedBooleanToBinaryMapping[1:] = 1  # assuming model vit_b

    binarySamMask = summedBooleanToBinaryMapping[sumOfImageRegions]
    binarySamMask = binarySamMask.astype(np.uint8)
    _, binarySamMask = cv.threshold(binarySamMask, 0, 255, cv.THRESH_BINARY)  # anything more than 0 becomes full
    return binarySamMask

# function by Spidy20 at https://discuss.streamlit.io/t/how-to-download-image/3358/2
def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def main_loop():
    st.title("SAM Annotator.")

    # Import image
    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff', 'tif'])
    if not image_file:
        return
    image = Image.open(image_file)
    image = ensure_8_bit(image)
    image_array = np.array(image)
    image_rgba_pil = image.convert("RGBA")

    # Segment image with SAM
    sam_output = segment_with_sam(image_array)

    # creating a single-element container
    placeholder = st.empty()
    i = 0
    while True:
        # Create image overlay in RGBA PIL format
        sam_seg_rgba_array = create_rgba_array(sam_output)
        sam_seg_rgba_pil = Image.fromarray(sam_seg_rgba_array)
        image_overlay_pil = Image.alpha_composite(image_rgba_pil, sam_seg_rgba_pil)
        with placeholder.container():
            # Receive click
            value = streamlit_image_coordinates(image_overlay_pil, key=f'pil{i}')
            st.write(value)
            sam_output = remove_clicked_object(sam_output, value)
            binary_sam_mask = binarise_sam_output(sam_output)
        st.image(binary_sam_mask)
        ## Original image came from cv2 format, fromarray convert into PIL format
        result = Image.fromarray(binary_sam_mask)
        st.markdown(get_image_download_link(result, 'binary.png', 'Download ' + 'binary.png'), unsafe_allow_html=True)

        i += 1




    # sam_output_new = remove_clicked_object(sam_output, value)
    #
    # # Create image overlay in RGBA PIL format
    #
    # sam_seg_rgba_array_new = create_rgba_array(sam_output_new)
    # sam_seg_rgba_pil_new = Image.fromarray(sam_seg_rgba_array_new)
    # image_overlay_pil_new = Image.alpha_composite(image_rgba_pil, sam_seg_rgba_pil_new)
    #
    # # Plot for coordinate clicking
    # value = streamlit_image_coordinates(image_overlay_pil_new, key='pil2')
    # st.write(value)


def click_remover():
    image_path = '/home/franz/d2r1t3.tiff'
    image_array = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image_pil = Image.fromarray(image_array)

    # Segment image with SAM
    image_bgr = cv.cvtColor(image_array, cv.COLOR_GRAY2RGB)
    model = 'vit_b'
    sam_output = segment_with_sam(image_bgr, model, model_checkpoints[model])

    # Create image overlay in RGBA PIL format
    image_rgba_pil = image_pil.convert("RGBA")
    sam_seg_rgba_array = create_rgba_array(sam_output)
    sam_seg_rgba_pil = Image.fromarray(sam_seg_rgba_array)
    image_overlay_pil = Image.alpha_composite(image_rgba_pil, sam_seg_rgba_pil)

    # Take coordinate
    x = 906  # switch coordinates
    y = 966
    for i, sam_object in enumerate(sam_output):
        object_array = sam_object['segmentation']
        if object_array[x][y]:
            del sam_output[i]
            break


if __name__ == '__main__':
    main_loop()
    # click_remover()
