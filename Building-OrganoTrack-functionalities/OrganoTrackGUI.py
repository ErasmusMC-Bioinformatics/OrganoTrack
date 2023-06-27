import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from pathlib import Path

def main_loop():
    st.title("OrganoTrack")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff', 'tif'])
    if not image_file:
        return None


    original_image = Image.open(image_file)
    original_image = np.array(original_image)

    # Segmentation parameters
    extraBlur = False
    blurSize = 3
    segParams = [0.5, 250, 150, extraBlur, blurSize]
    exportPath = Path(
        '/home/franz/Documents/mep/data/for-creating-OrganoTrack/testing-OrganoTrack-full/export/segmented')
    saveSegParams = [False, exportPath]
    segmented = SegmentWithOrganoSegPy([original_image], segParams, saveSegParams)

    plt.figure(figsize=(20, 20))
    plt.imshow(image_file)
    plt.axis('off')
    plt.show()


    st.text("Original Image vs Processed Image")
    st.image([original_image, segmented[0]])


if __name__ == '__main__':
    main_loop()
