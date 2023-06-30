from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import ExportImageWithContours, Display, DisplayImages
from OrganoTrack.Filtering import FilterByFeature, RemoveBoundaryObjects
from OrganoTrack.Measuring import CalculateRoundness

from pathlib import Path
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
from skimage import measure, filters, morphology
import plotly
import plotly.express as px
import plotly.graph_objects as go

def Label(image):
    labeled = skimage.measure.label(image)
    return labeled


def CreatePropertyDF(labelledImage, property):
    dfSize = (np.max(labelledImage) + 1, 1)
    propertyDF = pd.DataFrame(np.ndarray(dfSize, dtype=str))

    imageObjects = skimage.measure.regionprops(labelledImage)
    for object in imageObjects:
        if property == 'roundness':
            propertyValue = CalculateRoundness(getattr(object, 'area'), getattr(object, 'perimeter'))
        else:
            propertyValue = getattr(object, property)
        objectLabel = object.label
        propertyDF.iloc[objectLabel, 0] = str(propertyValue)

    return propertyDF

def PlotBoxplotWithJitter(valuesDF, property, unit):
    valuesStrings = valuesDF.loc[1:, 0].values.tolist()
    valuesFloats = [float(value) for value in valuesStrings]
    valuesJitter = np.random.normal(1, 0.04, len(valuesFloats))

    plt.rcParams.update({'font.size': 20})
    plt.rcParams['figure.figsize'] = (3, 4)
    fig, ax = plt.subplots()
    ax.boxplot(valuesFloats, showfliers=False, widths=0.6)
    # ax.set_ylabel(f'{property.capitalize()} ({unit})')
    colourPalette = ['b', 'g', 'r', 'c']
    if property != 'area':
        ax.set_ylim([0, 1])
    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)
    ax.scatter(valuesJitter, valuesFloats, alpha=0.4, color=colourPalette[0])
    plt.tight_layout()
    plt.show()


def PlotPropertyBoxplot(binaryImage, property, unit):
    labelledImage = Label(binaryImage)

    propertyDF = CreatePropertyDF(labelledImage, property)

    PlotBoxplotWithJitter(propertyDF, property, unit)


def PlotWithPlotly(original, binary):
    img = original
    labels = Label(binary)

    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip')  # hover is only for label info

    props = measure.regionprops(labels, img)
    properties = ['area', 'eccentricity', 'solidity', 'perimeter']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labels.max()):
        label_i = props[index].label
        contour = measure.find_contours(labels == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)



# Set directories
imagesDir = Path('/home/franz/Documents/mep/report/results/measurement/input')
exportPath = Path('/home/franz/Documents/mep/report/results/measurement/output')
segmentedExportPath = exportPath / 'OrganoTrack-segmented'

# Import images
rawImages, imagesPaths = ReadImages(imagesDir)

# Get segmentations
if not os.path.exists(segmentedExportPath):
    extraBlur = False
    blurSize = 3
    displaySegmentationSteps = False
    segParams = [0.5, 250, 150, extraBlur, blurSize, displaySegmentationSteps]
    saveSegParams = [True, exportPath, imagesPaths]
    predictedImages = SegmentWithOrganoSegPy(rawImages, segParams, saveSegParams)
else:
    predictedImages, imagesPaths = ReadImages(segmentedExportPath)

# Display overlays
for i, (raw, prediction) in enumerate(zip(rawImages, predictedImages)):
    overlayed = ExportImageWithContours(raw, prediction)
    Display(str(i), overlayed, 1)
    # PlotWithPlotly(raw, prediction)


# PlotPropertyBoxplot(predictedImages[3], 'area', 'pixels')

cv.waitKey(0)