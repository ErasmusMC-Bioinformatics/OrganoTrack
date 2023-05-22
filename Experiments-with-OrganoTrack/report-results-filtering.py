from OrganoTrack.Detecting import SegmentWithOrganoSegPy
from OrganoTrack.Importing import ReadImages
from OrganoTrack.Displaying import ExportImageWithContours, Display
from OrganoTrack.Filtering import FilterByFeature, RemoveBoundaryObjects

from pathlib import Path
import cv2 as cv
from matplotlib.pyplot import plt
import pandas as pd
import skimage
import numpy as np


def PlotRegions(binaryImage):
    pass
    # if i == 0:  # area first
    #     labeledImages = [Label(image) for image in imagesInAnalysis]
    #     propertyToMeasure = ['roundness']
    #     propertyAndTimeDFs = []
    #     for propertyName in propertyToMeasure:
    #         timeDFs = []
    #         for k in range(len(labeledImages)):  # for each time point
    #
    #             size = (np.max(labeledImages[k]) + 1, 1)
    #             data = pd.DataFrame(np.ndarray(size, dtype=str))
    #
    #             regions = skimage.measure.regionprops(labeledImages[k])
    #             for region in regions:
    #                 if propertyName == 'roundness':
    #                     value = CalculateRoundness(getattr(region, 'area'), getattr(region, 'perimeter'))
    #                 else:
    #                     value = getattr(region, propertyName)
    #                 label = region.label
    #                 data.iloc[label, 0] = str(value)
    #             timeDFs.append(data)
    #         propertyAndTimeDFs.append(timeDFs)
    #
    #     roundnessMeasurements = [df.loc[1:, 0].values.tolist() for df in propertyAndTimeDFs[0]]
    #     roundnessMeasurementsFloat = [[float(i) for i in df] for df in roundnessMeasurements]
    #
    #     ys = []
    #     for count in range(len(roundnessMeasurementsFloat)):  # willm need to revaluate after each filtering
    #         ys.append(np.random.normal(count + 1, 0.04, len(roundnessMeasurementsFloat[count])))
    #
    #     # Plotting area
    #     plt.rcParams.update({'font.size': 15})
    #     fig4, ax4 = plt.subplots()
    #     ax4.boxplot(roundnessMeasurementsFloat, labels=plottingConditions, showfliers=False)
    #     ax4.set_ylabel('Roundness (a.u.)')
    #     ax4.set_ylim([0, 1])
    #     ax4.set_xlabel('Days after seeding')
    #     ax4.set_title('Organoid roundness at each time point')
    #     palette2 = ['b', 'g', 'r', 'c']
    #     for y, val2, c in zip(ys, roundnessMeasurementsFloat, palette2):
    #         ax4.scatter(y, val2, alpha=0.4, color=c)
    #     plt.tight_layout()
    #     fig4.show()

# Set directories
imagesDir = Path('/home/franz/Documents/mep/report/results/filtering/input')
exportPath = Path('/home/franz/Documents/mep/report/results/filtering')

# Import images
images, imagesPaths = ReadImages(imagesDir)

# Get segmentations
extraBlur = False
blurSize = 3
displaySegmentationSteps = False
segParams = [0.5, 250, 150, extraBlur, blurSize, displaySegmentationSteps]
saveSegParams = [False, exportPath, imagesPaths]
imagesInAnalysis = SegmentWithOrganoSegPy(images, segParams, saveSegParams)

# Display pre-filter
overlayed = ExportImageWithContours(images[0], imagesInAnalysis[0])
Display('pre filtering', overlayed, 0.5)

# Filter by features
filterBy = {'area': 450, 'roundness': 0.4}
for filterProp in list(filterBy.keys()):
    imagesInAnalysis = FilterByFeature(imagesInAnalysis, filterProp, filterBy[filterProp])
    filteredOverlay = ExportImageWithContours(images[0], imagesInAnalysis[0])
    Display(f'filtered by {filterProp}', filteredOverlay, 0.5)

# Filter by location
imagesInAnalysis = RemoveBoundaryObjects(imagesInAnalysis)

# Display final
filteredOverlay = ExportImageWithContours(images[0], imagesInAnalysis[0])
Display('post filtering', filteredOverlay, 0.5)
cv.waitKey(0)
