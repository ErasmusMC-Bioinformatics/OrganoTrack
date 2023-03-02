import matplotlib.ticker

from organotrack import organotrack
from scratch import *
from functions import display
from segmentation import segment
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

originalImages, originalDFs, analysedImages, objectFeatures = organotrack(experimentInfo)
# analysed Images = filteredImages
# Object Features = filteredDFs

display('filterd', analysedImages[0], 0.25)
cv.waitKey(0)



imageCondition = ['Ctrl',
                  'Cis 5 ',
                  'Tos .01',
                  'Cis 2',
                  'Cis 5+\nTos .01',
                  'Cis 2+\nTos .01']
#
newOrder = [1, 5, 2, 3, 6, 4]
sortedImageCondition = [x for _, x in sorted(zip(newOrder, imageCondition))]
sortedAnalysedImages = [x for _, x in sorted(zip(newOrder, analysedImages))]
sortedObjectFeatures = [x for _, x in sorted(zip(newOrder, objectFeatures))]

# Getting area values into a list of lists
areaMeasurements, xs, circularityMeasurements = [], [], []
for i in range(len(sortedObjectFeatures)):
    circularityMeasurements.append(sortedObjectFeatures[i]['Circularity'].tolist())
    areaMeasurements.append(sortedObjectFeatures[i]['Area'].tolist())
    xs.append(np.random.normal(i + 1, 0.04, sortedObjectFeatures[i]['Area'].values.shape[0]))
#
# # Converting list of lists into array of arrays for summing
# y = np.array([np.array(xi) for xi in areaMeasurements], dtype=object)
# areaSums = [np.sum(i) for i in y]
# plt.rcParams.update({'font.size': 15})
# # Object areas per condition
# fig, ax = plt.subplots()
# ax.boxplot(areaMeasurements, labels=sortedImageCondition, showfliers=False)
# ax.set_ylabel(r'Area ($pixels^2$)')
# ax.set_xlabel(r'Condition concentration ($\mu$M)')
# ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
# ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# ax.set_title('Object area per condition')
# palette = ['b', 'g', 'r', 'c', 'm', 'k']
# for x, val, c in zip(xs, areaMeasurements, palette):
#     ax.scatter(x, val, alpha=0.4, color=c)
# plt.tight_layout()
# fig.show()
#
# # Total area per condition
# fig1, ax1 = plt.subplots()
# ax1.bar(sortedImageCondition, areaSums)
# ax1.set_ylabel(r'Area ($pixels^2$)')
# ax1.set_xlabel(r'Condition concentration ($\mu$M)')
# # ax.ticklabel_format(axis='y', style='sci', useMathText=True)
# ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
# ax1.set_title('Total object area per condition')
# plt.tight_layout()
# fig1.show()
#
# # Total count per condition
# objectCount = np.array([len(i) for i in areaMeasurements])
# fig2, ax2 = plt.subplots()
# ax2.bar(sortedImageCondition, objectCount)
# ax2bars = ax.bar(sortedImageCondition, objectCount)
# ax2.bar_label(ax2bars)
# ax2.set_ylabel('Object count')
# ax2.set_ylim([0, 25])
# ax2.set_xlabel(r'Condition concentration ($\mu$M)')
# ax.set_xticklabels(objectCount.astype(int))
# ax2.set_title('Total object count per condition')
# plt.tight_layout()
# fig2.show()
#
# Circularity per condition
fig3, ax3 = plt.subplots()
ax3.boxplot(circularityMeasurements, labels=sortedImageCondition, showfliers=False)
ax3.set_ylabel('Circularity (a.u.)')
ax3.set_ylim([0, 1])
ax3.set_xlabel(r'Condition concentration ($\mu$M)')
ax3.set_title('Object circularity per condition')
palette = ['b', 'g', 'r', 'c', 'm', 'k']
for x, val, c in zip(xs, circularityMeasurements, palette):
    ax3.scatter(x, val, alpha=0.4, color=c)
plt.tight_layout()
fig3.show()





# call organotrack to analyse data
    # receive filtered images and included object DFs

# plot data

# /home/franz/Insync/ftapiac.96@gmail.com/Google Drive/mep/image-analysis-pipelines/OrganoTrack/code/scratch.txt