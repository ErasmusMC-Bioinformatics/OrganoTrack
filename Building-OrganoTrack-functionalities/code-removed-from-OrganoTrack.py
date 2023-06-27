'''
    Code after Segmentation
'''

# plottingProperties = ['area', 'roundness']
# plottingConditions = ['1', '3', '5', '7']
#
#
# labeledImages = [Label(image) for image in imagesInAnalysis]
#
# propertyAndTimeDFs = []
# for propertyName in plottingProperties:
#     timeDFs = []
#     for i in range(len(labeledImages)):     # for each time point
#
#         size = (np.max(labeledImages[i]) + 1, 1)
#         data = pd.DataFrame(np.ndarray(size, dtype=str))
#
#         regions = skimage.measure.regionprops(labeledImages[i])
#         for region in regions:
#             if propertyName == 'roundness':
#                 value = CalculateRoundness(getattr(region, 'area'), getattr(region, 'perimeter'))
#             else:
#                 value = getattr(region, propertyName)
#             label = region.label
#             data.iloc[label, 0] = str(value)
#         timeDFs.append(data)
#     propertyAndTimeDFs.append(timeDFs)
#
#
#
# areaMeasurements = [df.loc[1:,0].values.tolist() for df in propertyAndTimeDFs[0]]
# areaMeasurementsFloat = [[int(i) for i in df] for df in areaMeasurements]
#
# xs = []
# for i in range(len(areaMeasurementsFloat)):  # willm need to revaluate after each filtering
#     xs.append(np.random.normal(i + 1, 0.04, len(areaMeasurementsFloat[i])))
#
# # Plotting area
# plt.rcParams.update({'font.size': 15})
# fig3, ax3 = plt.subplots()
# ax3.boxplot(areaMeasurementsFloat, labels=plottingConditions, showfliers=False)
# ax3.set_ylabel('Area (pixels)')
# # ax3.set_ylim([0, 1])
# ax3.set_xlabel(r'Days after seeding')
# ax3.set_title('Organoid sizes at each time point')
# palette = ['b', 'g', 'r', 'c', 'm', 'k']
# for x, val, c in zip(xs, areaMeasurementsFloat, palette):
#     ax3.scatter(x, val, alpha=0.4, color=c)
# plt.tight_layout()
# fig3.show()

# for i in range(len(imagesInAnalysis)):
#     ExportImageWithContours(inputImages[i], imagesInAnalysis[i], imageNames[i], exportPath)
# print('done')


'''
    After filtering
'''

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
            #
            # if i == 2:
            #     print('entered into exporting')
            #     for j in range(len(imagesInAnalysis)):
            #         ExportImageWithContours(inputImages[j], imagesInAnalysis[j], imageNames[j], exportPath)
            # print('done')

            # if livePreview:
            #     print('h')
                # Display('Filtered by ' + filterCriteria[i], imagesInAnalysis[0], 0.5)
                # cv.waitKey(0)
                # SegmentWithOrganoSegPy(inputImages, saveSeg, exportPath, inputImagesPaths)

'''
    Tracking code
'''
# timelapseSets = [imagesInAnalysis[i * timePoints:(i + 1) * timePoints]  # thus, timelapse images are together
#                  for i in range((len(imagesInAnalysis) + timePoints - 1) // timePoints )]
# # list[0 * 4 : 1 * 4] grabs 0,1,2,3 images, ..., list[6*4 : 7*4] grabs 24,25,26,27 images
# # range((28 + 4 - 1)//4) = range(7)
# # timelapseSets = [[imagesInAnalysis[0], imagesInAnalysis[1], imagesInAnalysis[2], imagesInAnalysis[3]]]
# # line above from https://www.geeksforgeeks.org/break-list-chunks-size-n-python/. Compare other methods
# # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]
#
# imageNamesCollected = [imageNames[i * timePoints:(i + 1) * timePoints]
#                        for i in range((len(imageNames) + timePoints - 1) // timePoints )]
# # imageNamesCollected = [[imageNames[0], imageNames[1], imageNames[2], imageNames[3]]]
#
# trackedSets = [track(timelapseSet) for timelapseSet in timelapseSets]  # track expects a list timelapse images
# # [ tracked stack 1, tracked stack 2, ..., tracked 3D array n ]
#
# binaryTrackedSets = [ConvertLabelledImageToBinary(trackedSet) for trackedSet in trackedSets]
# # [ tracked 3D array 1, tracked 3D array 2, ..., tracked 3D array n ]
#
# binaryTrackedSetsList = [[binaryTrackedSet[i] for i in range(len(binaryTrackedSet))] for binaryTrackedSet in binaryTrackedSets]
# # [ [time lapse set 1], [t0, t1, t2, t3], ..., [timelapse set n] ]
#
# binaryTrackedList = list(chain(*binaryTrackedSetsList))
# # [ time lapse set 1, t0, t1, t2, t3, ..., timelapse set n ]
#



'''
    Plot data function at the end
'''

# if plotData:
#     properties = ['area', 'roundness', 'eccentricity', 'solidity']
#     propertyTables = [pd.read_excel(pathDataForPlotting, header=None, sheet_name=prop) for prop in properties]
#
#     # conditions
#     conditionsIndeces = (propertyTables[0].iloc[0]).index[(propertyTables[0].iloc[0]).notnull()].tolist()
#     conditions = (propertyTables[0].iloc[0])[(propertyTables[0].iloc[0]).notnull()].tolist()
#     conditions = [condition.split(" ") for condition in conditions]
#     for i in range(len(conditions)):
#         conditions[i][1] = float(conditions[i][1])
#     conditions[6][1] = 2e-07
#     conditionsConcentrations = [condition[1] for condition in conditions]
#     sortingOrder = list(range(len(conditionsConcentrations)))
#     zipped = zip(conditionsConcentrations, sortingOrder)
#     sortedConcentrations = sorted(zipped)
#     sortingOrder = [point[1] for point in sortedConcentrations]
#
#     conditionsNewOrder = [x for _, x in sorted(zip(sortingOrder, conditions))]
#
#     # properties
#     propertyAndConditionDFs = []
#     for i in range(len(propertyTables)):    # for each property
#         propertyDFs = []
#         for j in range(len(conditions)):        # for each condition
#             propertyDFs.append(propertyTables[i].iloc[:, conditionsIndeces[j]:conditionsIndeces[j]+5])
#         propertyAndConditionDFs.append(propertyDFs)
#
#     propertyAndConditionDFsWithoutNaNs = []
#     for i in range(len(propertyAndConditionDFs)):
#         propertyDFsWithoutNaNs = []
#         for j in range(len(propertyAndConditionDFs[i])):
#             a = propertyAndConditionDFs[i][j]
#             # a = a.iloc[3:, [1,4]]
#             propertyDFsWithoutNaNs.append(a[~a.isnull().any(axis=1)])
#         propertyAndConditionDFsWithoutNaNs.append(propertyDFsWithoutNaNs)
#
#
#     # Plotting size, roundness, eccentricity, solidity tracks for the 8 organoids
#     sizeDF = propertyAndConditionDFsWithoutNaNs[0][6]
#     sizeDFData = sizeDF.iloc[:, 1:]
#     sizeDF = sizeDF.reset_index()
#     sizeDFData = sizeDFData.values.astype(float)
#
#     plt.rcParams.update({'font.size': 15})
#     fig, ax = plt.subplots()
#     timepoints = np.arange(4)
#     for i in range(sizeDFData.shape[0]):
#         ax.plot(timepoints, sizeDFData[i, :], label=f'Object {sizeDF.iloc[i,1]}')
#     ax.set_xticks([0, 1, 2, 3])
#     ax.set_xticklabels([0, 1, 3, 6])
#     ax.set_xlabel('Time (days)')
#     ax.set_ylabel('Size (px)')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(str(exportPath / 'timecourseSizes.png'), dpi=300)
#     plt.show()
#
#     # Calculating fractional growth
#     areaDFs = propertyAndConditionDFsWithoutNaNs[0]
#     areaDFsSorted = [x for _, x in sorted(zip(sortingOrder, areaDFs))]
#
#     # change column names and calculate fractional growth
#     for i in range(len(conditionsNewOrder)):
#         areaDFsSorted[i].columns = ['index', 't1', 't2', 't3', 't4']
#         areaDFsSorted[i]['Frac Growth'] = areaDFsSorted[i]['t4'] / areaDFsSorted[i]['t1']
#
#     # calculate normalised average fractional growth
#     avgFracGrowthControl = areaDFsSorted[0]['Frac Growth'].mean()
#     xs = []
#     for i in range(len(conditionsNewOrder)):
#         areaDFsSorted[i]['Norm Frac Growth'] = areaDFsSorted[i]['Frac Growth'] / avgFracGrowthControl
#         xs.append(np.random.normal(i + 1, 0.04, areaDFsSorted[i]['Norm Frac Growth'].values.shape[0]))
#
#     normFracGrowthValues = [df['Norm Frac Growth'].tolist() for df in areaDFsSorted]
#
#     avgNormFracGrowthPerCondition = np.asarray([df['Norm Frac Growth'].mean() for df in areaDFsSorted])
#     stdNormFracGrowthPerCondition = np.asarray([df['Norm Frac Growth'].std() for df in areaDFsSorted])
#
#     # plot
#     plt.rcParams.update({'font.size': 15})
#     concentrations = [0, 0.2, 1, 2, 3, 5, 25]
#     conditionsToPlot = [" ".join(str(item) for item in alist[1:]) for alist in conditionsNewOrder]
#     condConcsToPlot = [str(alist[1]*1e6) for alist in conditionsNewOrder]
#     fig, ax = plt.subplots()
#     ax.errorbar(concentrations, avgNormFracGrowthPerCondition, yerr=stdNormFracGrowthPerCondition, capsize=5)
#     ax.set_ylabel('Avg. norm\'d fractional growth')
#     ax.set_xlabel(r'Cisplatin concentration ($\mu$M)')
#     ax.set_title('Organoid growth in cisplatin')
#     plt.tight_layout()
#     fig.show()
#
#     fig3, ax3 = plt.subplots(figsize=(20, 6))
#     ax3.boxplot(normFracGrowthValues, labels=concentrations, showfliers=False)
#     ax3.set_ylabel('Norm\'d fractional growth')
#     ax3.set_xlabel(r'Cisplatin concentration ($\mu$M)')
#     ax3.set_title('Organoid growth in cisplatin')
#     palette = ['b', 'g', 'r', 'c', 'm', 'k']
#     for x, val, c in zip(xs, normFracGrowthValues, palette):
#         ax3.scatter(x, val, alpha=0.4, color=c)
#     plt.tight_layout()
#     fig3.show()
#
#     print('ploting')