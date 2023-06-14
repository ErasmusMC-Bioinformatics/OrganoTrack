import matplotlib.pyplot as plt


def Plotting():
    pass

concentrations = {1: {'conc': 0,
                      'unit': 10**-6,
                      'reps': [(2,2), (3,2), (4,2)]},
                  2: {'conc': 0.2,
                      'unit': 10**-6,
                      'reps': [(2,8), (3,8), (4,8)]},
                  3: {'conc': 1,
                      'unit': 10**-6,
                      'reps': [(2,7), (3,7), (4,7)]},
                  4: {'conc': 2,
                      'unit': 10**-6,
                      'reps': [(2,6), (3,6), (4,6)]},
                  5: {'conc': 3,
                      'unit': 10**-6,
                      'reps': [(2,5), (3,5), (4,5)]},
                  6: {'conc': 5,
                      'unit': 10**-6,
                      'reps': [(2,4), (3,4), (4,4)]},
                  7: {'conc': 25,
                      'unit': 10**-6,
                      'reps': [(2,3), (3,3), (4,3)]}}

# collecting the tracked measurements for each condition across the replicates
areaMeasurementsPerConc = dict()
for concIndex in list(concentrations.keys()):
    dfsPerConc = []
    for well in concentrations[concIndex]['reps']:
        dfsPerConc.append(trackedMeasurementsPerWell['area'][well])
    areaMeasurementsPerConc[concIndex] = pd.concat(dfsPerConc)

# removing tracks without all four timepoint measurements
fullyTrackedAreaMeasurementsPerConc = dict()
for concIndex in list(areaMeasurementsPerConc.keys()):
    test = areaMeasurementsPerConc[concIndex]
    test = test.replace('', np.nan)
    df = test.dropna(subset=['t0', 't1', 't2', 't3'])
    fullyTrackedAreaMeasurementsPerConc[concIndex] = df


# Calculating fractional growth
for concIndex in list(fullyTrackedAreaMeasurementsPerConc.keys()):
    df = fullyTrackedAreaMeasurementsPerConc[concIndex]
    df = df.astype(float)
    df['fracGrowth'] = df['t3'] / df['t0']
    fullyTrackedAreaMeasurementsPerConc[concIndex] = df

# Calculating average control fractional growth
avgFracGrowthControl = fullyTrackedAreaMeasurementsPerConc[1]['fracGrowth'].mean()

# Normalising fractional growth
for concIndex in list(fullyTrackedAreaMeasurementsPerConc.keys()):
    df = fullyTrackedAreaMeasurementsPerConc[concIndex]
    df['normFracGrowth'] = df['fracGrowth'] / avgFracGrowthControl
    concentrations[concIndex]['trackedAreaJitter'] = \
        np.random.normal(concIndex, 0.04, df['normFracGrowth'].values.shape[0])
    fullyTrackedAreaMeasurementsPerConc[concIndex] = df

# Getting normalised values into a list
normFracGrowthValues = []
xs = []
for cIndex in list(fullyTrackedAreaMeasurementsPerConc.keys()):
    df = fullyTrackedAreaMeasurementsPerConc[cIndex]
    normFracGrowthValues.append(df['normFracGrowth'].tolist())
    xs.append(concentrations[cIndex]['trackedAreaJitter'])

concValues = [0, 0.2, 1, 2, 3, 5, 25]

trackedAreaNumPoints = [1461, 1134, 1058, 1088, 1204, 1328, 2623]
# Plot box plots of normalised fractional growth


plt.rcParams.update({'font.size': 20})
fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.boxplot(normFracGrowthValues, labels=concValues, showfliers=False)
ax3.set_ylabel('Normalised fractional growth')
ax3.set_xlabel(r'Cisplatin concentration ($\mu$M)')
yTicks = [1, 2, 3, 4, 5, 6]
ax3.set_yticks(yTicks)
ax3.set_yticklabels(yTicks)
palette = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
for x, val, c in zip(xs, normFracGrowthValues, palette):
    ax3.scatter(x, val, alpha=0.4, color=c)
labels = [f"{concValues[i]}\n(n={n})" for i, n in enumerate(trackedAreaNumPoints)]
ax3.set_xticklabels(labels)
plt.tight_layout()
fig3.show()

# Plot averages and
avgNormFracGrowthPerCondition = np.asarray([np.mean(df) for df in normFracGrowthValues])
stdNormFracGrowthPerCondition = np.asarray([np.std(df) for df in normFracGrowthValues])

# plot
plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(figsize=(8,5))
ax.errorbar(concValues, avgNormFracGrowthPerCondition, yerr=stdNormFracGrowthPerCondition, capsize=5)
ax.set_ylabel('Avg. norm\'d fractional growth (+ 1 SD)')
ax.set_xlabel(r'Cisplatin concentration ($\mu$M)')
plt.tight_layout()
fig.show()

# Violin plot
fig2, ax2 = plt.subplots(figsize=(12, 6))
violins = ax2.violinplot(normFracGrowthValues, showextrema=False)
ax2.set_ylabel('Normalised fractional growth')
ax2.set_xlabel(r'Cisplatin concentration ($\mu$M)')
ax2.set_xticks(np.arange(1, 8))
yTicks = [1, 2, 3, 4, 5, 6]
ax2.set_yticks(yTicks)
ax2.set_yticklabels(yTicks)
labels = [f"{concValues[i]}\n(n={n})" for i, n in enumerate(trackedAreaNumPoints)]
ax2.set_xticklabels(labels)
palette = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
for pc, color in zip(violins['bodies'], palette):
    pc.set_facecolor(color)
plt.tight_layout()
fig2.show()

# Eccentricy / Roundness / Solidity change over time at each concentration
eccentricityMeasures = trackedMeasurementsPerWell['eccentricity']

# removing tracks without all four timepoint measurements
for well in list(eccentricityMeasures.keys()):
    df = eccentricityMeasures[well]
    df = df.replace('', np.nan)
    df = df.dropna(subset=['t0', 't1', 't2', 't3'])
    eccentricityMeasures[well] = df

# generate list of lists of Ecc measurements for each concentration, together with a jitter list
eccentricityPerConcAndTime = dict()
for concIndex in list(concentrations.keys()):
    eccentricityPerConcAndTime[concIndex] = dict()
    eccentricityPerConcAndTime[concIndex]['data'] = []
    eccentricityPerConcAndTime[concIndex]['jitter'] = []
    concReps = concentrations[concIndex]['reps']  # list of tuples
    jitterIndex = 1
    for timePoint in ['t0', 't1', 't2', 't3']:
        for well in concReps:
            # eccentricityPerConcAndTime[concIndex].append(eccentricityMeasures[well][timePoint].astype(float).tolist())
            data = eccentricityMeasures[well][timePoint].astype(float).tolist()
            eccentricityPerConcAndTime[concIndex]['data'].append(data)
            jitter = np.random.normal(jitterIndex, 0.04, len(data))
            eccentricityPerConcAndTime[concIndex]['jitter'].append(jitter)
            jitterIndex += 1

conc = 1

def PlotSubplot(conc, ax):
    data = eccentricityPerConcAndTime[conc]['data']
    xs = eccentricityPerConcAndTime[conc]['jitter']

    colors = ['b', 'g', 'r', 'c']  #, 'm', 'k', 'y']

    # Plot the boxplots
    ax.boxplot(data, showfliers=False)
    ax.set_ylabel('Eccentricity')
    ax.set_xlabel('Days elapsed after seeding, replicates')
    ax.set_ylim([0, 1])

    # Scatter plots
    for i, (x, val) in enumerate(zip(xs, data)):
        color = colors[i // 3]
        ax.scatter(x, val, alpha=0.4, c=color)

    # Set x-axis tick labels
    timePoints = ['d1', 'd2', 'd2', 'd7']
    xTickLabels = [f'{timePoint}, rep {rep}' for timePoint in timePoints for rep in [1, 2, 3]]
    xTickLabels = [currentLabel+f'\n(n={len(data[i])})' for i, currentLabel in enumerate(xTickLabels)]
    ax.set_xticklabels(xTickLabels, rotation=45, ha='right')

    plt.tight_layout()

plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(40, 32))

# Add the first subplot (1 row, 1 column, position 1)
ax1 = axes[0, 0]
ax1.set_title('Control')
PlotSubplot(1, ax1)

# Add the other six subplots (3 rows, 2 columns)
ax2 = axes[1, 0]
ax2.set_title('Cis 0.2 uM')
PlotSubplot(2, ax2)

ax3 = axes[1, 1]
ax3.set_title('Cis 1 uM')
PlotSubplot(3, ax3)

ax4 = axes[2, 0]
ax4.set_title('Cis 2 uM')
PlotSubplot(4, ax4)

ax5 = axes[2, 1]
ax5.set_title('Cis 3 uM')
PlotSubplot(5, ax5)

ax6 = axes[3, 0]
ax6.set_title('Cis 5 uM')
PlotSubplot(6, ax6)

ax7 = axes[3, 1]
ax7.set_title('Cis 25 uM')
PlotSubplot(7, ax7)

# Remove the extra subplots
fig.delaxes(axes[0, 1])

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()


# Now, look at each day across all concentrations
def GatherFeatureAcrossReplicatesAndConcentrationsForEachTimepoint(featureMeasures, concentrations):
    featureAcrossRepsAndConcsForEachTimePoint = dict()  # e.g. = {'t0' = {'data' = [...], 'jitter' = [...],
                                                        #         't1' = {'data' = [...], 'jitter' = [...]}
                                                        # ... = 21 feature measures
                                                        #  b/c for each rep of 3 for each conc of 7

    for timePoint in ['t0', 't1', 't2', 't3']:
        featureAcrossRepsAndConcsForEachTimePoint[timePoint] = dict()
        featureAcrossRepsAndConcsForEachTimePoint[timePoint]['data'] = []
        featureAcrossRepsAndConcsForEachTimePoint[timePoint]['jitter'] = []

        scatterJitterIndex = 1
        for concIndex in list(concentrations.keys()):
            concReps = concentrations[concIndex]['reps']  # list of tuples
            for well in concReps:
                featureData = featureMeasures[well][timePoint].astype(float).tolist()
                featureAcrossRepsAndConcsForEachTimePoint[timePoint]['data'].append(featureData)
                scatterJitter = np.random.normal(scatterJitterIndex, 0.04, len(featureData))
                featureAcrossRepsAndConcsForEachTimePoint[timePoint]['jitter'].append(scatterJitter)
                scatterJitterIndex += 1

    return featureAcrossRepsAndConcsForEachTimePoint

def SubPlotFeatureAcrossRepsAndConcsForEachTimePoint(featureData, timePoint, ax):
    data = featureData[timePoint]['data']
    xs = featureData[timePoint]['jitter']

    colors = ['b', 'g', 'r', 'c', 'm', 'k', 'y']

    # Plot the boxplots
    ax.boxplot(data, showfliers=False)
    ax.set_ylabel('Eccentricity')
    ax.set_xlabel(r'Cis conc. ($\mu$M), replicates')
    ax.set_ylim([0, 1])

    # Scatter plots
    for i, (x, val) in enumerate(zip(xs, data)):
        color = colors[i // 3]
        ax.scatter(x, val, alpha=0.4, c=color)

    # Set x-axis tick labels
    # timePoints = ['d1', 'd2', 'd2', 'd7']
    # xTickLabels = [f'{timePoint}, rep {rep}' for timePoint in timePoints for rep in [1, 2, 3]]
    # xTickLabels = [currentLabel+f'\n(n={len(data[i])})' for i, currentLabel in enumerate(xTickLabels)]
    # ax.set_xticklabels(xTickLabels, rotation=45, ha='right')
    plt.tight_layout()


plt.rcParams.update({'font.size': 20})
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(50, 15))

# Add the first subplot (1 row, 1 column, position 1)
ax1 = axes[0, 0]
fig1, ax1 = plt.subplots(figsize=(20,6))
ax1.set_title('Day 1')
SubPlotFeatureAcrossRepsAndConcsForEachTimePoint(eccentricityAcrossRepsAndConcsForEachTimePoint, 't0', ax1)
plt.show()

ax2 = axes[0, 1]
ax2.set_title('Day 2')
SubPlotFeatureAcrossRepsAndConcsForEachTimePoint(eccentricityAcrossRepsAndConcsForEachTimePoint, 't1', ax2)

# Add the other six subplots (3 rows, 2 columns)
ax3 = axes[1, 0]
ax3.set_title('Day 4')
SubPlotFeatureAcrossRepsAndConcsForEachTimePoint(eccentricityAcrossRepsAndConcsForEachTimePoint, 't2', ax3)

ax4 = axes[1, 1]
ax4.set_title('Day 7')
SubPlotFeatureAcrossRepsAndConcsForEachTimePoint(eccentricityAcrossRepsAndConcsForEachTimePoint, 't3', ax4)

# Adjust the spacing between subplots
plt.tight_layout()

# Show the figure
plt.show()
