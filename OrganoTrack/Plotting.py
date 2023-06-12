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
eccentricityMeasurementsPerConc = dict()
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
