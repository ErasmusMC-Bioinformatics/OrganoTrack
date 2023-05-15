import matplotlib.pyplot as plt
import pandas as pd

organoTrack_seg_scores_EMC_Dir = '/home/franz/Documents/mep/data/for-creating-OrganoTrack/training-dataset/preliminary-gt-dataset/predictions/OrganoTrack-seg-scores-EMC.xlsx'
organoTrack_seg_scores_OrganoID_OriData_Dir = '/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/OriginalData/export/OrganoTrack-seg-scores-OrganoID-OriginalData.xlsx'
organoTrack_seg_scores_OrganoID_MouseOrgs_Dir = '/home/franz/Documents/mep/data/published-data/OrganoID-data/combinedForOrganoTrackTesting/MouseOrganoids/Export/OrganoTrack-seg-scores-OrganoID-MouseOrganoids.xlsx'


organoTrack_seg_scores_dataset1_EMC = pd.read_excel(organoTrack_seg_scores_EMC_Dir, header=1).drop(columns='Unnamed: 0').to_numpy()
organoTrack_seg_scores_dataset2_OrganoID_OriData = pd.read_excel(organoTrack_seg_scores_OrganoID_OriData_Dir, header=1).drop(columns='Unnamed: 0').to_numpy()
organoTrack_seg_scores_dataset3_OrganoID_MouseOrgs = pd.read_excel(organoTrack_seg_scores_OrganoID_MouseOrgs_Dir, header=1).drop(columns='Unnamed: 0').to_numpy()


organoTrack_f1 = [organoTrack_seg_scores_dataset1_EMC[:, 0],
                  organoTrack_seg_scores_dataset2_OrganoID_OriData[:, 0],
                  organoTrack_seg_scores_dataset3_OrganoID_MouseOrgs[:, 0]]

organoTrack_iou = [organoTrack_seg_scores_dataset1_EMC[:, 1],
                  organoTrack_seg_scores_dataset2_OrganoID_OriData[:, 1],
                  organoTrack_seg_scores_dataset3_OrganoID_MouseOrgs[:, 1]]

ticks = ['EMC', 'OrganoID Original', 'OrganoID Mouse Orgs']

# plot
# for i in range(len(conditionsNewOrder)):
#     areaDFsSorted[i]['Norm Frac Growth'] = areaDFsSorted[i]['Frac Growth'] / avgFracGrowthControl
#     xs.append(np.random.normal(i + 1, 0.04, areaDFsSorted[i]['Norm Frac Growth'].values.shape[0]))

plt.rcParams.update({'font.size': 15})

# fig3, ax3 = plt.subplots()
# ax3.boxplot([1], labels=concentrations, showfliers=False)
# ax3.set_ylabel('Norm\'d fractional growth')
# ax3.set_xlabel(r'Cisplatin concentration ($\mu$M)')
# ax3.set_title('Organoid growth in cisplatin')
# palette = ['b', 'g', 'r', 'c', 'm', 'k']
# # for x, val, c in zip(xs, normFracGrowthValues, palette):
# #     ax3.scatter(x, val, alpha=0.4, color=c)
# plt.tight_layout()
# fig3.show()

print('ploting')


print('h')