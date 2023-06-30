from OrganoTrack.OrganoTrack import run_OrganoTrack
from pathlib import Path

def template():
    # Import
    import_path = Path(' ')  # enter absolute path
    identifiers = {'row': 'r',  # character identifier in the image name for well row
                   'column': 'c',
                   'field': 'f',
                   'position': 'p',
                   'time_point': 'sk'}
    export_path = Path(' ')  # enter absolute path

    # Segmentation
    segment = False
    intensity_threshold = 0.5  # multiplication factor of Otsu threshold within adaptive thresholding
    max_window_size = 250  # maximum window size considered for adaptive thresholding
    size_threshold_px = 150  # threshold for object sizes, below which they are removed
    extra_blur = False
    blur_size = 3  # For the extra blurring step, size of structuring element
    seg_parameters = [intensity_threshold, max_window_size, size_threshold_px, extra_blur, blur_size]
    export_segmentation = False
    params_to_save_segmentations = [export_segmentation, export_path]
    path_to_segmented_imgs = export_path / 'Harmony-segmented'

    # Selection of organoids
    remove_boundary_objects = True
    filter_organoids_by_morphology = True
    morph_filter_criteria = ['area', 150]  # list of morph measure to filter by, with min threshold as the next element

    # Track organoids
    track_organoids = True

    # Measure organoids
    export_tracked_organoid_measurements = True
    morph_properties_to_measure = ['area', 'roundness', 'eccentricity', 'solidity']

    run_OrganoTrack(import_path, identifiers, export_path,
                    segment, seg_parameters, params_to_save_segmentations, path_to_segmented_imgs,
                    remove_boundary_objects, filter_organoids_by_morphology, morph_filter_criteria,
                    track_organoids,
                    export_tracked_organoid_measurements, morph_properties_to_measure)

def harmony_segmented_all_cis_data():
    # Import
    import_path = Path(
        '/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/import')
    identifiers = {'row': 'R',
                   'column': 'C',
                   'field': 'F',
                   'position': 'P',
                   'timePoint': 'T'}
    export_path = Path(
        '/home/franz/Documents/mep/data/experiments/220405-Cis-drug-screen/Harmony-masks-with-analysis-220318-106TP24-15BME-CisGemCarbo-v4/export')

    # Segmentation
    segment = False
    extra_blur = False
    blur_size = 3
    seg_parameters = [0.5, 250, 150, extra_blur, blur_size]
    params_to_save_segmentations = [False, export_path]
    path_to_segmented_imgs = export_path / 'Harmony-segmented'

    # Selection of organoids
    remove_boundary_objects = True
    filter_organoids_by_morphology = True
    morph_filter_criteria = ['area', 150]

    # Track organoids
    track_organoids = True

    # Measure organoids
    export_tracked_organoid_measurements = True
    morph_properties_to_measure = ['area', 'roundness', 'eccentricity', 'solidity']

    run_OrganoTrack(import_path, identifiers, export_path, segment, seg_parameters, params_to_save_segmentations,
                    path_to_segmented_imgs, remove_boundary_objects, filter_organoids_by_morphology,
                    morph_filter_criteria, track_organoids, export_tracked_organoid_measurements,
                    morph_properties_to_measure)

def cisplatin_drug_screen():
    # Import
    import_path = Path('C:/Users/franz/Documents/mep/OrganoTrack/experiment-cisplatin-drug-screen/import')  # enter absolute path
    identifiers = {'row': 'R',  # character identifier in the image name for well row
                   'column': 'C',
                   'field': 'F',
                   'position': 'P',
                   'time_point': 'T'}
    export_path = Path('C:/Users/franz/Documents/mep/OrganoTrack/experiment-cisplatin-drug-screen/export')  # enter absolute path

    # Segmentation
    segment = False
    intensity_threshold = 0.5  # multiplication factor of Otsu threshold within adaptive thresholding
    max_window_size = 250  # maximum window size considered for adaptive thresholding
    size_threshold_px = 150  # threshold for object sizes, below which they are removed
    seg_parameters = [intensity_threshold, max_window_size, size_threshold_px]
    export_segmentation = False
    params_to_save_segmentations = [export_segmentation, export_path]
    path_to_segmented_imgs = export_path / 'Harmony-segmented'

    # Selection of organoids
    remove_boundary_objects = True
    filter_organoids_by_morphology = True
    morph_filter_criteria = ['area', 150]  # list of morph measure to filter by, with min threshold as the next element

    # Track organoids
    track_organoids = True

    # Measure organoids
    export_tracked_organoid_measurements = True
    morph_properties_to_measure = ['area', 'roundness', 'eccentricity', 'solidity']

    run_OrganoTrack(import_path, identifiers, export_path,
                    segment, seg_parameters, params_to_save_segmentations, path_to_segmented_imgs,
                    remove_boundary_objects, filter_organoids_by_morphology, morph_filter_criteria,
                    track_organoids,
                    export_tracked_organoid_measurements, morph_properties_to_measure)


if __name__ == '__main__':
    cisplatin_drug_screen()
