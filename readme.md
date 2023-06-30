# OrganoTrack: A developing platform for producing dose response curves from measuring temporal organoid morphology

## Purpose
OrganoTrack was created to measure the morphology of single organoids, from brightfield images, over time.
These organoids, derived from patient tumours, were cultured with varying drug concentrations and imaged with a brightfield microscope over time.
Having these temporal morphological measurements, dose-response plots showing the effect of increasing drug concentrations on PDTO morphology can be made.

Therefore, OrganoTrack receives well-plate-based brightfield images and the conditions of the well plate lyaout.
The output of OrganoTrack is a spreadsheet with tables for each well, and sheets for each measurement.
Each table has the measurements of single organoids at each time point.

## Installation

Overview: to set up OrganoTrack source dependencies, create an empty Conda environment and install all packages listed in requirements.txt.

1) Install Anaconda (https://www.anaconda.com/products/distribution)
2) Open the Anaconda prompt and create a new environment
   ```
   >> conda create -n organotrack python=3.9
   >> conda activate organotrack
   ```
3) Download OrganoTrack and extract it to a directory of your choosing. You may also clone the repository instead.
4) In the Anaconda prompt, navigate to the OrganoTrack root directory (which contains the readme.md file)

   ```
   >> cd path/to/OrganoTrack/directory
   ```

5) Install all OrganoTrack requirements:
   ```
   >> pip install -r requirements.txt
   ```

## Usage

To use OrganoTrack, a specific folder structure for the experiment needs to be made, 
and OrganoTrack needs to be executed from a Python IDE.  


### Creating experiment folder structure

1) Create an experiment directory with your desired name, e.g. cisplatin-drug-response
2) Within this experiment directory, create two directories called 'import' and 'export'
3) Within the import directory, there should be two items. The fist is an 'images' directory containing all the experiment images. The second item is the plate_layout file in .csv, .tsv, or any MS Excel format.

**Image naming note:**

The images should be named iteratively, starting with the well, the well field, z position, and imaging time point.
For example, the name "R2C2_F10P1T0_220405-106TP24-15BME-CisGemCarbo-Harmony-mask" corresponds to
an image captured from row (R) 2, column (C) 2, field (F) 10 of 25 well fields
position (P) 1 and time point (T) 0. The rest of the image name contains experiment information.

**Plate layout note:**

Use a format for the plate layout as structured in plate_layout.xlsx.
The plate layout has two tables, the first for the well plate conditions,
and the second for the condition concentrations.

For a trial execution of OrganoTrack, some experiment data is provided in the cisplatin-drug-screen directory.

### Executing OrganoTrack from a Python IDE

For a template script, see function template() in executingOrganoTrack.py (within Experiments-with-OrganoTrack).


## Exported data
Currently, the export of measurements works only if organoids were tracked.
Otherwise, measurements will be exported per single image.

Input data requirements:

- For example, images exported from the Opera Phenix Plus are named "r02c02f01p01-ch1sk1fk1fl1.tiff"
- This image corresponds to, of a 96-well plate, row 2, column 2, field 1, position 1 within z-stack, channel 1, time point (sk) 1. The rest is not yet understood.
- The images read by OrganoTrack are stored into a nested dictionary. The first nest considers a tuple of (row, column) for each well
- The second nest considers the field number, the values of which are lists of the time point images.
- For correctly storing the timepoint images in the right order (i.e. 1, 2, 3, 4, ...), the images must be named by the well, field, time point order
- Such that if they are sorted alphanumerically, the time point images of one field for one well will be read one after the other.
- Thus, another naming format is r2c2f1sk1, r2c2f1sk2, r2c2f1sk3, r2c2f1sk4. Thus, the information is ordered by well, field, time point
- And alphanumeric sorting should sort the images in this order.
- Ensure that each image is unique in terms of: well, field, position, timepoint

