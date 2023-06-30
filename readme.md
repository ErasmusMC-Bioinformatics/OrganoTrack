# OrganoTrack: A developing platform for producing dose response curves from measuring temporal organoid morphology

## Purpose
OrganoTrack was created to measure the morphology of single organoids, from brightfield images, over time.
These organoids, derived from patient tumours, were cultured with varying drug concentrations and imaged with a brightfield microscope over time.
Having these temporal morphological measurements, dose-response plots showing the effect of increasing drug concentrations on PDTO morphology can be made.

Therefore, OrganoTrack receives well-plate-based brightfield images and the conditions of the well plate lyaout.
The output of OrganoTrack is a spreadsheet with morphological measurements for each organoid over time.

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
The export of OrganoTrack can then be interpreted.

For a trial execution of OrganoTrack, some experiment data is provided in experiment-cisplatin-drug-screen.

### Creating experiment folder structure

1) Create an experiment directory with your desired name, e.g. experiment-cisplatin-drug-response
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

### Executing OrganoTrack from a Python IDE

To execute OrganoTrack, load the organotrack environment onto your IDE.
After the organotrack environment is loaded, follow these instructions for a basic execution of OrganoTrack: 
1) Open executingOrganoTrack.py from Using-OrganoTrack.
2) Copy the template() function, paste it on the same python script. Rename the copy to your desired name, e.g. cisplatin_drug_screen.
3) Within your function copy, define the following variables:
   - import_path: the absolute path of the experiment import directory (on Windows, ensure that '/' is used within the path.)
   - identifiers: enter the characters that, within the image name, identify the row, column, field, position, and time point of the image
   - export_path: the absolute path of the experiment export directory (on Windows, ensure that '/' is used within the path.)
4) Call this function under main, and execute the Python script.

### Interpreting results from OrganoTrack execution

The experiment export directory will be updated with a trackedMeasures.xlxs file.
This file will contain a number of sheets, each for a specific morphological measure.
Within each sheet, there will be a table for each well.
Each table contains the measure of each organoid at each time point that it is found.



