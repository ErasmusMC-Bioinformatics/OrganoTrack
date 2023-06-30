# OrganoTrack: A developing platform for measuring morphological changes of single organoids across time lapse brightfield imaging, well plate based experiments

## Purpose of OrganoTrack
OrganoTrack was created to measure the morphology of single organoids, from brightfield images, over time.
These organoids, derived from patient tumours, were cultured with varying drug concentrations and imaged with a brightfield microscope over time.
Having these temporal morphological measurements, dose-response plots showing the effect of increasing drug concentrations on PDTO morphology can be made.
Therefore, OrganoTrack receives well-plate-based brightfield images and the conditions of the well plate lyaout.
The output of OrganoTrack is a spreadsheet with tables for each well, and sheets for each measurement.
Each table has the measurements of single organoids at each time point.

## Installing OrganoTrack
Create an environment by installing from the organotrack.yml file.
Use conda or mamba, and execute: conda env create -f organotrack.yml

   >> conda create -n OrganoID python=3.9
   >> activate OrganoID

## Structuring experiment folder
One directory should be dedicated per experiment.
Within this directory, there should be two directories called 'import' and 'export'.

Within the import sub-directory, there should be two items.
One item is another sub-directory called 'images', wherein all the experimental images lie.
The second item is the plate_layout file. Supported formats include .csv, .tsv, .xls, .xlsx, .xlsm, .xlsb, .odf, .ods, and .odt.

Within the export sub-directory, files exported through the analysis of OrganoTrack should exist.
It is thus empty for a new experiment.

## Executing OrganoTrack
First, the images and plate layout of the experiment should be stored within a directory as described above in 'Structuring experiment folder'.
OrganoTrack can be executed from a Python script.
For a template script, see function template() in executingOrganoTrack.py (within Experiments-with-OrganoTrack).


## Exported data
Currently, the export of measurements works only if organoids were tracked.
Otherwise, measurements will be exported per single image.

Input data requirements:
- Images should be named iteratively, starting with information for the well, well area, and finally time point.
- For example, images exported from the Opera Phenix Plus are named "r02c02f01p01-ch1sk1fk1fl1.tiff"
- This image corresponds to, of a 96-well plate, row 2, column 2, field 1, position 1 within z-stack, channel 1, time point (sk) 1. The rest is not yet understood.
- The images read by OrganoTrack are stored into a nested dictionary. The first nest considers a tuple of (row, column) for each well
- The second nest considers the field number, the values of which are lists of the time point images.
- For correctly storing the timepoint images in the right order (i.e. 1, 2, 3, 4, ...), the images must be named by the well, field, time point order
- Such that if they are sorted alphanumerically, the time point images of one field for one well will be read one after the other.
- Thus, another naming format is r2c2f1sk1, r2c2f1sk2, r2c2f1sk3, r2c2f1sk4. Thus, the information is ordered by well, field, time point
- And alphanumeric sorting should sort the images in this order.
- Ensure that each image is unique in terms of: well, field, position, timepoint

Plate layout reads uM, not μM

Functionality to add: convert A2 well format to row, column format, or vice versa.
Ensure robustness across any naming convention
- e.g. D4 1, D4 2 (2 same well, field, timepoint, different position)
- F4 10x: Well, 10x mag not needed for reading, but later for area calculation
- E5 10x 1.tif, E5 10x 2.tif, E5 10x 3.tif: Same well, 10x magnification (not needed for reading), 3 different fields
- Allow for consideration of 10x or 4x: Thus, well, magnification, field, position, timepoint

Get their lazy filenaming. Find a pattern. Make reading fit that. Ultimately:
- It's a well plate, with wells, magnifications, fields, positions, timepoints 

All naming patterns I have:
- r02c02f01p01-ch1sk1fk1fl1:                                         I know this one
- P117T-P9-C1-10x:                                                   Extra-wellByLetter&Number-MagnificationX
- Haga2Tp14_FotoFranz:                                               Extra. No well information
- E10 1:                                                             wellByLetter&Number position/field (cannot change, will confuse software)
- B4 10x. Format:                                                    wellByLetter&Number MagnificationX
- E4 10x 3. Format:                                                  wellByLetter&Number MagnificationX field number
- R2C2_F1P1T3_220405-106TP24-15BME-CisGemCarbo-Harmony-mask. Format: RowNumColumnNum_FieldNumPositionNumTimeNum_Extra

True patterns: 
wells, magnifications, fields, positions, timepoints (no channels currently)

