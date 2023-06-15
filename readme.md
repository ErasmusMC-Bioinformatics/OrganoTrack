# OrganoTrack: A developing platform for preclinical chemotherapy response prediction based on drug-induced morphological changes of tumour organoids derived from muscle-invasive bladder cancer patients




Input data requirements:

- Input images should be within a folder called images, within a parent folder. The directory of this parent folder should be given, and 'images' folder should be the first element in a list of items/directories in the parent folder

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
