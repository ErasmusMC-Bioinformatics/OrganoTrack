import numpy as np
import cv2 as cv

def rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

def display(title, img, sf=1):
    img = rescale(img, sf)
    cv.imshow(title, img)

'''
for each well
    for each timepoint
        for each field
            read image  
        combine all 25 images in the right order
'''


dir = '/home/franz/Documents/mep/data/organoid-images/drug-screen-april-05/Images'




'''
General code:
1. Prompt: is well divided into images that need to be stitched? Y/N
2. Prompt: How many images? 25
3. Prompt: How many rows and columns? 5 x 5
4. Prompt: Enter order of stitching: 2, 4, 5,...
5. Prompt: Name format in images? rxxcyyfzzp01-ch1sktfk1fl1.tiff: xx row, yy col, zz field, t time point
Receive 
'''

stitching_order = [ '02', '03', '04', '05', '06',
                    '11', '10', '09', '08', '07',
                    '12', '13', '01', '14', '15',
                    '20', '19', '18', '17', '16',
                    '21', '22', '23', '24', '25' ]

"""
List order = [   0,  1,  2,  3,  4,
                 5,  6,  7,  8,  9,
                10, 11, 12, 13, 14,
                15, 16, 17, 18, 19,
                20, 21, 22, 23, 24 ]
"""


all_fields = []
for field in range(len(stitching_order)):
    all_fields.append(cv.imread(dir + "/r02c02f"+stitching_order[field]+"p01-ch1sk1fk1fl1.tiff"))

'''
Concatenate rows
'''
row_fields = []
rows = 5
columns = 5
# for row in range(rows):
#     row_concat =
#     for col in range(columns):
#         row_concat = np.concatenate()
#     row_fields.append(np.concatenate((all_fields[0], all_fields[1], all_fields[2], all_fields[3], all_fields[4]), axis=1))
first_row = np.concatenate( (all_fields[0], all_fields[1], all_fields[2], all_fields[3], all_fields[4]), axis=1)
second_row = np.concatenate( (all_fields[5], all_fields[6], all_fields[7], all_fields[8], all_fields[9]), axis=1)
third_row = np.concatenate( (all_fields[10], all_fields[11], all_fields[12], all_fields[13], all_fields[14]), axis=1)
fourth_row = np.concatenate( (all_fields[15], all_fields[16], all_fields[17], all_fields[18], all_fields[19]), axis=1)
fifth_row = np.concatenate( (all_fields[20], all_fields[21], all_fields[22], all_fields[23], all_fields[24]), axis=1)

full_image = np.concatenate((first_row, second_row, third_row, fourth_row, fifth_row), axis=0)
display('well 1', full_image, 0.2)

cv.waitKey(0)