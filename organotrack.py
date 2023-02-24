from stitch import read_images
'''
    Welcome
'''
print("Welcome to OrganoTrack")

'''
    Read data
'''

# print("Enter naming structure of data")
# < Naming structure
# Determine appropriate data structure

dataDirectory = input("Enter data directory: ")
fields = input("Enter number of fields: ")
wells = input("Enter wells to study")
experiment = 'drug_screen_april-05'
positions = 1
channels = 1
time_points = 0
imData = read_images(dataDirectory, wells, positions, channels, fields)
# Read images according to data structure
    # If image not grayscale, read as grayscale in the same bit depth

'''
    Segmentation
'''

# print("Choose from options")
# < OrganoSeg / OrganoID / Optimise parameters and segment

# If OrganoSeg / OrganoID:
    # segmentation = segment(images)

# Elif optimise parameters:
    # for each parameter combination:
        # calculate segmentation performance
            # (user can input % of dataset to test this on)
            # Does decreasing to 8-bit make the operation run faster? Does it affect seg performance?
        # store segmenation performance for each combination

    # return combination with highest performance

    # segmentation = segment_with_optimal_params(images)

    # further processing:
        # ID every object
        # remove noise
        # smoothen
        # include hole closing of everything (but keep the identity of out of focus organoids)


# Report segmentation metrics (using GT dataset)


'''
    User selection / Filtering / Measuring features
'''
# Remove out of focus organoids


# Remove border objects


# Select unmerged organoids (only if there is time in the data)
# If one object in timepoint t+1 overlaps with more than 1 object in timepoint t:
    # remove that object from the image of timepoint t+1


# Measure features of each organoid


# Filter out organoids that do not meet a range in: size, circularity, STAR, SER, etc.


# Remaining: organoids of interest, with their measurements available (export data)


'''
    Data exporting
'''
# Export to CSV

# Plotting
