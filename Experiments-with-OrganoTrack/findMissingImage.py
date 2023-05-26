import os

source_directory = 'C:/Users/franz/OneDrive/Documents/work-Franz/mep/cisplatinDataset/input/images' # Replace with the path to your directory containing the images

# Create a set of expected image names
expected_images = set()
for row in range(2, 5):
    for column in range(2, 9):
        for area in range(1, 26):
            for timepoint in range(1, 5):
                image_name = f'r{row:02}c{column:02}f{area:02}p01-ch1sk{timepoint}fk1fl1.tiff'
                expected_images.add(image_name)

# Get the actual image names in the source directory
actual_images = set(os.listdir(source_directory))

# Find the missing image
missing_images = expected_images - actual_images

# Print the missing image names
for missing_image in missing_images:
    print(f"Missing image: {missing_image}")
    # Missing is r04c08f11p01-ch1sk4fk1fl1.tiff