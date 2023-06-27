import os
import shutil

source_directory = 'C:/Users/franz/OneDrive/Documents/work-Franz/mep/cisplatinDataset/input/images'  # Replace with the path to your current directory
destination_directory = 'C:/Users/franz/OneDrive/Documents/work-Franz/mep/cisplatinDataset/extra-images'  # Replace with the path to the new directory

# Create the destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Define the range of rows and columns to keep
desired_rows = range(2, 5)
desired_columns = range(2, 9)

# Iterate over the files in the source directory
for filename in os.listdir(source_directory):
    filepath = os.path.join(source_directory, filename)
    if os.path.isfile(filepath):
        # Extract the row and column from the filename
        row = int(filename[1:3].lstrip("0"))
        column = int(filename[4:6].lstrip("0"))

        # Check if the image belongs to the desired rows and columns
        if row not in desired_rows or column not in desired_columns:
            # Move the file to the destination directory
            shutil.move(filepath, os.path.join(destination_directory, filename))
            print(f"Moved {filename} to the destination directory.")