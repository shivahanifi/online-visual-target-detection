# Path to the additional frames of the setting
folder_path = '/Users/shiva.hnf/Documents/IIT/VTD_retrain/VTD_dumper_Shiva_modified/VTD_dumper_Shiva/session_5/pringles-masterchef-sugarbox-bleach-mustard/session5-shiva-pringles/rgb_input_00005'
# The starting numebr for naming the frames (The last name in the TXT file +1)
starting_number = 234
# Path to save the renamed images
new_folder_path = '/Users/shiva.hnf/Documents/IIT/VTD_retrain/VTD_dumper_Shiva_modified/VTD_dumper_Shiva/session_5/pringles-masterchef-sugarbox-bleach-mustard/session5-shiva-mustard'

import os
import shutil

# Create the new folder
new_folder_name = 'renamed_rgb_input'
new_folder_path = os.path.join(folder_path, new_folder_name)
os.makedirs(new_folder_path, exist_ok=True)

# Get the list of files in the original folder
files = os.listdir(folder_path)

# Sort the files based on their current names
files.sort()

# Iterate over each file
for index, file in enumerate(files):
    if file.endswith('.ppm'):
        # Construct the new name for the file
        new_name = '{:08d}.ppm'.format(starting_number + index)
        
        # Construct the full file paths
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(new_folder_path, new_name)
        
        # Copy the file to the new folder with the new name
        shutil.copy2(old_path, new_path)


        