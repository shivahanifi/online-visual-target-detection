from logging import root
import os
import shutil

def take_every_n_file(input_folder, output_folder, n):
    """reduce the number of files by only selecting every nth file.
    
    :param input_folder: str - path to the input folder.
    :param output_folder: str - path to the output folder.
    :param n: int - to select every nth file.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get a list of files and directories in the input folder
    entries = os.listdir(input_folder)
    entries.sort()
    
    # Filter out directories and logs from the list
    files = [entry for entry in entries if os.path.isfile(os.path.join(input_folder, entry)) and 'log' not in entry]
    
    # Loop through the files and copy every Nth file to the output folder
    for i in range(0, len(files), n):
        file = files[i]
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        
        # Copy the file to the output folder    
        shutil.copy2(input_path, output_path)
        print(f"Copied {file} to {output_path}")

#input_folder = '/Users/shiva.hnf/Desktop/ObjectAttention_w:json files/Andrea/session_1/mustard/rgb_input'
#output_folder = '/Users/shiva.hnf/Desktop/ObjectAttention_w:json files/Andrea/session_1/mustard/rgb_input/subsampled'
#take_every_n_file(input_folder, output_folder, 2)


def subsample_files(root_folder,n):
    """Reduce the number of files by only selecting every nth file for all the folders in the participant's folder.
    
    :param root_folder: str - path to the input folder.
    :param n: int - to select every nth file.
    """
    # Iterate over sessions
    for session_folder in os.listdir(root_folder):
        session_path = os.path.join(root_folder, session_folder)
        
        # Check if the item in the root folder is a directory
        if os.path.isdir(session_path):
            # Iterate over settings
            for setting_folder in os.listdir(session_path):
                setting_path = os.path.join(session_path, setting_folder)
                
                # Check if the item in the session folder is a directory
                if os.path.isdir(setting_path):
                    # Iterate over target objects
                    for object_folder in os.listdir(setting_path):
                        object_path = os.path.join(setting_path, object_folder)
                        
                        # Check if the item in the setting folde is  a directory
                        if os.path.isdir(object_path):
                            
                            for obj_subfolder in os.listdir(object_path):
                                obj_subfolder_path = os.path.join(object_path, obj_subfolder)
                                print('obj_subfolder_path: ' + obj_subfolder_path)
                                if os.path.isdir(obj_subfolder_path):
                                    print('obj_subfolder_path is folder')
                                    print('obj_subfolder: ' + obj_subfolder)
                                    if 'rgb' in obj_subfolder and 'openpose' not in obj_subfolder:
                                        print('passed if statement')
                                        # Create the output folder if it doesn't exist
                                        image_subsampled_path = os.path.join(root_folder,session_folder + "_subsampled",setting_folder,object_folder)
                                        if not os.path.exists(image_subsampled_path):
                                            os.makedirs(image_subsampled_path)  
                                        # Get tge list of images
                                        image_files = [file for file in os.listdir(obj_subfolder_path) if os.path.isfile(os.path.join(obj_subfolder_path, file)) and 'log' not in file]
                                        image_files.sort()
                                        for i in range(0, len(image_files), n):
                                            file = image_files[i]
                                            image_input_path = os.path.join(obj_subfolder_path, file)
                                                                       
                                            # Copy the file to the output folder    
                                            shutil.copy2(image_input_path, image_subsampled_path)

                                    if 'json' in obj_subfolder:
                                        # Create the output folder if it doesn't exist
                                        jason_subsampled_path = os.path.join(root_folder,session_folder + "_subsampled",setting_folder,"json", object_folder + "_json")
                                        if not os.path.exists(jason_subsampled_path):
                                            os.makedirs(jason_subsampled_path) 
                                        # Get the list of JSON files
                                        json_files = [file for file in os.listdir(obj_subfolder_path) if os.path.isfile(os.path.join(obj_subfolder_path, file))]
                                        json_files.sort()
                                        for i in range(0, len(json_files), n):
                                            file = json_files[i]
                                            json_input_path = os.path.join(obj_subfolder_path, file)
                                                                        
                                            # Copy the file to the output folder    
                                            shutil.copy2(json_input_path, jason_subsampled_path)
                                            
root_folder = '/Users/shiva.hnf/Desktop/ObjectAttention_w_json_files/Stefano'
subsample_files(root_folder,2)