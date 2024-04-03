import os
import shutil
import xml.etree.ElementTree as ET
import json
import numpy as np
import math
import glob

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
JOINTS_POSE_FACE = [0, 15, 16, 17, 18]

# Path to the dataset location 
root_path = '/Users/shiva.hnf/Desktop/under_process'

def read_openpose_from_json(json_filename):
    with open(json_filename) as data_file:
        loaded = json.load(data_file)

        poses = []
        conf_poses = []
        faces = []
        conf_faces = []

        for arr in loaded["people"]:
            conf_poses.append(arr["pose_keypoints_2d"][2::3]) #seq[start:end:step]
            arr_poses = np.delete(arr["pose_keypoints_2d"], slice(2, None, 3))
            poses.append(list(zip(arr_poses[::2], arr_poses[1::2]))) #respectively X and Y

            conf_faces.append(arr["face_keypoints_2d"][2::3])
            arr_faces = np.delete(arr["face_keypoints_2d"], slice(2, None, 3))  # remove confidence values from the array
            faces.append(list(zip(arr_faces[::2], arr_faces[1::2])))

    return poses, conf_poses, faces, conf_faces

def compute_centroid(points):
    mean_x = np.mean([p[0] for p in points])
    mean_y = np.mean([p[1] for p in points])

    if mean_x >= IMAGE_WIDTH:
        mean_x = IMAGE_WIDTH-1
    if mean_x < 0:
        mean_x = 0
    if mean_y >= IMAGE_HEIGHT:
        mean_y = IMAGE_HEIGHT-1
    if mean_y < 0:
        mean_y = 0

    return [mean_x, mean_y]

def joint_set(p):
     return p is not None and (p[0] != 0.0 or p[1] != 0.0)

def get_openpose_bbox(pose):

    n_joints_set = [pose[0][joint] for joint in JOINTS_POSE_FACE if joint_set(pose[0][joint])]
    if n_joints_set:
        centroid = compute_centroid(n_joints_set)

        min_x = min([joint[0] for joint in n_joints_set])
        max_x = max([joint[0] for joint in n_joints_set])
        min_x -= (max_x - min_x) * 0.2
        max_x += (max_x - min_x) * 0.2

        width = max_x - min_x

        min_y = centroid[1] - (width/3)*2
        max_y = centroid[1] + (width/3)*2

        min_x = math.floor(max(0, min(min_x, IMAGE_WIDTH)))
        max_x = math.floor(max(0, min(max_x, IMAGE_WIDTH)))
        min_y = math.floor(max(0, min(min_y, IMAGE_HEIGHT)))
        max_y = math.floor(max(0, min(max_y, IMAGE_HEIGHT)))

        return min_x, min_y, max_x, max_y
    else:
        #print("Joint set empty!")
        return None, None, None, None
  
  
def gaze_target_info(xml_path, gaze_target):
    """Extract the information related to the gaze target.
    
    :param target_xml_path: str - string indicating the path in which 
                            object bboxes annotation in the form of XML file is stored.
    :param gaze_target: str - string indicating the name of the gazed object.
    :return: gaze_x: str - string indicating the x position of the targeted pixel.
    :return: gaze_y: str - string indicating the y position of the targeted pixel.
    """  
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = root.findall('object')
    for obj in objects:
        if obj.find('name').text == gaze_target:
            xmin_obj = int(obj.find('bndbox/xmin').text)
            xmax_obj = int(obj.find('bndbox/xmax').text)
            ymin_obj = int(obj.find('bndbox/ymin').text)
            ymax_obj = int(obj.find('bndbox/ymax').text)    
            gaze_x = str((xmin_obj + xmax_obj) / 2)
            gaze_y = str((ymin_obj + ymax_obj) / 2)
    return gaze_x, gaze_y


def put_frames_in_order(root_path):
    # Iterate over participants
    for participant_folder in os.listdir(root_path):
        if '.DS' not in participant_folder:
            participant_path = os.path.join(root_path, participant_folder)
        
            # Check if the item in the root folder is a directory
            if os.path.isdir(participant_path):
                #print(f"participant_path : {participant_path}")         
                # Iterate over sessions
                for session_folder in os.listdir(participant_path):
                    if '.DS' not in session_folder:
                        session_path = os.path.join(participant_path, session_folder)
                    
                        # Check if the item in the participant folder is a directory
                        if os.path.isdir(session_path):
                            #print(f"session_path : {session_path}") 
                            # Iterate over settings
                            for setting_folder in os.listdir(session_path):
                                if '.DS' not in setting_folder:
                                    setting_path = os.path.join(session_path, setting_folder)
                                    # Number of objects in the previous object folder
                                    frame_count = 0
                                    json_count = 0
                                    # Check if the item in the session folder is a directory
                                    if os.path.isdir(setting_path):
                                        #print(f"setting_path : {setting_path}")
                                        # Iterate over target objects
                                        for object_folder in os.listdir(setting_path):
                                                if '.DS' not in object_folder and 'xml' not in object_folder:
                                                    object_path = os.path.join(setting_path, object_folder)
                                                    
                                                    # Path to save the annotation TXT file
                                                    txt_path = os.path.join(setting_path,'S01.txt')
                                                    
                                                    # Path to XML annotation
                                                    xml_file = glob.glob(os.path.join(setting_path, '*.xml'))
                                                    xml_path = os.path.join(setting_path, xml_file[0])
                                                
                                                    # Check if the item in the setting folder is  a directory
                                                    if os.path.isdir(object_path):
                                                        #print(f"object_path : {object_path}")
                                                        for obj_subfolder in os.listdir(object_path):
                                                            if '.DS' not in obj_subfolder:
                                                                obj_subfolder_path = os.path.join(object_path, obj_subfolder)
                                                                if os.path.isdir(obj_subfolder_path):
                                                                    
                                                                    #print(f"obj_subfolder_path : {obj_subfolder_path}")
                                                                    if 'rgb' in obj_subfolder and 'openpose' not in obj_subfolder:
                                                                        obj_frames = [frame for frame in os.listdir(obj_subfolder_path) if frame.lower().endswith('.ppm')]
                                                                        obj_frames.sort()
                                                                        obj_frames_sampled =obj_frames[::2]
                                                                        
                                                                        for frame in obj_frames_sampled:
                                                                            frame_path = os.path.join(obj_subfolder_path, frame)
                                                                            padded_frame_count = str(frame_count).zfill(5)
                                                                            new_frame_name = str(padded_frame_count) + '.' + frame.split('.')[1]
                                                                            new_frame_path = os.path.join(setting_path, "subsampled_frames", new_frame_name)
                                                                            # Create the output folder if it doesn't exist
                                                                            if not os.path.exists(os.path.join(setting_path, "subsampled_frames")):
                                                                                os.makedirs(os.path.join(setting_path, "subsampled_frames"))
                                                                            # Copy the frames to the new path
                                                                            shutil.copy2(frame_path, os.path.join(setting_path, "subsampled_frames"))
                                                                            #rename the frames
                                                                            os.rename(os.path.join(setting_path, "subsampled_frames", frame), new_frame_path)
                                                                            #print(f"renamed {frame} to {new_frame_name}. frame count is: {frame_count}")
                                                                            frame_count +=1
                                                                            
                                                                    if 'json' in obj_subfolder:
                                                                        obj_jsons = [obj_json for obj_json in os.listdir(obj_subfolder_path) if obj_json.lower().endswith('.json')]
                                                                        obj_jsons.sort()
                                                                        obj_jsons_sampled = obj_jsons[::2]
                                                                        
                                                                        for obj_json in obj_jsons_sampled:
                                                                            json_path = os.path.join(obj_subfolder_path, obj_json)
                                                                            padded_json_count = str(json_count).zfill(5)
                                                                            new_json_name = str(padded_json_count) + '.' + obj_json.split('.')[1]
                                                                            new_json_path = os.path.join(setting_path, "subsampled_json", new_json_name)
                                                                        
                                                                            # Create the output folder if it doesn't exist
                                                                            if not os.path.exists(os.path.join(setting_path, "subsampled_json")):
                                                                                os.makedirs(os.path.join(setting_path, "subsampled_json"))
                                                                            
                                                                            shutil.copy2(json_path, os.path.join(setting_path, "subsampled_json"))
                                                                            
                                                                            #rename the json files
                                                                            os.rename(os.path.join(setting_path, "subsampled_json",obj_json),new_json_path)
                                                                            json_count +=1
                                                                            #print(f"renamed {obj_json} to {new_json_name}. JSON count is {json_count}")
                                                                            
                                                                            # Preparing the annotation file
                                                                            # Get the gaze target X,Y from object annotation XML
                                                                            gaze_target = str(object_folder)    
                                                                            gaze_x, gaze_y = gaze_target_info(xml_path, gaze_target)                                    
                                            
                                                                            poses, conf_poses, faces, conf_faces = read_openpose_from_json(new_json_path)
                                                                            min_x, min_y, max_x, max_y = get_openpose_bbox(poses)
                                                                            line_to_write = str(new_json_name.split('_')[0]) + ', ' + str(min_x) + ', ' + str(min_y) + ', ' + str(max_x) + ', ' + str(max_y) + ', ' + str(gaze_x) + ', ' + str(gaze_y) + '\n'
                                                                            with open(txt_path, 'a') as f:
                                                                                f.write(line_to_write)
                                                    print(f"Subsampling, renaming, and annotation creation for participant {participant_folder}, {session_folder}, setting: {setting_folder}, and the target object {object_folder} is completed")    
                                                            
put_frames_in_order(root_path)
                                                
                                            
                                            