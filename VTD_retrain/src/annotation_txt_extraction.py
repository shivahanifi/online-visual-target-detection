import xml.etree.ElementTree as ET
import os
import json
import numpy as np
import math

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
JOINTS_POSE_FACE = [0, 15, 16, 17, 18]
# Path to the dataset location 
root_folder = '/Users/shiva.hnf/Desktop/ObjectAttention_w:json files/Andrea/session_1_subsampled/masterchef-masterchef/masterchef'
# Path to JSON files
json_path = '/Users/shiva.hnf/Desktop/ObjectAttention_w:json files/Andrea/session_1_subsampled/masterchef-masterchef/json/masterchef_json'
# XML path
xml_path = '/Users/shiva.hnf/Desktop/ObjectAttention_augmentation/Andrea/session_1/masterchef- masterchef/00000005.xml'
# Gaze target
gaze_target = 'masterchef'
# TXT path
txt_path = '/Users/shiva.hnf/Desktop/ObjectAttention_w:json files/Andrea/session_1_subsampled/masterchef-masterchef/S11.txt'



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

gaze_x, gaze_y = gaze_target_info(xml_path, gaze_target)                                    

json_files = os.listdir(json_path)
json_files.sort()
for json_file in json_files:
    json_file_path = os.path.join(json_path, json_file)
    poses, conf_poses, faces, conf_faces = read_openpose_from_json(json_file_path)
    min_x, min_y, max_x, max_y = get_openpose_bbox(poses)
    line_to_write = str(json_file.split('_')[0]) + ', ' + str(min_x) + ', ' + str(min_y) + ', ' + str(max_x) + ', ' + str(max_y) + ', ' + str(gaze_x) + ', ' + str(gaze_y) + '\n'
    with open(txt_path, 'a') as f:
        f.write(line_to_write)