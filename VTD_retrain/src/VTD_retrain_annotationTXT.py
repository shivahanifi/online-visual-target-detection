# Code to prepare annotation TXT files required for retraining the VTD architecture

#----------------- FILL THE INFO REQUIRED --------------------#

# Path to the frames
frame_path = '/Users/shiva.hnf/Documents/IIT/VTD_retrain/VTD_dumper_Shiva_modified/VTD_dumper_Shiva/session_5/pringles-masterchef-sugarbox-bleach-mustard/session5-shiva-pringles/renamed_rgb_input'
# Path to the head bbox annotation XML file
#head_xml_path = '/Users/shiva.hnf/Documents/IIT/VTD_retrain/VTD_dumper_Shiva_modified/VTD_dumper_Shiva/session_5/pringles-masterchef-sugarbox-bleach-mustard/session5-shiva-pringles/Head_bbox_annotation/00000003.xml'
# Gaze target name
gaze_target = 'pringles'
# Path for the annotation XML
xml_path = '/Users/shiva.hnf/Documents/IIT/VTD_retrain/VTD_dumper_Shiva_modified/VTD_dumper_Shiva/session_5/pringles-masterchef-sugarbox-bleach-mustard/session5-shiva-pringles/Object_annotations/00000000.xml'
# Path to save the TXT file
txt_path = '/Users/shiva.hnf/Documents/IIT/VTD_retrain/VTD_dumper_Shiva_modified/VTD_dumper_Shiva/session_5/pringles-masterchef-sugarbox-bleach-mustard/S52.txt'


import xml.etree.ElementTree as ET
import os

def head_bbox_info(xml_path):
    """Extract the information related to the head bbox.
    
    :param head_xml_path: str - string indicating the path in which 
                            head bbox annotation in the form of XML file is stored.
    :return filename: str - string indicating the name of the file
    :return xmin: str 
    :return xmax: str 
    :return ymin: str 
    :return ymax: str 
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    for ann in root.iter('annotation'):
        filename = ann.find('filename').text
        objects = root.findall('object')
        for obj in objects:
            if obj.find('name').text == 'head':
                xmin = ann.find('object/bndbox/xmin').text
                xmax = ann.find('object/bndbox/xmax').text
                ymin = ann.find('object/bndbox/ymin').text
                ymax = ann.find('object/bndbox/ymax').text
    
    return filename, xmin, xmax, ymin, ymax

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

def frame_name(frame_path):
    frames = os.listdir(frame_path)
    frames.sort()
    # Filter out only the image files
    image_files = [file for file in frames if file.endswith('.ppm')]
    # Extract the image names
    frame_names = [os.path.splitext(file)[0] for file in image_files]
    return frame_names

def create_new_txt(txt_path):
    f = open(txt_path,'w') 
    f.write('frame x_min y_min x_max y_max gaze_x gaze_y\n')
    f.close()
    
    
filename, xmin, xmax, ymin, ymax = head_bbox_info(xml_path)
gaze_x, gaze_y = gaze_target_info(xml_path, gaze_target)
frame_names = frame_name(frame_path)

for frame in frame_names:
    line_to_write = frame.split('.')[0] + ', ' + xmin + ', ' + ymin + ', ' + xmax + ', ' + ymax + ', ' + gaze_x + ', ' + gaze_y + '\n'
    with open(txt_path, 'a') as f:
        f.write(line_to_write)