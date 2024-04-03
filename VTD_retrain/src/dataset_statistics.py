import os
import numpy as np
import cv2
import glob
import xml.etree.ElementTree as ET

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

root_path ='/Users/shiva.hnf/Desktop/ObjectAttention_augmentation'

density_rect_list = []
density_circle_list = []
total_frame_count = 0

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
                                # Check if the item in the session folder is a directory
                                if os.path.isdir(setting_path):
                                    #print(f"setting_path : {setting_path}")
                                    # Iterate over target objects
                                    for object_folder in os.listdir(setting_path):
                                        # Path to XML annotation
                                            xml_file = glob.glob(os.path.join(setting_path, '*.xml'))
                                            xml_path = os.path.join(setting_path, xml_file[0])
                                            if '.DS' not in object_folder and 'xml' not in object_folder:
                                                object_path = os.path.join(setting_path, object_folder)
                                                
                                                # Check if the item in the setting folder is  a directory
                                                if os.path.isdir(object_path):
                                                    canvas_rect = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                                                    canvas_circles = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
                                                    #print(f"object_path : {object_path}")
                                                    for obj_subfolder in os.listdir(object_path):
                                                        if '.DS' not in obj_subfolder:
                                                            obj_subfolder_path = os.path.join(object_path, obj_subfolder)
                                                            if os.path.isdir(obj_subfolder_path):    
                                                                #print(f"obj_subfolder_path : {obj_subfolder_path}")
                                                                if 'rgb' in obj_subfolder:
                                                                    obj_frames = [frame for frame in os.listdir(obj_subfolder_path) if frame.lower().endswith('.ppm')]
                                                                    obj_frames.sort()
                                                                    num_frames = len(obj_frames)
                                                                    tree = ET.parse(xml_path)
                                                                    root = tree.getroot()
                                                                    objects = root.findall('object')
                                                                    for obj in objects:
                                                                        if obj.find('name').text == str(object_folder) :
                                                                            xmin_obj = int(obj.find('bndbox/xmin').text)
                                                                            xmax_obj = int(obj.find('bndbox/xmax').text)
                                                                            ymin_obj = int(obj.find('bndbox/ymin').text)
                                                                            ymax_obj = int(obj.find('bndbox/ymax').text)
                                                                total_frame_count = total_frame_count + num_frames 
                                                                target_obj_center = ((xmin_obj + xmax_obj) // 2, (ymin_obj + ymax_obj) // 2)
                                                                radius = min((xmax_obj - xmin_obj) // 2, (ymax_obj - ymin_obj) // 2)
                                                                #radius = 30
                                                                
                                                                # Gaze target as object bounding boxes
                                                                canvas_rect[ymin_obj:ymax_obj, xmin_obj:xmax_obj] += 1
                                                                total_canvas_rect = canvas_rect * num_frames
                                                                density_rect_list.append(total_canvas_rect)
                                                                
                                                                # Gaze target as solid circle
                                                                cv2.circle(canvas_circles, target_obj_center, radius, 255, -1)
                                                                canvas_circles +=1    
                                                                total_canvas_circle = canvas_circles * num_frames
                                                                density_circle_list.append(total_canvas_circle)
                                                                print('canvases for the participant {}, setting {}, object {} created'.format(participant_folder, setting_folder, object_folder))


# All the images for gaze target as object bounding box
sum_image = np.zeros_like(density_rect_list[0], dtype=np.float32)
for image in density_rect_list:
    sum_image += image

object_location_density_rect = (sum_image/total_frame_count * 255).astype(np.uint8)
# Apply thresholding to make white regions more specific
#_, thresholded_image = cv2.threshold(object_location_density, 0, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding to enhance white regions
#_, thresholded_image = cv2.threshold(object_location_density, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Histogram equalization
eq_density = cv2.equalizeHist(object_location_density_rect)

cv2.imshow('object location density as rectangles',eq_density)
cv2.imwrite('/Users/shiva.hnf/Documents/IIT/visual-targets/VTD_retrain/img/object_location_density.png', eq_density)

# All the images for gaze target as solid circles
sum_image_circle = np.zeros_like(density_circle_list[0], dtype=np.float32)
for image in density_circle_list:
    sum_image_circle += image

object_location_density_circle = (sum_image_circle/total_frame_count * 255).astype(np.uint8)
inverted_density_circle = cv2.bitwise_not(object_location_density_circle)
eq_density_circle = cv2.equalizeHist(inverted_density_circle)

# Apply Gaussian blur to the image
#blurred_density_circle = cv2.GaussianBlur(eq_density_circle, (0, 0), sigmaX=1)

# Invert the colors


cv2.imshow('object location density as circles', eq_density_circle)
cv2.imwrite('/Users/shiva.hnf/Documents/IIT/visual-targets/VTD_retrain/img/object_location_density_circles.png', eq_density_circle)


cv2.waitKey(0)
cv2.destroyAllWindows()                                                             
                                                                
                                                                
                                    
                                                                    

                                                                    
                                                                    
                                                                