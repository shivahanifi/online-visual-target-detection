import yarp
import argparse, os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import math
import glob
import yarp
import PIL
import sys
import io
from PIL import Image
from PIL import ImageShow
from scipy.misc import imresize
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from functions.config_vt import *
from functions.utilities_vt import *


# Initialize YARP
yarp.Network.init()

class VisualTargetDetection(yarp.RFModule):
    def configure(self, rf):
        
        # GPU
        num_gpu = rf.find("num_gpu").asInt32()
        num_gpu_start = rf.find("num_gpu_start").asInt32()
        print('Num GPU: %d, GPU start: %d' % (num_gpu, num_gpu_start))
        init_gpus(num_gpu, num_gpu_start)    
        
        # Command port
        self.cmd_port = yarp.Port()
        self.cmd_port.open('/vtd/command:i')
        print('{:s} opened'.format('/vtd/command:i'))
        self.attach(self.cmd_port)
        
        # Input port and buffer for rgb image
        # Create the port and name it
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/vtd/image:i')
        # Create numpy array to receive the image 
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        # Wrap YARP image around the array
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1],
                                            self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:i'))
        
        
        # Input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/vtd/data:i')
        print('{:s} opened'.format('/vtd/data:i'))
        
        # Output port for bboxes
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/vtd/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1],
                                             self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:o'))

        # Propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/vtd/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/vtd/propag:o'))

        # Output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/vtd/pred:o')
        print('{:s} opened'.format('/vtd/pred:o'))
        
        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        # To get argumennts needed
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
        #parser.add_argument('--image_dir', type=str, help='images', default=FRAMES_DIR)
        #parser.add_argument('--head_dir', type=str, help='head bounding boxes', default=JSON_FILES)
        parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='heatmap')
        parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
        self.args = parser.parse_args()

        return True
    
    # Respond to a message
    def respond(self, command, reply):
        if command.get(0).asString() == 'quit':
            # command: quit
            #self.TRAIN = 0
            self.cleanup()
            reply.addString('Quit command sent')
        return True
    
    def cleanup(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True

    # Closes all the ports after execution
    def close(self):
        print('Cleanup function')
        self.in_port_human_image.close()
        self.in_port_human_depth.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_propag_depth.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True


    # Called after a quit command (Does nothing)
    def interruptModule(self):
        print('Interrupt function')
        self.in_port_human_image.close()
        self.in_port_human_data.close()
        self.out_port_human_image.close()
        self.out_port_propag_image.close()
        self.out_port_prediction.close()
        self.cmd_port.close()
        return True

    # Desired period between successive calls to updateModule()
    def getPeriod(self):
        return 0.001


    def updateModule(self):
    
        received_image = self.in_port_human_image.read()
        self.in_buf_human_image.copy(received_image)
        assert self.in_buf_human_array.__array_interface__['data'][0] == self.in_buf_human_image.getRawImage().__int__()

        # Convert the numpy array to a PIL image
        pil_image = Image.fromarray(self.in_buf_human_array)
        
       
        # To check the input
        #pil_image.save('/projects/test_images/pil_image.png')

        if pil_image:
            received_data = self.in_port_human_data.read()
            if received_data:
                try:
                    poses, conf_poses, faces, conf_faces = read_openpose_data(received_data)
                
                    if poses:
                        min_x, min_y, max_x, max_y = get_openpose_bbox(poses)

                        column_names = ['left', 'top', 'right', 'bottom']
                        line_to_write = [[min_x, min_y, max_x, max_y]]
                        df = pd.DataFrame(line_to_write, columns=column_names)
    
                        df['left'] -= (df['right']-df['left'])*0.1
                        df['right'] += (df['right']-df['left'])*0.1
                        df['top'] -= (df['bottom']-df['top'])*0.1
                        df['bottom'] += (df['bottom']-df['top'])*0.1


        #                 #logging.debug(df)
        #                 # set up data transformation
                        test_transforms = get_transform()

                        model = ModelSpatial()
                        model_dict = model.state_dict()
                        pretrained_dict = torch.load(self.args.model_weights)
                        pretrained_dict = pretrained_dict['model']
                        model_dict.update(pretrained_dict)
                        model.load_state_dict(model_dict)

                        model.cuda()
                        model.train(False)

                        with torch.no_grad():
                            for i in df.index:
                                frame_raw = pil_image
                            
                                width, height = frame_raw.size

                                head_box = [df.loc[i,'left'], df.loc[i,'top'], df.loc[i,'right'], df.loc[i,'bottom']]

                                head = frame_raw.crop((head_box)) # head crop

                                head = test_transforms(head) # transform inputs
                                frame = test_transforms(frame_raw)
                                head_channel = imutils.get_head_box_channel(head_box[0], head_box[1], head_box[2], head_box[3], width, height,
                                                                            resolution=input_resolution).unsqueeze(0)

                                head = head.unsqueeze(0).cuda()
                                frame = frame.unsqueeze(0).cuda()
                                head_channel = head_channel.unsqueeze(0).cuda()

                                # forward pass
                                raw_hm, _, inout = model(frame, head_channel, head)

                                #logging.debug(raw_hm)
                                #logging.debug(inout)

                                # heatmap modulation
                                # raw_hm = raw_hm.cpu().detach().numpy() * 255
                                # raw_hm = raw_hm.squeeze()
                                # inout = inout.cpu().detach().numpy()
                                # inout = 1 / (1 + np.exp(-inout))
                                # inout = (1 - inout) * 255
                                # norm_map = imresize(raw_hm, (height, width)) - inout

                                # vis
                                plt.close()
                                plt.rcParams["figure.figsize"] = [20.00,20.00]
                                fig = plt.figure()
                                fig.canvas.manager.window.move(0,0)
                                plt.axis('off')
                                plt.imshow(frame_raw)

                                ax = plt.gca()
                                rect = patches.Rectangle((head_box[0], head_box[1]), head_box[2]-head_box[0], head_box[3]-head_box[1], linewidth=2, edgecolor=(0,1,0), facecolor='none')
                                ax.add_patch(rect)

        #                         if self.args.vis_mode == 'arrow':
        #                             if inout < self.args.out_threshold: # in-frame gaze
        #                                 pred_x, pred_y = evaluation.argmax_pts(raw_hm)
        #                                 norm_p = [pred_x/output_resolution, pred_y/output_resolution]
        #                                 circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
        #                                 ax.add_patch(circ)
        #                                 plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))

        #                                 #logging.debug(args.vis_mode)
        #                                 #logging.debug(inout)
        #                                 #logging.debug(args.out_threshold)
        #                                 #logging.debug(norm_p)

        #                         else:
        #                             plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

                                #plt.savefig('bbox.png', dpi=300)
                                #plt.show(block=False)
                                fig.canvas.draw()
                                img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.unit8)
                                yarp_image = yarp.ImageFloat()
                                yarp_image.setExternal(img_array, img_array.shape[1], img_array.shape[0])
                                self.out_port_human_image.write(yarp_image)
                                plt.pause(1)
        #                         plt.savefig('/home/r1-user/code_sh/new_new/attention-target-detection/data/demo/offLine_output/fig{0}.png'.format(i))

                                        # self.out_buf_human_array[:, :] = self.in_buf_human_array
                                        # self.out_port_human_image.write(self.out_buf_human_image)

                                print('DONE!')
                except:
                    print("An error occured")
                              

if __name__ == '__main__':
    rf = yarp.ResourceFinder()
    rf.setVerbose(True)
    rf.setDefaultContext("VisualTargetDetection")
    rf.setDefaultConfigFile('../app/config/.ini')
    rf.configure(sys.argv)

    # Run module
    manager = VisualTargetDetection()
    manager.runModule(rf)