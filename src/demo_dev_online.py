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
from PIL import Image
from scipy.misc import imresize
from model import ModelSpatial
from utils import imutils, evaluation
from config import *
from functions.config_vt import *
from functions.utilities_vt import *

import logging
logging.basicConfig(filename='my.log', level=logging.DEBUG)

yarp.Network.init()

class VisualTargetDetection(yarp.RFModule):
    def configure(self, rf):
        
        #gpu
        num_gpu = rf.find("num_gpu").asInt32()
        num_gpu_start = rf.find("num_gpu_start").asInt32()
        print('Num GPU: %d, GPU start: %d' % (num_gpu, num_gpu_start))
        init_gpus(num_gpu, num_gpu_start)    
        
        # input port for rgb image
        self.in_port_human_image = yarp.BufferedPortImageRgb()
        self.in_port_human_image.open('/vtd/image:i')
        self.in_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.in_buf_human_image = yarp.ImageRgb()
        self.in_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1],
                                            self.in_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:i'))
        
        # input port for depth
        self.in_port_human_depth = yarp.BufferedPortImageFloat()
        self.in_port_human_depth_name = '/vtd/depth:i'
        self.in_port_human_depth.open(self.in_port_human_depth_name)
        self.in_buf_human_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.in_buf_human_depth = yarp.ImageFloat()
        self.in_buf_human_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.in_buf_human_depth.setExternal(self.in_buf_human_depth_array.data, self.in_buf_human_depth_array.shape[1],
                                            self.in_buf_human_depth_array.shape[0])
        print('{:s} opened'.format('/vtd/depth:i'))
        
        # input port for openpose data
        self.in_port_human_data = yarp.BufferedPortBottle()
        self.in_port_human_data.open('/vtd/data:i')
        print('{:s} opened'.format('/vtd/data:i'))
        
        # output port for bboxes
        self.out_port_human_image = yarp.Port()
        self.out_port_human_image.open('/vtd/image:o')
        self.out_buf_human_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_human_image = yarp.ImageRgb()
        self.out_buf_human_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_human_image.setExternal(self.out_buf_human_array.data, self.out_buf_human_array.shape[1],
                                             self.out_buf_human_array.shape[0])
        print('{:s} opened'.format('/vtd/image:o'))

        # propag input image
        self.out_port_propag_image = yarp.Port()
        self.out_port_propag_image.open('/vtd/propag:o')
        self.out_buf_propag_image_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
        self.out_buf_propag_image = yarp.ImageRgb()
        self.out_buf_propag_image.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_image.setExternal(self.out_buf_propag_image_array.data, self.out_buf_propag_image_array.shape[1],
                                             self.out_buf_propag_image_array.shape[0])
        print('{:s} opened'.format('/vtd/propag:o'))

        # propag input depth
        self.out_port_propag_depth = yarp.Port()
        self.out_port_propag_depth.open('/vtd/depth:o')
        self.out_buf_propag_depth_array = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 1), dtype=np.float32)
        self.out_buf_propag_depth = yarp.ImageFloat()
        self.out_buf_propag_depth.resize(IMAGE_WIDTH, IMAGE_HEIGHT)
        self.out_buf_propag_depth.setExternal(self.out_buf_propag_depth_array.data, self.out_buf_propag_depth_array.shape[1],
                                              self.out_buf_propag_depth_array.shape[0])
        print('{:s} opened'.format('/vtd/depth:o'))

        # output port for the selection
        self.out_port_prediction = yarp.Port()
        self.out_port_prediction.open('/vtd/pred:o')
        print('{:s} opened'.format('/vtd/pred:o'))

        # output for the logger for the state machine
        self.out_port_state = yarp.Port()
        self.out_port_state.open('/vtd/state:o')
        print('{:s} opened'.format('/vtd/state:o'))
        
        self.human_image = np.ones((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)

        return True
        
    def updateModule(self):
        
        received_image = self.in_port_human_image.read()
        received_depth = self.in_port_human_depth.read(False)
        
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weights', type=str, help='model weights', default='model_demo.pt')
        parser.add_argument('--image_dir', type=str, help='images', default=FRAMES_DIR)
        #parser.add_argument('--head_dir', type=str, help='head bounding boxes', default=JSON_FILES)
        parser.add_argument('--vis_mode', type=str, help='heatmap or arrow', default='heatmap')
        parser.add_argument('--out_threshold', type=int, help='out-of-frame target dicision threshold', default=100)
        args = parser.parse_args()

        listOfJson = sorted(glob.glob(JSON_FILES + '/*.json'))
        json_log = logging.debug(listOfJson)

        #extracting image names
        idx = []
        for b in listOfJson:
            txt = b.split('/')
            txt1 = txt[6].split('_')
            txt2 = txt1[0] + '.jpg'
            idx.append(txt2)

        def _get_transform():
            transform_list = []
            transform_list.append(transforms.Resize((input_resolution, input_resolution)))
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            return transforms.Compose(transform_list)


        def run():
            column_names = ['frame', 'left', 'top', 'right', 'bottom']
            df = pd.DataFrame()
            for j in listOfJson:
                #print(j)
                poses, conf_poses, faces, conf_faces = read_openpose_from_json(j)
                #logging.debug(poses)
                min_x, min_y, max_x, max_y = get_openpose_bbox(poses)
                #logging.debug(min_x)
                #line_to_write = j + ',' + min_x + ',' + min_y + ',' + max_x + ',' + max_y + '\n'
                line_to_write = [[j, min_x, min_y, max_x, max_y]]
                df_tmp = pd.DataFrame(line_to_write, columns=column_names)
                #print(df_tmp)
                df = df.append(df_tmp, ignore_index=True)
            logging.debug(df)
            df['left'] -= (df['right']-df['left'])*0.1
            df['right'] += (df['right']-df['left'])*0.1
            df['top'] -= (df['bottom']-df['top'])*0.1
            df['bottom'] += (df['bottom']-df['top'])*0.1


            #logging.debug(df)
            # set up data transformation
            test_transforms = _get_transform()

            model = ModelSpatial()
            model_dict = model.state_dict()
            pretrained_dict = torch.load(args.model_weights)
            pretrained_dict = pretrained_dict['model']
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            model.cuda()
            model.train(False)

            with torch.no_grad():
                for i in df.index:
                    frame_raw = Image.open(os.path.join(args.image_dir, str(idx[i])))
                    frame_raw = frame_raw.convert('RGB')
                    width, height = frame_raw.size

                    #logging.debug(width)
                    #logging.debug(height)

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
                    raw_hm = raw_hm.cpu().detach().numpy() * 255
                    raw_hm = raw_hm.squeeze()
                    inout = inout.cpu().detach().numpy()
                    inout = 1 / (1 + np.exp(-inout))
                    inout = (1 - inout) * 255
                    norm_map = imresize(raw_hm, (height, width)) - inout

                    #logging.debug(norm_map)

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

                    if args.vis_mode == 'arrow':
                        if inout < args.out_threshold: # in-frame gaze
                            pred_x, pred_y = evaluation.argmax_pts(raw_hm)
                            norm_p = [pred_x/output_resolution, pred_y/output_resolution]
                            circ = patches.Circle((norm_p[0]*width, norm_p[1]*height), height/50.0, facecolor=(0,1,0), edgecolor='none')
                            ax.add_patch(circ)
                            plt.plot((norm_p[0]*width,(head_box[0]+head_box[2])/2), (norm_p[1]*height,(head_box[1]+head_box[3])/2), '-', color=(0,1,0,1))

                            #logging.debug(args.vis_mode)
                            #logging.debug(inout)
                            #logging.debug(args.out_threshold)
                            #logging.debug(norm_p)

                    else:
                        plt.imshow(norm_map, cmap = 'jet', alpha=0.2, vmin=0, vmax=255)

                    plt.show(block=False)
                    plt.pause(1)
                    plt.savefig('/home/r1-user/code_sh/new_new/attention-target-detection/data/demo/offLine_output/fig{0}.png'.format(i))

                print('DONE!')


if __name__ == "__main__":
    run()
