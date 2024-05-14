# Copyright (c) OpenMMLab. All rights reserved.
import os
from types import new_class
import cv2
import time
import copy
import torch
import numpy as np

import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
from mmdet.datasets.builder import PIPELINES

from torchvision import transforms as T

import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
import io
from scipy.ndimage import label as sep_mask


#Changes to load one image, and the filenames are already known.
@PIPELINES.register_module()
class LoadMultiImages(object):
    def __init__(self,
                 images_filename,
                 camera_names,
                ):
        self.images_filename = images_filename
        self.camera_names=camera_names
        self.file_client_args=dict(backend='disk')
        self.file_client = mmcv.FileClient(**self.file_client_args)
    def __call__(self, results, route_index, idx):
        #The image is joined already, separate it for this network.
        image=np.array(Image.open(self.images_filename[route_index][idx]))
        results['img'] = np.array([image[:,512*i:512*(i+1)] for i in range(len(self.camera_names))])
        results["img_filename"] = self.images_filename[route_index][idx]
        return results
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(is_local={self.is_local}, '
        return repr_str



@PIPELINES.register_module()
class LoadPoints(LoadPointsFromFile):
    def __init__(self,
                 **kwargs,
                ):
        super().__init__(**kwargs)

    def _load_points(self, pts_filename):
        points = np.load(pts_filename, allow_pickle=True)
        return points

#Our depth images are converted already in float.
@PIPELINES.register_module()
class LoadDepth(LoadMultiImages):
    def __init__(self,
                 images_filename,
                 camera_names,
                ):
        super().__init__(images_filename,camera_names)
    def __call__(self, results, route_index, idx):
        #The image is joined already, separate it for this network.
        image=np.load(self.images_filename[route_index][idx]).reshape(128,-1)
        results['depth'] = np.array([image[:,512*i:512*(i+1)] for i in range(len(self.camera_names))])
        return results

## Segment Traffic Light
def red_green_yellow(rgb_image):
    hsv = cv2.cvtColor(rgb_image[:, None, :], cv2.COLOR_RGB2HSV)
    avg_saturation = int(hsv[:,:,1].mean()) # Sum the brightness values   
    sat_low = int(avg_saturation * 1.1)#1.3)
    val_low = 140
    # Green
    lower_green = np.array([70,sat_low,val_low])
    upper_green = np.array([100,255,255])
    sum_green = cv2.inRange(hsv, lower_green, upper_green).astype(np.bool8).sum()
    # Red
    lower_red = np.array([150,sat_low,val_low])
    upper_red = np.array([180,255,255])
    sum_red = cv2.inRange(hsv, lower_red, upper_red).astype(np.bool8).sum()
    if sum_red < 3 and sum_green < 3:
        return 0 #not sure or yellow
    if sum_red >= sum_green:
        return 1# Red
    return 2 # Green


@PIPELINES.register_module()
class LoadSeg(LoadMultiImages):
    def __init__(self,
                 images_filename,
                 camera_names,
                 converter,
                ):
        super().__init__(images_filename,camera_names)
        self.seg_converter = np.uint8(converter)
    
    def __call__(self, results, route_index, idx):
        seg = Image.open(self.images_filename[route_index][idx]).convert('L')
        seg = self.seg_converter[seg]
        results['seg']=np.array([seg[:,512*i:512*(i+1)] for i in range(len(self.camera_names))])
        return results

