import os
from unittest import result
import cv2
import copy
import mmcv
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T

from nuscenes.eval.common.utils import Quaternion
import pickle
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS, CustomDataset
from mmdet3d.datasets.pipelines import Compose

from .base_dataset import BaseDataset
import json
import math
import io
from pathlib import Path
from .pipelines.loading import *
from .pipelines.formating import CarlaCollect
import csv

@DATASETS.register_module()
class CarlaDataset(BaseDataset):
    r"""Carla Dataset.
    """

    def __init__(self, root, cfg, full_queue_pipeline):
        super(CarlaDataset, self).__init__()
        self.cfg=cfg
        # From Think Twice
        self.resize_to21x21 = T.Resize(size=(21, 21))
        self.history_query_index_lis = cfg["history_query_index_lis"]
        self.full_queue_pipeline = Compose(full_queue_pipeline)
        self.is_local=True

        # From our data loader
        self.images = []
        self.depth_img = []
        self.seg_img = []
        self.measurements = []
        self.commands = []
        self.lidar=[]
        self.route_info=[]
        route_idx=0
        self.routes=[]
        data = os.listdir(root)
        
        self.max_speed=cfg["max_speed"]
        self.max_steer=cfg["max_steer"]
        self.init_empty=False
        self.zero_results={}

        for route in data:
            self.routes.append(os.path.join(root, route))

        count=0
        for route in self.routes:
            
            route_dir = Path(route)

            route_len = len(os.listdir(route_dir / "rgb_images"))
            route_measurements = []
            route_commands = []

            measurement= route_dir / "measurements.csv"
            with open(measurement) as csvfile:
                file = csv.DictReader(csvfile)
                for row in file:
                    for key in row.keys():
                        row[key]=float(row[key])
                    route_measurements.append(row)

            command = route_dir / "commands.csv"
            with open(command) as csvfile:
                file = csv.DictReader(csvfile)
                for row in file:
                    for key in row.keys():
                        row[key]=float(row[key])
                    route_commands.append(row)

            # Loads by sequence
            seq_images, seq_depth_img, seq_seg_img, seq_lidar = [],[], [], []
            for idx in range(1,route_len+1):
                count+=1
                seq_images.append(route_dir / "rgb_images" / ("%d.png" % (idx)))
                seq_depth_img.append(route_dir / "depth_images" / ("%d.npy" % (idx)))
                seq_seg_img.append(route_dir / "semantic_images" / ("%d.png" % (idx)))
                seq_lidar.append(route_dir / "lidar" / ("%d.npy" % (idx)))
                self.route_info.append(route_idx)
            self.measurements.append(route_measurements)
            self.commands.append(route_commands)
            self.images.append(seq_images)
            self.depth_img.append(seq_depth_img)
            self.seg_img.append(seq_seg_img)
            self.lidar.append(seq_lidar)
            route_idx+=1

            assert(len(self.measurements[-1])==len(self.commands[-1]) and len(self.measurements[-1])==len(self.images[-1]))
        
        # There is a complex "memory leak"/performance issue when using Python objects like lists in a Dataloader that is loaded with multiprocessing, num_workers > 0
        # A summary of that ongoing discussion can be found here https://github.com/pytorch/pytorch/issues/13246#issuecomment-905703662
        # A workaround is to store the string lists as numpy byte objects because they only have 1 refcount.
        self.image_loader=LoadMultiImages(self.images,cfg["camera_names"])
        self.depth_loader=LoadDepth(self.depth_img,cfg["camera_names"])
        self.seg_loader=LoadSeg(self.seg_img,cfg["camera_names"],cfg["seg_converter"])
        self.points_loader=LoadPoints(coord_type='LIDAR', load_dim=4, use_dim=4)
        self.formatting=CarlaCollect(keys=[
                'img', 'points', 'depth', 'seg', 'waypoints', 'target_point',
                'speed', "future_speed", "steer",
                'action_acker_speed', 'action_acker_steer',
                'future_action_acker_speed', 'future_action_acker_steer'])
        print("Loading %d frames from %d folders"%(count, len(self.routes)))
        self.data_length = count
        #For debugging
        if cfg["is_dev"]:
            self.images = [self.images[0][:2]]
            self.depth_img = [self.depth_img[0][:2]]
            self.seg_img = [self.seg_img[0][:2]]
            self.lidar = [self.lidar[0][:2]]
            self.image_loader=LoadMultiImages(self.images,cfg["camera_names"])
            self.depth_loader=LoadDepth(self.depth_img,cfg["camera_names"])
            self.seg_loader=LoadSeg(self.seg_img,cfg["camera_names"],cfg["seg_converter"])
            self.points_loader=LoadPoints(coord_type='LIDAR', load_dim=4, use_dim=4)
            self.measurements = [self.measurements[0][:2]]
            self.commands = [self.commands[0][:2]]
            self.data_length=len(self.measurements[0])
        self.flag = np.ones(len(self), dtype=np.uint8)
            

    
    def load_json(self, fname):
        if self.is_local:
            with open(fname, "r") as f:
                return json.load(f)
        else:
            return json.loads(self.client.get(fname))

    def load_npy(self, fname):
        if self.is_local:
            with open(fname, "rb") as f:
                return np.load(fname, allow_pickle=True)
        else:
            return np.load(io.BytesIO(self.client.get(fname)), allow_pickle=True)

    def offset_then_rotate(self, target_2d_world_coor, ref_2d_wolrd_coor, ref_yaw):
        final_coor = target_2d_world_coor - ref_2d_wolrd_coor
        R = np.array([
            [np.cos(ref_yaw), -np.sin(ref_yaw)],
            [np.sin(ref_yaw), np.cos(ref_yaw)]
        ])
        return np.einsum("ij,kj->ki", R.T, final_coor)

    def get_waypoints(self, measurements, id, pred_length, rotMatrix):
        waypoints = []
        speed = []
        robot_x, robot_y = float(measurements[id]["robot_pose_x"]), float(measurements[id]["robot_pose_y"])
        for i in range(id+1,min(id+pred_length+1,len(measurements))):
            wx, wy = float(measurements[i]["robot_pose_x"]), float(measurements[i]["robot_pose_y"])
            #Transform to local positions
            waypoint = np.dot(rotMatrix,[wx-robot_x, wy-robot_y])
            waypoints.append(waypoint)
            speed.append(measurements[i]["speed"])
        while len(waypoints)<pred_length: waypoints.append([0.0,0.0])
        
        return np.array(waypoints), np.array(speed)

    ## Preprocess for single frame
    def get_data_info(self, route_idx, idx, is_current):
        results = {}
        measurements = self.measurements[route_idx][idx]
        ego_theta = -measurements["robot_pose_yaw"]*np.pi/180.0
        results["input_theta"] = ego_theta
        results["input_x"] = measurements["robot_pose_x"] ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        results["input_y"] = -measurements["robot_pose_y"] ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        ego_xy = np.stack([results["input_x"], results["input_y"]], axis=-1)
        rotMatrix = np.array([[np.cos(ego_theta), -np.sin(ego_theta)], 
							[np.sin(ego_theta),  np.cos(ego_theta)]])
        if is_current:
            waypoints, future_speed = self.get_waypoints(self.measurements[route_idx], idx, self.cfg["pred_len"], rotMatrix)
            results['waypoints'] = waypoints
            results["future_speed"] = future_speed

        results["speed"] = measurements["speed"]/self.max_speed
        results["steer"] = measurements['steering_angle']/self.max_steer
        results["can_bus"] = np.zeros(18)
        results["can_bus"][0] = results["input_x"] ## Gloabal Coordinate
        results["can_bus"][1] = results["input_y"] ## Global Coordinate
        accel = np.array([measurements["accelerometer_x"],measurements["accelerometer_y"],measurements["accelerometer_z"]])
        accel[:2] = self.offset_then_rotate(np.array(accel[:2])[np.newaxis, :], np.array([0, 0]), ego_theta).squeeze(0)
        results["can_bus"][7:10] = accel
        results["can_bus"][10:13] = [0,0,0]
        results["can_bus"][13] = measurements["speed"]
        results["can_bus"][-2] = ego_theta
        results["can_bus"][-1] = ego_theta / np.pi * 180
        
        x_target = measurements["target_pos_x"]  ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        y_target = -measurements["target_pos_y"]  ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        target_point=np.array([x_target,y_target])
        results['target_point'] = np.dot(rotMatrix,target_point)
        results['target_point_aim'] = results["target_point"]

        results["pts_filename"] = self.lidar[route_idx][idx]
        if is_current:
            commands = self.commands[route_idx]
            results['action_acker_speed']=np.array([commands[idx]["desired_speed"]/self.cfg["max_speed"]])
            results['action_acker_steer']=np.array([commands[idx]["desired_steering_angle"]/self.max_steer])
            future_action_acker_speed, future_action_acker_steer = [],[]
            for j in range(idx+1,min(idx+self.cfg["pred_len"],len(commands))):
                future_action_acker_speed.append(float(commands[j]["desired_speed"])/self.max_speed)
                future_action_acker_steer.append(float(commands[j]["desired_steering_angle"])/self.max_steer)
            while len(future_action_acker_speed)<self.cfg["pred_len"]-1:
                future_action_acker_speed.append(0.0)
                future_action_acker_steer.append(0.0)
            results['future_action_acker_speed']=np.array(future_action_acker_speed).reshape(-1,1)
            results['future_action_acker_steer']=np.array(future_action_acker_steer).reshape(-1,1)
        
        #Load images and LiDAR
        self.image_loader(results,route_idx,idx)
        self.depth_loader(results,route_idx,idx)
        self.seg_loader(results,route_idx,idx)
        self.points_loader(results)
        results=self.formatting(results)

        return results

    def prepare_train_data(self, route_idx, index):
        """Returns the item at index idx. """
        data_queue = []
        # temporal aug      # random choose 3 frames from last 4 frames
        #prev_indexs_list = list(range(index-self.queue_length, index))
        #random.shuffle(prev_indexs_list)
        #prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        prev_indexs_list = self.history_query_index_lis
        
        input_dict = self.get_data_info(route_idx, index, is_current=True)
        if input_dict is None:
            return None

        #example = self.pipeline(input_dict)
        data_queue.insert(0, input_dict)
        # Load prev frames, not load the current
        for i in prev_indexs_list[:-1][::-1]:
            input_dict = self.get_data_info(route_idx, index + i, is_current=False)
            input_dict["is_local"] = self.is_local
            # example = self.pipeline(input_dict)
            data_queue.insert(0, copy.deepcopy(input_dict))
        return union2one(self.full_queue_pipeline, data_queue)
    
    def __getitem__(self, index):
        prev_indexs_list = self.history_query_index_lis
        route_idx= self.route_info[index]
        for i in range(route_idx):
            index-=len(self.measurements[i])
        while prev_indexs_list[0]+index<0:
            index+=1
        while True:
            data = self.prepare_train_data(route_idx, index)
            return data


    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example



def get_ego_shift(delta_x, delta_y, ego_angle):
    # obtain rotation angle and shift with ego motion
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi)
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) 
    return shift_x, shift_y


### Multiple frames
def union2one(full_queue_pipeline, queue):
    queue = full_queue_pipeline(queue)
    imgs_list = torch.stack([each['img'] for each in queue])
    points_size = [each['points'].data.shape[0] for each in queue]
    metas_map = []
    prev_pos = None
    prev_angle = None
    for i, each in enumerate(queue):
        meta = copy.deepcopy(each['img_metas'].data)
        meta['points_size'] = points_size
        if i == 0: ## Current Frame
            meta['prev_bev'] = False
            prev_pos = copy.deepcopy(meta['can_bus'][:3])
            prev_angle = copy.deepcopy(meta['can_bus'][-1])
            meta['can_bus'][:3] = 0
            meta['can_bus'][-1] = 0
        else:
            meta['prev_bev'] = True
            tmp_pos = copy.deepcopy(meta['can_bus'][:3])
            tmp_angle = copy.deepcopy(meta['can_bus'][-1])
            meta['can_bus'][:3] -= prev_pos
            meta['can_bus'][-1] -= prev_angle
            prev_pos = copy.deepcopy(tmp_pos)
            prev_angle = copy.deepcopy(tmp_angle)
        metas_map.append(meta)

    # sweep2key transformation
    metas_map[-1]['curr2key'] = torch.eye(4)
    metas_map[-1]['currlidar2keycam'] = metas_map[-1]['lidar2cam']
    key_x, key_y = queue[-1]['img_metas'].data['can_bus'][:2]
    key_yaw = queue[-1]['img_metas'].data['can_bus'][-2]
    for i in range(len(queue)-2, -1, -1):
        curr_x = queue[i]['img_metas'].data['can_bus'][0]
        curr_y = queue[i]['img_metas'].data['can_bus'][1]
        curr2key_x, curr2key_y = get_ego_shift(
            key_x - curr_x,
            key_y - curr_y,
            key_yaw / np.pi * 180
        )
        curr_yaw = queue[i]['img_metas'].data['can_bus'][-2]
        curr2key_angle = key_yaw - curr_yaw 
        # get transmation mats
        R = torch.eye(4)
        R[:2,:2] = torch.Tensor([[np.cos(curr2key_angle), np.sin(curr2key_angle)],
                    [-np.sin(curr2key_angle), np.cos(curr2key_angle)]])
        T = torch.eye(4)
        T[0,3], T[1,3] = curr2key_x, curr2key_y
        curr2key = R @ T
        metas_map[i]['curr2key'] = curr2key
        metas_map[i]['currlidar2keycam'] = metas_map[i]['lidar2cam'] @ curr2key

    # dense-fusion
    points = queue[-1]['points'].data     
    points = torch.cat([points,torch.zeros(points.shape[0],1)], dim=1)
    points[:,4] = 0
    points_list = [points]
    for i in range(len(queue)-2, -1, -1):
        points_sweep = copy.deepcopy(queue[i]['points'].data)
        points_sweep = torch.cat([points_sweep,torch.zeros(points_sweep.shape[0],1)], dim=1)
        curr2key = metas_map[i]['curr2key']
        points_sweep[:, :4] = (curr2key@points_sweep[:, :4].T).T
        timestamp = i-(len(queue)-1)
        points_sweep[:,4] = timestamp
        points_list.append(points_sweep)
        points = torch.cat(points_list).unsqueeze(0)
        queue[-1]['points'] = DC(points, cpu_only=False, stack=True)
        
    queue[-1]['img'] = DC(imgs_list,
                            cpu_only=False, stack=True)
    queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
    queue = queue[-1]
    return queue
