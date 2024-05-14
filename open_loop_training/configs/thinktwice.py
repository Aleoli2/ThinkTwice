_base_ = [
    './_base_/default_runtime.py',
]

dev_max_sample_per_town = {"town01":10, "town02":10, "town03":10, "town04":10, "town05":10, "town06":10, "town07":10, "town10":10}
local_dev_train = ["town01_00", ]
local_dev_val = ["town01_00", ]

tcp_max_sample_per_town = {"town01":50384, "town02":55943, "town03":42771, "town04":47954, "town05":53684, "town06":48415, "town07":51549, "town10":59898} ## tcp num
train_towns = ["01", "03", "04", "06"]
val_twons =  ["02", "05"]
index_per_down = ["val"] + [str(_).zfill(2) for _ in range(0, 12)]

tcp_train = []
for town in train_towns:
    for town_index in index_per_down:
        tcp_train.append("town"+town + "_" + town_index)

tcp_val = []
for town in val_twons:
    for town_index in index_per_down:
        tcp_val.append("town"+town + "_" + town_index)

full_towns = ["01", "02", "03", "04", "05", "06", "07", "10"]
full_train = []
full_val = []
for town in full_towns:
    for town_index in index_per_down:
        if "val" == town_index:
            full_val.append("town"+town + "_" + town_index)
        else:
            full_train.append("town"+town + "_" + town_index)
max_sample_per_town_full = {"town01":1e9, "town02":1e9, "town03":1e9, "town04":1e9, "town05":1e9, "town06":1e9, "town07":1e9, "town10":1e9}

root_dir_all = "/carla_dataset/dataset/"

train_dir=root_dir_all+"train/"
val_dir=root_dir_all+"val/"

plugin = True
plugin_dir = 'code/'

img_aug = False
SyncBN=False
point_cloud_range = [-8.0, -19.2, -4.0, 30.4, 19.2, 10.0] ## The same as Roach
cfg = dict(
    max_speed=1.3,
    max_steer=24.0,
    ###From TCP
    pred_len = 4, # future waypoints predicted
    turn_KP = 0.75,
    turn_KI = 0.75,
    turn_KD = 0.3,
    turn_n = 40, # buffer size
    speed_KP = 5.0,
    speed_KI = 0.5,
    speed_KD = 1.0,
    speed_n = 40, # buffer size
    brake_speed = 0.4, # desired speed below which brake is triggered
    brake_ratio = 1.1, # ratio of speed to desired speed at which brake is triggered
    clip_delta = 0.25, # maximum change in speed input to logitudinal controller
    aim_dist = 4.0, # distance to search around for aim point
    angle_thresh = 0.3, # outlier control detection angle
    dist_thresh = 10, # target point y-distance for outlier filtering
    speed_weight = 0.05,
    value_weight = 0.001,
    features_weight = 0.05,
    img_aug=img_aug,

    ## ThinkTwice configuration
    undistort = False, ## Use the intrinsics of undistorted images
    unreal_coord = False, ## Use the coordinate system of Carla - Unreal https://carla.readthedocs.io/en/0.9.10/core_actors/
    is_dev=False, ## Turn it into True when running on a small dataset for debug
    is_local=True, ## Ignore it since we train our model on a cluster with ceph
    is_full=False, ## Set it as True will use the size of dataset exactly the same as TCP, while set it as False will use all possible data recorded in the ../dataset/dataset_metadata.pkl

    refine_num=5,
    total_epochs = 60, ## For 16 A100; With less GPUs, you could train less epochs

    ceph_conf="~/petreloss.conf",
    FPN_out_channels=[256, 256, 256, 256],
    point_cloud_range=point_cloud_range,
    SyncBN=SyncBN,
)

ckpt_interval = 1
batch_size_per_gpu = 6 ## 2 for 3090, 3 for V100, 8 for A100
if cfg["is_dev"]:
    cfg["train_town"] = local_dev_train
    cfg["val_town"] = local_dev_val
    cfg["max_sample_per_town"] = dev_max_sample_per_town
    num_worker_per_gpu = 0
else:
    if cfg["is_full"]:
        cfg["train_town"] = full_train
        cfg["val_town"] = full_val
        cfg["max_sample_per_town"] = max_sample_per_town_full
    else:
        cfg["train_town"] = tcp_train
        cfg["val_town"] = tcp_val
        cfg["max_sample_per_town"] = tcp_max_sample_per_town
    num_worker_per_gpu = 4

bev_h = 21 ## The number of BEV grid
bev_w = 21 ## The number of BEV grid
cfg["history_query_index_lis"] = [-1, 0] ## The index of frames used for BEV encoder
cfg["queue_length"] = len(cfg["history_query_index_lis"]) ## The index of frames used for BEV encoder

camera_list = ['rgb_left', 'rgb_front', 'rgb_right', 'rgb_back']
num_cams = len(camera_list)
cfg['camera_names'] = camera_list
cfg["num_cams"] = num_cams
cfg['use_depth'] = True ## During Training
cfg['use_seg'] = True ## During Training
cfg['seg_label_idxs'] = [0,1,2,3,4]
cfg["num_seg_type"] = 5
cfg['seg_converter'] = [
        0,    # unlabeled     =   0u
        1,   # road          =   1
        2,   # sidewalk      =   2
        0,   # bilding      =   3
        0,   # wall          =   4
        0,   # fence         =   5
        0,   # pole          =   6
        0,   # traffic light =   7
        0,   # traffic sign  =   8
        0,   # vegetation    =   9
        2,   # terrain       =  10
        4,   # sky           =  11
        3,   # pedestrian    =  12
        3,   # rider         =  13
        0,   # Car           =  14
        0,   # trck         =  15
        0,   # bs           =  16
        0,   # train         =  17
        0,   # motorcycle    =  18
        0,   # bicycle       =  19
        # custom
        0,   # static        =  20
        0,   # dynamic       =  21
        0,   # other         =  22
        0,   # water         =  23
        1,   # road line     =  24
        2,   # grond        =  25
        2,   # bridge        =  26
        1,   # rail track    =  27
        1    # gard rail    =  28
    ]
##From BEVDepth: https://github.com/Megvii-BaseDetection/BEVDepth, data augmentation
ida_aug_conf = {
    'resize_lim': (0.56, 0.6255),
    'final_dim':(128,2048),
    'rot_lim': (0, 0),
    'H': 128,
    'W': 2048,
    'rand_flip': True,
    'bot_pct_lim': (0.0, 0.0),
}
cfg["img_size"] = ida_aug_conf['final_dim']

model = dict(
    type='EncoderDecoder',
    decoder=dict(
        type='ThinkTwiceDecoder',
        config=cfg,
        bev_h = bev_h,
        bev_w = bev_w,
    ),
    img_encoder=dict(
        type='LSS',
        ## Same BEV range with Roach
        x_bound=[-8.0, 30.4, 1.8285],
        y_bound=[-19.2, 19.2, 1.8285],
        z_bound=[-4, 10, 14],
        d_bound=[1.0, 41.0, 0.5],
        final_dim=cfg["img_size"],
        output_channels=256,
        downsample_factor=16,
        queue_len=cfg['queue_length'],
        img_backbone_conf = dict(
                                type='ResNet',
                                depth=50,
                                frozen_stages=-1,
                                out_indices=[0, 1, 2, 3],
                                norm_eval=False,
                                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
                            ),
        img_neck_conf = dict(
                        type='PAFPN',
                        in_channels=[256, 512, 1024, 2048],
                        num_outs=4,
                        out_channels=256,
                    ),
        depth_net_conf=dict(in_channels=512, mid_channels=512),
        seg_net_conf=dict(in_channels=512, out_channels=cfg["num_seg_type"]+1),
        fpn_in_channels=[256, 256, 256, 256],
    ),
    lidar_encoder=dict(
        type='LidarNet',
        pts_voxel_layer=dict(
            max_num_points=10,
            voxel_size=[0.0571428, 0.0571428, 0.2],
            max_voxels=(120000, 160000),
            point_cloud_range = point_cloud_range),
        pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
        pts_middle_encoder=dict(
            type='SparseEncoder_fp32',
            in_channels=5,
            sparse_shape=[41, 672, 672],
            output_channels=128,
            order=('conv', 'norm', 'act'),
            encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                        128)),
            encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
            block_type='basicblock'),
        pts_backbone=dict(
            type='SECOND',
            in_channels=256,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False)),
        pts_neck=dict(
            type='SECONDFPN',
            in_channels=[128, 256],
            out_channels=[256, 256],
            upsample_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
            upsample_cfg=dict(type='deconv', bias=False),
            use_conv_for_no_stride=True),
    ),
    use_depth=cfg['use_depth'],
    num_cams=num_cams,
    train_cfg=cfg,
    test_cfg=cfg,
)

dataset_type = 'CarlaDataset'
file_client_args = dict(backend='disk')
#fp16 = dict(loss_scale="dynamic")  ## Use amp if you are familiar with debugging overflow gradients

train_pipeline = [
    dict(type='LoadPoints', is_local=cfg["is_local"],  coord_type='LIDAR', load_dim=4, use_dim=4, ceph_conf=cfg["ceph_conf"]), 
    dict(type='LoadMultiImages', 
            is_local=cfg["is_local"],
            camera_names=camera_list, ceph_conf=cfg["ceph_conf"]),
    dict(type='LoadDepth', is_local=cfg["is_local"], camera_names=camera_list, ceph_conf=cfg["ceph_conf"]),
    dict(type='LoadSeg', is_local=cfg["is_local"], camera_names=camera_list, seg_label_idxs=cfg['seg_label_idxs'], ceph_conf=cfg["ceph_conf"]),
    dict(type='CarlaFormatBundle'),
    dict(type='CarlaCollect', keys=[
                'img', 'points', 'depth', 'seg',
                'waypoints', 'target_point', 'target_command', 'target_command_raw',
                'speed', "future_speed", 'value', "future_value",
                'feature', 'future_feature', "grid_feature", "future_grid_feature",
                'action_sigma', 'action_mu',
                'future_action_mu', 'future_action_sigma'],)
]
test_pipeline = [
    dict(type='LoadPoints', is_local=cfg["is_local"], coord_type='LIDAR', load_dim=4, use_dim=4, ceph_conf=cfg["ceph_conf"]),
    dict(type='LoadMultiImages', 
            is_local=cfg["is_local"],
            camera_names=camera_list, ceph_conf=cfg["ceph_conf"]),
    dict(type='LoadDepth', is_local=cfg["is_local"], camera_names=camera_list, ceph_conf=cfg["ceph_conf"]),
    dict(type='LoadSeg', is_local=cfg["is_local"], camera_names=camera_list, seg_label_idxs=cfg['seg_label_idxs'], ceph_conf=cfg["ceph_conf"]),
    dict(type='CarlaFormatBundle'),
    dict(type='CarlaCollect', keys=[
                'img', 'points', 'depth', 'seg',
                'waypoints', 'target_point', 'target_command', 'target_command_raw',
                'speed', "future_speed", 'value', "future_value", 
                'feature', 'future_feature', "grid_feature", "future_grid_feature",
                'action_sigma', 'action_mu',
                'future_action_mu', 'future_action_sigma'])
]


train_full_queue_pipeline = [
    dict(type='IDAImageTransform', cfg=cfg, ida_aug_conf=ida_aug_conf, is_train=True and img_aug),
    dict(type='ImageTransformMulti', aug=True, batch_size=batch_size_per_gpu),
]
val_full_queue_pipeline = [
    dict(type='IDAImageTransform', cfg=cfg, ida_aug_conf=ida_aug_conf, is_train=False),
    dict(type='ImageTransformMulti', aug=False, batch_size=batch_size_per_gpu),
]

data = dict(
    samples_per_gpu=batch_size_per_gpu,
    workers_per_gpu=num_worker_per_gpu,
    train=dict(
        type=dataset_type,
        cfg=cfg,
        root=train_dir,
        full_queue_pipeline=train_full_queue_pipeline,
        ),
    val=dict(
        type=dataset_type,
        cfg = cfg,
        root=val_dir,
        full_queue_pipeline=val_full_queue_pipeline,
    ),
    test=dict(
        type=dataset_type,
        cfg = cfg,
        root=val_dir,
        full_queue_pipeline=val_full_queue_pipeline,
    ),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)


optimizer = dict(
    type='AdamW',
    lr=1e-4, ## 1e-4 - 16 A100; For less GPUs, please use smaller lr such as 3e-5 for stable training
    weight_decay=1e-7,)

optimizer_config = dict(grad_clip=dict(max_norm=100, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = cfg["total_epochs"]
evaluation = dict(interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

if cfg["is_dev"]:
    log_interval = 1
else:
    log_interval = 100

log_config = dict(
    interval=log_interval,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=ckpt_interval)