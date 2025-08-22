# coding=utf-8
# Copyright 2025 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Tri-plane Tokenizer for 3D Out-Door Semantic Segmentation.

Internship Project: Tan Wang
"""
_base_ = ['../../_base_/custom_nus-3d.py', '../../_base_/default_runtime.py']

sync_bn = False
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

class_names = [
    'empty',
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
]
num_class = len(class_names)

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# occ_size = [256, 256, 32]

# for point cloud, we change it to [128, 128, 16]
# occ_size = [128, 128, 16]
# occ_size = [2048, 2048, 160]
# occ_size = [2048, 2048, 64]

# downsample ratio in [x, y, z] when generating 3D volumes in LSS
# lss_downsample = [2, 2, 2]

# voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
# voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
# voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]
# voxel_size = [voxel_x, voxel_y, voxel_z]
voxel_size = [0.05, 0.05, 0.05]
max_epoch = 80

data_config = {
    'cams': [
        'CAM_FRONT_LEFT',
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT',
        'CAM_BACK',
        'CAM_BACK_RIGHT',
    ],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),
    # image-view augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}


numc_trans = 128
# voxel_channels = [128, 256, 512, 1024]
voxel_channels = [128, 256, 512, 1024]
voxel_num_layer = [2, 2, 2, 2]
voxel_strides = [1, 2, 2, 2]
# voxel_strides = [2, 2, 2, 2]
voxel_out_indices = (0, 1, 2, 3)
voxel_out_channels = 192
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

# settings for mask2former head
mask2former_num_queries = 100
mask2former_feat_channel = voxel_out_channels
mask2former_output_channel = voxel_out_channels
mask2former_pos_channel = mask2former_feat_channel / 3  # divided by ndim
mask2former_num_heads = voxel_out_channels // 32

model = dict(
    type='OccupancyFormer3DPT',
    voxel_type='minkunet',
    max_voxels=None,
    batch_first=True,
    point_clip=False,
    # batch_first = False,
    pts_voxel_layer=dict(
        max_num_points=-1,  # 64,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(-1, -1),
    ),
    # max_voxels=(30000, 40000)),
    pts_voxel_backbone=dict(
        type='SpUNet-v1m1',
        in_channels=4,
        num_classes=16,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
    ),
    pts_middle_encoder=None,
    # pts_middle_encoder=dict(

    img_view_transformer=None,
    img_neck=None,
    img_bev_encoder_backbone=None,
    img_bev_encoder_neck=None,
    pts_bbox_head=dict(
        type='SparseUNetHead',
        channels=96,
        num_classes=16,
        loss_decode=dict(type='CrossEntropyLoss'),
        ignore_index=255,
        # loss_decode=dict(type='mmdet.CrossEntropyLoss', avg_non_ignore=True),
        # ignore_index=19
    ),
)

dataset_type = 'CustomNuScenesOccLSSDataset'
data_root = 'data/nuscenes'
nusc_class_metas = 'projects/configs/_base_/nuscenes.yaml'

# bda_aug_conf = dict(
#     rot_lim=(-180, 180),
#     scale_lim=(0.9, 1.1),
#     flip_p=0.5,)

bda_aug_conf = dict(
    rot=dict(angle=(-1, 1), axis='z', center=[0, 0, 0], p=0.5),
    scale=dict(scale=[0.9, 1.1]),
    flip_p=dict(p=0.5),
    jitter=dict(sigma=0.005, clip=0.02),
)

train_pipeline = [
    dict(
        type='LoadNuscPointsAnnotations_NewAug',
        is_train=True,
        point_cloud_range=point_cloud_range,
        bda_aug_conf=bda_aug_conf,
        cls_metas=nusc_class_metas,
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=['points_occ'],
        meta_keys=['pc_range', 'occ_size'],
    ),
]

test_pipeline = [
    dict(
        type='LoadNuscPointsAnnotations_NewAug',
        is_train=False,
        point_cloud_range=point_cloud_range,
        bda_aug_conf=bda_aug_conf,
        cls_metas=nusc_class_metas,
    ),
    dict(
        type='OccDefaultFormatBundle3D',
        class_names=class_names,
        with_label=False,
    ),
    dict(
        type='Collect3D',
        keys=['points_occ'],
        meta_keys=[
            'pc_range',
            'occ_size',
            'sample_idx',
            'timestamp',
            'scene_token',
            'img_filenames',
            'scene_name',
        ],
    ),
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
)

test_config = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file='data/nuscenes/nuscenes_infos_temporal_val.pkl',
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=None,
    pc_range=point_cloud_range,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='data/nuscenes/nuscenes_infos_temporal_train.pkl',
        # ann_file='data/nuscenes/nuscenes_infos_temporal_val.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        occ_size=None,
        pc_range=point_cloud_range,
    ),
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# embed_multi = dict(lr_mult=1.0, decay_mult=0.0)

lr = 0.002

optimizer = dict(type='AdamW', lr=lr, weight_decay=0.005)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=int(0.05 * max_epoch),
    warmup_ratio=0.1,
    min_lr=1e-5,
)

# param_scheduler = [
#     dict(

#     dict(
#         type='CosineAnnealingLR',
#         begin=0,
#         T_max=15,
#         by_epoch=True,
#         eta_min=1e-5,
#         convert_to_iter_based=True)
# ]

# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     weight_decay=0.01,
#     eps=1e-8,
#     betas=(0.9, 0.999),
#     paramwise_cfg=dict(
#         custom_keys={
#             'query_embed': embed_multi,
#             'query_feat': embed_multi,
#             'level_embed': embed_multi,
#             'absolute_pos_embed': dict(decay_mult=0.),
#             'relative_position_bias_table': dict(decay_mult=0.),
#         },
#         norm_decay_mult=0.0))

optimizer_config = dict()
# optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))

# learning policy
# lr_config = dict(
#     policy='step',
#     step=[20, 23],
# )

checkpoint_config = dict(max_keep_ckpts=1, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=max_epoch)

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='nuScenes_lidarseg_mean',
    rule='greater',
)
