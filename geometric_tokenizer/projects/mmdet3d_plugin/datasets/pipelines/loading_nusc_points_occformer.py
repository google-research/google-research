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
import numpy as np
from projects.mmdet3d_plugin.datasets.pipelines.transform_pointcept import RandomFlip, RandomJitter, RandomRotate, RandomScale, RandomShift
import os
import torch
from torchvision import transforms
import yaml

class LoadNuscPointsAnnotationsOccformer(object):
  """Module for Tri-plane Tokenizer."""
  def __init__(
      self,
      data_root='data/nuscenes',
      is_train=False,
      is_test_submit=False,
      voxel_size=None,
      point_cloud_range=None,
      bda_aug_conf=None,
      cls_metas='nuscenes.yaml',
      prepare_voxel_label=True,
  ):
    self.is_train = is_train
    self.is_test_submit = is_test_submit
    self.cls_metas = cls_metas
    with open(cls_metas, 'r') as stream:
      nusc_cls_metas = yaml.safe_load(stream)
      self.learning_map = nusc_cls_metas['learning_map']

    self.data_root = data_root
    self.bda_aug_conf = bda_aug_conf

    # voxel settings
    self.point_cloud_range = np.array(point_cloud_range)


    # self.unoccupied_id = unoccupied_id

    # create full-resolution occupancy labels

    self.voxel_size = voxel_size
    self.prepare_voxel_label = prepare_voxel_label

    self.remove_label_0 = True
    if self.remove_label_0:
      print('NOTE: REMOVE LABEL "0" IN SEMSEG - set as 255')

  def sample_3d_augmentation(self):
    """Generate 3d augmentation values based on bda_config."""


    rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
    scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
    flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
    flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
    flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)

    return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

  def __call__(self, results):
    # for test-submission of nuScenes LiDAR Segmentation
    if self.is_test_submit:
      (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          gt_depths,
          sensor2sensors,
      ) = results['img_inputs']
      bda_rot = torch.eye(3).float()
      results['img_inputs'] = (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          bda_rot,
          gt_depths,
          sensor2sensors,
      )

      pts_filename = results['pts_filename']
      points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
          -1, 5
      )[Ellipsis, :3]
      points_label = np.zeros((points.shape[0], 1))  # placeholder
      lidarseg = np.concatenate([points, points_label], axis=-1)
      results['points_occ'] = torch.from_numpy(lidarseg).float()

      return results

    lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
    points_label = np.fromfile(
        lidarseg_labels_filename, dtype=np.uint8
    ).reshape([-1, 1])

    ### NOTE: REMOVE LABEL "0" IN SEMSEG - set as 255
    points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
    points_label = points_label - 1
    points_label[points_label == -1] = 255

    pts_filename = results['pts_filename']

    points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
        -1, 5
    )
    points_coord = points[Ellipsis, :3]
    strength = (
        points[:, 3].reshape([-1, 1]) / 127.5 - 1
    )  # scale strength to [-1, 1]
    points = np.concatenate([points_coord, strength], axis=-1)
    lidarseg = np.concatenate([points, points_label], axis=-1)

    if self.is_train:
      _, bda_rot = self.sample_3d_augmentation()
    else:
      bda_rot = torch.eye(3).float()

    # transform points
    points_coord = points_coord @ bda_rot.t().numpy()
    lidarseg[:, :3] = points_coord

    results['points_occ'] = torch.from_numpy(lidarseg).float()

    return results


class LoadNuscPointsAnnotationsOccformerNewAug(object):
  """Module for Tri-plane Tokenizer."""
  def __init__(
      self,
      data_root='data/nuscenes',
      is_train=False,
      is_test_submit=False,
      point_cloud_range=None,
      bda_aug_conf=None,
      cls_metas='nuscenes.yaml',
      prepare_voxel_label=False,
      unoccupied_id=17,
      grid_size=None,
  ):
    self.is_train = is_train
    self.is_test_submit = is_test_submit
    self.cls_metas = cls_metas
    with open(cls_metas, 'r') as stream:
      nusc_cls_metas = yaml.safe_load(stream)
      self.learning_map = nusc_cls_metas['learning_map']

    self.data_root = data_root
    self.bda_aug_conf = bda_aug_conf

    # voxel settings
    self.point_cloud_range = np.array(point_cloud_range)


    self.unoccupied_id = unoccupied_id

    # create full-resolution occupancy labels
    if grid_size is not None:
      self.grid_size = np.array(grid_size)
      self.voxel_size = (
          self.point_cloud_range[3:] - self.point_cloud_range[:3]
      ) / self.grid_size
    # self.voxel_size = voxel_size
    self.prepare_voxel_label = prepare_voxel_label

    self.remove_label_0 = True
    if self.remove_label_0:
      print(
          'NOTE: REMOVE LABEL "0" IN SEMSEG - set as 255, but for occformer, we'
          ' have un-occupy label as 0'
      )

    ### build the transform

    # self.random_scale = RandomScale(scale=self.bda_aug_conf['scale_lim'])
    # self.random_flip = RandomFlip(p=self.bda_aug_conf['flip_p'])
    # self.random_jitter = RandomJitter(sigma=0.005, clip=0.02)
    self.random_rotate = RandomRotate(**self.bda_aug_conf['rot'])
    self.random_scale = RandomScale(**self.bda_aug_conf['scale'])
    self.random_flip = RandomFlip(**self.bda_aug_conf['flip_p'])
    self.random_jitter = RandomJitter(**self.bda_aug_conf['jitter'])
    if 'shift' in self.bda_aug_conf:
      self.random_shift = RandomShift(**self.bda_aug_conf['shift'])
      transform_list = [
          self.random_shift,
          self.random_rotate,
          self.random_scale,
          self.random_flip,
          self.random_jitter,
      ]
    else:
      transform_list = [
          self.random_rotate,
          self.random_scale,
          self.random_flip,
          self.random_jitter,
      ]

    self.point_transform = transforms.Compose(transform_list)

  def __call__(self, results):
    # for test-submission of nuScenes LiDAR Segmentation
    if self.is_test_submit:
      (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          gt_depths,
          sensor2sensors,
      ) = results['img_inputs']
      bda_rot = torch.eye(3).float()
      results['img_inputs'] = (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          bda_rot,
          gt_depths,
          sensor2sensors,
      )

      pts_filename = results['pts_filename']
      points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
          -1, 5
      )[Ellipsis, :3]
      points_label = np.zeros((points.shape[0], 1))  # placeholder
      lidarseg = np.concatenate([points, points_label], axis=-1)
      results['points_occ'] = torch.from_numpy(lidarseg).float()

      return results

    lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
    points_label = np.fromfile(
        lidarseg_labels_filename, dtype=np.uint8
    ).reshape([-1, 1])

    ### NOTE: REMOVE LABEL "0" IN SEMSEG - set as 255
    points_label = np.vectorize(self.learning_map.__getitem__)(points_label)

    if not self.prepare_voxel_label:
      # points_label = points_label - 1
      points_label[points_label == 0] = 255  # now 0 is the un-occupy

    pts_filename = results['pts_filename']

    points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
        -1, 5
    )
    points_coord = points[Ellipsis, :3]
    strength = (
        points[:, 3].reshape([-1, 1]) / 127.5 - 1
    )  # scale strength to [-1, 1]
    points = np.concatenate([points_coord, strength], axis=-1)
    lidarseg = np.concatenate([points, points_label], axis=-1)

    if self.is_train:
      point_dict = self.point_transform({'coord': points_coord})
    else:
      # bda_rot = torch.eye(3).float()
      point_dict = {'coord': points_coord}

    # # transform points
    # points_coord = points_coord @ bda_rot.t().numpy()
    lidarseg[:, :3] = point_dict['coord']

    if self.prepare_voxel_label:

      # 0: noise, 1-16 normal classes, 17 unoccupied (empty)
      empty_id = self.unoccupied_id
      processed_label = np.ones(self.grid_size, dtype=np.uint8) * empty_id

      # convert label_0 to label_255 (ignored)
      processed_label[processed_label == 0] = 255
      # convert empty to label id 0
      processed_label[processed_label == empty_id] = 0

      # output: bda_mat, point_occ, and voxel_occ
      results['gt_occ'] = torch.from_numpy(processed_label).long()
      # results['occformer_grid'] = torch.from_numpy(points_grid_ind)

    results['points_occ'] = torch.from_numpy(lidarseg).float()

    return results


class LoadNuscPointsAnnotationsOccformerWithSweep(object):
  """Module for Tri-plane Tokenizer."""
  def __init__(
      self,
      data_root='data/nuscenes',
      is_train=False,
      is_test_submit=False,
      voxel_size=None,
      point_cloud_range=None,
      bda_aug_conf=None,
      unoccupied_id=17,
      cls_metas='nuscenes.yaml',
      prepare_voxel_label=True,
      sweeps_num=10,
      remove_close=False,
      test_mode=False,
  ):
    self.is_train = is_train
    self.is_test_submit = is_test_submit
    self.cls_metas = cls_metas
    with open(cls_metas, 'r') as stream:
      nusc_cls_metas = yaml.safe_load(stream)
      self.learning_map = nusc_cls_metas['learning_map']

    self.data_root = data_root
    self.bda_aug_conf = bda_aug_conf

    # voxel settings
    self.point_cloud_range = np.array(point_cloud_range)

    self.transform_center = (
        self.point_cloud_range[:3] + self.point_cloud_range[3:]
    ) / 2
    self.unoccupied_id = unoccupied_id

    # create full-resolution occupancy labels

    self.voxel_size = voxel_size

    self.prepare_voxel_label = prepare_voxel_label
    self.sweeps_num = sweeps_num
    self.remove_close = remove_close
    self.test_mode = test_mode

  def sample_3d_augmentation(self):
    """Generate 3d augmentation values based on bda_config."""


    rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
    scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
    flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
    flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
    flip_dz = np.random.uniform() < self.bda_aug_conf.get('flip_dz_ratio', 0.0)

    return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz

  def _remove_close(self, points, radius=1.0):
    """Removes point too close within a certain radius from origin.
    """
    if isinstance(points, np.ndarray):
      points_numpy = points
    else:
      raise NotImplementedError
    x_filt = np.abs(points_numpy[:, 0]) < radius
    y_filt = np.abs(points_numpy[:, 1]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    return points[not_close]

  def __call__(self, results):
    # for test-submission of nuScenes LiDAR Segmentation
    if self.is_test_submit:
      (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          gt_depths,
          sensor2sensors,
      ) = results['img_inputs']
      bda_rot = torch.eye(3).float()
      results['img_inputs'] = (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          bda_rot,
          gt_depths,
          sensor2sensors,
      )

      pts_filename = results['pts_filename']
      points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
          -1, 5
      )[Ellipsis, :3]
      points_label = np.zeros((points.shape[0], 1))  # placeholder
      lidarseg = np.concatenate([points, points_label], axis=-1)
      results['points_occ'] = torch.from_numpy(lidarseg).float()

      return results

    lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
    points_label = np.fromfile(
        lidarseg_labels_filename, dtype=np.uint8
    ).reshape([-1, 1])
    points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
    pts_filename = results['pts_filename']

    points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
        -1, 5
    )[Ellipsis, :3]
    lidarseg = np.concatenate([points, points_label], axis=-1)

    sweep_points_list = []
    # import pdb; pdb.set_trace()
    if len(results['sweeps']) == 0:
      pass
      # print('There is one sample without any sweeps!!')
      # print(results['sweeps'])
      # import pdb; pdb.set_trace()
    if len(results['sweeps']) <= self.sweeps_num:
      choices = np.arange(len(results['sweeps']))
    elif self.test_mode:
      choices = np.arange(self.sweeps_num)
    else:
      choices = np.random.choice(
          len(results['sweeps']), self.sweeps_num, replace=False
      )
    # print(choices)
    for idx in choices:
      sweep = results['sweeps'][idx]
      # points_sweep = self._load_points(sweep['data_path'])
      points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32)
      points_sweep = np.copy(points_sweep).reshape(-1, 5)
      if self.remove_close:
        points_sweep = self._remove_close(points_sweep)
      # sweep_ts = sweep['timestamp'] / 1e6
      points_sweep[:, :3] = (
          points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
      )
      points_sweep[:, :3] += sweep['sensor2lidar_translation']
      # points_sweep[:, 4] = ts - sweep_ts
      # points_sweep = points.new_point(points_sweep)
      sweep_points_list.append(points_sweep)
    points_sweep = (
        np.concatenate(sweep_points_list, axis=0)[:, :3]
        if len(sweep_points_list) > 0
        else np.empty((0, 3))
    )
    # points = points[:, self.use_dim]



    if self.is_train:
      _, bda_rot = self.sample_3d_augmentation()
    else:
      bda_rot = torch.eye(3).float()

    points = points @ bda_rot.t().numpy()
    lidarseg[:, :3] = points
    points_sweep = points_sweep @ bda_rot.t().numpy()

    results['points_occ'] = torch.from_numpy(lidarseg).float()
    results['points_sweep'] = torch.from_numpy(points_sweep).float()

    return results


class LoadNuscPointsAnnotationsOccformerNewAugWithSweep(object):
  """Module for Tri-plane Tokenizer."""
  def __init__(
      self,
      data_root='data/nuscenes',
      is_train=False,
      is_test_submit=False,
      voxel_size=None,
      point_cloud_range=None,
      bda_aug_conf=None,
      cls_metas='nuscenes.yaml',
      prepare_voxel_label=True,
      sweeps_num=10,
      remove_close=False,
      test_mode=False,
  ):
    self.is_train = is_train
    self.is_test_submit = is_test_submit
    self.cls_metas = cls_metas
    with open(cls_metas, 'r') as stream:
      nusc_cls_metas = yaml.safe_load(stream)
      self.learning_map = nusc_cls_metas['learning_map']

    self.data_root = data_root
    self.bda_aug_conf = bda_aug_conf

    # voxel settings
    self.point_cloud_range = np.array(point_cloud_range)


    # self.unoccupied_id = unoccupied_id

    # create full-resolution occupancy labels

    self.voxel_size = voxel_size
    self.prepare_voxel_label = prepare_voxel_label
    self.sweeps_num = sweeps_num
    self.remove_close = remove_close
    self.test_mode = test_mode

    self.remove_label_0 = True
    if self.remove_label_0:
      print('NOTE: REMOVE LABEL "0" IN SEMSEG - set as 255')

    ### build the transform
    self.random_rotate = RandomRotate(**self.bda_aug_conf['rot'])
    self.random_scale = RandomScale(**self.bda_aug_conf['scale'])
    self.random_flip = RandomFlip(**self.bda_aug_conf['flip_p'])
    self.random_jitter = RandomJitter(**self.bda_aug_conf['jitter'])
    if 'shift' in self.bda_aug_conf:
      self.random_shift = RandomShift(**self.bda_aug_conf['shift'])
      transform_list = [
          self.random_shift,
          self.random_rotate,
          self.random_scale,
          self.random_flip,
          self.random_jitter,
      ]
    else:
      transform_list = [
          self.random_rotate,
          self.random_scale,
          self.random_flip,
          self.random_jitter,
      ]

    self.point_transform = transforms.Compose(transform_list)

  def __call__(self, results):
    # for test-submission of nuScenes LiDAR Segmentation
    if self.is_test_submit:
      (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          gt_depths,
          sensor2sensors,
      ) = results['img_inputs']
      bda_rot = torch.eye(3).float()
      results['img_inputs'] = (
          imgs,
          rots,
          trans,
          intrins,
          post_rots,
          post_trans,
          bda_rot,
          gt_depths,
          sensor2sensors,
      )

      pts_filename = results['pts_filename']
      points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
          -1, 5
      )[Ellipsis, :3]
      points_label = np.zeros((points.shape[0], 1))  # placeholder
      lidarseg = np.concatenate([points, points_label], axis=-1)
      results['points_occ'] = torch.from_numpy(lidarseg).float()

      return results

    lidarseg_labels_filename = os.path.join(self.data_root, results['lidarseg'])
    points_label = np.fromfile(
        lidarseg_labels_filename, dtype=np.uint8
    ).reshape([-1, 1])

    ### NOTE: REMOVE LABEL "0" IN SEMSEG - set as 255
    points_label = np.vectorize(self.learning_map.__getitem__)(points_label)
    points_label = points_label - 1
    points_label[points_label == -1] = 255

    pts_filename = results['pts_filename']

    points = np.fromfile(pts_filename, dtype=np.float32, count=-1).reshape(
        -1, 5
    )
    points_coord = points[Ellipsis, :3]
    strength = (
        points[:, 3].reshape([-1, 1]) / 127.5 - 1
    )  # scale strength to [-1, 1]
    points = np.concatenate([points_coord, strength], axis=-1)
    lidarseg = np.concatenate([points, points_label], axis=-1)

    sweep_points_list = []
    # import pdb; pdb.set_trace()
    if len(results['sweeps']) == 0:
      pass
      # print('There is one sample without any sweeps!!')
      # print(results['sweeps'])
      # import pdb; pdb.set_trace()
    if len(results['sweeps']) <= self.sweeps_num:
      choices = np.arange(len(results['sweeps']))
    elif self.test_mode:
      choices = np.arange(self.sweeps_num)
    else:
      choices = np.random.choice(
          len(results['sweeps']), self.sweeps_num, replace=False
      )
    # print(choices)
    for idx in choices:
      sweep = results['sweeps'][idx]
      # points_sweep = self._load_points(sweep['data_path'])
      points_sweep = np.fromfile(sweep['data_path'], dtype=np.float32)
      points_sweep = np.copy(points_sweep).reshape(-1, 5)
      if self.remove_close:
        points_sweep = self._remove_close(points_sweep)
      # sweep_ts = sweep['timestamp'] / 1e6
      points_sweep[:, :3] = (
          points_sweep[:, :3] @ sweep['sensor2lidar_rotation'].T
      )
      points_sweep[:, :3] += sweep['sensor2lidar_translation']
      # normalize strength
      points_sweep[:, 3] = points_sweep[:, 3] / 127.5 - 1
      sweep_points_list.append(points_sweep)
    points_sweep = (
        np.concatenate(sweep_points_list, axis=0)
        if len(sweep_points_list) > 0
        else np.empty((0, 5))
    )



    if self.is_train:
      # point_dict = self.point_transform({'coord': points_coord})
      point_dict = None
      point_sweep_dict = None
    else:
      # bda_rot = torch.eye(3).float()
      point_dict = {'coord': points_coord}
      point_sweep_dict = {'coord': points_sweep[:, :3]}

    # # transform points
    # points_coord = points_coord @ bda_rot.t().numpy()
    lidarseg[:, :3] = point_dict['coord']
    points_sweep[:, :3] = point_sweep_dict['coord']

    results['points_occ'] = torch.from_numpy(lidarseg).float()
    results['points_sweep'] = torch.from_numpy(points_sweep).float()

    return results
