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

# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
"""General data loader."""

import glob
import multiprocessing as mp
import os

import dataset.augmentation as t
from dataset.voxelizer import Voxelizer
import numpy as np
import SharedArray as SA
import torch
from torch.utils import data


def sa_create(name, var):
  x = SA.create(name, var.shape, dtype=var.dtype)
  x[Ellipsis] = var[Ellipsis]
  x.flags.writeable = False
  return x


def collation_fn(batch):

  coords, feats, labels = list(zip(*batch))

  for i in range(len(coords)):
    coords[i][:, 0] *= i

  return torch.cat(coords), torch.cat(feats), torch.cat(labels)


def collation_fn_eval_all(batch):
  """Collation function for evaluation."""

  coords, feats, labels, inds_recons = list(zip(*batch))
  inds_recons = list(inds_recons)

  accmulate_points_num = 0
  for i in range(len(coords)):
    coords[i][:, 0] *= i
    inds_recons[i] = accmulate_points_num + inds_recons[i]
    accmulate_points_num += coords[i].shape[0]

  return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(
      inds_recons)


class ScanNet3D(data.Dataset):
  """General data loader."""

  SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
  ROTATION_AUGMENTATION_BOUND = ((-np.pi / 64, np.pi / 64),
                                 (-np.pi / 64, np.pi / 64), (-np.pi, np.pi))
  TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
  ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

  ROTATION_AXIS = 'z'
  LOCFEAT_IDX = 2

  def __init__(self,
               datapathprefix='Data',
               voxelsize=0.05,
               split='train',
               aug=False,
               memcacheinit=False,
               identifier=1233,
               loop=1,
               data_aug_color_trans_ratio=0.1,
               data_aug_color_jitter_std=0.05,
               data_aug_hue_max=0.5,
               data_aug_saturation_max=0.2,
               eval_all=False,
               input_color=True):
    super().__init__()
    self.split = split
    self.identifier = identifier
    self.data_paths = sorted(
        glob.glob(os.path.join(datapathprefix, split, '*.pth')))
    self.input_color = input_color
    self.voxelsize = voxelsize
    self.aug = aug
    self.loop = loop
    self.eval_all = eval_all
    dataset_name = datapathprefix.split('/')[-1]  # scannet_3d | scannet200_3d
    self.dataset_name = dataset_name
    self.use_shm = memcacheinit

    self.voxelizer = Voxelizer(
        voxel_size=voxelsize,
        clip_bound=None,
        use_augmentation=True,
        scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
        rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
        translation_augmentation_ratio_bound=self
        .TRANSLATION_AUGMENTATION_RATIO_BOUND)

    if aug:
      prevoxel_transform_train = [
          t.ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)
      ]
      self.prevoxel_transforms = t.Compose(prevoxel_transform_train)
      input_transforms = [
          t.RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False),
          t.ChromaticAutoContrast(),
          t.ChromaticTranslation(data_aug_color_trans_ratio),
          t.ChromaticJitter(data_aug_color_jitter_std),
          t.HueSaturationTranslation(data_aug_hue_max, data_aug_saturation_max),
      ]
      self.input_transforms = t.Compose(input_transforms)

    if memcacheinit and (not os.path.exists(
        '/dev/shm/openscene_%s_%s_%06d_locs_%08d' %
        (dataset_name, split, identifier, 0))):
      print('[*] Starting shared memory init ...')
      print('No. CPUs: ', mp.cpu_count())
      for i, (locs, feats, labels) in enumerate(
          torch.utils.data.DataLoader(
              self.data_paths,
              collate_fn=lambda x: torch.load(x[0]),
              num_workers=min(16, mp.cpu_count()),
              shuffle=False)):
        labels[labels == -100] = 255
        labels = labels.astype(np.uint8)
        # Scale color to 0-255
        if np.isscalar(
            feats
        ) and feats == 0:  #! no color in the input point cloud, e.g nuscenes
          feats = np.zeros_like(locs)
        feats = (feats + 1.) * 127.5
        sa_create(
            'shm://openscene_%s_%s_%06d_locs_%08d' %
            (dataset_name, split, identifier, i), locs)
        sa_create(
            'shm://openscene_%s_%s_%06d_feats_%08d' %
            (dataset_name, split, identifier, i), feats)
        sa_create(
            'shm://openscene_%s_%s_%06d_labels_%08d' %
            (dataset_name, split, identifier, i), labels)
      print('[*] %s (%s) loading 3D points done (%d)! ' %
            (datapathprefix, split, len(self.data_paths)))

  def __getitem__(self, index_long):

    index = index_long % len(self.data_paths)
    if self.use_shm:
      locs_in = SA.attach(
          'shm://openscene_%s_%s_%06d_locs_%08d' %
          (self.dataset_name, self.split, self.identifier, index)).copy()
      feats_in = SA.attach(
          'shm://openscene_%s_%s_%06d_feats_%08d' %
          (self.dataset_name, self.split, self.identifier, index)).copy()
      labels_in = SA.attach(
          'shm://openscene_%s_%s_%06d_labels_%08d' %
          (self.dataset_name, self.split, self.identifier, index)).copy()
    else:
      locs_in, feats_in, labels_in = torch.load(self.data_paths[index])
      labels_in[labels_in == -100] = 255
      labels_in = labels_in.astype(np.uint8)
      if np.isscalar(
          feats_in
      ) and feats_in == 0:  # no color in the input point cloud, e.g nuscenes
        feats_in = np.zeros_like(locs_in)
      feats_in = (feats_in + 1.) * 127.5

    locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
    locs, feats, labels, inds_reconstruct = self.voxelizer.voxelize(
        locs, feats_in, labels_in)
    if self.eval_all:
      labels = labels_in
    if self.aug:
      locs, feats, labels = self.input_transforms(locs, feats, labels)
    coords = torch.from_numpy(locs).int()
    coords = torch.cat(
        (torch.ones(coords.shape[0], 1, dtype=torch.int), coords), dim=1)
    if self.input_color:
      feats = torch.from_numpy(feats).float() / 127.5 - 1.
    else:
      feats = torch.ones(coords.shape[0], 3)
    labels = torch.from_numpy(labels).long()

    if self.eval_all:
      return coords, feats, labels, torch.from_numpy(inds_reconstruct).long()
    return coords, feats, labels

  def __len__(self):
    return len(self.data_paths) * self.loop
