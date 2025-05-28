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
"""General data loader for features."""

import copy
import glob
import os

from dataset.scannet3d import ScanNet3D
import numpy as np
import SharedArray as SA
import torch


class ScanNet3DFeat(ScanNet3D):
  """General data loader for features."""

  def __init__(
      self,
      datapathprefix='Data',
      voxelsize=0.05,
      split='train',
      aug=False,
      memcacheinit=False,
      identifier=7791,
      loop=1,
      eval_all=False,
      val_benchmark=False,
      overfit=False,
      feat_type='lseg',
      no_voxelization=False,
      define_slot=None,
      input_color=True,
  ):
    super().__init__(
        datapathprefix=datapathprefix,
        voxelSize=voxelsize,
        split=split,
        aug=aug,
        memcacheinit=memcacheinit,
        identifier=identifier,
        loop=loop,
        eval_all=eval_all,
        input_color=input_color)
    self.aug = aug
    self.val_benchmark = val_benchmark
    self.overfit = overfit
    if self.val_benchmark:
      self.offset = 0

    self.feat_type = feat_type
    self.no_voxelization = no_voxelization
    # for looping over every slot in features, can probably remove later
    self.define_slot = define_slot
    # decide whether we use color point cloud or not
    self.input_color = input_color

    # prepare for 3D features
    if feat_type == 'lseg':
      self.datapathfeat = datapathprefix + '_lseg'
    elif feat_type == 'lseg_random':
      self.datapathfeat = datapathprefix + '_lseg_random'
    elif feat_type == 'lseg_random_cluster':
      self.datapathfeat = datapathprefix + '_lseg_random_cluster'
    elif feat_type == 'lseg_random_average':
      self.datapathfeat = datapathprefix + '_lseg_random_average'
    elif feat_type == 'lseg_random_gt_guided':
      self.datapathfeat = datapathprefix + '_lseg_random_gt_guided'
    elif feat_type == 'osegclip_random_average':
      self.datapathfeat = datapathprefix + '_osegclip_random_average'
    elif feat_type == 'osegclip_random_average_test':
      self.datapathfeat = datapathprefix + '_osegclip_random_average_test'
      self.datapathfeat = '/home/songyou/disk2/matterport_3d_osegclip_random_average_test'
    elif feat_type == 'osegclip_single':
      self.datapathfeat = datapathprefix + '_osegclip_single'
    elif feat_type == '05sec_osegclip':
      self.datapathfeat = datapathprefix + '_osegclip'
    elif feat_type == '05sec_lseg':
      self.datapathfeat = datapathprefix + '_lseg'
    else:
      raise NotImplementedError

    if os.path.exists(os.path.join(self.datapathfeat, 'occurrance.npy')):
      self.list_occur = np.load(
          os.path.join(self.datapathfeat, 'occurrance.npy'))
    elif 'nuscenes' in self.dataset_name:
      self.list_occur = None
    else:
      self.list_occur = []
      for x in self.data_paths:
        if 'scannet' in self.dataset_name:
          scene_name = x[:-15].split('/')[-1]
        elif 'matterport' in self.dataset_name:
          scene_name = x[:-4].split('/')[-1]
          scene_name = x[:-4].split('/')[-1]
        else:
          raise NotImplementedError
        ps = glob.glob(os.path.join(self.datapathfeat, scene_name + '_*.pt'))
        self.list_occur.append(len(ps))
      ind = np.where(np.array(self.list_occur) != 0)[
          0]  # some scenes in matterport have no features at all
      if np.any(np.array(self.list_occur) == 0):
        data_paths, list_occur = [], []
        for i in ind:
          data_paths.append(self.data_paths[i])
          list_occur.append(self.list_occur[i])
        self.data_paths = data_paths
        self.list_occur = list_occur

  def __getitem__(self, index_long):
    if self.overfit:
      self.split = 'train'
      index_long = 0

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
      else:
        feats_in = (feats_in + 1.) * 127.5

    # load 3D features
    if self.dataset_name == 'scannet_3d':
      scene_name = self.data_paths[index][:-15].split('/')[-1]
    else:
      scene_name = self.data_paths[index][:-4].split('/')[-1]

    if 'nuscenes' not in self.dataset_name:
      n_occur = self.list_occur[index]
      if n_occur > 1:
        nn_occur = np.random.randint(n_occur)
      elif n_occur == 1:
        nn_occur = 0
      else:
        raise NotImplementedError

      if self.define_slot is not None:
        nn_occur = self.define_slot

      processed_data = torch.load(
          os.path.join(self.datapathfeat, scene_name + '_%d.pt' % (nn_occur,)))
    else:
      # no repeated file
      processed_data = torch.load(
          os.path.join(self.datapathfeat, scene_name + '.pt'))

    flag_mask_merge = False
    if len(processed_data.keys()) > 2:
      feat_3d, mask_visible, mask_chunk = processed_data[
          'feat'], processed_data['mask'], processed_data['mask_full']
      mask = torch.zeros(feat_3d.shape[0], dtype=torch.bool)
      mask[mask_visible] = True  # mask out invisible points
    elif len(processed_data.keys()) == 2:
      flag_mask_merge = True
      feat_3d, mask_chunk = processed_data['feat'], processed_data['mask_full']
      if isinstance(mask_chunk,
                    np.ndarray):  # if the mask itself is a numpy array
        mask_chunk = torch.from_numpy(mask_chunk)
      mask = copy.deepcopy(mask_chunk)
      if self.split != 'train':  # val or test for matterport3d & nuscenes
        feat_3d_new = torch.zeros((locs_in.shape[0], feat_3d.shape[1]),
                                  dtype=feat_3d.dtype)
        feat_3d_new[mask] = feat_3d
        feat_3d = feat_3d_new
        mask_chunk = torch.ones_like(
            mask_chunk)  # every point needs to be evaluted

    if len(feat_3d.shape) > 2:
      feat_3d = feat_3d[Ellipsis, 0]

    locs = self.prevoxel_transforms(locs_in) if self.aug else locs_in
    if self.no_voxelization:
      locs = locs[mask_chunk]
      feats = feats_in[mask_chunk]
      labels = labels_in[mask_chunk]
      inds_reconstruct = np.zeros((labels.shape[0]))
      locs *= 1000
    # use randomly sampled points for supervision
    elif self.split == 'train' and 'random' in self.feat_type and not flag_mask_merge:
      feat_3d = feat_3d[mask]  # get features for visible points
      locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
          locs_in, feats_in, labels_in, return_ind=True)
      mask_chunk[
          mask_chunk.clone()] = mask  # seems to be an issue with PyTorch 1.9
      vox_ind = torch.from_numpy(vox_ind)
      mask = mask_chunk[
          vox_ind]  # voxelized visible mask for entire point cloud

      mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
      index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
      index1[mask_ind] = mask_ind

      tt = index1[vox_ind]
      chunk_ind = tt[tt != -1]

      ttt = torch.zeros(mask_chunk.shape[0])
      ttt[mask_ind] = 1
      tttt = torch.cumsum(ttt, dim=0, dtype=int)
      indices = tttt[chunk_ind] - 1

      feat_3d = feat_3d[indices]
    elif self.split == 'train' and flag_mask_merge:
      locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
          locs_in, feats_in, labels_in, return_ind=True)
      vox_ind = torch.from_numpy(vox_ind)
      mask = mask_chunk[
          vox_ind]  # voxelized visible mask for entire point cloud
      mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
      index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
      index1[mask_ind] = mask_ind

      tt = index1[vox_ind]
      chunk_ind = tt[tt != -1]

      ttt = torch.zeros(mask_chunk.shape[0])
      ttt[mask_ind] = 1
      tttt = torch.cumsum(ttt, dim=0, dtype=int)
      indices = tttt[chunk_ind] - 1
      feat_3d = feat_3d[indices]
    else:
      locs, feats, labels, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
          locs[mask_chunk],
          feats_in[mask_chunk],
          labels_in[mask_chunk],
          return_ind=True)
      vox_ind = torch.from_numpy(vox_ind)
      feat_3d = feat_3d[vox_ind]
      mask = mask[vox_ind]
      # vox_ind = vox_ind.cuda(non_blocking=True)

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
      return coords, feats, labels, feat_3d, mask, torch.from_numpy(
          inds_reconstruct).long()
    return coords, feats, labels, feat_3d, mask


def collation_fn(batch):
  """Collation function."""
  coords, feats, labels, feat_3d, mask_chunk = list(zip(*batch))

  for i in range(len(coords)):
    coords[i][:, 0] *= i

  return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(
      feat_3d), torch.cat(mask_chunk)


def collation_fn_eval_all(batch):
  """Collation function for evaluation."""
  coords, feats, labels, feat_3d, mask, inds_recons = list(zip(*batch))
  inds_recons = list(inds_recons)
  # pdb.set_trace()

  accmulate_points_num = 0
  for i in range(len(coords)):
    coords[i][:, 0] *= i
    inds_recons[i] = accmulate_points_num + inds_recons[i]
    accmulate_points_num += coords[i].shape[0]

  return torch.cat(coords), torch.cat(feats), torch.cat(labels), torch.cat(
      feat_3d), torch.cat(mask), torch.cat(inds_recons)
