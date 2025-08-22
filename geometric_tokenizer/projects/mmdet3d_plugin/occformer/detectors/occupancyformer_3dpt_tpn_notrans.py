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
import collections
from typing import List

import torch
import torch.nn.functional as F
import builder
import simple_expansion_wrapper

from .bevdepth import BEVDepth

try:
  import torch_scatter
except:
  print('Can not import the `torch_scatter` function !')
try:
  from torch_geometric.utils import scatter
except:
  print('Can not import the `scatter` function via geometric !')
import spconv.pytorch as spconv

import numpy as np
import time
import pdb


class OccupancyFormer3DPTTPNNoTrans(BEVDepth):

  def __init__(
      self,
      voxel_type='hard',
      max_voxels=None,
      pts_voxel_backbone=None,
      batch_first=True,
      use_voxel_grid=False,
      point_clip=True,
      dynamic_voxel_coord=False,
      fnv=False,
      ignore_idx=-1,
      point_downsample=False,
      point_downsample_rate=False,
      pretrain=None,
      use_pre_gt=False,
      two_res_output=False,
      map_back_sequence=False,
      trans_out_channels=None,
      before_classifier_channels=None,
      occ_size=None,
      occ_size_highres=None,
      feature_combination='addition',
      spconv_outrange_pred=False,
      transenc_spconvdec=False,
      output_voxel_label=False,
      output_map=False,
      tpn_head=None,
      **kwargs,
  ):
    super().__init__(**kwargs)

    if pts_voxel_backbone is not None:
      self.pts_voxel_backbone = builder.BACKBONES.build(pts_voxel_backbone)
    if tpn_head is not None:
      self.tpn_head = builder.HEADS.build(tpn_head)
    self.voxel_type = voxel_type
    self.max_voxels = max_voxels
    self.record_time = False
    self.batch_first = batch_first
    self.use_voxel_grid = use_voxel_grid  # if use voxel grid label for seg head
    self.point_clip = point_clip
    self.dynamic_voxel_coord = dynamic_voxel_coord
    self.fnv = fnv
    self.ignore_idx = ignore_idx
    self.point_downsample = point_downsample
    self.point_downsample_rate = point_downsample_rate
    self.use_pre_gt = use_pre_gt
    self.two_res_output = (
        two_res_output  # output both spconv encode and decode feature
    )
    self.transenc_spconvdec = (
        transenc_spconvdec  # transformer ecoder -> spconv decoder
    )
    self.occ_size = occ_size
    self.occ_size_highres = occ_size_highres
    self.feature_combination = feature_combination
    self.spconv_outrange_pred = (
        spconv_outrange_pred  # if use spconv out-of-range point prediction
    )
    self.output_voxel_label = output_voxel_label
    self.output_map = output_map
    self.map_back_sequence = map_back_sequence
    self.time_stats = collections.defaultdict(list)
    self.simple_expansion_wrapper = simple_expansion_wrapper(
        self.pts_voxel_layer.point_cloud_range[:3],
        self.pts_voxel_layer.point_cloud_range[3:],
        self.pts_voxel_layer.voxel_size,
    )


  def bev_encoder(self, x):
    x = self.img_bev_encoder_backbone(x)
    x = self.img_bev_encoder_neck(x)
    return x

  def extract_3dpt_feat(self, pts, points_label=None, points_strength=None):

    ### voxel feature encoder
    voxels, num_points, coors, addition_dict = self.voxelize(
        pts,
        points_label=points_label,
        points_strength=points_strength,
        dynamic_voxel_coord=self.dynamic_voxel_coord,
        output_label=self.output_voxel_label,
        output_map=self.output_map,
    )
    if (
        'voxel_strength' in addition_dict
        and addition_dict['voxel_strength'] is not None
    ):
      voxels = torch.cat(
          [voxels, addition_dict['voxel_strength'].unsqueeze(-1)], dim=-1
      )

    if hasattr(self, 'pts_voxel_encoder'):
      voxel_features, _ = self.pts_voxel_encoder(
          voxels, num_points, coors
      )
    else:
      voxel_features = voxels

    if hasattr(self, 'pts_neck'):
      assert isinstance(voxel_features, list)
      voxel_features_highres = self.pts_neck(voxel_features[:3])
      if type(voxel_features_highres) in [list, tuple]:
        _ = voxel_features_highres[
            0
        ]  # back to high-res feature map
      voxel_features = voxel_features[-1]
      ### prepare the high-res label (point label)
      # gt_occ_highres = torch.cat(points_label, dim=0)

    elif hasattr(self, 'pts_voxel_backbone'):
      if getattr(self.pts_voxel_backbone, 'name', None) == 'SpUNet-v1m1-2res':
        _, voxel_features = self.pts_voxel_backbone(
            voxel_features, coors
        )
        _ = addition_dict['voxel_semantic_mask'].long()
      elif self.transenc_spconvdec:
        voxel_features, skips = self.pts_voxel_backbone(voxel_features, coors)
        _ = addition_dict['voxel_semantic_mask'].long()
      else:
        if (
            self.pts_voxel_backbone.mode == 'mean'
        ):  ## only output the highres feature
          _ = self.pts_voxel_backbone(
              voxel_features, coors
          )
          _ = addition_dict['voxel_semantic_mask'].long()
        else:
          voxel_features = self.pts_voxel_backbone(voxel_features, coors)

    ### 3d voxel scatter
    if hasattr(self, 'pts_middle_encoder'):
      try:
        voxel_features_value, voxel_downsample_coors = (
            voxel_features.features,
            voxel_features.indices,
        )
      except:
        voxel_features_value = voxel_features


      ## 1. first we need to do the possible downsampling to low-res feat
      if self.point_downsample:

        bs = coors[-1, 0] + 1

        if self.point_clip:
          voxel2points_map, points2voxel_map, voxelize_down_coors = (
              self.get_pooling_map_batch(
                  bs,
                  coors,
                  torch.Tensor(self.point_downsample_rate).to(coors.device),
                  dynamic_voxel_coord=self.dynamic_voxel_coord,
              )
          )
        else:  # if not use point clip, full range of points
          if (
              getattr(self.pts_voxel_backbone, 'name', None)
              == 'SpUNet-v1m1-2res'
          ):
            # directly from encoder feature -> transformer
            voxel2points_map, points2voxel_map, voxelize_down_coors = (
                self.get_pooling_map_batch(
                    bs,
                    coors,
                    torch.Tensor(self.point_downsample_rate).to(coors.device),
                    dynamic_voxel_coord=self.dynamic_voxel_coord,
                )
            )
            (
                voxel2points_map_bound,
                points2voxel_map_bound,
                voxelize_down_coors_bound,
            ) = self.simple_expansion_wrapper.get_pooling_map_batch_lowres(
                addition_dict['pts_min_discrete'],
                bs,
                voxelize_down_coors,
                torch.Tensor(self.point_downsample_rate).to(coors.device),
                dynamic_voxel_coord=self.dynamic_voxel_coord,
            )

          else:
            # first unet decoder to high res, then mean pooling -> transformer
            voxel2points_map, points2voxel_map, voxelize_down_coors = (
                self.simple_expansion_wrapper.get_pooling_map_batch(
                    addition_dict['pts_min_discrete'],
                    bs,
                    coors,
                    torch.Tensor(self.point_downsample_rate).to(coors.device),
                    dynamic_voxel_coord=self.dynamic_voxel_coord,
                )
            )

        if self.pts_voxel_backbone.mode == 'SparseMaxPool':
          # sort the voxel downsample results if necessary
          voxel_features_list, voxel_downsample_coors_list = [], []
          for current_bs in range(bs):
            voxel_downsample_coors = None
            bs_mask = voxel_downsample_coors[:, 0] == current_bs
            voxel_downsample_coors_, voxel_feature_ = (
                voxel_downsample_coors[bs_mask],
                voxel_features_value[bs_mask],
            )
            voxel_downsample_coors_hash = (
                self.ravel_hash(voxel_downsample_coors_[:, 1:].cpu().numpy()))
            voxel_downsample_sort = voxel_downsample_coors_hash.argsort()
            voxel_downsample_coors_, voxel_feature_ = (
                voxel_downsample_coors_[voxel_downsample_sort],
                voxel_feature_[voxel_downsample_sort],
            )
            voxel_features_list.append(voxel_feature_)
            voxel_downsample_coors_list.append(voxel_downsample_coors_)

          voxel_features_value, coors = torch.cat(
              voxel_features_list, dim=0
          ), torch.cat(voxel_downsample_coors_list, dim=0)
          addition_dict['voxel_semantic_mask'] = addition_dict[
              'voxel_semantic_mask'
          ][voxel2points_map]

        elif self.pts_voxel_backbone.mode == 'mean':
          if (
              getattr(self.pts_voxel_backbone, 'name', None)
              == 'SpUNet-v1m1-2res'
          ):
            # directly from encoder feature -> transformer
            (
                voxel2points_map_lowres,
                points2voxel_map_lowres,
                voxelize_down_coors_lowres,
            ) = self.simple_expansion_wrapper.get_pooling_map_batch_lowres_nosort(
                addition_dict['pts_min_discrete'],
                bs,
                None,
                torch.Tensor(self.point_downsample_rate).to(coors.device),
                dynamic_voxel_coord=self.dynamic_voxel_coord,
            )
            voxel_features_value = torch_scatter.scatter_mean(
                voxel_features_value,
                torch.Tensor(points2voxel_map_lowres).to(coors).long(),
                dim=0,
            )
            coors = voxelize_down_coors_lowres[voxel2points_map_lowres]
            addition_dict['voxel_semantic_mask'] = addition_dict[
                'voxel_semantic_mask'
            ][voxel2points_map_lowres]
          else:
            voxel_features_value = torch_scatter.scatter_mean(
                None,
                torch.Tensor(points2voxel_map).to(coors).long(),
                dim=0,
            )
            coors = voxelize_down_coors[voxel2points_map]
            addition_dict['voxel_semantic_mask'] = addition_dict[
                'voxel_semantic_mask'
            ][voxel2points_map]

      ## 2. then grid the feature
      if not self.batch_first:
        coors = coors[:, [3, 0, 1, 2]]
      if self.voxel_type == 'minkunet':
        coors = coors[:, [0, 3, 1, 2]]  # convert the coors to (z, x, y)
      elif self.voxel_type == 'hard':
        coors = coors[:, [0, 1, 3, 2]]  # (z,y,x)->(z,x,y)
      batch_size = (
          coors[-1, 0].item() + 1
      )  # if self.batch_first else coors[-1, -1].item()+1

      if not self.pts_middle_encoder.only_label_scatter:
        x = self.pts_middle_encoder(voxel_features_value, coors, batch_size)
      else:
        x = voxel_features_value

      # if self.voxel_type == 'minkunet' and not self.use_pre_gt:
      if not self.use_pre_gt:
        gt_occ = addition_dict['voxel_semantic_mask'].long()
        gt_occ_grid = self.pts_middle_encoder(
            gt_occ.unsqueeze(1), coors, batch_size, channel_revise=1
        )  # default use 0 for the unoccupy padding
        gt_occ_grid = gt_occ_grid.squeeze(1)  # squeeze the label channel
      else:
        gt_occ = None
        gt_occ_grid = None
    else:
      # x = voxel_features_highres
      gt_occ = addition_dict['voxel_semantic_mask'].long()
      gt_occ_grid = None


    if self.map_back_sequence:
      if 'gt_occ_highres' not in locals():
        coors_highres = coors
      else:
        if 'points2voxel_map_bound' in locals():
          coors_highres = coors[None]
        else:
          if 'points2voxel_map' not in locals():
            points2voxel_map = addition_dict['point2voxel_map']
          coors_highres = coors[None]
      map_batch_idx, map_coors = (
          coors_highres[:, 0].long(),
          coors_highres[:, [2, 3, 1]].long(),
      )  # z,x,y -> x,y,z]
      map_coors_x, map_coors_y, map_coors_z = (
          map_coors[:, 0],
          map_coors[:, 1],
          map_coors[:, 2],
      )
      x = None
      x = x[0][map_batch_idx, :, map_coors_x, map_coors_y, map_coors_z]

    if self.two_res_output:
      if self.training:
        voxel_features_highres = None
        voxel_features_highres = voxel_features_highres.view(
            coors.shape[0], 3, *voxel_features_highres.shape[-3:]
        )
        tpn_plane = [
            voxel_features_highres[:, 2, :],
            voxel_features_highres[:, 0, :],
            voxel_features_highres[:, 1, :],
        ]
        points_norm = None
        pts_feature_highres = self.tpn_head(tpn_plane, points=points_norm)
        x = pts_feature_highres
        ## select the non-null points
        voxel_num, points_batch_num, _ = x.shape
        points_idx = []
        for idx, num_i in enumerate(num_points):
          points_idx += list(
              range(idx * points_batch_num, idx * points_batch_num + num_i)
          )
        x = x.reshape(voxel_num * points_batch_num, -1)[points_idx]
        gt_occ = (
            addition_dict['voxel_semantic_all']
            .view(voxel_num * points_batch_num, -1)[points_idx]
            .squeeze()
            .long()
        )

      else:
        # for testing, use all the points for the inference
        voxel_num = coors.shape[0]
        pts_feature_highres, points_voxel_labels = [], []
        voxel_features_highres = None
        voxel_features_highres = voxel_features_highres.view(
            coors.shape[0], 3, *voxel_features_highres.shape[-3:]
        )
        for voxel_i in range(voxel_num):
          points_voxel_i_idx = addition_dict['voxel2point_map'][voxel_i]
          points_voxel_i_coors = addition_dict['res_processed'][
              points_voxel_i_idx
          ]
          points_voxel_i_labels = addition_dict['res_processed_label'][
              points_voxel_i_idx
          ]

          ## get interplate feature
          voxel_size = (
              torch.Tensor(self.pts_voxel_layer.voxel_size)
              .view(1, 1, 3)
              .to(points_voxel_i_coors)
          )
          points_coors_norm = points_voxel_i_coors % voxel_size / voxel_size
          tpn_plane_voxel_i = [
              voxel_features_highres[voxel_i : voxel_i + 1, 2, :],
              voxel_features_highres[voxel_i : voxel_i + 1, 0, :],
              voxel_features_highres[voxel_i : voxel_i + 1, 1, :],
          ]
          pts_feature_highres_voxel_i = self.tpn_head(
              tpn_plane_voxel_i, points=points_coors_norm
          )

          pts_feature_highres.append(pts_feature_highres_voxel_i.squeeze(0))
          points_voxel_labels.append(points_voxel_i_labels)

        x = torch.cat(pts_feature_highres, dim=0)
        gt_occ = torch.cat(points_voxel_labels, dim=0).long()

    elif (
        self.transenc_spconvdec
    ):  # use spconv decoder after the transformer encoder
      voxel_downsample_coors = None
      map_batch_idx, map_coors = (
          voxel_downsample_coors[:, 0].long(),
          voxel_downsample_coors[:, [1, 2, 3]].long(),
      )
      map_coors_x, map_coors_y, map_coors_z = (
          map_coors[:, 0],
          map_coors[:, 1],
          map_coors[:, 2],
      )
      x = None
      x = x[0][map_batch_idx, :, map_coors_x, map_coors_y, map_coors_z]
      x = self.lateral_convs(x)
      ## send the x back to the spconv decoder
      voxel_features = voxel_features.replace_feature(x)
      skips = None
      x = [
          self.pts_voxel_backbone(
              voxel_features, coors, skips=skips, ver='decode'
          ).features
      ]
      gt_occ_highres=None
      gt_occ = gt_occ_highres

    x = None
    if not isinstance(x, list):
      x = [x]
    return x, gt_occ, gt_occ_grid, coors, addition_dict

  def ravel_hash(self, x):
    """Get voxel coordinates hash for np.unique.

    Args:
        x (np.ndarray): The voxel coordinates of points, Nx3.

    Returns:
        np.ndarray: Voxels coordinates hash.
    """
    assert x.ndim == 2, x.shape

    x = x - np.min(x, axis=0)
    x = x.astype(np.uint64, copy=False)
    xmax = np.max(x, axis=0).astype(np.uint64) + 1

    h = np.zeros(x.shape[0], dtype=np.uint64)
    for k in range(x.shape[1] - 1):
      h += x[:, k]
      h *= xmax[k + 1]
    h += x[:, -1]
    return h

  @classmethod
  def fnv_hash_vec(arr):
    """FNV64-1A"""
    assert arr.ndim == 2
    # Floor first for negative coordinates
    arr = arr.copy()
    arr = arr.astype(np.uint64, copy=False)
    hashed_arr = np.uint64(14695981039346656037) * np.ones(
        arr.shape[0], dtype=np.uint64
    )
    for j in range(arr.shape[1]):
      hashed_arr *= np.uint64(1099511628211)
      hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
    return hashed_arr

  def sparse_quantize(
      self,
      coords,
      return_index = False,
      return_inverse = False,
      return_index_all = False,
      dynamic_voxel_coord = False,
  ):
    """Sparse Quantization for voxel coordinates used in Minkunet.

    Args:
        coords (np.ndarray): The voxel coordinates of points, Nx3.
        return_index (bool): Whether to return the indices of the unique coords,
          shape (M,).
        return_inverse (bool): Whether to return the indices of the original
          coords, shape (N,).

    Returns:
        List[np.ndarray]: Return index and inverse map if return_index and
        return_inverse is True.
    """
    key = self.ravel_hash(coords)
    idx_sort = np.argsort(key)
    # key_sort = key[idx_sort]
    _, indices, inverse_indices, count = np.unique(
        key, return_index=True, return_inverse=True, return_counts=True
    )
    # _, indices2, inverse_indices2, count2 = np.unique(
    #     key_sort, return_index=True, return_inverse=True, return_counts=True)
    if dynamic_voxel_coord:  # randomly choose a point coord as the voxel coord
      idx_select = (
          np.cumsum(np.insert(count, 0, 0)[0:-1])
          + np.random.randint(0, count.max(), count.size) % count
      )
      indices = idx_sort[idx_select]

    # coords = coords[indices]
    outputs = []
    if return_index:
      outputs += [indices]
    if return_inverse:
      outputs += [inverse_indices]
    if return_index_all:
      point_cnt = 0
      voxel2points_all = []
      for voxel_i in range(len(indices)):
        voxel2points_all.append(
            idx_sort[point_cnt : point_cnt + count[voxel_i]]
        )
        point_cnt += count[voxel_i]
      return outputs, voxel2points_all
    else:
      return outputs

  def retain_elements_in_range(self, points, a, b, dim=-1):
    mask = (points >= a) & (points < b)
    mask = torch.all(mask, dim=dim)
    filtered_points = points[mask]
    return filtered_points, mask

  def get_pooling_map(self, res, voxel_size, dynamic_voxel_coord):
    res_coors = torch.floor(res[:, :3] / voxel_size).int()
    res_coors -= res_coors.min(0)[0]

    res_coors_numpy = res_coors.cpu().numpy()
    inds, point2voxel_map = self.sparse_quantize(
        res_coors_numpy,
        return_index=True,
        return_inverse=True,
        dynamic_voxel_coord=dynamic_voxel_coord,
    )
    return inds

  def get_pooling_map_batch(self, bs, res, voxel_size, dynamic_voxel_coord):
    # for bs>1 situation, res: [bs, x, y, z]
    inds_all, point2voxel_map_all, res_coors_all = [], [], []
    cnt = 0
    voxel_cnt = 0
    for current_bs in range(bs):
      bs_mask = res[:, 0] == current_bs
      res_bs = res[bs_mask][:, 1:]
      res_coors = torch.floor(res_bs / voxel_size).int()
      # res_coors -= res_coors.min(0)[0]
      res_coors_all.append(
          torch.cat([res[bs_mask][:, 0].unsqueeze(1), res_coors], dim=1)
      )

      res_coors_numpy = res_coors.cpu().numpy()
      inds, point2voxel_map = self.sparse_quantize(
          res_coors_numpy,
          return_index=True,
          return_inverse=True,
          dynamic_voxel_coord=dynamic_voxel_coord,
      )
      inds_all.append(inds + cnt)
      point2voxel_map_all.append(point2voxel_map + voxel_cnt)
      cnt += res_coors.shape[0]
      voxel_cnt += inds.shape[0]

    return (
        np.concatenate(inds_all, axis=0),
        np.concatenate(point2voxel_map_all, axis=0),
        torch.cat(res_coors_all, dim=0),
    )

  @torch.no_grad()
  def voxelize(
      self,
      points,
      points_label=None,
      points_strength=None,
      dynamic_voxel_coord=False,
      output_label=False,
      output_map=False,
  ):
    if self.voxel_type == 'hard':
      """Apply hard voxelization to points."""
      (
          voxels,
          coors,
          num_points,
          point2voxel_maps,
          voxel2point_maps,
          range_masks,
          res_processed,
          res_processed_label,
      ) = ([], [], [], [], [], [], [], [])
      voxel_cnt, point_cnt = 0, 0
      for res_idx, res in enumerate(points):
        if output_label:
          res = torch.cat([res, points_label[res_idx].unsqueeze(-1)], dim=-1)
        res_voxels, voxel_coors, res_num_points = self.pts_voxel_layer(res)
        if output_map:
          res_label = res[:, -1]
          res = res[:, :3]
          voxel_size = points[0].new_tensor(self.pts_voxel_layer.voxel_size)
          if self.point_clip:
            ## rm the point out of range
            min_bound = res.new_tensor(
                self.pts_voxel_layer.point_cloud_range[:3]
            )
            max_bound = res.new_tensor(
                self.pts_voxel_layer.point_cloud_range[3:]
            )
            # eps = 1e-4
            # max_bound = res.new_tensor(
            #     self.pts_voxel_layer.point_cloud_range[3:]) - eps
            res, mask = self.retain_elements_in_range(res, min_bound, max_bound)
            res_coors = torch.floor(
                (
                    res[:, :3]
                    - torch.Tensor(
                        self.pts_voxel_layer.point_cloud_range[:3]
                    ).to(res)
                )
                / voxel_size
            )
            assert self.occ_size is not None
            res_coors, _ = self.retain_elements_in_range(
                res_coors,
                torch.Tensor([0, 0, 0]).to(res),
                torch.Tensor(self.occ_size).to(res),
            )
            res_coors = res_coors.int()
            # res_coors = torch.floor(res[:, :3] / voxel_size).int()
            # res_coors -= (torch.Tensor(self.pts_voxel_layer.point_cloud_range[:3]).to(res) / voxel_size).int()
            res_label = res_label[mask]

          else:  # for no points clip
            # res_coors = torch.round(res_clamp[:, :3] / voxel_size).int()
            res_coors = torch.floor(res[:, :3] / voxel_size).int()
            res_min = res_coors.min(0)[0]
            res_coors -= res_min

          res_coors_numpy = res_coors.cpu().numpy()
          [inds, point2voxel_map], voxel2point_map = self.sparse_quantize(
              res_coors_numpy,
              return_index=True,
              return_inverse=True,
              return_index_all=True,
              dynamic_voxel_coord=dynamic_voxel_coord,
          )
          # get the map to map back to minkunet sort
          sort_idx = self.ravel_hash(
              voxel_coors[:, [2, 1, 0]].cpu().numpy()
          ).argsort()
          res_voxels, voxel_coors, res_num_points = (
              res_voxels[sort_idx],
              voxel_coors[sort_idx],
              res_num_points[sort_idx],
          )
          try:
            assert (
                res_coors[inds][:, 0] == voxel_coors[:, -1]
            ).all()  # quick check
          except:
            print(res_coors[inds][:, 0])
            print(voxel_coors[:, -1])
            torch.save(voxel_coors, 'work_dirs/wrong_voxel_coors.torch')
            torch.save(res_coors[inds], 'work_dirs/wrong_res_coors.torch')

          ### consider batch size for `point2voxel_map`
          point2voxel_map += voxel_cnt
          point2voxel_map = torch.from_numpy(point2voxel_map).cuda().long()
          if point_cnt > 0:
            voxel2point_map = [i + point_cnt for i in voxel2point_map]

          voxel_cnt += len(inds)
          point_cnt += len(res)

        voxels.append(res_voxels)
        coors.append(voxel_coors)
        num_points.append(res_num_points)
        res_processed.append(res)
        if output_map:
          point2voxel_map, voxel2point_map, res_label, mask = None, None, None, None
          point2voxel_maps.append(point2voxel_map)
          voxel2point_maps += voxel2point_map
          res_processed_label.append(res_label)
          if self.point_clip:
            range_masks.append(mask)
      voxels = torch.cat(voxels, dim=0)
      num_points = torch.cat(num_points, dim=0)
      point2voxel_maps = (
          torch.cat(point2voxel_maps, dim=0) if output_map else None
      )
      range_masks = (
          torch.cat(range_masks, dim=0)
          if output_map and self.point_clip
          else None
      )
      res_processed = torch.cat(res_processed, dim=0)
      res_processed_label = (
          torch.cat(res_processed_label, dim=0) if output_map else None
      )
      coors_batch = []
      for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
      coors_batch = torch.cat(coors_batch, dim=0)

      if output_label:
        num_dim = voxels.shape[0]
        voxel_semantic = torch.zeros(
            num_dim,
        ).to(voxels)
        voxel_semantic_all = voxels[:, :, -1]

        for idx in range(num_dim):
          voxel_semantic[idx] = voxel_semantic_all[
              idx, : num_points[idx]
          ].mode()[
              0
          ]  # select the max to be the label
        voxels = voxels[:, :, :3]
        voxel_semantic = voxel_semantic.long()
      else:
        voxel_semantic_all = None
        voxel_semantic = None

      return (
          voxels,
          num_points,
          coors_batch,
          {
              'voxel_semantic_mask': voxel_semantic,
              'voxel_semantic_all': voxel_semantic_all,
              'point2voxel_map': point2voxel_maps,
              'voxel2point_map': voxel2point_maps,
              'range_mask': range_masks,
              'res_processed': res_processed,
              'res_processed_label': res_processed_label,
          },
      )

    elif self.voxel_type == 'dynamic':
      """Apply dynamic voxelization to points.

      Args:
          points (list[torch.Tensor]): Points of each sample.

      Returns:
          tuple[torch.Tensor]: Concatenated points, number of points
              per voxel, and coordinates.
      """
      voxels, coors, num_points = [], [], []
      for res in points:
        res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
        voxels.append(res_voxels)
        coors.append(res_coors)
        num_points.append(res_num_points)
      voxels = torch.cat(voxels, dim=0)
      num_points = torch.cat(num_points, dim=0)
      coors_batch = []
      for i, coor in enumerate(coors):
        coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
        coors_batch.append(coor_pad)
      coors_batch = torch.cat(coors_batch, dim=0)
      return voxels, num_points, coors_batch, {}

    elif self.voxel_type == 'minkunet':
      assert points_label is not None
      voxel_cnt = 0
      (
          voxels,
          coors,
          voxel_semantic_masks,
          point2voxel_maps,
          range_masks,
          voxel_strengths,
      ) = ([], [], [], [], [], [])
      pts_min_discretes = []
      voxel_size = points[0].new_tensor(self.pts_voxel_layer.voxel_size)
      for i, (res, pts_semantic_label) in enumerate(zip(points, points_label)):
        if self.point_clip:
          ## rm the point out of range
          min_bound = res.new_tensor(self.pts_voxel_layer.point_cloud_range[:3])
          eps = 1e-4
          max_bound = (
              res.new_tensor(self.pts_voxel_layer.point_cloud_range[3:]) - eps
          )
          # res = torch.clamp(res, min_bound, max_bound).unique(dim=0)
          res, mask = self.retain_elements_in_range(res, min_bound, max_bound)
          pts_semantic_label = pts_semantic_label[mask]

          res_coors = torch.floor(
              (
                  res[:, :3]
                  - torch.Tensor(self.pts_voxel_layer.point_cloud_range[:3]).to(
                      res
                  )
              )
              / voxel_size
          ).int()

          # res_coors -= res_coors.min(0)[0]
          res_min = (
              torch.Tensor(self.pts_voxel_layer.point_cloud_range[:3]).to(res)
              / voxel_size
          ).int()

        else:  # for no points clip
          # res_coors = torch.round(res_clamp[:, :3] / voxel_size).int()
          res_coors = torch.floor(res[:, :3] / voxel_size).int()
          res_min = res_coors.min(0)[0]
          res_coors -= res_min

        pts_min_discretes.append(res_min.unsqueeze(0))

        res_coors_numpy = res_coors.cpu().numpy()
        inds, point2voxel_map = self.sparse_quantize(
            res_coors_numpy,
            return_index=True,
            return_inverse=True,
            dynamic_voxel_coord=dynamic_voxel_coord,
        )

        ### consider batch size for `point2voxel_map`
        point2voxel_map += voxel_cnt
        point2voxel_map = torch.from_numpy(point2voxel_map).cuda()

        if self.training and self.max_voxels is not None:
          if len(inds) > self.max_voxels:
            inds = np.random.choice(inds, self.max_voxels, replace=False)
        inds = torch.from_numpy(inds).cuda()
        voxel_semantic_mask = pts_semantic_label[inds]
        voxel_strength = (
            points_strength[i][inds] if points_strength is not None else None
        )
        # if hasattr(data_sample.gt_pts_seg, 'pts_semantic_mask'):
        #     data_sample.gt_pts_seg.voxel_semantic_mask \
        #         = data_sample.gt_pts_seg.pts_semantic_mask[inds]
        res_voxel_coors = res_coors[inds]
        res_voxels = res[inds]
        if self.batch_first:
          res_voxel_coors = F.pad(
              res_voxel_coors, (1, 0), mode='constant', value=i
          )
          batch_idx = res_voxel_coors[:, 0]
        else:
          res_voxel_coors = F.pad(
              res_voxel_coors, (0, 1), mode='constant', value=i
          )
          batch_idx = res_voxel_coors[:, -1]
        point2voxel_map = point2voxel_map.long()
        voxels.append(res_voxels)
        coors.append(res_voxel_coors)
        voxel_semantic_masks.append(voxel_semantic_mask)
        voxel_strengths.append(voxel_strength)
        point2voxel_maps.append(point2voxel_map)
        if self.point_clip:
          mask = None
          range_masks.append(mask)

        voxel_cnt += len(inds)

      voxels = torch.cat(voxels, dim=0)
      coors = torch.cat(coors, dim=0)
      voxel_semantic_masks = torch.cat(voxel_semantic_masks, dim=0)
      point2voxel_maps = torch.cat(point2voxel_maps, dim=0)
      range_masks = torch.cat(range_masks, dim=0) if self.point_clip else None
      voxel_strengths = (
          torch.cat(voxel_strengths, dim=0)
          if points_strength is not None
          else None
      )
      addition_dict = {'pts_min_discrete': torch.cat(pts_min_discretes, dim=0)}
      addition_dict.update({
          'voxel_semantic_mask': voxel_semantic_masks,
          'point2voxel_map': point2voxel_maps,
          'range_mask': range_masks,
          'voxel_strength': voxel_strengths,
      })
      return voxels, None, coors, addition_dict  # coors: [x,y,z,bs]

  def extract_img_feat(self, img, img_metas):
    """Extract features of images."""

    if self.record_time:
      torch.cuda.synchronize()
      t0 = time.time()

    x = self.image_encoder(img[0])
    img_feats = x.clone()

    rots, trans, intrins, post_rots, post_trans, bda = img[1:7]

    mlp_input = self.img_view_transformer.get_mlp_input(
        rots, trans, intrins, post_rots, post_trans, bda
    )
    geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]

    x, depth = self.img_view_transformer([x] + geo_inputs)


    x = self.bev_encoder(x)
    if type(x) is not list:
      x = [x]

    return x, depth, img_feats

  def extract_feat(
      self,
      points,
      img=None,
      img_metas=None,
      points_label=None,
      points_strength=None,
  ):
    """Extract features from images and points."""
    # voxel_feats = self.extract_img_feat(img, img_metas)
    voxel_feats, new_gt_occ, gt_occ_grid, coors, addition_dict = (
        self.extract_3dpt_feat(
            points, points_label=points_label, points_strength=points_strength
        )
    )
    return voxel_feats, new_gt_occ, gt_occ_grid, coors, addition_dict

  def forward_pts_train(
      self,
      pts_feats,
      gt_occ=None,
      points_occ=None,
      img_metas=None,
      img_feats=None,
      points_uv=None,
      gt_occ_grid=None,
      voxel_coors=None,
      addition_dict=None,
      **kwargs,
  ):
    if self.record_time:
      torch.cuda.synchronize()
      t0 = time.time()

    losses = self.pts_bbox_head.forward_train(
        voxel_feats=pts_feats,
        voxel_labels=gt_occ,
        img_metas=img_metas,
        gt_occ=gt_occ,
        points=points_occ,
        img_feats=img_feats,
        points_uv=points_uv,
        scatter=getattr(self, 'pts_middle_encoder', None),
        voxel_coors=voxel_coors,
        addition_dict=addition_dict,
        **kwargs,
    )


    return losses

  def forward_train(
      self,
      points=None,
      img_metas=None,
      img_inputs=None,
      gt_occ=None,
      points_occ=None,
      points_uv=None,
      points_sweep=None,
      **kwargs,
  ):
    points, points_label = [
        points_i[:, :3].contiguous() for points_i in points_occ
    ], [points_i[:, -1].contiguous() for points_i in points_occ]
    if points_occ[0].shape[1] == 5:
      points_strength = [
          points_i[:, -2].contiguous() for points_i in points_occ
      ]
    # points, points_strength, points_label = [points_i[:,:3].contiguous(), points_i[:,-2].contiguous(), points_i[:,-1].contiguous() for points_i in points_occ]
    if points_sweep is not None:
      if points_sweep[0].shape[1] == 5:
        points_sweep_strength = [
            points_sweep_i[:, -2].contiguous()
            for points_sweep_i in points_sweep
        ]
      points_anno_len = [len(points_i) for points_i in points]
      points_sweep_len = [
          len(points_sweep_i) for points_sweep_i in points_sweep
      ]
      points = [
          torch.cat([points_i, points_sweep_i[:, :3]])
          for (points_i, points_sweep_i) in zip(points, points_sweep)
      ]
      points_strength = None
      points_sweep_strength = None
      if points_strength is not None:
        points_strength = [
            torch.cat([points_strength_i, points_sweep_strength_i])
            for (points_strength_i, points_sweep_strength_i) in zip(
                points_strength, points_sweep_strength
            )
        ]
      points_label = [
          torch.cat([
              points_label_i,
              (torch.ones(points_sweep_len[idx]) * self.ignore_idx).type_as(
                  points_label_i
              ),
          ])
          for idx, points_label_i in enumerate(points_label)
      ]
      # nosweep_mask = [torch.cat([torch.ones(points_anno_len[idx]), torch.zeros(points_sweep_len[idx])]).type_as(points_i) for idx, points_i in enumerate(points)]
      points_occ = [
          torch.cat([points_i, points_label_i.unsqueeze(1)], dim=-1)
          for (points_i, points_label_i) in zip(points, points_label)
      ]

    points_strength = None
    # extract bird-eye-view features from perspective images
    voxel_feats, new_gt_occ, gt_occ_grid, voxel_coors, addition_dict = (
        self.extract_feat(
            points,
            img=img_inputs,
            img_metas=img_metas,
            points_label=points_label,
            points_strength=points_strength,
        )
    )

    # if new_gt_occ is not None and self.voxel_type == 'minkunet':
    if new_gt_occ is not None:
      if self.use_voxel_grid:
        gt_occ = gt_occ_grid
      else:
        gt_occ = new_gt_occ

    # training losses
    losses = dict()

    if self.record_time:
      torch.cuda.synchronize()
      t0 = time.time()


    losses_occupancy = self.forward_pts_train(
        voxel_feats,
        gt_occ,
        points_occ,
        img_metas,
        img_feats=None,
        points_uv=points_uv,
        gt_occ_grid=gt_occ_grid,
        voxel_coors=voxel_coors,
        addition_dict=addition_dict,
        **kwargs,
    )
    losses.update(losses_occupancy)

    if self.record_time:
      # logging latencies
      avg_time = {
          key: sum(val) / len(val) for key, val in self.time_stats.items()
      }
      sum_time = sum(list(avg_time.values()))
      out_res = ''
      for key, val in avg_time.items():
        out_res += '{}: {:.4f}, {:.1f}, '.format(key, val, val / sum_time)

      print(out_res)

    return losses

  def forward_test(
      self,
      img_metas=None,
      img_inputs=None,
      **kwargs,
  ):
    return self.simple_test(img_metas, img_inputs, **kwargs)

  def simple_test(
      self,
      img_metas,
      img=None,
      rescale=False,
      points_occ=None,
      gt_occ=None,
      points_uv=None,
      points_sweep=None,
  ):
    points, points_label = [
        points_i[:, :3].contiguous() for points_i in points_occ
    ], [points_i[:, -1].contiguous() for points_i in points_occ]
    if points_occ[0].shape[1] == 5:
      points_strength = [
          points_i[:, -2].contiguous() for points_i in points_occ
      ]
    if points_sweep is not None:
      if points_sweep[0].shape[1] == 5:
        points_sweep_strength = [
            points_sweep_i[:, -2].contiguous()
            for points_sweep_i in points_sweep
        ]
      points_anno_len = [len(points_i) for points_i in points]
      points_sweep_len = [
          len(points_sweep_i) for points_sweep_i in points_sweep
      ]
      points = [
          torch.cat([points_i, points_sweep_i[:, :3]])
          for (points_i, points_sweep_i) in zip(points, points_sweep)
      ]
      points_strength = None
      points_sweep_strength = None
      if points_strength is not None:
        points_strength = [
            torch.cat([points_strength_i, points_sweep_strength_i])
            for (points_strength_i, points_sweep_strength_i) in zip(
                points_strength, points_sweep_strength
            )
        ]
      points_label = [
          torch.cat([
              points_label_i,
              (torch.ones(points_sweep_len[idx]) * self.ignore_idx).type_as(
                  points_label_i
              ),
          ])
          for idx, points_label_i in enumerate(points_label)
      ]
      points_occ = [
          torch.cat([points_i, points_label_i.unsqueeze(1)], dim=-1)
          for (points_i, points_label_i) in zip(points, points_label)
      ]

    points_strength = None
    voxel_feats, new_gt_occ, gt_occ_grid, voxel_coors, addition_dict = (
        self.extract_feat(
            points,
            img=img,
            img_metas=img_metas,
            points_label=points_label,
            points_strength=points_strength,
        )
    )
    output = self.pts_bbox_head.simple_test(
        voxel_feats=voxel_feats,
        voxel_labels=new_gt_occ,
        points=points_occ,
        img_metas=img_metas,
        img_feats=None,
        points_uv=points_uv,
        # gt_occ=gt_occ,
        gt_occ=new_gt_occ,
        scatter=getattr(self, 'pts_middle_encoder', None),
        voxel_coors=voxel_coors,
        addition_dict=addition_dict,
    )

    # if new_gt_occ is not None and self.voxel_type == 'minkunet':
    if new_gt_occ is not None:
      if self.use_voxel_grid:
        gt_occ = gt_occ_grid
      else:
        gt_occ = new_gt_occ

    # evaluate nusc lidar-seg
    logits_num = output['output_points'].shape[1]
    if output['output_points'] is not None and points_occ is not None:
      if logits_num == 17:
        output['output_points'] = (
            torch.argmax(output['output_points'][:, 1:], dim=1) + 1
        )
      else:
        output['output_points'] = (
            torch.argmax(output['output_points'], dim=1) + 1
        )

      target_points = gt_occ.unsqueeze(-1)
      if logits_num == 16:
        # target + 1 to match the language label
        target_points[:, -1] += 1
      output['evaluation_semantic'] = self.simple_evaluation_semantic(
          output['output_points'], target_points, img_metas
      )
      output['target_points'] = target_points

    # evaluate voxel
    if output['output_voxels'] is not None:
      output_voxels = output['output_voxels'][0]
      target_occ_size = img_metas[0]['occ_size']

      if (output_voxels.shape[-3:] != target_occ_size).any():
        output_voxels = F.interpolate(
            output_voxels,
            size=tuple(target_occ_size),
            mode='trilinear',
            align_corners=True,
        )
    else:
      output_voxels = None

    output['output_voxels'] = output_voxels
    output['target_voxels'] = gt_occ

    return output

  def post_process_semantic(self, pred_occ):
    if type(pred_occ) == list:
      pred_occ = pred_occ[-1]

    score, clses = torch.max(torch.softmax(pred_occ, dim=1), dim=1)
    return clses

  def simple_evaluation_semantic(self, pred, gt, img_metas):
    # pred = torch.argmax(pred, dim=1).cpu().numpy()
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    gt = gt[:, -1].astype(np.int)
    unique_label = np.arange(16)
    hist = None

    return hist
