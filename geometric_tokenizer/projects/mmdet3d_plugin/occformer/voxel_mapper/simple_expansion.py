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
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

"""
need to return a map function between high-res and low-res voxel
"""


class simple_expansion_wrapper:
  # ___|_|_|___
  # ___|_|_|___
  # ___|_|_|___
  #   | | |

  def __init__(self, min_range, max_range, voxel_size, fnv=False):
    self.min_range = min_range  # x,y,z
    self.max_range = max_range  # x,y,z
    self.voxel_size = voxel_size
    self.fnv = fnv

  def forward_map(
      self, bs, pts_min_discrete, coors_discrete, voxel_downsample_rate=None
  ):
    """To build the map function from high-res -> low-res voxels

    coors_discrete: high-res voxel discrete coors
    """
    min_range_discrete = torch.floor(
        torch.Tensor(self.min_range) / torch.Tensor(self.voxel_size)
    ).int()
    min_range_discrete = (
        min_range_discrete.to(coors_discrete) - pts_min_discrete[bs]
    )
    max_range_discrete = torch.floor(
        (torch.Tensor(self.max_range) - 1e-4) / torch.Tensor(self.voxel_size)
    ).int()
    max_range_discrete = (
        max_range_discrete.to(coors_discrete) - pts_min_discrete[bs]
    )
    if voxel_downsample_rate is not None:
      voxel_downsample_rate = voxel_downsample_rate.int()
      min_range_discrete, max_range_discrete = (
          min_range_discrete / voxel_downsample_rate
      ).ceil().int(), (max_range_discrete / voxel_downsample_rate).floor().int()
      ### for easy check
      assert (max_range_discrete - min_range_discrete).tolist() == [
          128 - 1,
          128 - 1,
          10 - 1,
      ]

    # new discrete coors for x,y,z-axis
    map_x = torch.clamp(
        coors_discrete[:, 0],
        min=min_range_discrete[0],
        max=max_range_discrete[0],
    ).unsqueeze(1)
    map_y = torch.clamp(
        coors_discrete[:, 1],
        min=min_range_discrete[1],
        max=max_range_discrete[1],
    ).unsqueeze(1)
    map_z = torch.clamp(
        coors_discrete[:, 2],
        min=min_range_discrete[2],
        max=max_range_discrete[2],
    ).unsqueeze(1)
    mapped_coors_discrete = torch.cat([map_x, map_y, map_z], dim=-1)

    ## back to the zero-starting for ease of mapping calculation
    mapped_coors_discrete -= min_range_discrete
    return mapped_coors_discrete

  def get_pooling_map_batch_lowres(
      self, pts_min_discrete, bs, res, voxel_size, dynamic_voxel_coord
  ):
    # for bs>1 situation, res: [bs, x, y, z]
    inds_all, point2voxel_map_all, res_coors_all = [], [], []
    cnt = 0
    voxel_cnt = 0
    for current_bs in range(bs):
      bs_mask = res[:, 0] == current_bs
      res_bs = res[bs_mask][:, 1:]

      res_coors = self.forward_map(
          current_bs, pts_min_discrete, res_bs, voxel_size
      )
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

  def get_pooling_map_batch_lowres_nosort(
      self, pts_min_discrete, bs, res, voxel_size, dynamic_voxel_coord
  ):
    # for bs>1 situation, res: [bs, x, y, z]
    res_new = torch.zeros_like(res).to(res)
    for current_bs in range(bs):
      bs_mask = res[:, 0] == current_bs
      res_bs = res[bs_mask][:, 1:]

      res_coors = self.forward_map(
          current_bs, pts_min_discrete, res_bs, voxel_size
      )
      res_new[bs_mask] = torch.cat(
          [res[bs_mask][:, 0].unsqueeze(1), res_coors], dim=1
      )

    res_numpy = res_new.cpu().numpy()
    inds, point2voxel_map = self.sparse_quantize(
        res_numpy,
        return_index=True,
        return_inverse=True,
        dynamic_voxel_coord=dynamic_voxel_coord,
    )

    return inds, point2voxel_map, res_new

  def get_pooling_map_batch(
      self, pts_min_discrete, bs, res, voxel_size, dynamic_voxel_coord
  ):
    # for bs>1 situation, res: [bs, x, y, z]
    inds_all, point2voxel_map_all, res_coors_all = [], [], []
    cnt = 0
    voxel_cnt = 0
    for current_bs in range(bs):
      bs_mask = res[:, 0] == current_bs
      res_bs = res[bs_mask][:, 1:]
      # # assert res_bs[:,0].max()<self.occ_size_highres[0] and res_bs[:,1].max()<self.occ_size_highres[1] and res_bs[:,2].max()<self.occ_size_highres[2]
      # # we need to use torch.clip to restrict the maximum range of the high res coords
      # # res_bs = torch.clamp(res_bs, min=torch.Tensor([0,0,0]).to(res_bs), max=torch.Tensor([i-1 for i in self.occ_size_highres]).to(res_bs))
      # res_coors = torch.floor(res_bs / voxel_size).int()
      # # res_coors -= res_coors.min(0)[0]
      res_coors = self.forward_map(current_bs, pts_min_discrete, res_bs)
      res_coors = torch.floor(res_coors / voxel_size).int()
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

  def sparse_quantize(
      self,
      coords,
      return_index = False,
      return_inverse = False,
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
    return outputs

  def ravel_hash(self, x):
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
