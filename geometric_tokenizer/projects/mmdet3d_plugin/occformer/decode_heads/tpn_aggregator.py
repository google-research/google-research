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
import torch.nn.functional as F
import torch.nn as nn

class TPNAggregator(nn.modules):

  def __init__(self, tpv_h, tpv_w, tpv_z, scale_h=2, scale_w=2, scale_z=2):
    super().__init__()
    self.tpv_h = tpv_h
    self.tpv_w = tpv_w
    self.tpv_z = tpv_z
    self.scale_h = scale_h
    self.scale_w = scale_w
    self.scale_z = scale_z

  def forward(self, tpv_list, points=None):
    tpv_hw, tpv_zh, tpv_wz = tpv_list
    bs, c, _, _ = tpv_hw.shape

    if self.scale_h != 1 or self.scale_w != 1:
      tpv_hw = F.interpolate(
          tpv_hw,
          size=(self.tpv_h * self.scale_h, self.tpv_w * self.scale_w),
          mode='bilinear',
      )
    if self.scale_z != 1 or self.scale_h != 1:
      tpv_zh = F.interpolate(
          tpv_zh,
          size=(self.tpv_z * self.scale_z, self.tpv_h * self.scale_h),
          mode='bilinear',
      )
    if self.scale_w != 1 or self.scale_z != 1:
      tpv_wz = F.interpolate(
          tpv_wz,
          size=(self.tpv_w * self.scale_w, self.tpv_z * self.scale_z),
          mode='bilinear',
      )

    if points is not None:
      # points: bs, n, 3
      _, n, _ = points.shape
      points = points.reshape(bs, 1, n, 3)
      ## torch grid sample, range [-1, 1]
      points = points * 2 - 1

      # sample_loc = points[:, :, :, [0, 1]]
      sample_loc = points[:, :, :, [1, 0]]
      tpv_hw_pts = F.grid_sample(tpv_hw, sample_loc).squeeze(2)  # bs, c, n
      # sample_loc = points[:, :, :, [1, 2]]
      sample_loc = points[:, :, :, [2, 1]]
      tpv_zh_pts = F.grid_sample(tpv_zh, sample_loc).squeeze(2)
      # sample_loc = points[:, :, :, [2, 0]]
      sample_loc = points[:, :, :, [2, 0]]
      tpv_wz_pts = F.grid_sample(tpv_wz, sample_loc).squeeze(2)

      fused_pts = tpv_hw_pts + tpv_zh_pts + tpv_wz_pts
      fused_pts = fused_pts.permute(0, 2, 1)
      return fused_pts
