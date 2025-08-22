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

# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
from torch import nn
import math
import warnings
import numpy as np


class RbfLayer(nn.Module):

  def __init__(self,
               n,
               init_c=0.5,
               init_r=0.001,
               const=5.0,
               n_to_print=0,
               with_mlp=False,
               softmax_temp=1.0,
               weight_min=0.0):
    super().__init__()
    self.n = n
    self.const = const
    assert self.const > 0
    self.init_c = init_c
    self.init_r = init_r

    self.call_cnt = 0
    self.n_to_print = n_to_print

    self.with_mlp = with_mlp
    assert not self.with_mlp
    self.softmax_temp = softmax_temp
    self.weight_min = weight_min

  def setup_loc(self, x, freeze):
    self.centers = nn.Parameter(
        (torch.rand((self.n, 3)) - 0.5) * 2 * self.init_c,
        requires_grad=(not freeze))
    self.radii = nn.Parameter(torch.rand(self.n, 3) * self.init_r)
    self.rotations = nn.Parameter((torch.rand((self.n, 3)) - 0.5) * 2 * math.pi)

  @staticmethod
  def rpy2mat(rpy):
    cosines = torch.cos(rpy)
    sines = torch.sin(rpy)
    cx, cy, cz = cosines.unbind(-1)
    sx, sy, sz = sines.unbind(-1)
    # pyformat: disable
    rotation = torch.stack(
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
         sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
         -sy, cy * sx, cy * cx], dim=-1)
    # pyformat: enable
    rotation = rotation.view(*rotation.shape[:-1], 3, 3)
    return rotation

  @staticmethod
  def decode_covariance_rpy(radii, rotations, invert=False):
    DIV_EPSILON = 1e-8
    d = 1.0 / (radii + DIV_EPSILON) if invert else radii
    diag = torch.diag_embed(d)
    rotation = RbfLayer.rpy2mat(rotations)
    return torch.matmul(torch.matmul(rotation, diag), rotation.transpose(1, 2))

  @staticmethod
  def eval_rbf(world_xyz, centers, radii, rotations, temp=1.0):
    assert len(world_xyz.shape) == 2
    world_xyz = world_xyz[None, Ellipsis]

    diff = world_xyz - centers.unsqueeze(-2)
    x, y, z = diff.unbind(-1)

    inv_cov = RbfLayer.decode_covariance_rpy(radii, rotations, invert=True)
    inv_cov = inv_cov.view(*inv_cov.shape[:-2], 1, 9)

    c00, c01, c02, _, c11, c12, _, _, c22 = inv_cov.unbind(-1)
    dist = (
        x * (c00 * x + c01 * y + c02 * z) + y * (c01 * x + c11 * y + c12 * z) +
        z * (c02 * x + c12 * y + c22 * z))
    dist = torch.exp(-0.5 * dist / temp)[Ellipsis, None]
    return dist

  def calc_weights(self, world_xyz):
    const = self.const
    centers = self.centers
    radii = torch.abs(self.radii)
    rotations = self.rotations

    rbfs = self.eval_rbf(world_xyz, centers, radii, rotations,
                         self.softmax_temp)
    rbfs = rbfs * const  # [N_n, N_samples, 1]
    assert len(rbfs.shape) == 3

    return rbfs

  def blend(self, rbfs, rgbs, dists, topk, sigma_only=False, sem_logits=None):
    centers = self.centers

    no_pen_thresh = self.weight_min
    pen = 0.001 * (rbfs - no_pen_thresh).clamp(min=0).sum(dim=0).mean()

    # TODO: configurable bbox
    bbox_min = torch.tensor([[-self.init_c, -self.init_c, -self.init_c]],
                            device=centers.device)
    bbox_max = torch.tensor([[self.init_c, self.init_c, self.init_c]],
                            device=centers.device)
    c_above_max = (centers - bbox_max).clamp(min=0).sum()
    c_below_max = (bbox_min - centers).clamp(min=0).sum()
    pen = pen + 1.0 * (c_above_max + c_below_max)

    alpha = 1.0 - torch.exp(-torch.relu(rgbs[Ellipsis, -1:]) * dists)
    rgb = rgbs[Ellipsis, :3]
    rgba = torch.cat((rgb, alpha), dim=-1)

    if 0 < topk < self.n:
      ret = torch.topk(rbfs, topk, dim=0)
      rbfs = ret.values
      idx = ret.indices

      if sem_logits is not None:
        sem_logits = sem_logits[idx.view(-1)].view(topk, rbfs.size(1), -1)
    elif sem_logits is not None:
      sem_logits = sem_logits[:, None, :]

    rbfs = rbfs * (rbfs >= self.weight_min)

    rbf_sums = rbfs.sum(dim=0, keepdim=True)  # [N_n, N_samples, 1]

    if self.n == 1:
      outputs = rgba.sum(dim=0)
      sem_outputs = None if sem_logits is None else \
          sem_logits[:, None, :].sum(dim=0)
    else:
      outputs = (rbfs / (rbf_sums + 1e-7) * rgba).sum(dim=0)
      sem_outputs = None if sem_logits is None else \
          (rbfs / (rbf_sums + 1e-7) * sem_logits).sum(dim=0)

    return outputs, pen, sem_outputs

  def blend_cls(self,
                rbfs,
                rgbs,
                dists,
                topk,
                sigma_only=False,
                sem_logits=None,
                cls_id=0,
                nerflets=None):
    nerflets_cls_id = nerflets.sem_logits.argmax(dim=1, keepdims=True)
    nerflets_cls_id_filter = (nerflets_cls_id == cls_id)
    rbfs = rbfs.squeeze(-1)
    rbfs.mul_(nerflets_cls_id_filter)
    return self.blend(rbfs[Ellipsis, None], rgbs, dists, topk, sigma_only,
                      sem_logits)

  def forward(self):
    raise NotImplementedError

  def dumps(self, path=None):
    """Encodes a ldif to a string, and optionally writes it to disk.

        A description of the file format:
        Line 1: SIF
        Line 2: Three ints separated by spaces. In order:
          1) The number of blobs.
          2) The version ID for the blob types. I added this to be safe since
             last time when we updated to add rotation it broke all the old txt
             files. For now it will always be zero, which means the following
             eleven explicit parameters will be given per blob (in order):
               1 constant. float.
               3 centers (XYZ). float.
               3 radii (XYZ diagonals). float.
               3 radii (roll-pitch-yaw rotations). float.
               1 symmetry ID type. int. For now it will be either 0 or 1:
                   Zero: Not symmetric.
                    One: Left-right (XY-plane) symmetry.
          3) The number of implicit parameters per blob. So it will likely
             be between 0-256.
        After the first two lines, there is a line for each blob.
         Each line will have the explicit parameters followed by the implicit
         parameters. They are space separated.
        Args:
         sif_vector: The SIF vector to encode as a np array. Has shape
           (element_count, element_length).
         path: dump path

        Returns:
          A string encoding of v in the ldif v1 file format.
    """
    shape_count = self.centers.size(0)
    centers = self.centers.cpu()
    radii = torch.sqrt(self.radii.clamp(min=0)).cpu()
    rotations = self.rotations.cpu()
    implicit_len = 0
    header = 'SIF\n%i %i %i\n' % (shape_count, 0, implicit_len)
    out = header
    for row_idx in range(shape_count):
      row_vec = torch.cat([
          torch.tensor([self.const]), centers[row_idx, :], radii[row_idx, :],
          rotations[row_idx, :]
      ])
      row = ' '.join(10 * ['%.9g']) % tuple(row_vec.tolist())
      symmetry = 0
      row += ' %i' % symmetry
      row += '\n'
      out += row
    if path is not None:
      with open(path, 'wt') as f:
        f.write(self.dumps())
    return out


coverage_funcs = {'rbf': RbfLayer, 'avg': AverageLayer}

if __name__ == '__main__':
  rbf_layer = RbfLayer(64)
  xyz = torch.rand(4096, 3) - 0.5
  rgbs = torch.rand(64, 4096, 4)
  dists = torch.rand(4096, 1)

  ret, pen = rbf_layer(xyz, rgbs, dists)
  print(ret)
  print(pen)
  print(ret.shape)
  print(pen.shape)
