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

"""3D projected noise for triplane model."""
import numpy as np
import torch
from torch_utils import persistence


@persistence.persistent_class
class NoiseGenerator(torch.nn.Module):
  """wrapper for 3D noise tensor."""

  def __init__(self, x_dim, y_dim, z_dim, box_x, box_y, box_z, seed=None):
    super().__init__()
    self.x_dim = x_dim
    self.y_dim = y_dim
    self.z_dim = z_dim
    self.box_x = box_x
    self.box_y = box_y
    self.box_z = box_z
    self.seed = seed

  def get_noise(self, batch_size, device):
    seed = self.seed
    # grid sample expects BxCxDxHxW dimensions
    if seed is None:
      noise = torch.randn(
          batch_size, 1, self.z_dim, self.y_dim, self.x_dim, device=device
      )
    else:
      rng = np.random.RandomState(seed)
      noise = rng.normal(
          size=(batch_size, 1, self.z_dim, self.y_dim, self.x_dim)
      )
      noise = torch.from_numpy(noise).float()
    return {
        'noise': noise.to(device),
        'box_x': self.box_x,
        'box_y': self.box_y,
        'box_z': self.box_z,
    }


def sample_noise(noise_input, points):
  """runs gridsample on noise tensor given xyz points."""
  # points.shape = [batch, H*W*num_samples, 3]
  noise = noise_input['noise']
  box_x = noise_input['box_x']
  box_y = noise_input['box_y']
  box_z = noise_input['box_z']
  # normalize the points by x_dim // 2, y_dim // 2, z_dim // 2
  x = points[:, :, [0]] / (box_x / 2)
  y = points[:, :, [1]] / (box_y / 2)
  z = points[:, :, [2]] / (box_z / 2)
  points_norm = torch.cat([x, y, z], dim=2)
  # gridsample the noise
  # input= N,C,D(z),H(y),W(x)
  # grid= N, Dout, Hout=1, Wout=1, 3(x,y,z)
  # output =N, C, Dout, Hout=1, Wout=1
  points_norm = points_norm[:, :, None, None, :]
  noise_render = torch.nn.functional.grid_sample(
      input=noise,
      grid=points_norm,
      mode='bilinear',
      align_corners=False,
      padding_mode='reflection',
  )
  # return the rendered noise
  noise_render = noise_render[:, :, :, 0, 0]  # NxCxDout
  noise_render = noise_render.permute(0, 2, 1)  # NxDoutxC
  return noise_render


def build_soat_noise(G, grid_size):  # pylint: disable=invalid-name
  noise_gen = NoiseGenerator(
      G.plane_resolution * grid_size,
      G.plane_resolution,
      G.plane_resolution * grid_size,
      G.rendering_kwargs['box_warp'] * grid_size,
      G.rendering_kwargs['box_warp'],
      G.rendering_kwargs['box_warp'] * grid_size,
  )
  return noise_gen
