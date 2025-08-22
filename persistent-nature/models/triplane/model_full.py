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

"""Wrapper for triplane and sky model."""
import pickle

from models.sky import model_sky
import torch
from utils.utils import interpolate


class ModelFull(torch.nn.Module):
  """Extendable triplane wrapper."""

  def __init__(self, land_model_path, sky_model_path):
    """Initialize wrapper for full model, consisting of land and sky.

    Args:
        land_model_path: str containing the path to land model, using the
          triplane backbone
        sky_model_path: str containing the path to sky model
    """
    super().__init__()
    self.land_model_path = land_model_path
    self.sky_model_path = sky_model_path
    with open(self.land_model_path, 'rb') as f:
      ckpt = pickle.load(f)
      self.ground = ckpt['G_ema'].eval()  # torch.nn.Module
      self.ground.rendering_kwargs['white_back'] = False
      if 'world2cam_poses' in ckpt:
        # access training poses, if available
        self.world2cam_poses = ckpt['world2cam_poses']
      else:
        self.world2cam_poses = None
    with open(self.sky_model_path, 'rb') as f:
      sky = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module
      self.sky = model_sky.ModelSky(sky.G).eval()
    self.z_dim = self.ground.z_dim
    self.c_dim = self.ground.c_dim

  def forward(self, z, c, render_sky=False, **kwargs):
    """Generate image.

    Generates an image based on sampled latent code z and c (which
    contains the camera parameter information)

    Args:
        z: torch.Tensor of shape [batch_size, z_dim]
        c: torch.Tensor of shape [batch_size, 25], which is a vector of the
          flattened extrinsics and intrinsics matrix
        render_sky: bool, whether to render sky based on the generated terrain
          image
        **kwargs: additional arguments into the triplane terrain model

    Returns:
        outs: a dictionary of image outputs and associated intermediate
          outputs such as depth, sky mask, and low resolution outputs
    """
    outs = self.ground(z, c, **kwargs)
    if render_sky:
      img = outs['image'][:, :3]
      sky_z = z[:, : self.sky.G.z_dim]
      mask = interpolate(outs['weights_raw'], img.shape[-1])
      mask = torch.sigmoid((mask - 0.9) * 100)  # this sharpens the mask
      img_w_gray_sky = img * mask
      sky = self.sky(
          z=sky_z, c=None, img=img_w_gray_sky, acc=mask, multiply=False
      )
      composite = sky * (1 - mask) + img * mask
      outs['composite'] = composite
      outs['sky'] = sky
      outs['sky_mask'] = mask
    return outs
