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

"""Wrapper for land and sky together."""
import pickle

from models.sky import model_sky
import torch
from utils import sky_util
from utils.utils import interpolate


class ModelFull(torch.nn.Module):
  """Wraps land and sky models."""

  def __init__(self, land_model_path, sky_model_path, depth_noise=False):
    """Initialize wrapper for full model, consisting of land and sky.

    Args:
        land_model_path: str containing the path to land model, using the layout
          feature backbone
        sky_model_path: str containing the path to sky model
        depth_noise: bool of whether to use depth-based noise blending
    """

    super().__init__()
    self.land_model_path = land_model_path
    self.sky_model_path = sky_model_path

    self.depth_noise = depth_noise

    with open(self.land_model_path, 'rb') as f:
      self.terrain_model = pickle.load(f)['G_ema'].eval()
      self.orig_render_settings = {
          'nerf_out_res': (
              self.terrain_model.layout_model.layout_decoder_kwargs.nerf_out_res
          ),
          'samples_per_ray': (
              self.terrain_model.layout_model.layout_decoder_kwargs.samples_per_ray
          ),
          'nerf_far': self.terrain_model.layout_model.layout_decoder_kwargs.far,
          'patch_size': self.terrain_model.layout_model.patch_size,
      }

    with open(self.sky_model_path, 'rb') as f:
      # reload the class to get pretrained encoder loaded
      g_sky = pickle.load(f)['G_ema'].eval()
      g_sky = model_sky.ModelSky(g_sky.G)
      self.sky_model = g_sky

  def set_nerf_params(
      self,
      nerf_out_res=None,
      samples_per_ray=None,
      nerf_far=None,
      patch_size=None,
  ):
    if samples_per_ray is not None:
      self.terrain_model.layout_model.layout_decoder_kwargs.samples_per_ray = (
          samples_per_ray
      )
    if nerf_out_res is not None:
      self.terrain_model.layout_model.layout_decoder_kwargs.nerf_out_res = (
          nerf_out_res
      )
    if nerf_far is not None:
      self.terrain_model.layout_model.layout_decoder_kwargs.far = nerf_far
    if patch_size is not None:
      self.terrain_model.layout_model.patch_size = patch_size

  def composite_sky(self, rgb_in, sky_mask, sky_out):
    rgb_in = interpolate(rgb_in, sky_out.shape[-1])
    sky_mask = interpolate(sky_mask, sky_out.shape[-1])
    return rgb_in * sky_mask + (1 - sky_mask) * sky_out

  def sharpen_mask(self, acc):
    return torch.sigmoid((acc - 0.5) * 10)

  def forward(
      self,
      z,
      c,
      camera_params,
      truncation=1.0,
      truncation_cutoff=None,
      nerf_kwargs={},
      upsampler_kwargs={},
      sky_texture=None,
  ):  # pylint: disable=dangerous-default-value
    """Generates an image based on sampled latent code z and camera parameters.

    Args:
        z: torch.Tensor of shape [batch_size, z_dim]
        c: torch.Tensor of shape [batch_size, c_dim] or None, the class
          conditioning input of StyleGAN
        camera_params: dictionary containing keys K and optionally Rt, K is a
          torch.Tensor of shape [batch_size, 3, 3] of intrinsics, Rt is a
          torch.Tensor of shape [batch_size, 4, 4] of extrinsics
        truncation: float, StyleGAN truncation value
        truncation_cutoff: int or None, StyleGAN truncation cutoff
        nerf_kwargs: dictionary of additional arguments to rendering in layout
          generator model
        upsampler_kwargs: dictionary of additional arguments to upsampler
        sky_texture: torch.Tensor of shape [batch_size, 3, height, weight] or
          None, the sky image that is composited with the terrain image

    Returns:
        outputs: a dictionary of the composite image outputs and associated
          intermediate outputs such as terrain image, depth, sky mask,
          low-resolution outputs, and intermediate rendering outputs
    """
    mapping_kwargs = dict(
        truncation_psi=truncation,
        truncation_cutoff=truncation_cutoff,
        update_emas=False,
    )
    upsampler_ws, feature, rgb_zc, extras = self.terrain_model.mapping(
        z, c, camera_params, **mapping_kwargs, **nerf_kwargs
    )

    fake_depth = extras['depth']
    fake_acc = extras['acc']

    layout_model = self.terrain_model.layout_model
    upsampler = self.terrain_model.upsampler

    if upsampler.synthesis.default_noise_mode == '3dnoise':
      if self.depth_noise:
        # blend two sizes of the projected noise
        size = layout_model.layout_decoder_kwargs.nerf_out_res
        noise_input = extras['layout_noise']
        noise_input = noise_input.view(-1, size, size)[:, None]
        down_size = upsampler.input_resolution
        noise_input_far = interpolate(interpolate(noise_input, down_size), size)
        weights = fake_depth.clamp(0, 1)
        weight_min = 0.1
        weight_max = 0.3
        mix_weights = (weights.clamp(weight_min, weight_max) - weight_min) / (
            weight_max - weight_min
        )
        noise_input = (
            mix_weights * noise_input + (1 - mix_weights) * noise_input_far
        )
      else:
        size = layout_model.layout_decoder_kwargs.nerf_out_res
        noise_input = extras['layout_noise']
        noise_input = noise_input.view(-1, size, size)[:, None]
    else:
      noise_input = None

    # update layout noise with depth-aware blend if needed
    extras['layout_noise'] = noise_input
    rgb_up = self.terrain_model.synthesis(
        upsampler_ws,
        interpolate(feature, upsampler.input_resolution),
        interpolate(rgb_zc, upsampler.input_resolution),
        extras=extras,
        **upsampler_kwargs
    )

    # if rgb_up includes depth channel
    if rgb_up.shape[1] > 3:
      depth_up = rgb_up[:, 3:4]
    else:
      depth_up = None

    # if rgb_up includes acc channel
    if rgb_up.shape[1] > 4:
      acc_up = rgb_up[:, 4:]
    else:
      acc_up = None

    rgb_up = rgb_up[:, :3]

    outputs = {
        'rgb_thumb': rgb_zc[:, :3],
        'depth_thumb': fake_depth,
        'acc_thumb': fake_acc,
        'rgb_up': rgb_up,
        'depth_up': depth_up,
        'acc_up': acc_up,
        'acc_sharpen': (
            self.sharpen_mask(acc_up)
            if acc_up is not None
            else self.sharpen_mask(fake_acc)
        ),
    }
    # copy Rt, K, z, layout, etc to outputs
    outputs['extras'] = extras

    outputs['rgb_up'] = rgb_up[:, :3]

    # copy noise input (includes depth blending)
    if upsampler.synthesis.default_noise_mode == '3dnoise':
      outputs['3dnoise'] = noise_input

    if sky_texture is not None:
      Rt = outputs['extras']['Rt']  # pylint: disable=invalid-name
      if sky_texture.shape == rgb_up.shape:
        # if sky is a single panel, use it directly
        sky_out = sky_texture
      else:
        sky_out = sky_util.sample_sky_from_Rt(
            sky_texture,
            Rt,
            rgb_up.shape[-1],
            self.terrain_model.layout_model.fov_mean,
        )
        sky_out = sky_out.to(rgb_up.device)
    else:
      sky_out = self.sky_model(
          z=z,
          c=c,
          img=rgb_up,
          acc=torch.ones_like(rgb_up),
          multiply=False,
          **mapping_kwargs
      )

    sky_mask = outputs['acc_sharpen']
    fake_rgb_overlay = self.composite_sky(rgb_up, sky_mask, sky_out)
    outputs['rgb_overlay_upsample'] = fake_rgb_overlay
    outputs['sky_mask'] = sky_mask
    outputs['sky_out'] = sky_out
    return outputs
