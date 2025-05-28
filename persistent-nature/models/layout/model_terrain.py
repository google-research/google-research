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

"""Wrapper for terrain model."""
import pickle
import dnnlib
import torch
from torch_utils import persistence
from utils import camera_util
from utils.utils import interpolate


@persistence.persistent_class
class ModelTerrain(torch.nn.Module):
  """Terrain wraps upsampler and layout model."""

  def __init__(self, layout_model_path, **upsampler_kwargs):
    """Initialize wrapper for upsampler refinement module.

    Args:
        layout_model_path: str containing the path to layout model
        **upsampler_kwargs: dictionary of inputs to initialize upsampler
    """
    super().__init__()
    self.upsampler = dnnlib.util.construct_class_by_name(**upsampler_kwargs)
    self.z_dim = self.upsampler.z_dim
    self.c_dim = self.upsampler.c_dim
    self.img_channels = self.upsampler.img_channels
    self.img_resolution = self.upsampler.img_resolution
    self.layout_model_path = layout_model_path

    with open(self.layout_model_path, 'rb') as f:
      self.layout_model = pickle.load(f)['G_ema'].eval()

    # sanity checks
    assert self.upsampler.z_dim == self.layout_model.layout_generator.w_dim
    assert (
        self.upsampler.synthesis.input_channels
        == self.layout_model.layout_decoder.init_kwargs.nerf_mlp_config.out_channel
    )

  def decode_layout(
      self,
      z,
      c,
      camera_params,
      noise_input=None,
      extras=[],
      **mapping_and_layout_kwargs
  ):  # pylint: disable=dangerous-default-value
    """Render the low-resolution image from the layout model."""

    self.layout_model.requires_grad_(False)
    # self.layout_model.patch_size = 1024 # 256
    self.layout_model.eval()

    # add layout noise if not provided as input arguments
    if self.upsampler.synthesis.default_noise_mode == '3dnoise':
      if 'layout_noise' not in extras:
        extras.append('layout_noise')

    feature_size = self.layout_model.layout_generator.img_resolution
    if noise_input is None:
      noise_input = torch.randn(
          z.shape[0], 1, feature_size, feature_size, device=z.device
      )
    # pylint: disable-next=invalid-name
    fake_rgb, fake_depth, fake_acc, Rt, K, layout, ws, extras = (
        self.layout_model.generate(
            z,
            camera_params=camera_params,
            c=c,
            extras=extras,
            noise_input=noise_input,
            **mapping_and_layout_kwargs
        )
    )
    return fake_rgb, fake_depth, fake_acc, Rt, K, layout, ws, extras

  def mapping(
      self,
      z,
      c,
      camera_params=None,
      truncation_psi=1,
      truncation_cutoff=None,
      update_emas=False,
      **kwargs
  ):
    """Get the low-resolution image and prepare inputs for upsampler."""

    mapping_kwargs = dict(
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
        update_emas=update_emas,
    )
    # compute camera params
    bs = z.shape[0]
    if camera_params is None:
      camera_params = camera_util.get_full_image_parameters(
          self.layout_model,
          self.layout_model.layout_decoder_kwargs.nerf_out_res,
          batch_size=bs,
          device=z.device,
          Rt=None,
      )
    # pylint: disable-next=invalid-name
    fake_rgb, fake_depth, fake_acc, Rt, K, layout, ws, extras = (
        self.decode_layout(
            z=z, c=c, camera_params=camera_params, **mapping_kwargs, **kwargs
        )
    )
    to_rgb_input = fake_rgb

    feature = extras['rgb_feature']
    if self.upsampler.synthesis.num_additional_feature_channels > 0:
      # concat depth to features and rgb
      feature = torch.cat([feature, fake_depth], dim=1)
      to_rgb_input = torch.cat([to_rgb_input, fake_depth], dim=1)
    if self.upsampler.synthesis.num_additional_feature_channels > 1:
      # concat acc to features and rgb
      feature = torch.cat([feature, fake_acc], dim=1)
      to_rgb_input = torch.cat([to_rgb_input, fake_acc], dim=1)

    extras = dict(
        z=z, K=K, Rt=Rt, layout=layout, depth=fake_depth, acc=fake_acc, **extras
    )

    upsampler_ws = self.upsampler.mapping(ws[:, 0], c, **mapping_kwargs)
    return upsampler_ws, feature, to_rgb_input, extras

  def synthesis(self, upsampler_ws, feature, to_rgb_input, extras, **kwargs):
    """Run the upsampler synthesis model."""

    feature_resize = interpolate(feature, self.upsampler.input_resolution)
    rgb_resize = interpolate(to_rgb_input, self.upsampler.input_resolution)
    feature_resize = feature_resize.detach()
    rgb_resize = rgb_resize.detach()

    if self.upsampler.synthesis.default_noise_mode == '3dnoise':
      size = self.layout_model.layout_decoder_kwargs.nerf_out_res
      noise_input = extras['layout_noise']
      noise_input = noise_input.view(-1, size, size)[:, None]  # B1HW
    else:
      noise_input = None
    rgb_up = self.upsampler.synthesis(
        upsampler_ws,
        feature_resize,
        rgb_resize,
        noise_input=noise_input,
        **kwargs
    )
    return rgb_up

  def forward(
      self,
      z,
      c,
      camera_params=None,
      truncation_psi=1,
      truncation_cutoff=None,
      update_emas=False,
      **kwargs
  ):
    """Generate refined terrain.

    Generates the refined terrain using the layout model and upsampler
    model, from sampled latent code z and camera parameters

    Args:
        z: torch.Tensor of shape [batch_size, z_dim]
        c: torch.Tensor of shape [batch_size, c_dim] or None, the class
          conditioning input of StyleGAN
        camera_params: dictionary containing keys K and optionally Rt, K is a
          torch.Tensor of shape [batch_size, 3, 3] of intrinsics, Rt is a
          torch.Tensor of shape [batch_size, 4, 4] of extrinsics. If
          camera_params is None then they will be inferred based on the fov and
          poses specified in the layout model
        truncation_psi: float, StyleGAN truncation value
        truncation_cutoff: int or None, StyleGAN truncation cutoff
        update_emas: bool, whether layer ema values should be updated, from
          StyleGAN training framework
        **kwargs: dictionary of additional arguments to StyleGAN layers

    Returns:
        outputs: a dictionary of the composite image outputs and associated
          intermediate outputs such as terrain image, depth, sky mask,
          low-resolution outputs, and intermediate rendering outputs
    """

    upsampler_ws, feature, rgb_thumbnail, extras = self.mapping(
        z,
        c,
        camera_params=camera_params,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
        update_emas=update_emas,
        **kwargs
    )

    # note: doesn't apply truncation params on upsampler
    rgb_up = self.synthesis(
        upsampler_ws,
        feature,
        rgb_thumbnail,
        extras,
        update_emas=update_emas,
        **kwargs
    )
    return rgb_up, rgb_thumbnail
