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

"""Wrapper for layout model."""
import dnnlib
from external.gsn.models.model_utils import RenderParams
import torch
from torch_utils import persistence
from utils import utils


# adapted from GSN: https://github.com/apple/ml-gsn/blob/main/models/gsn.py
@persistence.persistent_class
class ModelLayout(torch.nn.Module):
  """Wrapper for layout rendering."""

  def __init__(
      self,
      layout_generator_kwargs,
      layout_decoder_kwargs,
      rgb_projection_kwargs,
      img_res=32,
      patch_size=None,
      voxel_res=256,
      voxel_size=0.15,
      fov_mean=60,
      fov_std=0,
  ):
    """Initialize wrapper layout rendering model.

    The layout model consists of 2D ground plane features, a neural decoder
    network, and a rgb projection layer.

    Args:
        layout_generator_kwargs: dictionary of arguments to initialize the
          layout generator model, which is a StyleGAN
        layout_decoder_kwargs: dictionary of arguments to initialize the layout
          decoder model, which is an MLP architecture
        rgb_projection_kwargs: dictionary of arguments to initialize the rgb
          projection layer, which is a single linear layer
        img_res: int, rendering resolution
        patch_size: int or None, optional size of patches for neural rendering,
          helps to render larger images at inference time
        voxel_res: int, resolution of ground plane features
        voxel_size: float, unit size of each ground plane feature
        fov_mean: float, average FOV of camera
        fov_std: float, std of camera FOVs to sample, assuming normally
          distributed
    """
    super().__init__()
    self.layout_generator_kwargs = layout_generator_kwargs
    self.layout_decoder_kwargs = layout_decoder_kwargs
    self.rgb_projection_kwargs = rgb_projection_kwargs
    self.z_dim = layout_generator_kwargs.z_dim
    self.c_dim = layout_generator_kwargs.c_dim

    self.img_res = img_res
    self.patch_size = patch_size
    self.voxel_res = voxel_res
    self.voxel_size = voxel_size
    self.coordinate_scale = voxel_res * voxel_size
    self.fov_mean = fov_mean
    self.fov_std = fov_std

    assert layout_decoder_kwargs.img_res == img_res
    assert layout_decoder_kwargs.global_feat_res == voxel_res

    self.layout_generator = dnnlib.util.construct_class_by_name(
        **layout_generator_kwargs
    )
    self.layout_decoder = dnnlib.util.construct_class_by_name(
        **layout_decoder_kwargs
    )
    self.rgb_projection = dnnlib.util.construct_class_by_name(
        **rgb_projection_kwargs
    )

  def set_trajectory_sampler(self, trajectory_sampler):
    self.trajectory_sampler = trajectory_sampler

  def generate(
      self,
      z,
      c,
      camera_params,
      update_emas=False,
      truncation_psi=1.0,
      truncation_cutoff=None,
      cached_layout=None,
      extras=[],
      noise_input=None,
      **kwargs
  ):  # pylint: disable=dangerous-default-value
    # camera_params should be a dict with Rt and K
    # (if Rt is not present it will be sampled)
    nerf_out_res = self.layout_decoder_kwargs.nerf_out_res
    samples_per_ray = self.layout_decoder_kwargs.samples_per_ray
    batch_size = z.shape[0]

    # map 1D latent code z to style latent code w
    # (this style code is later used in upsampler)
    ws = self.layout_generator.mapping(
        z=z,
        c=c,
        update_emas=update_emas,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
    )

    # use inference layout if necessary
    if cached_layout is None:
      layout = self.layout_generator.synthesis(
          ws=ws, update_emas=update_emas, **kwargs
      )
    else:
      layout = cached_layout

    if 'Rt' not in camera_params.keys():
      # pylint: disable-next=invalid-name
      Rt = self.trajectory_sampler.sample_trajectories(
          self.layout_decoder, layout
      )
      camera_params['Rt'] = Rt[:, 0]  # Bx4x4

    if self.patch_size is None:
      # compute full image in one pass
      indices_chunks = [None]
    elif nerf_out_res <= self.patch_size:
      indices_chunks = [None]
    elif nerf_out_res > self.patch_size:
      # break the whole image into manageable pieces,
      # then compute each of those separately
      indices = torch.arange(nerf_out_res**2, device=z.device)
      indices_chunks = torch.chunk(
          indices, chunks=int(nerf_out_res**2 / self.patch_size**2)
      )

    rgb, depth, acc, extra_outputs = [], [], [], []
    for indices in indices_chunks:
      render_params = RenderParams(
          Rt=camera_params['Rt'].clone(),  # Bx4x4
          K=camera_params['K'].clone(),  # Bx3x3
          samples_per_ray=samples_per_ray,
          near=self.layout_decoder_kwargs.near,
          far=self.layout_decoder_kwargs.far,
          alpha_noise_std=self.layout_decoder_kwargs.alpha_noise_std,
          nerf_out_res=nerf_out_res,
          mask=indices,
      )
      patch_outputs = self.layout_decoder(
          local_latents=layout, render_params=render_params
      )
      rgb.append(patch_outputs['rgb'])  # shape [BT, HW, C]
      depth.append(patch_outputs['depth'])
      acc.append(patch_outputs['acc'])
      patch_extra_outputs = {}
      if 'checkerboard' in extras:
        xz = patch_outputs['xyz'][Ellipsis, [0, 2]]
        scaled_xz = xz / self.voxel_size
        grid = torch.round(scaled_xz).long() % 2 == 1
        checker = torch.logical_xor(grid[Ellipsis, 0], grid[Ellipsis, 1])
        patch_extra_outputs['checkerboard'] = torch.sum(
            patch_outputs['weights'] * checker, dim=2
        )
      if 'entropy' in extras:
        entropy = -torch.sum(
            patch_outputs['weights']
            * torch.log(patch_outputs['weights'].clip(1e-8)),
            dim=-1,
        )
        patch_extra_outputs['entropy'] = entropy
      if 'camera_points' in extras:
        patch_extra_outputs['camera_points'] = patch_outputs['xyz']
      if 'opacity_regularization' in extras:
        patch_extra_outputs['alpha'] = patch_outputs['alpha']
        patch_extra_outputs['dists'] = patch_outputs['dists']
        patch_extra_outputs['weights'] = patch_outputs['weights']
      if 'layout_noise' in extras:
        xz = patch_outputs['xyz'][Ellipsis, [0, 2]]
        scaled_xz = xz / (self.coordinate_scale / 2)  # [-1, 1]
        assert noise_input is not None
        if self.layout_decoder.inference_feat_res:
          # rescale the gridsample coordinates
          scaled_xz = (
              scaled_xz
              * self.layout_decoder.global_feat_res
              / self.layout_decoder.inference_feat_res
          )
          assert noise_input.shape[-1] == self.layout_decoder.inference_feat_res
        noise_render = torch.nn.functional.grid_sample(
            input=noise_input,
            grid=scaled_xz,
            mode='bilinear',
            align_corners=False,
            padding_mode='reflection',
        )  # B x 1 x num_pixels x num_samples
        patch_extra_outputs['layout_noise'] = torch.sum(
            patch_outputs['weights'][:, None] * noise_render, dim=3
        ).squeeze(1)
      extra_outputs.append(patch_extra_outputs)

    # combine image patches back into full images
    rgb = torch.cat(rgb, dim=1)
    depth = torch.cat(depth, dim=1)
    acc = torch.cat(acc, dim=1)
    if extras:
      extra_outputs = utils.concat_dict(extra_outputs)
    else:
      extra_outputs = {}

    depth = depth.view(batch_size, 1, nerf_out_res, nerf_out_res)
    acc = acc.view(batch_size, 1, nerf_out_res, nerf_out_res)
    rgb = rgb.permute(0, 2, 1).view(batch_size, -1, nerf_out_res, nerf_out_res)
    extra_outputs['rgb_feature'] = rgb.clone()

    rgb = self.rgb_projection(rgb)  # convert features to RGB
    Rt = patch_outputs['Rt']  # pylint: disable=invalid-name
    K = patch_outputs['K']  # pylint: disable=invalid-name

    return rgb, depth, acc, Rt, K, layout, ws, extra_outputs

  def forward(
      self,
      z,
      c,
      camera_params,
      update_emas=False,
      truncation_psi=1,
      truncation_cutoff=None,
      **kwargs
  ):
    """Generate rendered terrain.

    Generates a low-resolution terrain image from rendering the ground
    plane layout features.

    Args:
        z: torch.Tensor of shape [batch_size, z_dim]
        c: torch.Tensor of shape [batch_size, c_dim] or None, the class
          conditioning input of StyleGAN
        camera_params: dictionary containing keys K and optionally Rt, K is a
          torch.Tensor of shape [batch_size, 3, 3] of intrinsics, Rt is a
          torch.Tensor of shape [batch_size, 4, 4] of extrinsics
        update_emas: bool, whether layer ema values should be updated, from
          StyleGAN training framework
        truncation_psi: float, StyleGAN truncation value
        truncation_cutoff: int or None, StyleGAN truncation cutoff
        **kwargs: dictionary of additional arguments for neural rendering and
          layout feature synthesis

    Returns:
        rgb: torch.Tensor of shape [batch_size, 3, height, width] of the
          low-resolution terrain image
        A dictionary of associated rendering information, including the
          depth, sky mask, camera extrinsics and intrinsics, layout features,
          style-code from StyleGAN mapping, and additional rendering
          outputs
    """
    # pylint: disable-next=invalid-name
    rgb, depth, acc, Rt, K, layout, ws, extra_outputs = self.generate(
        z,
        c,
        camera_params,
        update_emas=update_emas,
        truncation_psi=truncation_psi,
        truncation_cutoff=truncation_cutoff,
        **kwargs
    )
    return rgb, {
        'rgb': rgb,
        'depth': depth,
        'acc': acc,
        'Rt': Rt,
        'K': K,
        'layout': layout,
        'ws': ws,
        'extra_outputs': extra_outputs,
    }
