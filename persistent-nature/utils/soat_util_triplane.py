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

# pylint: disable=invalid-name,g-multiple-import
"""utils for soat interpolation on triplane model."""
import math

import numpy as np
import torch
from torch_utils.ops import upfirdn2d
from utils import camera_util, sky_util
from utils import soat_renderer_triplane
from utils.soat_util import interpolate_features, bilinear_interpolate_features, concat_overlapping
from utils.utils import interpolate

G = None
soat_renderer = None


def init_soat_model(G_saved):
  global G
  global soat_renderer
  G = G_saved.eval().requires_grad_(False)
  G.rendering_kwargs['white_back'] = False  # disable white back
  # note: white_back doesn't matter when using sigmoid(mask-0.9)
  soat_renderer = soat_renderer_triplane.ImportanceRendererSOAT(
      G.img_resolution
  )
  return G


def construct_intrinsics(fov):
  focal = 0.5 / math.tan((fov / 2) * math.pi / 180)
  intrinsics = torch.tensor(
      [[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]]
  )
  return intrinsics


def sample_random_camera(fov, box_warp, seed=None):
  """sample a random camera with the box distance."""
  if seed is None:
    rng = np.random.RandomState()
  else:
    rng = np.random.RandomState(seed)
  initial_x = rng.uniform(-box_warp / 2, box_warp / 2)
  initial_y = 0.0
  initial_z = rng.uniform(-box_warp / 2, box_warp / 2)
  initial_theta = rng.uniform(0, 360)
  initial_psi = 0.0
  sampled_camera = camera_util.Camera(
      float(initial_x),
      float(initial_y),
      float(initial_z),
      float(initial_theta),
      float(initial_psi),
  )
  world2cam_matrix = camera_util.pose_from_camera(sampled_camera)
  cam2world_matrix = world2cam_matrix.inverse()
  intrinsics = construct_intrinsics(fov)
  return sampled_camera, cam2world_matrix, intrinsics


def prepare_zs(seed, grid_h=4, grid_w=4, device='cuda'):
  if seed is not None:
    torch.manual_seed(seed)
  zs = torch.randn(grid_h, grid_w, 1, G.z_dim, device=device)
  c = torch.zeros(1, 25, device=device)
  return zs, c


def interpolate_zs(zs, weight=0.7):
  """blend a grid of zs.

  interpolate a grid of random zs so that they are closer to each other
  helps to generate more coherent content

  Args:
    zs: torch.Tensor HxWx1xC
    weight: float

  Returns:
    blended latents
  """

  grid_size = zs.shape[0]
  zmid = zs[[grid_size // 2], [grid_size // 2], :, :]
  zinterp = zmid * weight + zs * (1 - weight)
  zinterp = (
      zinterp
      / torch.norm(zinterp, dim=-1, keepdim=True)
      * torch.norm(zs, dim=-1, keepdim=True)
  )
  zs = zinterp
  return zs


def prepare_ws(zs, cs, truncation_psi=1, truncation_cutoff=None):
  return G.mapping(
      zs, cs, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
  )


def split_ws(ws):
  ws = ws.to(torch.float32)
  w_idx = 0
  block_ws = []
  for res in G.backbone_xz.synthesis.block_resolutions:
    block = getattr(G.backbone_xz.synthesis, f'b{res}')
    block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
    w_idx += block.num_conv
  return block_ws


def generate_xz(zs, c, device='cuda', truncation_psi=1, truncation_cutoff=None):
  """SOAT stitch XZ plane."""
  frames = []
  mapping_kwargs = dict(
      truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
  )
  force_fp32 = False
  grid_h, grid_w, _, _ = zs.shape
  for h in range(1, grid_h):
    row = []
    for w in range(1, grid_w):
      ws00 = split_ws(prepare_ws(zs[h - 1, w - 1], c, **mapping_kwargs))
      ws01 = split_ws(prepare_ws(zs[h - 1, w], c, **mapping_kwargs))
      ws10 = split_ws(prepare_ws(zs[h, w - 1], c, **mapping_kwargs))
      ws11 = split_ws(prepare_ws(zs[h, w], c, **mapping_kwargs))
      img = None
      for i, res in enumerate(G.backbone_xz.synthesis.block_resolutions):
        block = getattr(G.backbone_xz.synthesis, f'b{res}')
        ws00_iter = ws00[i].unbind(dim=1)
        ws01_iter = ws01[i].unbind(dim=1)
        ws10_iter = ws10[i].unbind(dim=1)
        ws11_iter = ws11[i].unbind(dim=1)
        dtype = (
            torch.float16
            if block.use_fp16 and not force_fp32
            else torch.float32
        )
        memory_format = (
            torch.channels_last
            if block.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if block.in_channels == 0:
          out00 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws00[i].shape[0], 1, 1, 1)
          )
          out01 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws01[i].shape[0], 1, 1, 1)
          )
          out10 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws10[i].shape[0], 1, 1, 1)
          )
          out11 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(ws11[i].shape[0], 1, 1, 1)
          )

          out00 = block.conv1(out00, ws00_iter[0])
          out01 = block.conv1(out01, ws01_iter[0])
          out10 = block.conv1(out10, ws10_iter[0])
          out11 = block.conv1(out11, ws11_iter[0])

          # feature
          out0 = torch.cat([out00, out01], dim=3)
          out1 = torch.cat([out10, out11], dim=3)
          out = torch.cat([out0, out1], dim=2)

          # rgb skip connection
          skip00 = block.torgb(out, ws00_iter[1])
          skip01 = block.torgb(out, ws01_iter[1])
          skip10 = block.torgb(out, ws10_iter[1])
          skip11 = block.torgb(out, ws11_iter[1])
          skip = bilinear_interpolate_features(skip00, skip01, skip10, skip11)
          img = img.add_(skip) if img is not None else skip
        else:
          out = out.to(dtype=dtype, memory_format=memory_format)

          out00 = block.conv0(out, ws00_iter[0])
          out01 = block.conv0(out, ws01_iter[0])
          out10 = block.conv0(out, ws10_iter[0])
          out11 = block.conv0(out, ws11_iter[0])
          out = bilinear_interpolate_features(out00, out01, out10, out11)

          out00 = block.conv1(out, ws00_iter[1])
          out01 = block.conv1(out, ws01_iter[1])
          out10 = block.conv1(out, ws10_iter[1])
          out11 = block.conv1(out, ws11_iter[1])
          out = bilinear_interpolate_features(out00, out01, out10, out11)

          skip00 = block.torgb(out, ws00_iter[2])
          skip01 = block.torgb(out, ws01_iter[2])
          skip10 = block.torgb(out, ws10_iter[2])
          skip11 = block.torgb(out, ws11_iter[2])
          skip = bilinear_interpolate_features(skip00, skip01, skip10, skip11)

          if img is not None:
            img = upfirdn2d.upsample2d(img, block.resample_filter)
          img = (
              img.add_(skip.to(dtype=torch.float32))
              if img is not None
              else skip
          )

      row.append(img)
    frames.append(row)
  rows_concat = [concat_overlapping(row, dim=3) for row in frames]
  concat = concat_overlapping(rows_concat, dim=2)
  return concat.to(device)


def generate_xy(zs, c, device='cuda', truncation_psi=1, truncation_cutoff=None):
  """SOAT stitch all XY planes."""

  mapping_kwargs = dict(
      truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
  )
  grid_h, grid_w, _, _ = zs.shape
  planes = []
  force_fp32 = False

  for h in range(0, grid_h):  # keep the z axis fixed
    row = []
    for w in range(1, grid_w):  # stitch along the x axis
      w1_in = split_ws(prepare_ws(zs[h, w - 1], c, **mapping_kwargs))
      w2_in = split_ws(prepare_ws(zs[h, w], c, **mapping_kwargs))
      img = None
      for i, res in enumerate(G.backbone_xy.synthesis.block_resolutions):
        block = getattr(G.backbone_xy.synthesis, f'b{res}')
        w1_iter = w1_in[i].unbind(dim=1)
        w2_iter = w2_in[i].unbind(dim=1)
        dtype = (
            torch.float16
            if block.use_fp16 and not force_fp32
            else torch.float32
        )
        memory_format = (
            torch.channels_last
            if block.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if block.in_channels == 0:
          out1 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(w1_in[i].shape[0], 1, 1, 1)
          )
          out2 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(w2_in[i].shape[0], 1, 1, 1)
          )
          out1 = block.conv1(out1, w1_iter[0])
          out2 = block.conv1(out2, w2_iter[0])
          # feature
          out = torch.cat([out1, out2], dim=3)
          # rgb skip connection
          skip1 = block.torgb(out, w1_iter[1])
          skip2 = block.torgb(out, w2_iter[1])
          skip = interpolate_features(skip1, skip2)
          img = img.add_(skip) if img is not None else skip
        else:
          out = out.to(dtype=dtype, memory_format=memory_format)
          out1 = block.conv0(out, w1_iter[0])
          out2 = block.conv0(out, w2_iter[0])
          out = interpolate_features(out1, out2)

          out1 = block.conv1(out, w1_iter[1])
          out2 = block.conv1(out, w2_iter[1])
          out = interpolate_features(out1, out2)

          skip1 = block.torgb(out, w1_iter[2])
          skip2 = block.torgb(out, w2_iter[2])
          skip = interpolate_features(skip1, skip2)
          if img is not None:
            img = upfirdn2d.upsample2d(img, block.resample_filter)
          img = (
              img.add_(skip.to(dtype=torch.float32))
              if img is not None
              else skip
          )
      row.append(img)
    planes.append(row)
  concat = [concat_overlapping(row, dim=3).to(device) for row in planes]
  return concat


def generate_yz(zs, c, device='cuda', truncation_psi=1, truncation_cutoff=None):
  """SOAT stitch all YZ planes."""
  mapping_kwargs = dict(
      truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
  )
  grid_h, grid_w, _, _ = zs.shape
  planes = []
  force_fp32 = False

  for w in range(0, grid_w):  # keep the x axis fixed
    row = []
    for h in range(1, grid_h):  # stitch along the z axis
      w1_in = split_ws(prepare_ws(zs[h - 1, w], c, **mapping_kwargs))
      w2_in = split_ws(prepare_ws(zs[h, w], c, **mapping_kwargs))
      img = None
      for i, res in enumerate(G.backbone_yz.synthesis.block_resolutions):
        block = getattr(G.backbone_yz.synthesis, f'b{res}')
        w1_iter = w1_in[i].unbind(dim=1)
        w2_iter = w2_in[i].unbind(dim=1)
        dtype = (
            torch.float16
            if block.use_fp16 and not force_fp32
            else torch.float32
        )
        memory_format = (
            torch.channels_last
            if block.channels_last and not force_fp32
            else torch.contiguous_format
        )
        if block.in_channels == 0:
          out1 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(w1_in[i].shape[0], 1, 1, 1)
          )
          out2 = (
              block.const.to(dtype=dtype, memory_format=memory_format)
              .unsqueeze(0)
              .repeat(w2_in[i].shape[0], 1, 1, 1)
          )
          out1 = block.conv1(out1, w1_iter[0])
          out2 = block.conv1(out2, w2_iter[0])
          # feature
          out = torch.cat(
              [out1, out2], dim=2
          )  # NOTE: changed the dimension here
          # rgb skip connection
          skip1 = block.torgb(out, w1_iter[1])
          skip2 = block.torgb(out, w2_iter[1])
          skip = interpolate_features(skip1, skip2)
          img = img.add_(skip) if img is not None else skip
        else:
          out = out.to(dtype=dtype, memory_format=memory_format)
          out1 = block.conv0(out, w1_iter[0])
          out2 = block.conv0(out, w2_iter[0])
          out = interpolate_features(out1, out2)

          out1 = block.conv1(out, w1_iter[1])
          out2 = block.conv1(out, w2_iter[1])
          out = interpolate_features(out1, out2)

          skip1 = block.torgb(out, w1_iter[2])
          skip2 = block.torgb(out, w2_iter[2])
          skip = interpolate_features(skip1, skip2)
          if img is not None:
            img = upfirdn2d.upsample2d(img, block.resample_filter)
          img = (
              img.add_(skip.to(dtype=torch.float32))
              if img is not None
              else skip
          )
      row.append(img)
    planes.append(row)
  concat = [concat_overlapping(row, dim=2).to(device) for row in planes]
  return concat


def render_planes(
    planes,
    ws,
    cam2world_matrix,
    intrinsics,
    noise_input,
    depth_blend_noise=False,
):
  """get features from planes given camera information."""
  neural_rendering_resolution = None
  if neural_rendering_resolution is None:
    neural_rendering_resolution = G.neural_rendering_resolution
  else:
    G.neural_rendering_resolution = neural_rendering_resolution

  # Create a batch of rays for volume rendering
  ray_origins, ray_directions = G.ray_sampler(
      cam2world_matrix, intrinsics, neural_rendering_resolution
  )
  N, _, _ = ray_origins.shape

  # check noise_input shape
  noise = noise_input['noise']  # NCDHW
  assert planes[1].shape[2] == noise.shape[2]  # D
  assert planes[0][0].shape[2] == noise.shape[3]  # H (planes is y,x shape)
  assert planes[1].shape[3] == noise.shape[4]  # W

  (
      feature_samples,
      depth_samples,
      disp_samples,
      weights_samples,
      noise_samples,
  ) = soat_renderer(
      planes,
      G.decoder,
      ray_origins,
      ray_directions,
      G.rendering_kwargs,
      noise_input,
  )

  # Reshape into 'raw' neural-rendered image
  H = W = G.neural_rendering_resolution
  feature_image = (
      feature_samples.permute(0, 2, 1)
      .reshape(N, feature_samples.shape[-1], H, W)
      .contiguous()
  )
  depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
  disp_image = disp_samples.permute(0, 2, 1).reshape(N, 1, H, W)
  weights_image = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)
  noise_image = noise_samples.permute(0, 2, 1).reshape(N, 1, H, W)

  # blend the noise as a function of depth
  if depth_blend_noise:
    size = G.neural_rendering_resolution
    down_size = 16
    noise_image_far = interpolate(interpolate(noise_image, down_size), size)
    weights = disp_image.clamp(0, 1)
    weight_min = 0.1
    weight_max = 0.2
    mix_weights = (weights.clamp(weight_min, weight_max) - weight_min) / (
        weight_max - weight_min
    )
    noise_image = (
        mix_weights * noise_image + (1 - mix_weights) * noise_image_far
    )

  # Run superresolution to get final image
  rgb_image = feature_image[:, :3]
  rgb_and_disp_image = torch.cat([rgb_image, disp_image], dim=1)
  # G.superresolution will modify rgb_and_disp_image in
  # place --> make a copy of it
  sr_image_and_disp = G.superresolution(
      rgb_and_disp_image.clone(),
      feature_image,
      ws,
      noise_mode=G.rendering_kwargs['superresolution_noise_mode'],
      noise_input=noise_image,
  )
  # sr_image = sr_image_and_disp[:, :3]
  sr_disp = sr_image_and_disp[:, 3:]

  outputs = {
      'image': sr_image_and_disp,
      'disp': sr_disp,
      'image_raw': rgb_and_disp_image,  # rgb_image,
      'depth_raw': depth_image,
      'disp_raw': disp_image,
      'weights_raw': weights_image,
      'noise_raw': noise_image,
  }
  return outputs


def generate_frame(
    G,
    camera,
    planes,
    ws,
    intrinsics_matrix,
    noise_input,
    sky_texture=None,
    fov=60,
    to_cpu=False,
):  # pylint: disable=redefined-outer-name
  """generate image outputs from plane features and camera info."""
  world2cam_matrix = camera_util.pose_from_camera(camera)[None].cuda()
  cam2world_matrix = camera_util.pose_from_camera(camera)[None].inverse().cuda()
  outs = render_planes(
      planes, ws, cam2world_matrix, intrinsics_matrix, noise_input
  )
  img = outs['image'][:, :3]
  raw = outs['image_raw']
  sky_mask = interpolate(outs['weights_raw'], img.shape[-1])
  depth = raw[:, 3:]  # disparity
  horizon = camera_util.land_fraction(sky_mask)
  near = camera_util.near_fraction(depth, near_depth=0.12, near_spread=0.05)

  mask = interpolate(outs['weights_raw'], img.shape[-1])
  mask = torch.sigmoid((mask - 0.9) * 100)  # this sharpens the mask
  img_w_gray_sky = img * mask

  outputs = {
      'image': img,
      'mask': mask,
      'depth': depth,
      'image_w_gray_sky': img_w_gray_sky,
      'world2cam_matrix': world2cam_matrix,
      'cam2world_matrix': cam2world_matrix,
      'thumb': raw[:, :3],
      'noise': outs['noise_raw'],
  }

  if sky_texture is not None:
    sky_res = img.shape[-1]
    _, ray_directions = G.ray_sampler(
        cam2world_matrix, intrinsics_matrix, sky_res
    )
    sky_img = sky_util.sample_sky_from_viewdirs(
        sky_texture, ray_directions, img.shape[-1], fov
    )
    composite = sky_util.composite_sky(img, mask, sky_img)
    outputs['composite'] = composite

  if to_cpu:  # useful for videos
    outputs = {k: v.cpu() for k, v in outputs.items()}
  return outputs, horizon, near
