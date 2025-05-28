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

"""Blur and alignment utilities."""
from external.gsn.models.nerf_utils import meshgrid_xy
import numpy as np
import torch
from utils.utils import interpolate


def gaussian2d(size, sigma=0.5):
  x = torch.arange(size) - size // 2
  y = torch.arange(size) - size // 2
  xx, yy = torch.meshgrid(x, y)
  weights = torch.exp((-(xx**2) - yy**2) / (2 * sigma**2))
  weights_norm = torch.sum(weights)
  return weights / weights_norm


def gaussian1d(size, sigma=0.5):
  x = torch.arange(size) - size // 2
  weights = torch.exp((-(x**2)) / (2 * sigma**2))
  weights_norm = torch.sum(weights)
  return weights / weights_norm


def applyfilter2d(tensor, weights):
  # weights [K, K]
  # tensor = [B, C, H, W]
  # apply same filter channel-wise
  weights = weights.to(tensor.device)
  _, c, _, _ = tensor.shape
  k = weights.shape[0]
  pad = k // 2
  tensor = torch.nn.functional.pad(tensor, (pad, pad, pad, pad), mode='reflect')
  tensor = torch.nn.functional.conv2d(tensor, weights[None][None], groups=c)
  return tensor


# pylint: disable=invalid-name
def align_frames(rgb_src, Rt_src, Rt_tgt, depth_tgt, fov, far_bound):
  """Backwards warp rgb_src to rgb_tgt using the depth of tgt image.

  Args:
    rgb_src: rgb image
    Rt_src: src camera
    Rt_tgt: tgt camera
    depth_tgt: tgt depth
    fov: camera fov
    far_bound: maximum depth to clip

  Returns:
    warped src image and mask of valid region
  """

  _, _, h, w = rgb_src.shape
  depth_tgt = interpolate(depth_tgt, (h, w)).clamp(1 / far_bound, 1)
  depth_tgt = 1 / depth_tgt  # converts disparity to depth

  fx = (w / 2) / np.tan(np.deg2rad(fov) / 2)
  fy = -(h / 2) / np.tan(np.deg2rad(fov) / 2)  # positive y is top of image

  x = torch.arange(w)
  y = torch.arange(h)
  ii, jj = meshgrid_xy(x, y)

  # opengl convention
  # +x = right of image
  # +y = top of image
  # +z = behind image
  xy = torch.stack(
      [(ii - w * 0.5) / fx, (jj - h * 0.5) / fy, -torch.ones_like(ii)], dim=-1
  )[None]

  # adjust pixel centers for grid_sample
  xy[Ellipsis, 0] += 0.5 * 1 / fx  # 1/w
  xy[Ellipsis, 1] += 0.5 * 1 / fy  # 1/h

  # convert nerf image coordinates to [-1, 1] grid_sample coordinates
  # (keep them positive)
  scale_factor_x = fx / (w * 0.5)
  scale_factor_y = abs(fy / (h * 0.5))

  xy = xy.to(rgb_src.device)

  # warp tgt points to src coordinates (for backwards warp)
  xyd = xy * depth_tgt.permute(0, 2, 3, 1)
  xyd = xyd[Ellipsis, None]  # BHW31
  Rt_tgt2src = torch.matmul(Rt_src, Rt_tgt.inverse())
  Rt_tgt2src = Rt_tgt2src[:, None]  # B1144
  xyd_src = (
      torch.matmul(Rt_tgt2src[Ellipsis, :3, :3], xyd) + Rt_tgt2src[Ellipsis, :3, -1:]
  )  # BHW31

  # convert to opencv convention for grid sample
  opengl_to_opencv = torch.eye(3)
  opengl_to_opencv[1, 1] = -1
  opengl_to_opencv[2, 2] = -1
  opengl_to_opencv = opengl_to_opencv[None][None][None]  # 11133
  opengl_to_opencv = opengl_to_opencv.to(xyd_src.device)
  xyd_src = torch.matmul(opengl_to_opencv, xyd_src)
  xy_src = xyd_src[Ellipsis, :2, 0] / xyd_src[Ellipsis, 2:, 0]

  # scale it before grid sample
  xy_src[Ellipsis, 0] *= scale_factor_x
  xy_src[Ellipsis, 1] *= scale_factor_y

  # do backwards warp on rgb_src
  src_warped = torch.nn.functional.grid_sample(
      rgb_src, xy_src, mode='bilinear', padding_mode='reflection'
  )
  src_mask = torch.nn.functional.grid_sample(
      torch.ones_like(rgb_src), xy_src, mode='bilinear'
  )
  return src_warped, src_mask
# pylint: enable=invalid-name


def masked_temporal_average(aligned_frames, aligned_masks, weights):
  # TCHW,  T
  weights = weights.view(-1, 1, 1, 1).to(aligned_frames.device)
  masked_weights = weights * aligned_masks
  masked_weights = masked_weights / masked_weights.sum(dim=0, keepdim=True)
  output = (aligned_frames * masked_weights).sum(dim=0, keepdim=True)
  return output


def smooth_mask(outputs):
  """Fits a polynomial to sky mask boundary.

  Not used in paper; but looks visually a bit nicer.

  Args:
    outputs: dictionary of rendered results

  Returns:
    outputs: the same dictionary, with smooth mask added
  """
  mask_input = outputs['acc_sharpen']
  size = mask_input.shape[-1]
  assert size == 256
  diff_img = torch.diff(mask_input, dim=2)
  amax = torch.argmax(diff_img, dim=2)
  amax_np = (size - 1) - amax.cpu().numpy().squeeze()  # top down
  x_np = np.arange(len(amax_np))
  coefs = np.polyfit(x_np, amax_np, 7)
  fitted = np.polyval(coefs, x_np)
  ramp_weights = np.ones_like(amax_np)
  # ramp_weights[:11] = np.linspace(0, 1, 11)
  # ramp_weights[-11:] = np.linspace(1, 0, 11)
  weighted = ramp_weights * fitted + (1 - ramp_weights) * amax_np
  index = (size - 1) - np.round(weighted).astype(np.uint8)
  tmp = np.zeros((mask_input.shape[-2], mask_input.shape[-1]))
  out = np.put_along_axis(tmp, index[None], np.ones_like(index), 0)
  out = np.cumsum(tmp, axis=0)
  mask_smooth = torch.from_numpy(out)[None][None].to(mask_input.device).float()
  mask_smooth = applyfilter2d(mask_smooth, gaussian2d(9, 2))
  outputs['rgb_overlay_upsample'] = (
      outputs['rgb_up'] * mask_smooth + (1 - mask_smooth) * outputs['sky_out']
  )
  return outputs
