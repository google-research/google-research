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

"""Interpolation functions."""

import jax.numpy as jnp

Array = jnp.ndarray


def interpolate_2d_nearest(feat_grid, xy):
  """Nearest interpolation of 2D feature grid.

  Out of range coordinate will be clamp to within the range [-0.5, 0.5]^2.

  Args:
    feat_grid: Shape (H, W, D)
    xy: Shape (2). The coordinate of the output feature. The range of the
      coordinate should be [-0.5, 0.5]^2, where [-0.5, -0.5] is the upper-left
      corner; [0.5, 0.5] is the lower-right corner.

  Returns:
    Feature vector with shape (,D).
  """
  xy = jnp.clip(xy, -0.5, 0.5)
  # 1. Find nearest neighbor for each [xy] in the grid with size defined
  #    by the shape of [feat_grid].
  h, w = feat_grid.shape[-3], feat_grid.shape[-2]
  ix, iy = (xy[0] + 0.5) * (h - 1), (xy[1] + 0.5) * (w - 1)
  ix, iy = ix.astype(jnp.int64), iy.astype(jnp.int64)

  # 2. Use the nearest neighbor index from step-1 to retrieve [feat_grid]
  return feat_grid[ix, iy, :]


def interpolate_2d_bilinear(feat_grid, xy):
  """Bilinear interpolation of 2D feature grid.

  1. Out of range coordinate will be clamp to within the range [-0.5, 0.5]^2.
  2. Assume [feat_grid] has shape (h, w, d), then
    2.1 If h > 1 and w > 1 (i.e. there exists a cell)
    2.2 If h = 1, then 1D linearly interpolate (w, d) feature.
    2.3 If w = 1, then 1D linearly interpolate (h, d) feature.

  Args:
    feat_grid: Shape (H, W, D)
    xy: Shape (..., 2). The coordinate of the output feature. The range of the
      coordinate should be [-0.5, 0.5]^2, where [-0.5, -0.5] is the upper-left
      corner; [0.5, 0.5] is the lower-right corner.

  Returns:
    Feature vector with shape (,D).
  """
  h, w = feat_grid.shape[-3], feat_grid.shape[-2]
  if h == 1 and w == 1:
    return feat_grid[0, 0]

  hw = jnp.array([h, w])
  ixy = (jnp.clip(xy, -0.5, 0.5) + 0.5) * (
      hw - 1
  )  # Range [0, h-1] and [0, w-1]
  if h == 1:
    return jnp.interp(ixy[Ellipsis, 1], jnp.arange(w), feat_grid[0])

  if w == 1:
    return jnp.interp(ixy[Ellipsis, 0], jnp.arange(h), feat_grid[1])

  # For case h > 1 and w > 1, we will first find features at the four corners.
  x_upper = jnp.ceil(jnp.clip(ixy, min=1)).astype(jnp.int64)
  x_lower = x_upper - 1

  # Now interpolate first the x-axis, then the y-axis
  print(
      x_lower[Ellipsis, 0].shape,
      feat_grid[x_lower[Ellipsis, 0]].shape,
      jnp.arange(w).shape,
  )
  feat1 = jnp.interp(ixy[Ellipsis, 1], jnp.arange(w), feat_grid[x_lower[Ellipsis, 0]])
  feat2 = jnp.interp(ixy[Ellipsis, 1], jnp.arange(w), feat_grid[x_upper[Ellipsis, 0]])
  feat = jnp.interp(
      ixy[Ellipsis, 0] - jnp.ceil(ixy[Ellipsis, 0]),
      jnp.arange(2),
      jnp.array([feat1, feat2]),
  )
  return feat


def interpolate_2d(xy_out, feat_grid, interp_type="nearest"):
  if interp_type == "nearest":
    return interpolate_2d_nearest(feat_grid, xy_out)

  if interp_type == "bilinear":
    return interpolate_2d_bilinear(feat_grid, xy_out)

  print(f"Interpolation type {interp_type} not supported.")
  raise NotImplementedError


def interpolate_3d(xyz_out, feat_grid):
  raise NotImplementedError
