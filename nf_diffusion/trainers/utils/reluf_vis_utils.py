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

"""Fast visualization utilities for ReluF."""

import einops
import jax
import jax.numpy as jnp

from nf_diffusion.models.utils import instant_ngp_utils


def render_volumetric_orthographic(
    density, rgb, scene_scale=0.5, white_bkgd=True
):
  """Compute volumetric rendering weights with orthograhpic projection.

  Args:
    density: Sigma output of the field with shape (..., H, W, D, 1).
    rgb: Color field with shape (..., H, W, D, 3).
    scene_scale: The scale of the scene voxel (half).
    white_bkgd: Whether render the background as white.

  Returns:
    Dictionary maps (axis_idx, axis_inver) to a rendered image.
  """
  out_lst = {}
  for i in range(3):
    for sign in [-1, 1]:
      res = density.shape[-(i + 2)]
      delta = 2 * scene_scale / float(res)
      density_delta = density * delta
      if i == 0:
        density_delta = einops.rearrange(
            density_delta, "... h w d 1 -> ... w d h"
        )
        curr_rgb = einops.rearrange(rgb, "... h w d c -> ... w d h c")
      elif i == 1:
        density_delta = einops.rearrange(
            density_delta, "... h w d 1 -> ... h d w"
        )
        curr_rgb = einops.rearrange(rgb, "... h w d c -> ... h d w c")
      else:
        density_delta = einops.rearrange(
            density_delta, "... h w d 1 -> ... h w d"
        )
        curr_rgb = einops.rearrange(rgb, "... h w d c -> ... h w d c")
      if sign < 0:
        density_delta = density_delta[Ellipsis, ::-1]
        curr_rgb = curr_rgb[Ellipsis, ::-1, :]

      density_delta_shifted = jnp.concatenate(
          [jnp.zeros_like(density_delta[Ellipsis, :1]), density_delta[Ellipsis, :-1]],
          axis=-1,
      )
      alpha = 1.0 - jnp.exp(-density_delta)
      trans = jnp.exp(-jnp.cumsum(density_delta_shifted, axis=-1))
      weights = alpha * trans
      acc = jnp.sum(weights, axis=-1)
      rendered_img = jnp.sum(weights[Ellipsis, None] * curr_rgb, axis=-2)

      # Composite onto the background color.
      if white_bkgd:
        rendered_img = rendered_img + (1.0 - acc[Ellipsis, None])
      out_lst[(i, sign)] = (weights, rendered_img)
  return out_lst


def visualize_orthographic_projection(
    vox,
    scene_scale=0.5,
    white_bkgd=True,
    preconditioner=10.0,
    offset=-5,
    color_act=True,
):
  """Visualize orthographic projection."""
  sig = instant_ngp_utils.safe_exp(vox[Ellipsis, :1] * preconditioner + offset)
  if color_act:
    rgb = jax.nn.sigmoid(vox[Ellipsis, 1:4] * preconditioner)
  else:
    rgb = vox[Ellipsis, 1:4] * preconditioner
  out_dict = render_volumetric_orthographic(sig, rgb, scene_scale, white_bkgd)
  res = vox.shape[-2]
  bs = vox.shape[:-4]
  grid = jnp.zeros(bs + (3 * res, 4 * res, 3))
  x_axis_idx = 1
  y_axis_idx = 2
  z_axis_idx = 0
  for (px, py), _, axis_idx, axis_inv in [
      ((0, 1), "up", y_axis_idx, -1),
      ((1, 0), "back", z_axis_idx, -1),
      ((1, 1), "left", x_axis_idx, 1),
      ((1, 2), "front", z_axis_idx, 1),
      ((1, 3), "right", x_axis_idx, -1),
      ((2, 1), "bottom", y_axis_idx, 1),
  ]:
    _, rendered_img = out_dict[(axis_idx, axis_inv)]
    grid = grid.at[
        Ellipsis, res * px : res * (px + 1), res * py : res * (py + 1), :
    ].set(jnp.array(rendered_img))
  return grid


def create_center_slices(vox):
  h, w, d = vox.shape[-4:-1]
  h_slice = vox[Ellipsis, h // 2, :, :, :]
  w_slice = vox[Ellipsis, :, w // 2, :, :]
  d_slice = vox[Ellipsis, :, :, d // 2, :]
  return {
      "h": h_slice,
      "w": w_slice,
      "d": d_slice,
  }
