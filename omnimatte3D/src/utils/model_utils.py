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

"""Helper functions/classes for model definition."""

from einops import reduce
import jax.numpy as jnp
import mediapy


def gradient(disp):
  """Computes the gradient."""
  grad_x = jnp.abs(disp[Ellipsis, :-1, :] - disp[Ellipsis, 1:, :])
  grad_y = jnp.abs(disp[Ellipsis, :-1, :, :] - disp[Ellipsis, 1:, :, :])

  return grad_x, grad_y


def normalize_depth(depth):
  assert depth.shape[-1] == 1
  axes = (-3, -2)
  min_d, max_d = (
      jnp.min(depth, axis=axes, keepdims=True),
      jnp.max(depth, axis=axes, keepdims=True),
  )
  return (depth - min_d) / (max_d - min_d)


def normalize(x):
  max_val = reduce(x, '... c -> ... 1', 'max')
  min_val = reduce(x, '... c -> ... 1', 'min')
  out = (x - min_val) / (max_val - min_val)
  return out


def scale_disp(disp, min_depth, max_depth):
  """Convert network's sigmoid output to scene range."""
  min_disp = 1 / max_depth
  max_disp = 1 / min_depth
  scaled_disp = min_disp + (max_disp - min_disp) * disp
  return scaled_disp


def compute_psnr(mse):
  """Compute psnr value given mse (we assume the maximum pixel value is 1).

  Args:
    mse: float, mean square error of pixels.

  Returns:
    psnr: float, the psnr value.
  """
  return -10.0 * jnp.log(mse) / jnp.log(10.0)


def save_img(img, pth):
  """Save an image to disk.

  Args:
    img: jnp.ndarry, [height, width, channels].
    pth: string, path to save the image to.
  """
  mediapy.write_image(pth, img)
