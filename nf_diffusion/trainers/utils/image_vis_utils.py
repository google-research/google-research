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

"""Image visualization utilities."""

import mediapy as media
import numpy as np


def create_images(v, max_num_images=64):
  """Create square images from a tensor."""
  v = np.array(v)
  assert len(v.shape) >= 3
  if len(v.shape) == 3:
    v = v[None, Ellipsis]
  v = v.reshape(-1, *v.shape[-3:])
  if v.shape[0] > max_num_images:
    v = v[:max_num_images]
  h, w, c = v.shape[1:]
  bs = v.shape[0]
  nr = nc = int(np.floor(np.sqrt(bs)))
  img = v[: nr * nc]
  img = img.reshape(nr, nc, h, w, c).swapaxes(1, 2).reshape(nr * h, nc * w, c)
  return np.array(img)


def save_images(img, path):
  """Save an image tensor to disc.

  Assume:
  1. img is either (N, H, W, C) or (H, W, C)
  2. img range from [0, 1].
  3. saving directory already exists.

  Args:
    img: Image tensor of shape (N, H, W, C) or (H, W, C); range from [0, 1].
    path: path to save the image. The directory of the path should exist.
  """
  assert len(img.shape) < 5
  if len(img.shape) == 4:
    img = create_images(img)
  if img.max() > 1 or img.min() < 0:
    img = (img - img.min()) / (img.max() - img.min())
  out_img_arr = np.clip(np.array(img) * 255, a_min=0, a_max=255).astype(
      np.uint8
  )
  if out_img_arr.shape[-1] == 1:
    out_img_arr = out_img_arr[Ellipsis, 0]
  media.write_image(path, out_img_arr)
