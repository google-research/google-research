# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

# python3
"""Visualization utility functions."""

# pylint: disable=g-bad-import-order, unused-import, g-multiple-import
# pylint: disable=line-too-long, missing-docstring, g-importing-member
from collections import OrderedDict
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tensorflow.compat.v1 import gfile


def add_padding(images, pad):
  n, h, w, ch = images.shape
  new_images = np.ones((n, h + 2 * pad, w + 2 * pad, ch)) * 0.5
  for i in range(len(images)):
    new_images[i, pad:-pad, pad:-pad] = images[i]
  return new_images


def grid(images, num_cols, pad=1):
  if pad > 0:
    images = add_padding(images, pad)

  n, h, w, ch = images.shape
  c = num_cols
  r = n // num_cols
  images = images[:r * c].reshape(r, c, h, w, ch).transpose((0, 2, 1, 3, 4))
  images = images.reshape(r * h, c * w, ch)

  if ch == 3: return images
  if ch == 1: return images[Ellipsis, 0]


def generate(x1, x2, gen, z_dim, num_rows_per_set, num_cols):
  xs = []

  x = x1.numpy()[:num_rows_per_set * num_cols]
  xs += [x]
  xs += [x[:num_cols] * 0 + 0.5]  # Black border

  x = x2.numpy()[:num_rows_per_set * num_cols]
  xs += [x]
  xs += [x[:num_cols] * 0 + 0.5]  # Black border

  for dim in range(z_dim):
    for _ in range(num_rows_per_set):
      z = np.tile(np.random.randn(1, z_dim), (num_cols, 1)).astype(np.float32)
      z[:, dim] = norm.ppf(np.linspace(0.01, 0.99, num_cols))
      x = gen(z).numpy()
      xs += [x]
    xs += [x * 0 + 0.5]
  del xs[-1]

  return np.concatenate(xs, 0)


def ablation_visualization(
    x1, x2, gen, z_dim, basedir, global_step, figsize=(20, 20), show=False):
  images = generate(x1, x2, gen, z_dim, 3, 12)
  plt.figure(figsize=figsize)
  plt.imshow(grid(images, 12, 1), cmap='Greys_r', interpolation=None)
  plt.axis('off')
  if show:
    plt.show()

  filename = os.path.join(basedir, 'ablation_{:09d}.png'.format(global_step))
  with gfile.GFile(filename, mode='w') as f:
    plt.savefig(f, dpi=100, bbox_inches='tight')

  plt.close()
