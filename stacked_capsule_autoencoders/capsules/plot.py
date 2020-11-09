# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Constellation plotting tools."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import sonnet as snt
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp


_COLORS = """
    #a6cee3
    #1f78b4
    #b2df8a
    #33a02c
    #fb9a99
    #e31a1c
    #fdbf6f
    #ff7f00
    #cab2d6
    #6a3d9a
    #ffff99
    #b15928""".split()


def hex_to_rgb(value):
  value = value.lstrip('#')
  lv = len(value)
  return tuple(int(value[i:i + lv // 3], 16) / 255.
               for i in range(0, lv, lv // 3))


_COLORS = [c.strip() for c in _COLORS]
_COLORS = _COLORS[1::2] + _COLORS[::2]
_COLORS = np.asarray([hex_to_rgb(c) for c in _COLORS], dtype=np.float32)


def gaussian_blobs(params, height, width, norm='sum'):
  """Creates gaussian blobs on a canvas.

  Args:

    params: [B, 4] tensor, where entries represent (y, x, y_scale, x_scale).
    height: int, height of the output.
    width: int, width of the output.
    norm: type of normalization to use; must be a postfix of some
      tf.reduce_.* method.

  Returns:
    Tensor of shape [B, height, width].
  """

  params = tf.expand_dims(params, -1)
  uy, ux, sy, sx = tf.split(params, 4, -2)

  rows = tf.range(tf.to_int32(height))
  rows = tf.to_float(rows)[tf.newaxis, :, tf.newaxis]

  cols = tf.range(tf.to_int32(width))
  cols = tf.to_float(cols)[tf.newaxis, tf.newaxis, :]

  dy = (rows - uy) / sy
  dx = (cols - ux) / sx

  z = tf.square(dy) + tf.square(dx)
  mask = tf.exp(-.5 * z)

  # normalize so that the contribution of each blob sums to one
  # change this to `tf.reduce_max` if you want max value to be one
  norm_func = getattr(tf, 'reduce_{}'.format(norm))
  mask /= norm_func(mask, (1, 2), keep_dims=True) + 1e-8  # pylint:disable=not-callable
  return mask


def gaussian_blobs_const_scale(params, scale, height, width, norm='sum'):
  scale = tf.zeros_like(params[Ellipsis, :2]) + scale
  params = tf.concat([params[Ellipsis, :2], scale], -1)
  return gaussian_blobs(params, height, width, norm)


def denormalize_coords(coords, canvas_size, rounded=False):
  coords = (coords + 1.) / 2. * np.asarray(canvas_size)[np.newaxis]
  if rounded:
    coords = tf.round(coords)

  return coords


def render_by_scatter(size, points, colors=None, gt_presence=None):
  """Renders point by using tf.scatter_nd."""

  if colors is None:
    colors = tf.ones(points.shape[:-1].as_list() + [3], dtype=tf.float32)

  if gt_presence is not None:
    colors *= tf.cast(tf.expand_dims(gt_presence, -1), colors.dtype)

  batch_size, n_points = points.shape[:-1].as_list()
  shape = [batch_size] + list(size) + [3]
  batch_idx = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  batch_idx = snt.TileByDim([1], [n_points])(batch_idx)
  idx = tf.concat([batch_idx, tf.cast(points, tf.int32)], -1)

  return tf.scatter_nd(idx, colors, shape)


def render_constellations(pred_points,
                          capsule_num,
                          canvas_size,
                          gt_points=None,
                          n_caps=2,
                          gt_presence=None,
                          pred_presence=None,
                          caps_presence_prob=None):
  """Renderes predicted and ground-truth points as gaussian blobs.

  Args:
    pred_points: [B, m, 2].
    capsule_num: [B, m] tensor indicating which capsule the corresponding point
      comes from. Plots from different capsules are plotted with different
      colors. Currently supported values: {0, 1, ..., 11}.
    canvas_size: tuple of ints
    gt_points: [B, k, 2]; plots ground-truth points if present.
    n_caps: integer, number of capsules.
    gt_presence: [B, k] binary tensor.
    pred_presence: [B, m] binary tensor.
    caps_presence_prob: [B, m], a tensor of presence probabilities for caps.

  Returns:
    [B, *canvas_size] tensor with plotted points
  """

  # convert coords to be in [0, side_length]
  pred_points = denormalize_coords(pred_points, canvas_size, rounded=True)

  # render predicted points
  batch_size, n_points = pred_points.shape[:2].as_list()
  capsule_num = tf.to_float(tf.one_hot(capsule_num, depth=n_caps))
  capsule_num = tf.reshape(capsule_num, [batch_size, n_points, 1, 1, n_caps, 1])

  color = tf.convert_to_tensor(_COLORS[:n_caps])
  color = tf.reshape(color, [1, 1, 1, 1, n_caps, 3]) * capsule_num
  color = tf.reduce_sum(color, -2)
  color = tf.squeeze(tf.squeeze(color, 3), 2)

  colored = render_by_scatter(canvas_size, pred_points, color, pred_presence)

  # Prepare a vertical separator between predicted and gt points.
  # Separator is composed of all supported colors and also serves as
  # a legend.
  # [b, h, w, 3]
  n_colors = _COLORS.shape[0]
  sep = tf.reshape(tf.convert_to_tensor(_COLORS), [1, 1, n_colors, 3])
  n_tiles = int(colored.shape[2]) // n_colors
  sep = snt.TileByDim([0, 1, 3], [batch_size, 3, n_tiles])(sep)
  sep = tf.reshape(sep, [batch_size, 3, n_tiles * n_colors, 3])

  pad = int(colored.shape[2]) - n_colors * n_tiles
  pad, r = pad // 2, pad % 2

  if caps_presence_prob is not None:
    n_caps = int(caps_presence_prob.shape[1])
    prob_pads = ([0, 0], [0, n_colors - n_caps])
    caps_presence_prob = tf.pad(caps_presence_prob, prob_pads)
    zeros = tf.zeros([batch_size, 3, n_colors, n_tiles, 3], dtype=tf.float32)

    shape = [batch_size, 1, n_colors, 1, 1]
    caps_presence_prob = tf.reshape(caps_presence_prob, shape)

    prob_vals = snt.MergeDims(2, 2)(caps_presence_prob + zeros)
    sep = tf.concat([sep, tf.ones_like(sep[:, :1]), prob_vals], 1)

  sep = tf.pad(sep, [(0, 0), (1, 1), (pad, pad + r), (0, 0)],
               constant_values=1.)

  # render gt points
  if gt_points is not None:
    gt_points = denormalize_coords(gt_points, canvas_size, rounded=True)

    gt_rendered = render_by_scatter(canvas_size, gt_points, colors=None,
                                    gt_presence=gt_presence)

    colored = tf.where(tf.cast(colored, bool), colored, gt_rendered)
    colored = tf.concat([gt_rendered, sep, colored], 1)

  res = tf.clip_by_value(colored, 0., 1.)
  return res


def concat_images(img_list, sep_width, vertical=True):
  """Concatenates image tensors."""

  if vertical:
    sep = tf.ones_like(img_list[0][:, :sep_width])
  else:
    sep = tf.ones_like(img_list[0][:, :, :sep_width])

  imgs = []
  for i in img_list:
    imgs.append(i)
    imgs.append(sep)

  imgs = imgs[:-1]

  return tf.concat(imgs, 2 - vertical)


def apply_cmap(brightness, cmap):
  indices = tf.cast(brightness * 255.0, tf.int32)
  # Make sure the indices are in the right range. Comes in handy for NaN values.
  indices = tf.clip_by_value(indices, 0, 256)

  cm = matplotlib.cm.get_cmap(cmap)
  colors = tf.constant(cm.colors, dtype=tf.float32)
  return tf.gather(colors, indices)


def render_activations(activations, height, pixels_per_caps=2, cmap='gray'):
  """Renders capsule activations as a colored grid.

  Args:
    activations: [B, n_caps] tensor, where every entry is in [0, 1].
    height: int, height of the resulting grid.
    pixels_per_caps: int, size of a single grid cell.
    cmap: string: matplotlib-compatible cmap name.

  Returns:
    [B, height, width, n_channels] tensor.
  """

  # convert activations to colors
  if cmap == 'gray':
    activations = tf.expand_dims(activations, -1)

  else:
    activations = apply_cmap(activations, cmap)

  batch_size, n_caps, n_channels = activations.shape.as_list()

  # pad to fit a grid of prescribed hight
  n_rows = 1 + (height - pixels_per_caps) // (pixels_per_caps + 1)
  n_cols = n_caps // n_rows + ((n_caps % n_rows) > 0)
  n_pads = n_rows * n_cols - n_caps

  activations = tf.pad(activations, [(0, 0), (0, n_pads), (0, 0)],
                       constant_values=1.)

  # tile to get appropriate number of pixels to fil a pixel_per_caps^2 square
  activations = snt.TileByDim([2], [pixels_per_caps**2])(
      tf.expand_dims(activations, 2))

  activations = tf.reshape(activations, [batch_size, n_rows, n_cols,
                                         pixels_per_caps, pixels_per_caps,
                                         n_channels])

  # pad each cell with one white pixel on the bottom and on the right-hand side
  activations = tf.pad(activations, [(0, 0), (0, 0), (0, 0), (0, 1), (0, 1),
                                     (0, 0)], constant_values=1.)

  # concat along row and col dimensions
  activations = tf.concat(tf.unstack(activations, axis=1), axis=-3)
  activations = tf.concat(tf.unstack(activations, axis=1), axis=-2)

  # either pad or truncated to get the correct height
  if activations.shape[1] < height:
    n_pads = height - activations.shape[1]
    activations = tf.pad(activations, [(0, 0), (0, n_pads), (0, 0), (0, 0)])

  else:
    activations = activations[:, :height]

  return activations


def correlation(x, y):
  """Computes correlation between x and y.

  Args:
    x: [B, m],
    y: [B, n]

  Returns:
    corr_xy [m, n]
  """

  # [B, m+n]
  m = int(x.shape[-1])
  xy = tf.concat([x, y], -1)

  # [m+n, m+n]
  corr = tfp.stats.correlation(xy, sample_axis=0)
  corr_xy = corr[:m, m:]
  return corr_xy


def make_tsne_plot(caps_presence, labels, filename=None, save_kwargs=None):
  """Makes a TSNE plot."""

  # idx = np.random.choice(res.test.posterior_pres.shape[0], size=int(1e4),
  #                        replace=False)
  # points = res.train.posterior_pres[idx]
  # labels = res.train.label[idx]

  tsne = TSNE(2, perplexity=50)
  embedded = tsne.fit_transform(caps_presence)

  colors = np.asarray([
      166, 206, 227,
      31, 120, 180,
      178, 223, 138,
      51, 160, 44,
      251, 154, 153,
      227, 26, 28,
      253, 191, 111,
      255, 127, 0,
      202, 178, 214,
      106, 61, 154
  ], dtype=np.float32).reshape(10, 3) / 255.

  fig, ax = plt.subplots(1, 1, figsize=(6, 6))
  for i in range(10):
    idx = (labels == i)
    points_for_label = embedded[idx]
    ax.scatter(points_for_label[:, 0], points_for_label[:, 1], c=colors[i])

  if filename is not None:
    if save_kwargs is None:
      save_kwargs = dict(bbox_inches='tight', dpi=300)

    fig.savefig(filename, **save_kwargs)
    plt.close(fig)
