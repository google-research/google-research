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

"""Functions to perform a spatial transformation for Tensor.

This is a reimplementation of:
https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/spatial_transform_ops.py
in Jax.
"""

from typing import Mapping, Tuple

import flax.linen as nn
from jax import lax
from jax import nn as jnn
import jax.numpy as jnp

_EPSILON = 1e-8
Array = jnp.ndarray


def nearest_upsampling(data, scale):
  """Nearest neighbor upsampling implementation.

  Args:
    data: An array with a shape of [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.

  Returns:
    data_up: An array with a shape of
      [batch, height_in*scale, width_in*scale, channels]. Same dtype as input
      data.
  """
  batch_size, height, width, channels = data.shape

  # Instead of broadcasting with a 6-d tensor, we're using stacking here.
  output = jnp.stack([data] * scale, axis=3)
  output = jnp.stack([output] * scale, axis=2)
  return jnp.reshape(output,
                     [batch_size, height * scale, width * scale, channels])


def feature_bilinear_interpolation(features, kernel_y,
                                   kernel_x):
  """Feature bilinear interpolation.

  The RoIAlign feature f can be computed by bilinear interpolation
  of four neighboring feature points f0, f1, f2, and f3.

  f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                        [f10, f11]]
  f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
  f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
  kernel_y = [hy, ly]
  kernel_x = [hx, lx]

  Args:
    features: The features are in shape of [batch_size, num_boxes, output_size *
      2, output_size * 2, num_filters].
    kernel_y: an array of size [batch_size, boxes, output_size, 2, 1].
    kernel_x: an array of size [batch_size, boxes, output_size, 2, 1].

  Returns:
    A 5-D array representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].

  """
  batch_size, num_boxes, output_size, _, num_filters = (features.shape)
  if batch_size is None:
    batch_size = jnp.shape(features)[0]
  output_size = output_size // 2
  kernel_y = jnp.reshape(kernel_y, [batch_size, num_boxes, output_size * 2, 1])
  kernel_x = jnp.reshape(kernel_x, [batch_size, num_boxes, 1, output_size * 2])
  # Use implicit broadcast to generate the interpolation kernel. The
  # multiplier `4` is for avg pooling.
  interpolation_kernel = kernel_y * kernel_x * 4

  # Interpolate the gathered features with computed interpolation kernels.
  features *= jnp.expand_dims(
      interpolation_kernel, axis=-1).astype(features.dtype)
  features = jnp.reshape(
      features,
      [batch_size * num_boxes, output_size * 2, output_size * 2, num_filters])
  features = nn.avg_pool(features, (2, 2), (2, 2), 'VALID')
  features = jnp.reshape(
      features, [batch_size, num_boxes, output_size, output_size, num_filters])
  return features


def compute_grid_positions(
    boxes, boundaries, output_size,
    sample_offset):
  """Compute the grid position w.r.t.

  the corresponding feature map.

  Args:
    boxes: a 3-D array of shape [batch_size, num_boxes, 4] encoding the
      information of each box w.r.t. the corresponding feature map.
      boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
      corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
        in terms of the number of pixels of the corresponding feature map size.
    boundaries: a 3-D array of shape [batch_size, num_boxes, 2] representing
      the boundary (in (y, x)) of the corresponding feature map for each box.
      Any resampled grid points that go beyond the bounary will be clipped.
    output_size: a scalar indicating the output crop size.
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.

  Returns:
    kernel_y: an array of size [batch_size, boxes, output_size, 2, 1].
    kernel_x: an array of size [batch_size, boxes, output_size, 2, 1].
    box_grid_y0y1: an array of size [batch_size, boxes, output_size, 2]
    box_grid_x0x1: an array of size [batch_size, boxes, output_size, 2]
  """
  batch_size, num_boxes, _ = boxes.shape
  if batch_size is None:
    batch_size = jnp.shape(boxes)[0]
  box_grid_x = []
  box_grid_y = []
  for i in range(output_size):
    box_grid_x.append(boxes[:, :, 1] +
                      (i + sample_offset) * boxes[:, :, 3] / output_size)
    box_grid_y.append(boxes[:, :, 0] +
                      (i + sample_offset) * boxes[:, :, 2] / output_size)
  box_grid_x = jnp.stack(box_grid_x, axis=2)
  box_grid_y = jnp.stack(box_grid_y, axis=2)

  box_grid_y0 = jnp.floor(box_grid_y)
  box_grid_x0 = jnp.floor(box_grid_x)
  box_grid_x0 = jnp.maximum(0., box_grid_x0)
  box_grid_y0 = jnp.maximum(0., box_grid_y0)

  box_grid_x0 = jnp.minimum(box_grid_x0,
                            jnp.expand_dims(boundaries[:, :, 1], -1))
  box_grid_x1 = jnp.minimum(box_grid_x0 + 1,
                            jnp.expand_dims(boundaries[:, :, 1], -1))
  box_grid_y0 = jnp.minimum(box_grid_y0,
                            jnp.expand_dims(boundaries[:, :, 0], -1))
  box_grid_y1 = jnp.minimum(box_grid_y0 + 1,
                            jnp.expand_dims(boundaries[:, :, 0], -1))

  box_gridx0x1 = jnp.stack([box_grid_x0, box_grid_x1], axis=-1)
  box_gridy0y1 = jnp.stack([box_grid_y0, box_grid_y1], axis=-1)

  # The RoIAlign feature f can be computed by bilinear interpolation of four
  # neighboring feature points f0, f1, f2, and f3.
  # f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
  #                       [f10, f11]]
  # f(y, x) = (hy*hx)f00 + (hy*lx)f01 + (ly*hx)f10 + (lx*ly)f11
  # f(y, x) = w00*f00 + w01*f01 + w10*f10 + w11*f11
  ly = box_grid_y - box_grid_y0
  lx = box_grid_x - box_grid_x0
  hy = 1.0 - ly
  hx = 1.0 - lx
  kernel_y = jnp.reshape(
      jnp.stack([hy, ly], axis=3), [batch_size, num_boxes, output_size, 2, 1])
  kernel_x = jnp.reshape(
      jnp.stack([hx, lx], axis=3), [batch_size, num_boxes, output_size, 2, 1])

  return kernel_y, kernel_x, box_gridy0y1, box_gridx0x1


def get_grid_one_hot(box_gridy0y1, box_gridx0x1,
                     feature_height,
                     feature_width):
  """Get grid_one_hot from indices and feature_size."""
  (batch_size, num_boxes, output_size, _) = box_gridx0x1.shape
  if batch_size is None:
    batch_size = jnp.shape(box_gridx0x1)[0]

  y_indices = jnp.reshape(
      box_gridy0y1, [batch_size, num_boxes, output_size, 2]).astype(jnp.int32)
  x_indices = jnp.reshape(
      box_gridx0x1, [batch_size, num_boxes, output_size, 2]).astype(jnp.int32)

  # shape is [batch_size, num_boxes, output_size, 2, height]
  grid_y_one_hot = jnn.one_hot(y_indices, feature_height)
  # shape is [batch_size, num_boxes, output_size, 2, width]
  grid_x_one_hot = jnn.one_hot(x_indices, feature_width)

  return grid_y_one_hot, grid_x_one_hot


def multilevel_crop_and_resize(features,
                               boxes,
                               output_size = 7,
                               use_einsum_gather = False):
  """Crop and resize on multilevel feature pyramid.

  Generate the (output_size, output_size) set of pixels for each input box
  by first locating the box into the correct feature level, and then cropping
  and resizing it using the correspoding feature map of that level.

  Here is the step-by-step algorithm with use_einsum_gather=True:
  1. Compute sampling points and their four neighbors for each output points.
     Each box is mapped to [output_size, output_size] points.
     Each output point is averaged among #sampling_raitio^2 points.
     Each sampling point is computed using bilinear
     interpolation of its four neighboring points on the feature map.
  2. Gather output points separately for each level. Gather and computation of
     output points are done for the boxes mapped to this level only.
     2.1. Compute indices of four neighboring point of each sampling
          point for x and y separately of shape
          [batch_size, num_boxes, output_size, 2].
     2.2. Compute the interpolation kernel for axis x and y separately of
          shape [batch_size, num_boxes, output_size, 2, 1].
     2.3. The features are colleced into a
          [batch_size, num_boxes, output_size, output_size, num_filters]
          Tensor.
          Instead of a one-step algorithm, a two-step approach is used.
          That is, first, an intermediate output is stored with a shape of
          [batch_size, num_boxes, output_size, width, num_filters];
          second, the final output is produced with a shape of
          [batch_size, num_boxes, output_size, output_size, num_filters].

          Blinear interpolation is done during the two step gather:
          f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
                                [f10, f11]]
          [[f00, f01],
           [f10, f11]] = jnp.einsum(jnp.einsum(features, y_one_hot), x_one_hot)
          where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

          Note:
            a. Use one_hot with einsum to replace gather;
            b. Bilinear interpolation and averaging of
               multiple sampling points are fused into the one_hot vector.

  Args:
    features: A dictionary with key as pyramid level and value as features. The
      features are in shape of [batch_size, height_l, width_l, num_filters].
    boxes: A 3-D array of shape [batch_size, num_boxes, 4]. Each row represents
      a box with [y1, x1, y2, x2] in un-normalized coordinates.
    output_size: A scalar to indicate the output crop size.
    use_einsum_gather: use einsum to replace gather or not. Replacing einsum
      with gather can potentially improve performance.

  Returns:
    A 5-D array representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size, num_filters].
  """
  levels = list(features.keys())
  min_level = min(levels)
  max_level = max(levels)
  batch_size, max_feature_height, max_feature_width, num_filters = (
      features[min_level].shape)
  if batch_size is None:
    batch_size = jnp.shape(features[min_level])[0]
  _, num_boxes, _ = boxes.shape

  # Assigns boxes to the right level.
  box_width = boxes[:, :, 3] - boxes[:, :, 1]
  box_height = boxes[:, :, 2] - boxes[:, :, 0]
  areas_sqrt = jnp.sqrt(box_height * box_width)
  levels = jnp.floor(jnp.log(areas_sqrt / 224.0) / jnp.log(2.0)).astype(
      jnp.int32) + 4
  # Maps levels between [min_level, max_level].
  levels = jnp.minimum(max_level, jnp.maximum(levels, min_level))

  # Projects box location and sizes to corresponding feature levels.
  scale_to_level = jnp.float_power(2.0, levels.astype(jnp.float32)).astype(
      boxes.dtype)
  boxes /= jnp.expand_dims(scale_to_level, axis=2)
  box_width /= scale_to_level
  box_height /= scale_to_level
  boxes = jnp.concatenate([
      boxes[:, :, 0:2],
      jnp.expand_dims(box_height, -1),
      jnp.expand_dims(box_width, -1)
  ], axis=-1)

  if use_einsum_gather:
    def two_step_gather_per_level(features_level, mask):
      """Performs two-step gather using einsum for every level of features."""
      (_, feature_height, feature_width, _) = features_level.shape
      boundaries = jnp.tile(
          jnp.expand_dims(
              jnp.expand_dims(jnp.array([feature_height, feature_width]), 0),
              0), [batch_size, num_boxes, 1])
      boundaries = boundaries.astype(boxes.dtype)
      kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
          boxes, boundaries, output_size, sample_offset=0.5)

      # shape is:
      # [batch_size, num_boxes, output_size, 2, spatial_size]
      box_grid_y_one_hot, box_grid_x_one_hot = get_grid_one_hot(
          box_gridy0y1, box_gridx0x1, feature_height, feature_width)

      # # shape is [batch_size, num_boxes, output_size, spatial_size]
      box_grid_y_weight = jnp.sum(
          jnp.multiply(box_grid_y_one_hot, kernel_y), axis=-2)
      box_grid_x_weight = jnp.sum(
          jnp.multiply(box_grid_x_one_hot, kernel_x), axis=-2)

      # shape is [batch_size, num_boxes, output_size, width, feature]
      y_outputs = jnp.einsum('bhwf,bnyh->bnywf', features_level,
                             box_grid_y_weight.astype(features_level.dtype))

      # shape is [batch_size, num_boxes, output_size, output_size, feature]
      x_outputs = jnp.einsum('bnywf,bnxw->bnyxf', y_outputs,
                             box_grid_x_weight.astype(features_level.dtype))

      outputs = jnp.where(
          jnp.equal(mask, jnp.zeros_like(mask)), jnp.zeros_like(x_outputs),
          x_outputs)
      return outputs

    features_per_box = jnp.zeros(
        [batch_size, num_boxes, output_size, output_size, num_filters],
        dtype=features[min_level].dtype)
    for level in range(min_level, max_level + 1):
      level_equal = jnp.equal(levels, level)
      mask = jnp.tile(
          jnp.reshape(level_equal, [batch_size, num_boxes, 1, 1, 1]),
          [1, 1, output_size, output_size, num_filters])
      features_per_box += two_step_gather_per_level(features[level], mask)

    return features_per_box

  else:
    # Stack feature pyramid into a features_all of shape
    # [batch_size, levels, height, width, num_filters].
    features_all = []
    feature_heights = []
    feature_widths = []
    for level in range(min_level, max_level + 1):
      shape = features[level].shape
      feature_heights.append(shape[1])
      feature_widths.append(shape[2])
      # concatenate array of [batch_size, height_l * width_l, num_filters]
      # for each levels.
      features_all.append(
          jnp.reshape(features[level], [batch_size, -1, num_filters]))
    features_r2 = jnp.reshape(
        jnp.concatenate(features_all, 1), [-1, num_filters])

    # Calculate height_l * width_l for each level.
    level_dim_sizes = [
        feature_widths[i] * feature_heights[i]
        for i in range(len(feature_widths))
    ]
    # level_dim_offsets is accumulated sum of level_dim_size.
    level_dim_offsets = [0]
    for i in range(len(feature_widths) - 1):
      level_dim_offsets.append(level_dim_offsets[i] + level_dim_sizes[i])
    batch_dim_size = level_dim_offsets[-1] + level_dim_sizes[-1]
    level_dim_offsets = jnp.array(level_dim_offsets, jnp.int32)
    height_dim_sizes = jnp.array(feature_widths, jnp.int32)

    # Maps levels to [0, max_level-min_level].
    levels -= min_level
    level_strides = jnp.float_power(
        jnp.array([[2.0]]), levels.astype(jnp.float32))
    boundary = jnp.concatenate([
        jnp.expand_dims(
            jnp.array([[max_feature_height]], jnp.float32) / level_strides - 1,
            axis=-1),
        jnp.expand_dims(
            jnp.array([[max_feature_width]], jnp.float32) / level_strides - 1,
            axis=-1),
    ], axis=-1).astype(boxes.dtype)

    # Compute grid positions.
    kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
        boxes, boundary, output_size, sample_offset=0.5)

    x_indices = jnp.reshape(box_gridx0x1,
                            [batch_size, num_boxes, output_size * 2]).astype(
                                jnp.int32)
    y_indices = jnp.reshape(box_gridy0y1,
                            [batch_size, num_boxes, output_size * 2]).astype(
                                jnp.int32)

    batch_size_offset = jnp.tile(
        jnp.reshape(
            jnp.arange(batch_size) * batch_dim_size, [batch_size, 1, 1, 1]),
        [1, num_boxes, output_size * 2, output_size * 2])
    # Get level offset for each box. Each box belongs to one level.
    levels_offset = jnp.tile(
        jnp.reshape(level_dim_offsets[levels], [batch_size, num_boxes, 1, 1]),
        [1, 1, output_size * 2, output_size * 2])
    y_indices_offset = jnp.tile(
        jnp.reshape(y_indices * jnp.expand_dims(height_dim_sizes[levels], -1),
                    [batch_size, num_boxes, output_size * 2, 1]),
        [1, 1, 1, output_size * 2])
    x_indices_offset = jnp.tile(
        jnp.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
        [1, 1, output_size * 2, 1])
    indices = jnp.reshape(
        batch_size_offset + levels_offset + y_indices_offset + x_indices_offset,
        [-1])

    # performance.
    features_per_box = jnp.reshape(
        features_r2[indices],
        [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters])

    # Bilinear interpolation.
    features_per_box = feature_bilinear_interpolation(features_per_box,
                                                      kernel_y, kernel_x)
    return features_per_box


def selective_crop_and_resize(features,
                              boxes,
                              box_levels,
                              boundaries,
                              output_size = 7,
                              sample_offset = 0.5,
                              use_einsum_gather = False):
  """Crop and resize boxes on a set of feature maps.

  Given multiple features maps indexed by different levels, and a set of boxes
  where each box is mapped to a certain level, it selectively crops and resizes
  boxes from the corresponding feature maps to generate the box features.

  We follow the ROIAlign technique (see https://arxiv.org/pdf/1703.06870.pdf,
  figure 3 for reference). Specifically, for each feature map, we select an
  (output_size, output_size) set of pixels corresponding to the box location,
  and then use bilinear interpolation to select the feature value for each
  pixel.

  For performance, we perform the gather and interpolation on all layers as a
  single operation. In this op the multi-level features are first stacked and
  gathered into [2*output_size, 2*output_size] feature points. Then bilinear
  interpolation is performed on the gathered feature points to generate
  [output_size, output_size] RoIAlign feature map.

  Here is the step-by-step algorithm:
    1. The multi-level features are gathered into a
       [batch_size, num_boxes, output_size*2, output_size*2, num_filters]
       Tensor. The array contains four neighboring feature points for each
       vertex in the output grid.
    2. Compute the interpolation kernel of shape
       [batch_size, num_boxes, output_size*2, output_size*2]. The last 2 axis
       can be seen as stacking 2x2 interpolation kernels for all vertices in the
       output grid.
    3. Element-wise multiply the gathered features and interpolation kernel.
       Then apply 2x2 average pooling to reduce spatial dimension to
       output_size.

  Args:
    features: a 5-D array of shape [batch_size, num_levels, max_height,
      max_width, num_filters] where cropping and resizing are based.
    boxes: a 3-D array of shape [batch_size, num_boxes, 4] encoding the
      information of each box w.r.t. the corresponding feature map.
      boxes[:, :, 0:2] are the grid position in (y, x) (float) of the top-left
      corner of each box. boxes[:, :, 2:4] are the box sizes in (h, w) (float)
        in terms of the number of pixels of the corresponding feature map size.
    box_levels: a 3-D array of shape [batch_size, num_boxes, 1] representing
      the 0-based corresponding feature level index of each box.
    boundaries: a 3-D array of shape [batch_size, num_boxes, 2] representing
      the boundary (in (y, x)) of the corresponding feature map for each box.
      Any resampled grid points that go beyond the bounary will be clipped.
    output_size: a scalar indicating the output crop size.
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.
    use_einsum_gather: use einsum to replace gather or not. Replacing einsum
      with gather can improve performance when feature size is not large, einsum
      is friendly with model partition as well. Gather's performance is better
      when feature size is very large and there are multiple box levels.

  Returns:
    features_per_box: a 5-D array of shape
      [batch_size, num_boxes, output_size, output_size, num_filters]
      representing the cropped features.
  """
  (batch_size, num_levels, max_feature_height, max_feature_width,
   num_filters) = features.shape
  _, num_boxes, _ = boxes.shape
  kernel_y, kernel_x, box_gridy0y1, box_gridx0x1 = compute_grid_positions(
      boxes, boundaries, output_size, sample_offset)
  x_indices = jnp.reshape(
      box_gridx0x1, [batch_size, num_boxes, output_size * 2]).astype(jnp.int32)
  y_indices = jnp.reshape(
      box_gridy0y1, [batch_size, num_boxes, output_size * 2]).astype(jnp.int32)

  if use_einsum_gather:
    # Blinear interpolation is done during the last two gathers:
    #        f(y, x) = [hy, ly] * [[f00, f01], * [hx, lx]^T
    #                              [f10, f11]]
    #        [[f00, f01],
    #         [f10, f11]] = jnp.einsum(jnp.einsum(features, y_one_hot),
    #                                  x_one_hot)
    #       where [hy, ly] and [hx, lx] are the bilinear interpolation kernel.

    # shape is [batch_size, boxes, output_size, 2, 1]
    grid_y_one_hot, grid_x_one_hot = get_grid_one_hot(box_gridy0y1,
                                                      box_gridx0x1,
                                                      max_feature_height,
                                                      max_feature_width)

    # shape is [batch_size, num_boxes, output_size, height]
    grid_y_weight = jnp.sum(jnp.multiply(grid_y_one_hot, kernel_y), axis=-2)
    # shape is [batch_size, num_boxes, output_size, width]
    grid_x_weight = jnp.sum(jnp.multiply(grid_x_one_hot, kernel_x), axis=-2)

    # Gather for y_axis.
    # shape is [batch_size, num_boxes, output_size, width, features]
    features_per_box = jnp.einsum('bmhwf,bmoh->bmowf', features,
                                  grid_y_weight.astype(features.dtype))
    # Gather for x_axis.
    # shape is [batch_size, num_boxes, output_size, output_size, features]
    features_per_box = jnp.einsum('bmhwf,bmow->bmhof', features_per_box,
                                  grid_x_weight.astype(features.dtype))
  else:
    height_dim_offset = max_feature_width
    level_dim_offset = max_feature_height * height_dim_offset
    batch_dim_offset = num_levels * level_dim_offset

    batch_size_offset = jnp.tile(
        jnp.reshape(
            jnp.arange(batch_size) * batch_dim_offset, [batch_size, 1, 1, 1]),
        [1, num_boxes, output_size * 2, output_size * 2])
    box_levels_offset = jnp.tile(
        jnp.reshape(box_levels * level_dim_offset,
                    [batch_size, num_boxes, 1, 1]),
        [1, 1, output_size * 2, output_size * 2])
    y_indices_offset = jnp.tile(
        jnp.reshape(y_indices * height_dim_offset,
                    [batch_size, num_boxes, output_size * 2, 1]),
        [1, 1, 1, output_size * 2])
    x_indices_offset = jnp.tile(
        jnp.reshape(x_indices, [batch_size, num_boxes, 1, output_size * 2]),
        [1, 1, output_size * 2, 1])

    indices = jnp.reshape(
        batch_size_offset + box_levels_offset + y_indices_offset +
        x_indices_offset, [-1])

    features = jnp.reshape(features, [-1, num_filters])
    features_per_box = jnp.reshape(
        features[indices],
        [batch_size, num_boxes, output_size * 2, output_size * 2, num_filters])
    features_per_box = feature_bilinear_interpolation(features_per_box,
                                                      kernel_y, kernel_x)

  return features_per_box


def crop_mask_in_target_box(masks,
                            boxes,
                            target_boxes,
                            output_size,
                            sample_offset = 0.0,
                            use_einsum = True):
  """Crop masks in target boxes.

  Args:
    masks: An array with a shape of [batch_size, num_masks, height, width].
    boxes: a float array representing box cooridnates that tightly enclose
      masks with a shape of [batch_size, num_masks, 4] in un-normalized
      coordinates. A box is represented by [ymin, xmin, ymax, xmax].
    target_boxes: a float array representing target box cooridnates for masks
      with a shape of [batch_size, num_masks, 4] in un-normalized coordinates. A
      box is represented by [ymin, xmin, ymax, xmax].
    output_size: A scalar to indicate the output crop size. It currently only
      supports to output a square shape outputs.
    sample_offset: a float number in [0, 1] indicates the subpixel sample offset
      from grid point.
    use_einsum: Use einsum to replace gather in selective_crop_and_resize.

  Returns:
    A 4-D array representing feature crop of shape
    [batch_size, num_boxes, output_size, output_size].
  """
  batch_size, num_masks, height, width = masks.shape
  # Pad zeros on the boundary of masks.
  pad_value = jnp.array(0.0, dtype=masks.dtype)
  masks = lax.pad(masks, pad_value,
                  [(0, 0, 0), (0, 0, 0), (2, 2, 0), (2, 2, 0)])
  masks = jnp.reshape(masks, [batch_size, num_masks, height + 4, width + 4, 1])

  # Projects target box locations and sizes to corresponding cropped
  # mask coordinates.
  gt_y_min, gt_x_min, gt_y_max, gt_x_max = jnp.split(boxes, 4, axis=2)
  bb_y_min, bb_x_min, bb_y_max, bb_x_max = jnp.split(target_boxes, 4, axis=2)
  y_transform = (bb_y_min - gt_y_min) * height / (gt_y_max - gt_y_min +
                                                  _EPSILON) + 2
  x_transform = (bb_x_min - gt_x_min) * height / (gt_x_max - gt_x_min +
                                                  _EPSILON) + 2
  h_transform = (bb_y_max - bb_y_min) * width / (gt_y_max - gt_y_min + _EPSILON)
  w_transform = (bb_x_max - bb_x_min) * width / (gt_x_max - gt_x_min + _EPSILON)

  boundaries = jnp.concatenate([
      jnp.ones_like(y_transform) * ((height + 4) - 1),
      jnp.ones_like(x_transform) * ((width + 4) - 1)
  ],
                               axis=-1).astype(masks.dtype)

  # Reshape tensors to have the right shape for selective_crop_and_resize.
  transformed_boxes = jnp.concatenate(
      [y_transform, x_transform, h_transform, w_transform], -1)
  levels = jnp.tile(
      jnp.reshape(jnp.arange(num_masks), [1, num_masks]), [batch_size, 1])

  cropped_masks = selective_crop_and_resize(
      masks,
      transformed_boxes,
      levels,
      boundaries,
      output_size,
      sample_offset=sample_offset,
      use_einsum_gather=use_einsum)
  cropped_masks = jnp.squeeze(cropped_masks, axis=-1)

  return cropped_masks
