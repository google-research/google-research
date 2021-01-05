# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Utils for area computation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.layers import common_layers
import tensorflow.compat.v1 as tf


def area_bounds(length, max_area_width):
  """Compute the area bounds."""
  with tf.name_scope("compute_area_bounds"):
    start_list = []
    end_list = []
    for area_size in range(max_area_width):
      starts = tf.range(tf.maximum(length - area_size, 0))
      ends = starts + area_size + 1
      start_list.append(starts)
      end_list.append(ends)
    area_starts = tf.concat(start_list, axis=0)
    area_ends = tf.concat(end_list, axis=0)
    return area_starts, area_ends


def compute_sum_image(features, max_area_width):
  """Computes the vector sums of possible areas. (TODO: liyang) use t2t.

  Args:
    features: a tensor in shape of [batch_size, length, depth]
    max_area_width: a constant scalar.
  Returns:
    sum_image: vector sums of all the area combination.
    area_starts: the start position of each area.
    area_ends: the end position of each area.
  """
  with tf.name_scope("compute_sum_image", values=[features]):
    integral_image = tf.cumsum(features, axis=1, name="compute_integral_image")
    padded_integral_image = tf.pad(
        integral_image, [[0, 0], [1, 0], [0, 0]], constant_values=0)
    start_list = []
    end_list = []
    dst_images = []
    src_images = []
    shape = common_layers.shape_list(padded_integral_image)
    batch_size = shape[0]
    length = shape[1]
    for area_size in range(max_area_width):
      dst_images.append(padded_integral_image[:, area_size + 1:, :])
      src_images.append(padded_integral_image[:, :-area_size - 1, :])
      starts = tf.tile(tf.expand_dims(tf.range(
          tf.maximum(length - area_size - 1, 0)), 0), [batch_size, 1])
      ends = starts + area_size + 1
      start_list.append(starts)
      end_list.append(ends)
    sum_image = tf.subtract(tf.concat(dst_images, axis=1),
                            tf.concat(src_images, axis=1))
    area_starts = tf.concat(start_list, axis=1)
    area_ends = tf.concat(end_list, axis=1)
    return sum_image, area_starts, area_ends


def compute_alternative_span_rep(hiddens, features, max_area_width,
                                 hidden_size, advanced=False):
  """Computes the vector sums of possible areas. (TODO: liyang) use t2t.

  Args:
    hiddens: the hidden representation of features.
    features: a tensor in shape of [batch_size, length, depth].
    max_area_width: a constant scalar.
    hidden_size: the target hidden_size.
    advanced: whether to use advanced representations that includes start-end
      encoding, the weighted sum encoding and the size encoding.
  Returns:
    summary_features: representations for all the area combination in the shape
      of [batch_size, length, hidden_size].
  """
  with tf.name_scope("compute_start_end_image", values=[hiddens, features]):
    dst_images = [hiddens]
    src_images = [hiddens]
    # Starting from effective area size 2
    for area_size in range(max_area_width - 1):
      dst_images.append(hiddens[:, (area_size + 1):, :])
      src_images.append(hiddens[:, :-(area_size + 1), :])
    end_image = tf.concat(dst_images, axis=1)
    start_image = tf.concat(src_images, axis=1)
    start_end_image = tf.concat([end_image, start_image], axis=-1)
    if advanced:
      weights = tf.exp(tf.nn.sigmoid(tf.layers.dense(
          tf.layers.dense(hiddens, units=hidden_size,
                          activation=tf.nn.relu), units=1)))
      features = weights * features
      normalizers, area_starts, area_ends = compute_sum_image(
          weights, max_area_width)
      sum_images, _, _ = compute_sum_image(features, max_area_width)
      final_images = tf.math.divide_no_nan(sum_images, normalizers)
      sizes = area_ends - area_starts
      size_embeddings = tf.nn.embedding_lookup(
          params=tf.get_variable(
              name="span_len_w",
              shape=[max_area_width, max_area_width]),
          ids=sizes, name="embed_span_len")
      summary_features = tf.layers.dense(
          tf.concat([start_end_image, final_images, size_embeddings], axis=-1),
          units=hidden_size)
    else:
      summary_features = tf.layers.dense(
          start_end_image,
          units=hidden_size)
    return summary_features


def area_range_to_index(area_range, length, max_area_width):
  """Computes the indices of each area in the area expansion.

  Args:
    area_range: tensor in shape of [batch_size, 2]
    length: a scalar tensor gives the length of the original feature space.
    max_area_width: a constant scalar.
  Returns:
    indices: area indices tensor in shape of [batch_size]
  """
  with tf.control_dependencies([tf.assert_equal(tf.rank(area_range), 2),
                                tf.assert_equal(tf.shape(area_range)[1], 2)]):
    area_range = tf.cast(area_range, tf.int32)
  target_size = area_range[:, 1] - area_range[:, 0]
  with tf.control_dependencies([
      tf.assert_less(target_size, max_area_width + 1, summarize=100000)]):
    sizes = target_size - 1
    start_length = length
    pre_end_length = length - sizes + 1
    base = (start_length + pre_end_length) *\
        (start_length - pre_end_length + 1) // 2
    base = tf.where(
        tf.less_equal(target_size, 1),
        tf.zeros_like(target_size),
        base)
    offset = area_range[:, 0]
    return base + offset


def batch_gather(values, indices):
  """Gather slices from values.

  Args:
    values: a tensor in the shape of [batch_size, length, depth].
    indices: a tensor in the shape of [batch_size, slice_count] where
        slice_count < length.
  Returns:
    a tensor in the shape of [batch_size, slice_count, depth].
  """
  with tf.control_dependencies([
      tf.assert_equal(tf.rank(values), 3, message="values"),
      tf.assert_equal(tf.rank(indices), 2, message="indices"),
      tf.assert_equal(tf.shape(values)[0], tf.shape(indices)[0],
                      message="batch"),
  ]):
    shape = common_layers.shape_list(indices)
    depth = common_layers.shape_list(values)[-1]
  batch_indices = tf.reshape(tf.tile(
      tf.expand_dims(tf.range(shape[0]), [1]),
      [1, shape[1]]), [-1, 1])
  indices = tf.concat([batch_indices, tf.cast(
      tf.reshape(indices, [-1, 1]), tf.int32)], axis=-1)
  slices = tf.gather_nd(params=values, indices=indices)
  return tf.reshape(slices, [shape[0], shape[1], depth])


def query_area(query, area_encodings, area_bias):
  """Predicts a range of tokens based on the query.

  Args:
    query: a Tensor of shape [batch_size, length, depth]
    area_encodings: a tensor in shape of [batch_size, num_areas, depth]
    area_bias: a tensor in shape of [batch_size, num_areas].
  Returns:
    the logits to each area.
  """
  with tf.control_dependencies([tf.assert_equal(tf.rank(query), 3),
                                tf.assert_equal(tf.rank(area_encodings), 3),
                                tf.assert_equal(tf.shape(query)[-1],
                                                tf.shape(area_encodings)[-1]),
                                tf.assert_equal(tf.rank(area_bias), 2)]):
    dot_products = tf.matmul(query, tf.transpose(area_encodings, [0, 2, 1]))
    area_logits = dot_products + tf.expand_dims(area_bias, 1)
  return area_logits


def area_loss(logits, ranges, length, max_area_width, allow_empty=False):
  """Computes the loss regarding areas.

  Args:
    logits: the predictions of each area [batch_size, query_length, num_areas].
    ranges: the groundtruth [batch_size, query_length, 2].
    length: the length of the original tensor.
    max_area_width: the maximum area width.
    allow_empty: whether to allow empty refs.
  Returns:
    the loss.
  """
  num_areas = common_layers.shape_list(logits)[-1]
  ranges = tf.reshape(ranges, [-1, 2])
  indices = area_range_to_index(area_range=ranges,
                                length=length,
                                max_area_width=max_area_width)
  if allow_empty:
    indices = tf.where(
        tf.greater(ranges[:, 1], ranges[:, 0]), indices + 1,
        tf.zeros_like(indices))
  logits = tf.reshape(logits, [-1, num_areas])
  losses = tf.losses.sparse_softmax_cross_entropy(
      labels=indices, logits=logits,
      reduction=tf.losses.Reduction.NONE)
  with tf.control_dependencies([tf.assert_greater_equal(ranges[:, 1],
                                                        ranges[:, 0])]):
    if not allow_empty:
      mask = tf.greater(ranges[:, 1], ranges[:, 0])
      losses = losses * tf.cast(mask, tf.float32)
  return tf.reduce_mean(losses)


def area_to_refs(starts, ends, areas):
  return tf.concat([
      batch_gather(tf.expand_dims(starts, 2), areas),
      batch_gather(tf.expand_dims(ends, 2), areas)], axis=-1)
