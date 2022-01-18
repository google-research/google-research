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

"""Data loader which loads spoof data."""

import functools

import numpy as np
import tensorflow as tf

# pylint:disable=unused-import
from smurf.data.data_utils import evaluate
from smurf.data.data_utils import list_eval_keys


def _shift_images(image, include_flow, max_shift):
  """Create an artificial flow from a static image."""
  crop_to_height = image.shape[0] - max_shift * 2
  crop_to_width = image.shape[1] - max_shift * 2
  global_flow = tf.random.uniform([2], minval=-max_shift, maxval=max_shift)
  height_shift = global_flow[0]
  width_shift = global_flow[1]
  height_shift = tf.cast(height_shift, tf.int32)
  width_shift = tf.cast(width_shift, tf.int32)
  frame1 = tf.image.crop_to_bounding_box(
      tf.expand_dims(image, axis=0),
      offset_height=max_shift,
      offset_width=max_shift,
      target_height=crop_to_height,
      target_width=crop_to_width)[0]
  frame2 = tf.image.crop_to_bounding_box(
      tf.expand_dims(image, axis=0),
      offset_height=max_shift + height_shift,
      offset_width=max_shift + width_shift,
      target_height=crop_to_height, target_width=crop_to_width
  )[0]
  flow = tf.ones_like(frame1, tf.int32)
  flow = flow[:, :, 1:] * tf.reshape(
      tf.stack([height_shift, width_shift]), (1, 1, 2))
  images = tf.stack([frame1, frame2], axis=0)
  images = tf.cast(images, tf.float32)
  output = {'images': images}
  if include_flow:
    output['flow'] = tf.cast(flow, tf.float32)
    output['flow_valid'] = tf.ones_like(output['flow'][Ellipsis, 1:])
    occlusions = tf.ones((crop_to_height - tf.abs(height_shift),
                          crop_to_width - tf.abs(width_shift)))
    occlusions = tf.cond(
        height_shift > 0,
        lambda: tf.pad(occlusions, [[height_shift, 0], [0, 0]]),
        lambda: tf.pad(occlusions, [[0, -height_shift], [0, 0]]))
    occlusions = tf.cond(
        width_shift > 0,
        lambda: tf.pad(occlusions, [[0, 0], [width_shift, 0]]),
        lambda: tf.pad(occlusions, [[0, 0], [0, -width_shift]]))
    output['occlusions'] = tf.expand_dims(occlusions, axis=-1)
  return output


def make_dataset(path,
                 mode,
                 seq_len=2,
                 shuffle_buffer_size=0,
                 height=None,
                 width=None,
                 resize_gt_flow=True,
                 gt_flow_shape=None,
                 seed=41):
  """Creates a spoof dataset for testing purposes."""
  del path
  del shuffle_buffer_size
  del resize_gt_flow
  del gt_flow_shape
  del seed

  if seq_len != 2:
    raise ValueError('Only compatible with seq_len == 2.')
  max_shift = 12
  height = height or 256
  width = width or 512
  background = np.random.randn(height + max_shift * 2, width + max_shift * 2, 3)
  ds = tf.data.Dataset.from_tensor_slices([background])
  if 'train' in mode:
    ds = ds.repeat()
  else:
    ds = ds.repeat(10)

  include_flow = 'eval' in mode or 'sup' in mode
  ds = ds.map(functools.partial(_shift_images, include_flow=include_flow,
                                max_shift=max_shift))
  ds = ds.prefetch(10)

  return ds
