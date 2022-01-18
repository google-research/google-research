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

"""Dataset class to realize re-training with a multi-frame enhanced version."""

import os
from typing import Dict

import numpy as np
import tensorflow as tf

from smurf import smurf_utils


def _deserialize(raw_data, dtype, height, width, channels):
  return tf.reshape(
      tf.io.decode_raw(raw_data, dtype), [height, width, channels])


def _deserialize_png(raw_data):
  image_uint = tf.image.decode_png(raw_data)
  return tf.image.convert_image_dtype(image_uint, tf.float32)


class SmurfMultiframe:
  """Dataset for a multi-frame enhanced version."""

  def __init__(self):
    # Context and sequence features encoded in the dataset proto.
    self._context_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'flow_uv': tf.io.FixedLenFeature([], tf.string),
        'flow_valid': tf.io.FixedLenFeature([], tf.string),
        'flow_viz': tf.io.FixedLenFeature([], tf.string),
    }
    self._sequence_features = {
        'images': tf.io.FixedLenSequenceFeature([], tf.string)
    }

    # Prefix used for evaluation output.
    self._prefix = 'smurf'

    # List of what the eval dataset will return.
    self._eval_return = ['images', 'flow', 'flow_valid']

  def make_dataset(self,
                   path,
                   mode,
                   seq_len=2,
                   shuffle_buffer_size=0,
                   height=None,
                   width=None,
                   resize_gt_flow=True,
                   seed=41):
    """Make a dataset for multiframe training.

    Args:
      path: string, in the format of 'some/path/dir1,dir2,dir3' to load all
        files in some/path/dir1, some/path/dir2, and some/path/dir3.
      mode: string, only train-sup is supported.
      seq_len: only 2 is supported
      shuffle_buffer_size: int, size of the shuffle buffer; no shuffling if 0.
      height: int, height for reshaping the images
      width: int, width for reshaping the images
      resize_gt_flow: bool, whether or not to resize the ground truth flow.
      seed: int, controls the shuffling of the data shards.

    Returns:
      A tf.dataset of image sequences for training and ground truth flow
      in dictionary format.
    """
    if 'train' not in mode:
      raise NotImplementedError()
    if 'sup' not in mode:
      raise NotImplementedError()
    if seq_len != 2:
      raise NotImplementedError()
    # Split up the possibly comma seperated directories.
    if ',' in path:
      l = path.split(',')
      d = '/'.join(l[0].split('/')[:-1])
      l[0] = l[0].split('/')[-1]
      paths = [os.path.join(d, x) for x in l]
    else:
      paths = [path]

    # Generate list of filenames.
    # pylint:disable=g-complex-comprehension
    files = [os.path.join(d, f) for d in paths for f in tf.io.gfile.listdir(d)]
    num_files = len(files)
    rgen = np.random.RandomState(seed)
    rgen.shuffle(files)
    ds = tf.data.Dataset.from_tensor_slices(files)
    if shuffle_buffer_size:
      ds = ds.shuffle(num_files)
    ds = ds.map(tf.data.TFRecordDataset)
    # pylint:disable=g-long-lambda
    ds = ds.interleave(
        lambda x: x.map(
            lambda y: self.parse_train_supervised(y, height, width,
                                                  resize_gt_flow),
            num_parallel_calls=tf.data.experimental.AUTOTUNE),
        cycle_length=min(10, num_files),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if shuffle_buffer_size:
      ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()
    ds = ds.prefetch(10)
    return ds

  def parse_eval(self, proto):
    """Parse eval proto from byte-encoding to the correct type and shape.

    Args:
      proto: Encoded data in proto / tf-sequence-example.

    Returns:
      A dictionary containing:
        'images': a sequence of tf.Tensor images
        'flow': a ground truth flow field in uv format
        'flow_valid': a mask indicating which pixels have ground truth flow
    """

    # Parse context and image sequence from protobuffer.
    context_parsed, sequence_parsed = tf.io.parse_single_sequence_example(
        proto,
        context_features=self._context_features,
        sequence_features=self._sequence_features)

    # Deserialize images to float32 tensors.
    images = tf.map_fn(
        _deserialize_png, sequence_parsed['images'], dtype=tf.float32)

    # Deserialize flow and create corresponding mask.
    h = context_parsed['height']
    w = context_parsed['width']
    flow_uv = _deserialize(context_parsed['flow_uv'], tf.float32, h, w, 2)
    mask_valid = _deserialize(context_parsed['flow_valid'], tf.float32, h, w, 1)

    return {'images': images, 'flow': flow_uv, 'flow_valid': mask_valid}

  def parse_train_supervised(self, proto, height,
                             width,
                             resize_gt_flow):
    """Parse proto from byte-encoding to the correct type and shape.

    Args:
      proto: Encoded data in proto / tf-sequence-example format.
      height: Desired image height.
      width: Desired image width.
      resize_gt_flow: Indicates if ground truth flow should be resized.

    Returns:
      A dictionary containing:
        'images': a sequence of tf.Tensor images
        'flow': a ground truth flow field in uv format
        'flow_valid': a mask indicating which pixels have ground truth flow
    """
    parsed_data = self.parse_eval(proto)
    images = parsed_data['images']
    flow_uv = parsed_data['flow']
    mask_valid = parsed_data['flow_valid']

    # Resize images and flow.
    if height is not None and width is not None:
      images = smurf_utils.resize(images, height, width, is_flow=False)
      if resize_gt_flow:
        flow_uv, mask_valid = smurf_utils.resize(
            flow_uv, height, width, is_flow=True, mask=mask_valid)

    return {'images': images, 'flow': flow_uv, 'flow_valid': mask_valid}
