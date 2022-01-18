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

"""Dataset class for a simple datasets."""
# pylint:skip-file
import os

import tensorflow as tf

from smurf import smurf_utils


def _deserialize_png(raw_data):
  image_uint = tf.image.decode_png(raw_data)
  return tf.image.convert_image_dtype(image_uint, tf.float32)


class SimpleDataset():
  """Simple dataset that only holds an image triplet."""

  def __init__(self):
    super().__init__()
    # Context and sequence features encoded in the dataset proto.
    self._context_features = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
    }
    self._sequence_features = {
        'images': tf.io.FixedLenSequenceFeature([], tf.string)
    }

  def parse_train(self, proto, height, width):
    """Parse features from byte-encoding to the correct type and shape.

    Args:
      proto: Encoded data in proto / tf-sequence-example.
      height: int, desired image height.
      width: int, desired image width.

    Returns:
      A sequence of images as tf.Tensor of shape [2, height, width, 3].
    """
    _, sequence_parsed = tf.io.parse_single_sequence_example(
        proto,
        context_features=self._context_features,
        sequence_features=self._sequence_features)

    # Deserialize images to float32 tensors.
    images = tf.map_fn(
        _deserialize_png, sequence_parsed['images'], dtype=tf.float32)

    # Resize images.
    if height is not None and width is not None:
      images = smurf_utils.resize(images, height, width, is_flow=False)

    return {'images': images}

  def make_dataset(self,
                   path,
                   mode,
                   height=None,
                   width=None):
    """Make a dataset for training or evaluating SMURF.

    Args:
      path: string, in the format of 'some/path/dir1,dir2,dir3' to load all
        files in some/path/dir1, some/path/dir2, and some/path/dir3.
      mode: string, one of ['train', 'train-supervised', 'eval'] to switch
        between loading training data and evaluation data, which return
        different features.
      height: int, height for reshaping the images (only if for_eval=False)
        because reshaping for eval is more complicated and done in the evaluate
        function through SMURF inference.
      width: int, width for reshaping the images (only if for_eval=False).

    Returns:
      A tf.dataset of image sequences for training and ground truth flow
      in dictionary format.
    """
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
    ds = tf.data.Dataset.from_tensor_slices(files)
    if mode == 'multiframe':
      # Create a nested dataset.
      ds = ds.map(tf.data.TFRecordDataset)
      # pylint:disable=g-long-lambda
      ds = ds.interleave(
          lambda x: x.map(
              lambda y: self.parse_train(y, height, width),
              num_parallel_calls=tf.data.experimental.AUTOTUNE),
          cycle_length=min(10, num_files),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
      # Prefetch a number of batches because reading new ones can take much
      # longer when they are from new files.
      ds = ds.prefetch(10)

    return ds
