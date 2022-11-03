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

"""Load data."""

import os.path
import gin
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


@gin.configurable
class TFDSDataLoader:
  """Loads a dataset and provides convenient information."""

  def __init__(self,
               name='',
               dtype=tf.float32,
               shift = 0.1,
               augmentation = True):
    self.name = name
    self.dtype = dtype
    self.shift = shift
    self.augmentation = augmentation
    self._load()

  def convert(self, image, label):
    image = tf.cast(image, self.dtype)
    image = image / tf.cast(255.0, dtype=self.dtype)
    label = tf.one_hot(label, depth=self.output_size)
    return image, tf.cast(label, tf.int32)

  def random_shift(self, img):
    h, w = img.shape[:2]
    offset_h, offset_w = int(self.shift * h), int(self.shift * w)
    padded = tf.image.pad_to_bounding_box(
        img, offset_h, offset_w, h + offset_h, w + offset_w)
    return tf.image.random_crop(padded, img.shape)

  def augment(self, image, label):
    if self.augmentation:
      image = tf.image.random_flip_left_right(image)
      image = tf.py_function(self.random_shift, inp=[image], Tout=self.dtype)
    return image, label

  def _load(self):
    """Loads a tf.Dataset corresponding to the given name."""
    dataset, info = tfds.load(self.name, with_info=True, as_supervised=True)
    self.input_shape = info.features['image'].shape
    self.output_shape = (info.features['label'].num_classes,)
    self.output_size = self.output_shape[-1]
    self.num_examples = {t: s.num_examples for (t, s) in info.splits.items()}
    self.ds = dict()
    for tag in ['train', 'test']:
      self.ds[tag] = (
          dataset[tag].map(self.convert).map(self.augment).shuffle(10000))


@gin.configurable
class WarcraftDataLoader:
  """Loads a dataset and provides convenient information."""

  def __init__(self, folder=None, dtype=tf.float32, limit = -1):
    self.folder = folder
    self.dtype = dtype
    self.full = dict()
    self.ds = dict()
    self.num_examples = dict()
    for tag in ['train', 'test']:
      curr = []
      for suffix in ['maps', 'shortest_paths', 'vertex_weights']:
        filename = os.path.join(folder, '{}_{}.npy'.format(tag, suffix))
        with tf.io.gfile.GFile(filename, 'rb') as fp:
          tensor = np.load(fp)
          if suffix == 'maps':
            tensor = tensor.astype(np.int32) * 2
          else:
            tensor = tensor.astype(np.float32)
          if limit > 0:
            tensor = tensor[:limit]
          curr.append(tf.convert_to_tensor(tensor))
      self.full[tag] = tuple(curr)
      self.num_examples[tag] = curr[0].shape[0]

      ds = tf.data.Dataset.from_tensor_slices(self.full[tag])
      self.ds[tag] = ds.map(self.convert).shuffle(1024)

    self.input_shape = self.full['train'][0].shape[1:]
    self.output_shape = self.full['train'][1].shape[1:]

  def convert(self, im, label, info):
    return tf.cast(im, dtype=tf.float32) / 128 - 1.0, label, info
