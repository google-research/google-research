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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds


class DownsampledImagenet(object):

  def __init__(self, img_size, shuffle_count=int(1e6)):
    self._img_size = img_size
    self._img_shape = [img_size, img_size, 3]
    self._shuffle_count = shuffle_count

  @staticmethod
  def get_size(is_train):
    return 1281149 if is_train else 49999

  def _get_ds_name(self):
    s = self._img_size
    return 'downsampled_imagenet/{}x{}:2.0.0'.format(s, s)

  def _proc_and_batch(self, ds, batch_size):

    def _process_data(x_):
      img_ = tf.cast(x_['image'], tf.int32)
      img_.set_shape(self._img_shape)
      return {'image': img_, 'label': tf.constant(0, dtype=tf.int32)}

    ds = ds.map(_process_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds

  def train_input_fn(self, params):
    ds = tfds.load(self._get_ds_name(), split='train',
                   as_dataset_kwargs=dict(shuffle_files=True))
    ds = ds.repeat()
    ds = ds.shuffle(self._shuffle_count)
    return self._proc_and_batch(ds, params['batch_size'])

  def eval_input_fn(self, params):
    ds = tfds.load(self._get_ds_name(), split='validation',
                   as_dataset_kwargs=dict(shuffle_files=False))
    return self._proc_and_batch(ds, params['batch_size'])

  def train_one_pass_input_fn(self, params):
    ds = tfds.load(self._get_ds_name(), split='train',
                   as_dataset_kwargs=dict(shuffle_files=False))
    return self._proc_and_batch(ds, params['batch_size'])


def get_dataset(dataset_name, **kwargs):
  if dataset_name == 'imagenet32':
    return DownsampledImagenet(img_size=32, **kwargs)
  elif dataset_name == 'imagenet64':
    return DownsampledImagenet(img_size=64, **kwargs)
  else:
    raise ValueError('Unknown dataset: {}'.format(dataset_name))
