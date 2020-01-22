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

"""Image datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt
import numpy as np
import sonnet as snt
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import nest
import tensorflow_datasets as tfds


from stacked_capsule_autoencoders.capsules.data import tfrecords as _tfrecords


def create(which,
           batch_size,
           subset=None,
           n_replicas=1,
           transforms=None,
           **kwargs):
  """Creates data loaders according to the dataset name `which`."""

  func = globals().get('_create_{}'.format(which), None)
  if func is None:
    raise ValueError('Dataset "{}" not supported. Only {} are'
                     ' supported.'.format(which, SUPPORTED_DATSETS))

  dataset = func(subset, batch_size, **kwargs)

  if transforms is not None:
    if not isinstance(transforms, dict):
      transforms = {'image': transforms}

    for k, v in transforms.items():
      transforms[k] = snt.Sequential(nest.flatten(v))

  if transforms is not None or n_replicas > 1:

    def map_func(data):
      """Replicates data if necessary."""
      data = dict(data)

      if n_replicas > 1:
        tile_by_batch = snt.TileByDim([0], [n_replicas])
        data = {k: tile_by_batch(v) for k, v in data.items()}

      if transforms is not None:
        img = data['image']

        for k, transform in transforms.items():
          data[k] = transform(img)

      return data

    dataset = dataset.map(map_func)

  iter_data = dataset.make_one_shot_iterator()
  input_batch = iter_data.get_next()
  for _, v in input_batch.items():
    v.set_shape([batch_size * n_replicas] + v.shape[1:].as_list())

  return input_batch


def _create_mnist(subset, batch_size, **kwargs):
  return tfds.load(
      name='mnist', split=subset, **kwargs).repeat().batch(batch_size)



SUPPORTED_DATSETS = set(
    k.split('_', 2)[-1] for k in globals().keys() if k.startswith('_create'))
