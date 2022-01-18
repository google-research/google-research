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

"""Data config."""
import functools
from absl import flags
from monty.collections import AttrDict
import tensorflow.compat.v1 as tf
from stacked_capsule_autoencoders.capsules.data import constellation
from stacked_capsule_autoencoders.capsules.data import image
from stacked_capsule_autoencoders.capsules.data import preprocess

flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('canvas_size', 28, 'Canvas size.')


def get(config):
  """Returns the dataset."""

  if config.dataset == 'mnist':
    dataset = make_mnist(config)
  elif config.dataset == 'constellation':
    dataset = make_constellation(config)

  return dataset


def make_mnist(config):
  """Creates the MNIST dataset."""

  def to_float(x):
    return tf.to_float(x) / 255.

  transform = [to_float]

  if config.canvas_size != 28:
    transform.append(functools.partial(preprocess.pad_and_shift,
                                       output_size=config.canvas_size,
                                       shift=None))

  batch_size = config.batch_size
  res = AttrDict(
      trainset=image.create(
          'mnist', subset='train', batch_size=batch_size, transforms=transform),
      validset=image.create(
          'mnist', subset='test', batch_size=batch_size, transforms=transform))

  return res


def make_constellation(config):
  """Creates the constellation dataset."""

  dataset = constellation.create(
      batch_size=config.batch_size,
      shuffle_corners=True,
      gaussian_noise=.0,
      drop_prob=0.5,
      which_patterns=[[0], [1], [0]],
      rotation_percent=180 / 360.,
      max_scale=3.,
      min_scale=3.,
      use_scale_schedule=False,
      schedule_steps=0,
  )

  # data is created online, so there is no point in having
  # a separate dataset for validation
  res = AttrDict(trainset=dataset, validset=dataset)
  return res
