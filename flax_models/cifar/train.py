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

# Lint as: python3
"""Trains a model on cifar10 or cifar100."""

import os

from absl import app
from absl import flags
from absl import logging
import jax
from tensorflow.io import gfile

from flax_models.cifar.datasets import dataset_source as dataset_source_lib
from flax_models.cifar.models import load_model
from flax_models.cifar.training_utils import flax_training


FLAGS = flags.FLAGS


flags.DEFINE_enum('dataset', 'cifar10',
                  ['cifar10', 'cifar100', 'fashion_mnist', 'svhn'],
                  'Name of the dataset. Must be `cifar10`,  `cifar100` or  '
                  '`fashion_mnist`, or `svhn`.')
flags.DEFINE_enum(
    'model_name', 'WideResnet28x10',
    ['WideResnet28x10', 'WideResnet28x6_ShakeShake', 'Pyramid_ShakeDrop'],
    'Name of the model to train. Should be one of '
    '`WideResnet28x10`, `WideResnet28x6_ShakeShake`, '
    'or `Pyramid_ShakeDrop`.')
flags.DEFINE_integer('num_epochs', 200,
                     'How many epochs the model should be trained for.')
flags.DEFINE_integer(
    'batch_size', 128, 'Global batch size. If multiple '
    'replicas are used, each replica will receive '
    'batch_size / num_replicas examples. Batch size should be divisible by '
    'the number of available devices.')
flags.DEFINE_string(
    'output_dir', '', 'Directory where the checkpoints and the tensorboard '
    'records should be saved.')
flags.DEFINE_enum(
    'image_level_augmentations', 'basic', ['none', 'basic', 'autoaugment',
                                           'aa-only'],
    'Augmentations applied to the images. Should be `none` for '
    'no augmentations, `basic` for the standard horizontal '
    'flips and random crops, and `autoaugment` for the best '
    'AutoAugment policy for cifar10. For SVHN, aa-only should be use for '
    'autoaugment without random crops or flips.')
flags.DEFINE_enum(
    'batch_level_augmentations', 'none', ['none', 'cutout'],
    'Augmentations that are applied at the batch level. Should '
    'be `cutout` or `none`.')


def main(_):

  # As we gridsearch the weight decay and the learning rate, we add them to the
  # output directory path so that each model has its own directory to save the
  # results in. We also add the `run_seed` which is "gridsearched" on to
  # replicate an experiment several times.
  output_dir_suffix = os.path.join(
      'lr_' + str(FLAGS.learning_rate),
      'wd_' + str(FLAGS.weight_decay),
      'seed_' + str(FLAGS.run_seed))

  output_dir = os.path.join(FLAGS.output_dir, output_dir_suffix)

  if not gfile.Exists(output_dir):
    gfile.MakeDirs(output_dir)

  num_devices = jax.local_device_count()
  assert FLAGS.batch_size % num_devices == 0
  local_batch_size = FLAGS.batch_size // num_devices
  info = 'Total batch size: {} ({} x {} replicas)'.format(
      FLAGS.batch_size, local_batch_size, num_devices)
  logging.info(info)

  if FLAGS.dataset.lower() == 'cifar10':
    dataset_source = dataset_source_lib.Cifar10(FLAGS.batch_size,
                                                FLAGS.image_level_augmentations,
                                                FLAGS.batch_level_augmentations)
  elif FLAGS.dataset.lower() == 'cifar100':
    dataset_source = dataset_source_lib.Cifar100(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  elif FLAGS.dataset.lower() == 'fashion_mnist':
    dataset_source = dataset_source_lib.FashionMnist(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  elif FLAGS.dataset.lower() == 'svhn':
    dataset_source = dataset_source_lib.SVHN(
        FLAGS.batch_size, FLAGS.image_level_augmentations,
        FLAGS.batch_level_augmentations)
  else:
    raise ValueError('Available datasets: cifar10(0), fashion_mnist, svhn.')

  if 'cifar' in FLAGS.dataset.lower() or 'svhn' in FLAGS.dataset.lower():
    image_size = 32
    num_channels = 3
  else:
    image_size = 28  # For Fashion Mnist
    num_channels = 1

  num_classes = 100 if FLAGS.dataset.lower() == 'cifar100' else 10
  model, state = load_model.get_model(FLAGS.model_name,
                                      local_batch_size, image_size,
                                      num_classes, num_channels)
  # Learning rate will be overwritten by the lr schedule, we set it to zero.
  optimizer = flax_training.create_optimizer(model, 0.0)

  flax_training.train(optimizer, state, dataset_source, output_dir,
                      FLAGS.num_epochs)


if __name__ == '__main__':
  app.run(main)
