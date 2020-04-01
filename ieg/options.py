# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Define FLAGS of the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags


def define_basic_flags():
  """Defines basic flags."""

  flags.DEFINE_integer('max_iteration', 200000, 'Number of iteration')
  flags.DEFINE_float('learning_rate', 0.1, 'Learning rate')
  flags.DEFINE_integer('batch_size', 100, 'Batch size')
  flags.DEFINE_integer('val_batch_size', 100, 'Validation data batch size.')
  flags.DEFINE_integer('restore_step', 0, ('Checkpoint id.'
                                           '0: load the latest step ckpt.'
                                           '>0: load the assigned step ckpt.'))
  flags.DEFINE_enum('network_name', 'wrn28-10', ['resnet29', 'wrn28-10'],
                    'Network architecture name')
  flags.DEFINE_string('dataset', 'cifar100_uniform_0.8',
                      'Dataset schema: <dataset>_<noise-type>_<ratio>')
  flags.DEFINE_integer('seed', 12345, 'Seed for selecting validation set')
  flags.DEFINE_enum('method', 'ieg', ['ieg', 'l2r', 'supervised'],
                    'Method to deploy.')
  flags.DEFINE_float('momentum', 0.9,
                     'Use momentum optimizer and the same for meta update')
  flags.DEFINE_string('decay_steps', '500',
                      'Decay steps, format (integer[,<integer>,<integer>]')
  flags.DEFINE_float('decay_rate', 0.1, 'Decay steps')
  flags.DEFINE_float('eval_freq', 500, 'How many steps evaluate and save model')
  flags.DEFINE_string('checkpoint_path', '/tmp/ieg',
                      'Checkpoint saving root folder')
  flags.DEFINE_integer('warmup_epochs', 0, 'Warmup with standard training')
  flags.DEFINE_enum('lr_schedule', 'cosine',
                    ['cosine', 'custom_step', 'cosine_warmup', 'exponential'],
                    'Learning rate schedule.')
  flags.DEFINE_float('cos_t_mul', 1.5, 't_mul of cosine learning rate')
  flags.DEFINE_float('cos_m_mul', 0.9, 'm_mul of cosine learning rate')
  flags.DEFINE_bool('use_ema', True, 'Use EMA')

  # Method related arguments
  flags.DEFINE_float('meta_momentum', 0.9, 'Meta momentum.')
  flags.DEFINE_float('meta_stepsize', 0.1, 'Meta learning step size.')
  flags.DEFINE_float('ce_factor', 5,
                     'Weight of cross_entropy loss (p, see paper).')
  flags.DEFINE_float('consistency_factor', 20,
                     'Weight of KL loss (k, see paper)')
  flags.DEFINE_float(
      'probe_dataset_hold_ratio', 0.02,
      'Probe set holdout ratio from the training set (0.02 indicates 1000 images for CIFAR datasets).'
  )
  flags.DEFINE_float('grad_eps_init', 0.9, 'eps for meta learning init value')
  flags.DEFINE_enum(
      'aug_type', 'autoaug', ['autoaug', 'randaug', 'default'],
      'Fake autoaugmentation type. See dataset_utils/ for more details')
  flags.DEFINE_bool('post_batch_mode_autoaug', True,
                    'If true, apply batch augmentation.')
  flags.DEFINE_enum('mode', 'train', ['train', 'evaluation'],
                    'Train or evaluation mode.')
