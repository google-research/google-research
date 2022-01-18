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

# pylint: disable=logging-format-interpolation,unused-import

r"""Define all the relevant flags for these experiments in this file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import sys

from absl import flags
from absl import logging

import tensorflow.compat.v1 as tf


if 'gfile' not in sys.modules:
  gfile = tf.gfile

_flags = []


def define_string(name, default_value, helper):
  flags.DEFINE_string(name, default_value, helper)
  global _flags
  _flags.append(name)


def define_integer(name, default_value, helper):
  flags.DEFINE_integer(name, default_value, helper)
  global _flags
  _flags.append(name)


def define_float(name, default_value, helper):
  flags.DEFINE_float(name, default_value, helper)
  global _flags
  _flags.append(name)


def define_boolean(name, default_value, helper):
  flags.DEFINE_boolean(name, default_value, helper)
  global _flags
  _flags.append(name)


define_boolean('running_local_dev', False, '')
define_boolean('reset_output_dir', False, '')

define_string('load_checkpoint', None, '')
define_boolean('load_checkpoint_and_restart_global_step', False, '')

define_string('master', None, 'Should be `/bns/el-d/...`')
define_string('tpu_topology', '', 'Should be `2x2`, `4x4`, etc.')
define_boolean('use_tpu', False, '')
define_integer('num_infeed_threads', 4, '')
define_boolean('use_bfloat16', False, '')
define_integer('save_every', 1000, '')
define_integer('log_every', 10, '')

define_string('dataset_name', None, '')
define_integer('num_shards_per_worker', None, '')
define_string('task_mode', None, '')
define_string('output_dir', None, '')

define_float('batch_norm_decay', 0.99, '')
define_float('batch_norm_epsilon', 1e-3, '')
define_integer('batch_norm_batch_size', None, '')

define_integer('train_batch_size', 128, '')
define_integer('eval_batch_size', 128, '')

define_integer('image_size', 32, '')
define_integer('num_classes', 10, '')

define_integer('num_warmup_steps', 0, '')
define_integer('num_train_steps', 10000, '')
define_integer('num_decay_steps', 750, '')
define_float('weight_decay', 1e-4, '')
define_float('dense_dropout_rate', 0.1, '')
define_float('stochastic_depth_drop_rate', 0., '')
define_float('grad_bound', 1e9, '')

define_float('lr', 0.016, 'Per-256-examples start LR for RMSprop')
define_string('lr_decay_type', 'exponential', '')
define_string('optim_type', 'momentum', '')
define_string('model_type', 'wrn-28-2', '')
define_float('lr_decay_rate', 0.97, '')
define_float('rmsprop_rho', 0.9, '')
define_float('rmsprop_momentum', 0.9, '')
define_float('rmsprop_epsilon', 1e-3, '')

define_float('teacher_lr', 0.1, '')

define_float('ema_decay', 0., 'Set to 0 to not use moving_average')
define_integer('ema_start', 0, 'Step to start using ema at this step')

define_boolean('use_augment', False, None)
define_integer('augment_magnitude', 5, '')

define_string('inference_ckpt', None, '')
define_string('inference_ckpt_ensemble', None, 'Comma-separated list')
define_integer('inference_class_id', 0, '')

define_integer('inference_num_threads', None, '')
define_integer('inference_thread_index', 0, '')

define_integer('dataset_service_replicas', None, '')
define_string('dataset_service_bns_prefix', None, '')

define_float('label_smoothing', 0., '')

define_float('teacher_base_lr', 1e-3, '')
define_float('teacher_grad_bound', 20., '')
define_float('teacher_eps', 1e-7, '')
define_float('teacher_weight_decay', 1e-4, '')
define_float('teacher_temperature', 0.5, '')
define_float('teacher_grad_dot_moving_average_decay', 0.99, '')
define_float('teacher_init_range', 0.1, '')
define_float('teacher_one_hot_w', 0.9, '')
define_integer('teacher_num_samples', 10, '')

# UDA
define_integer('uda_warmup_steps', 2000, '')
define_integer('uda_data', 4, '')
define_integer('uda_steps', 10000, '')
define_float('uda_temp', 0.75, '')
define_float('uda_weight', 1., '')
define_float('uda_threshold', 0.6, '')

# MPL
define_float('mpl_temp', 0.75, '')
define_float('mpl_threshold', 0.6, '')
define_float('mpl_dot_product_bound', 0.1, '')
define_integer('mpl_student_wait_steps', 2500, '')
define_string('mpl_teacher_checkpoint', None, 'for MPL teacher')
define_string('mpl_student_checkpoint', None, 'for MPL student')

define_float('mpl_student_lr', 0.1, '')
define_integer('mpl_student_lr_wait_steps', 500, '')
define_integer('mpl_student_lr_warmup_steps', 5000, '')

define_float('mpl_teacher_lr', 0.1, '')
define_integer('mpl_teacher_lr_warmup_steps', 5000, '')


class HParams(object):
  """Implementing the interface of `tf.contrib.training.HParams`."""

  def __init__(self, **kwargs):
    self.params_dict = {}
    for k, v in kwargs.items():
      self.params_dict[k] = v

  def add_hparam(self, k, v):
    self.params_dict[k] = v

  def set_hparam(self, k, v):
    self.params_dict[k] = v

  def to_json(self, indent=0):
    return json.dumps(
        {k: self.params_dict[k] for k in sorted(self.params_dict.keys())},
        indent=indent)

  def __getattr__(self, k):
    return self.params_dict[k]

  def __contains__(self, k):
    return k in self.params_dict

  def __getitem__(self, k):
    return self.params_dict[k]

  def __setitem__(self, k, v):
    self.params_dict[k] = v


def _deduce_num_classes(params):
  """Set `num_classes` for `params`."""
  if 'imagenet' in params.dataset_name.lower():
    num_classes = 1000
  elif 'cifar100' in params.dataset_name.lower():
    num_classes = 100
  else:
    logging.info('Cannot infer `num_classes` from dataset {0}. Use 10'.format(
        params.dataset_name))
    num_classes = 10

  if 'num_classes' in params and num_classes != params.num_classes:
    logging.info('Replace `params.num_classes` from {0} to {1}'.format(
        params.num_classes, num_classes))
    params.set_hparam('num_classes', num_classes)


def build_params_from_flags():
  """Build and return a `tf.HParams` object."""
  FLAGS = flags.FLAGS  # pylint: disable=invalid-name

  # Make sure not to delete trained checkpoints
  if FLAGS.task_mode == 'evals':
    assert not FLAGS.reset_output_dir, '`eval` tasks cannot `reset_output_dir`'

  # Figure out `output_dir`
  output_dir = FLAGS.output_dir
  logging.info(f'Checkpoints are at: {output_dir}')

  # Create or delete `output_dir`
  if not gfile.IsDirectory(output_dir):
    logging.info(f'Path `{output_dir}` does not exist. Creating')
    gfile.MakeDirs(output_dir, mode=0o777)
  elif FLAGS.reset_output_dir:
    logging.info(f'Path `{output_dir}` exists. Removing')
    gfile.DeleteRecursively(output_dir)
    gfile.MakeDirs(output_dir, mode=0o777)

  global _flags
  params = HParams(
      inf=float('inf'),
      output_dir=output_dir,
  )
  for flag_name in _flags:
    flag_value = getattr(FLAGS, flag_name)
    if flag_name not in params:
      params.add_hparam(flag_name, flag_value)

  # Try to infer `num_classes` to avoid mistakes, eg. ImageNet with 10 classes.
  _deduce_num_classes(params)

  pretty_print_params = params.to_json(indent=2)
  logging.info(pretty_print_params)
  if params.task_mode not in ['inference', 'evals', 'eval_forever']:
    params_filename = os.path.join(params.output_dir, 'hparams.json')
    if not gfile.Exists(params_filename):
      with gfile.GFile(params_filename, 'w') as fout:
        fout.write(pretty_print_params)
        fout.flush()

  return params
