# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Various experiment tools.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import calendar
import time

from absl import logging

import numpy as np
import tensorflow.compat.v1 as tf


class GraphKeys(object):
  CUSTOM_UPDATE_OPS = 'CUSTOM_UPDATE_OPS'




def gradient_summaries(gvs, suppress_inf_and_nans=False):
  """Creates summaries for norm, mean and var of gradients."""
  gs = [gv[0] for gv in gvs]
  grad_global_norm = tf.global_norm(gs, 'gradient_global_norm')

  if suppress_inf_and_nans:
    is_nan_or_inf = tf.logical_or(tf.is_nan(grad_global_norm),
                                  tf.is_inf(grad_global_norm))

    grad_global_norm = tf.where(is_nan_or_inf,
                                tf.zeros_like(grad_global_norm) - 1.,
                                grad_global_norm)

  grad_abs_max, grad_abs_mean, grad_mean, grad_var = [0.] * 4
  n_grads = 1e-8
  for g, _ in gvs:
    if isinstance(g, tf.IndexedSlices):
      g = g.values

    if g is not None:
      current_n_grads = np.prod(g.shape.as_list())
      abs_g = abs(g)
      mean, var = tf.nn.moments(g, list(range(len(g.shape))))
      grad_abs_max = tf.maximum(grad_abs_max, tf.reduce_max(abs_g))
      grad_abs_mean += tf.reduce_sum(abs_g)
      grad_mean += mean * current_n_grads
      grad_var += var
      n_grads += current_n_grads

  tf.summary.scalar('grad/abs_max', grad_abs_max)
  tf.summary.scalar('grad/abs_mean', grad_abs_mean / n_grads)
  tf.summary.scalar('grad/mean', grad_mean / n_grads)
  tf.summary.scalar('grad/var', grad_var / n_grads)

  return dict(grad_global_norm=grad_global_norm)


def maybe_convert_dataset(dataset):
  if isinstance(dataset, tf.data.Dataset):
    dataset = dataset.repeat().make_one_shot_iterator().get_next()

  return dataset


def scalar_logs(tensor_dict, ema=False, group='', global_update=True):
  """Adds tensorboard logs for a dict of scalars, potentially taking EMAs."""

  tensor_dict = {k: tf.convert_to_tensor(v) for k, v in tensor_dict.items()}

  if ema:
    ema = tf.train.ExponentialMovingAverage(decay=ema, zero_debias=True)
    if global_update:
      update_op = ema.apply(tensor_dict.values())
      tf.add_to_collection(GraphKeys.CUSTOM_UPDATE_OPS, update_op)

  processed = dict()
  for k, v in tensor_dict.items():
    if ema:
      if global_update:
        v = ema.average(v)

      else:
        update_op = ema.apply([v])
        with tf.control_dependencies([update_op]):
          v = tf.identity(ema.average(v))

    processed[k] = v
    if group:
      k += '/{}'.format(group)

    tf.summary.scalar(k, v)

  return processed


def format_integer(number, group_size=3):
  """Formats integers into groups of digits."""
  assert group_size > 0

  number = str(number)
  parts = []

  while number:
    number, part = number[:-group_size], number[-group_size:]
    parts.append(part)

  number = ' '.join(reversed(parts))
  return number


def log_num_params():
  num_params = sum(
      [np.prod(v.shape.as_list(), dtype=int) for v in tf.trainable_variables()])

  num_params = format_integer(num_params)
  logging.info('Number of trainable parameters: %s', num_params)


def log_variables_by_scope():
  """Prints trainable variables by scope."""
  params = [(v.name, v.shape.as_list()) for v in tf.trainable_variables()]
  params = sorted(params, key=lambda x: x[0])

  last_scope = None
  scope_n_params = 0
  for _, (name, shape) in enumerate(params):

    current_scope = name.split('/', 1)[0]
    if current_scope != last_scope:
      if last_scope is not None:
        scope_n_params = format_integer(scope_n_params)
        logging.info('\t#  scope params = %s\n', scope_n_params)

      logging.info('scope: %s', current_scope)
      scope_n_params = 0

    last_scope = current_scope
    n_params = np.prod(shape, dtype=np.int32)
    scope_n_params += n_params
    logging.info('\t%s, %s', name, shape)

  logging.info('\t#  scope params = %s\n', format_integer(scope_n_params))


def clip_gradients(gvs, value_clip=0, norm_clip=0):
  """Clips gradients."""

  grads, vs = zip(*gvs)
  grads = list(grads)

  if value_clip > 0:
    for i, g in enumerate(grads):
      if g is not None:
        grads[i] = tf.clip_by_value(g, -value_clip, value_clip)

  if norm_clip > 0:
    n_params = sum(np.prod(g.shape) for g in grads if g is not None)
    # n_params is most likely tf.Dimension and cannot be converted
    # to float directly
    norm_clip *= np.sqrt(float(int(n_params)))

    grads_to_clip = [(i, g) for i, g in enumerate(grads) if g is not None]
    idx, grads_to_clip = zip(*grads_to_clip)
    clipped_grads = tf.clip_by_global_norm(grads_to_clip, norm_clip)[0]

    for i, g in zip(idx, clipped_grads):
      grads[i] = g

  return [item for item in zip(grads, vs)]
