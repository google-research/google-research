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

"""Utils for working with slim argscopes."""

import tensorflow.compat.v1 as tf
import tf_slim  # For tensorflow.contrib.layers

from nopad_inception_v3_fcn import network_params
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim


slim = contrib_slim


def get_conv_scope(params,
                   is_training = True):
  """Constructs an argscope for configuring CNNs.

  Note that the scope returned captures any existing scope from within which
  this function is called. The returned scope however is absolute and overrides
  any outside scope -- this implies that using it within a new scope renders the
  new scope redundant. Example:

    with slim.arg_scope(...) as existing_sc:
      sc = get_conv_scope(...)  # `sc` captures `existing_sc`.

    with slim.arg_scope(...) as new_sc:
      with slim.arg_scope(sc):
        ...  # This context does NOT capture `new_sc`; `sc` is absolute.

      # Correct way to capture `new_sc` by calling from within the scope.
      new_conv_sc = get_conv_scope(...)
      with slim.arg_scope(new_conv_sc):
        ...  # `new_conv_sc` captures `new_sc`.

  Args:
    params: `ParameterContainer` containing the model params.
    is_training: whether model is meant to be trained or not.

  Returns:
    sc: a `slim.arg_scope` that sets the context for convolutional layers based
      on `params` and the context from which `get_conv_scope` is called. Note
      that using `sc` overrides any outside `arg_scope`; see docstring for more
      info.
  """
  sc_gen = _get_base_scope(float(params.l2_weight_decay))
  with sc_gen:
    sc = contrib_framework.current_arg_scope()
  if params.batch_norm:
    batch_norm_sc_gen = _get_batch_norm_scope(
        is_training, decay=params.batch_norm_decay)
    sc = _update_arg_scope(sc, batch_norm_sc_gen)
  if params.dropout:
    dropout_sc_gen = _get_dropout_scope(
        is_training, keep_prob=params.dropout_keep_prob)
    sc = _update_arg_scope(sc, dropout_sc_gen)
  return sc


def _update_arg_scope(base_sc, override_sc_gen):
  """Override kwargs for ops in `base_sc` with those from `override_sc_gen`.

  Args:
    base_sc: base `arg_scope` containing ops mapped to their kwargs.
    override_sc_gen: a `slim.arg_scope` generator whose `arg_scope` will
      override the base scope.

  Returns:
    A new `arg_scope` that overrides 'base_sc` using overrides generated from
      `override_sc_gen`.
  """
  with slim.arg_scope(base_sc):
    with override_sc_gen:
      return contrib_framework.current_arg_scope()


def _get_base_scope(weight_decay=0.00004):
  """Defines the default arg scope.

  Args:
    weight_decay: The weight decay to use for regularizing the conv weights.

  Returns:
    A `slim.arg_scope` generator.
  """
  base_scope_args = _get_base_scope_args(weight_decay)
  sc_gen = slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose],
      **base_scope_args)
  return sc_gen


def _get_batch_norm_scope(is_training,
                          decay=0.9997,
                          init_stddev=0.1,
                          batch_norm_var_collection='moving_vars'):
  """Defines an arg scope for configuring batch_norm in conv2d layers.

  Args:
    is_training: Whether or not we're training the model.
    decay: Decay factor for moving averages used for eval.
    init_stddev: The standard deviation of the trunctated normal weight init.
    batch_norm_var_collection: The name of the collection for the batch norm
      variables.

  Returns:
    An `arg_scope` generator to induce batch_norm normalization in conv2d
      layers.
  """
  batch_norm_scope_args = _get_batch_norm_scope_args(is_training, decay,
                                                     init_stddev,
                                                     batch_norm_var_collection)
  sc_gen = slim.arg_scope(
      [slim.conv2d, slim.separable_conv2d, slim.conv2d_transpose],
      **batch_norm_scope_args)
  return sc_gen


def _get_dropout_scope(is_training, keep_prob=0.8):
  """Defines an arg scope for configuring dropout after slim.conv2d layers.

  Args:
    is_training: Whether or not we're training the model.
    keep_prob: The probability that each element is kept.

  Returns:
    An `arg_scope` generator to induce dropout normalization in slim.conv2d
      layers.
  """
  dropout_scope_args = _get_dropout_scope_args(is_training, keep_prob)
  sc_gen = slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                          **dropout_scope_args)
  return sc_gen


def _get_base_scope_args(weight_decay):
  """Returns arguments needed to initialize the base `arg_scope`."""
  regularizer = tf_slim.l2_regularizer(weight_decay)
  conv_weights_init = tf_slim.xavier_initializer_conv2d()
  base_scope_args = {
      'weights_initializer': conv_weights_init,
      'activation_fn': tf.nn.relu,
      'weights_regularizer': regularizer,
  }
  return base_scope_args


def _get_batch_norm_scope_args(is_training, decay, init_stddev,
                               batch_norm_var_collection):
  """Returns arguments needed to initialize the batch norm `arg_scope`."""
  batch_norm_params = {
      'is_training': is_training,
      # Decay for the moving averages.
      'decay': decay,
      # epsilon to prevent 0s in variance.
      'epsilon': 0.001,
      # collection containing the moving mean and moving variance.
      'variables_collections': {
          'beta': None,
          'gamma': None,
          'moving_mean': [batch_norm_var_collection],
          'moving_variance': [batch_norm_var_collection],
      },
      'zero_debias_moving_mean': False,
  }
  batch_norm_scope_args = {
      'normalizer_fn': slim.batch_norm,
      'normalizer_params': batch_norm_params,
      'weights_initializer': tf.truncated_normal_initializer(stddev=init_stddev)
  }
  return batch_norm_scope_args


def _get_dropout_scope_args(is_training, keep_prob):
  """Returns arguments needed to initialize the batch norm `arg_scope`."""
  dropout_scope_args = {
      'activation_fn': _get_relu_then_dropout(is_training, keep_prob),
  }
  return dropout_scope_args


def _get_relu_then_dropout(is_training, keep_prob):

  def relu_then_dropout(x):
    x = tf.nn.relu(x)
    return slim.dropout(x, is_training=is_training, keep_prob=keep_prob)

  return relu_then_dropout
