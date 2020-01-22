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

"""Helper functions for training and evaluating sparse ResNets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf
import state_of_sparsity.layers.l0_regularization as l0
import state_of_sparsity.layers.variational_dropout as vd
from tensorflow.contrib import summary


def format_tensors(*dicts):
  """Format metrics to be callable as tf.summary scalars on tpu's.

  Args:
    *dicts: A set of metric dictionaries, containing metric name + value tensor.

  Returns:
    A single formatted dictionary that holds all tensors.

  Raises:
   ValueError: if any tensor is not a scalar.
  """
  merged_summaries = {}
  for d in dicts:
    for metric_name, value in d.iteritems():
      shape = value.shape.as_list()
      if not shape:
        merged_summaries[metric_name] = tf.expand_dims(value, axis=0)
      elif shape == [1]:
        merged_summaries[metric_name] = value
      else:
        raise ValueError(
            'Metric {} has value {} that is not reconciliable'.format(
                metric_name, value))
  return merged_summaries


def host_call_fn(model_dir, **kwargs):
  """host_call function used for creating training summaries when using TPU.

  Args:
    model_dir: String indicating the output_dir to save summaries in.
    **kwargs: Set of metric names and tensor values for all desired summaries.

  Returns:
    Summary op to be passed to the host_call arg of the estimator function.
  """
  gs = kwargs.pop('global_step')[0]
  with summary.create_file_writer(model_dir).as_default():
    with summary.always_record_summaries():
      for name, tensor in kwargs.iteritems():
        summary.scalar(name, tensor[0], step=gs)
      return summary.all_summary_ops()


def mask_summaries(masks):
  metrics = {}
  for mask in masks:
    metrics['pruning/{}/sparsity'.format(
        mask.op.name)] = tf.nn.zero_fraction(mask)
  return metrics


def variational_dropout_dkl_loss(theta_logsigma2=None,
                                 reg_scalar=1.0,
                                 start_reg_ramp_up=0.,
                                 end_reg_ramp_up=1000.,
                                 warm_up=True,
                                 use_tpu=False):
  """Computes the KL divergance loss term for all parameters.

  Args:
    theta_logsigma2: if None, then the loss is computed over all entries in the
      THETA_LOGSIGMA2_COLLECTION.  Otherwise a list of tuples of (theta,
      log_sigma2).
    reg_scalar: The maximum value of the loss coefficient.
    start_reg_ramp_up: beginning of non-zero loss coefficient.
    end_reg_ramp_up: When the loss coefficient reaches reg_scalar.
    warm_up: If true, use the linear interpolation, otherwise always reg_scalar.
    use_tpu: Whether or not training is happening on the tpu.

  Returns:
    A Tensor containing the loss.
  """
  if theta_logsigma2 is None:
    theta_logsigma2 = tf.get_collection(vd.layers.THETA_LOGSIGMA2_COLLECTION)

  log_alphas = []
  for theta, log_sigma2 in theta_logsigma2:
    log_alphas.append(vd.common.compute_log_alpha(log_sigma2, theta))

  # Calculate the kl-divergence weight for this iteration
  step = tf.train.get_or_create_global_step()
  current_step_reg = tf.maximum(0.0,
                                tf.cast(step - start_reg_ramp_up, tf.float32))

  fraction_ramp_up_completed = tf.minimum(
      current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)

  # Compute the dkl over the parameters and weight it
  dkl_loss = tf.add_n([vd.nn.negative_dkl(log_alpha=a) for a in log_alphas])

  if warm_up:
    # regularizer intensifies over the course of ramp-up
    reg_scalar = fraction_ramp_up_completed * reg_scalar

  if not use_tpu:
    # Add summary for the kl-divergence weight and weighted loss
    tf.summary.scalar('reg_scalar', reg_scalar)

  dkl_loss = reg_scalar * dkl_loss

  return dkl_loss


def l0_regularization_loss(theta_logalpha=None,
                           reg_scalar=1.0,
                           start_reg_ramp_up=0.,
                           end_reg_ramp_up=1000.,
                           warm_up=True,
                           use_tpu=False):
  """Computes the l0-norm loss for the input parameters.

  Args:
    theta_logalpha: if None, then the loss is computed over all entries in the
      THETA_LOGALPHA_COLLECTION.  Otherwise a list of tuples of (theta,
      log_alpha).
    reg_scalar: The maximum value of the loss coefficient.
    start_reg_ramp_up: beginning of non-zero loss coefficient.
    end_reg_ramp_up: When the loss coefficient reaches reg_scalar.
    warm_up: If true, use the linear interpolation, otherwise always reg_scalar.
    use_tpu: Whether or not training is happening on the tpu.

  Returns:
    A Tensor containing the loss.
  """
  if theta_logalpha is None:
    theta_logalpha = tf.get_collection(l0.layers.THETA_LOGALPHA_COLLECTION)

  # Calculate the l0-norm weight for this iteration
  step = tf.train.get_or_create_global_step()
  current_step_reg = tf.maximum(
      0.0,
      tf.cast(step - start_reg_ramp_up, tf.float32))

  fraction_ramp_up_completed = tf.minimum(
      current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)

  # Compute the l0-norm over the parameters and weight it
  l0_norm_loss = tf.add_n([l0.nn.l0_norm(a) for (_, a) in theta_logalpha])

  if warm_up:
    # regularizer intensifies over the course of ramp-up
    reg_scalar = fraction_ramp_up_completed * reg_scalar

  if not use_tpu:
    # Add summary for the l0-norm weight
    tf.summary.scalar('reg_scalar', reg_scalar)

  l0_norm_loss = reg_scalar * l0_norm_loss
  return l0_norm_loss


def add_vd_pruning_summaries(theta_logsigma2=None, threshold=3.0):
  """Adds pruning summaries to training."""
  if theta_logsigma2 is None:
    theta_logsigma2 = tf.get_collection(vd.layers.THETA_LOGSIGMA2_COLLECTION)

  with tf.name_scope('vd_summaries'):
    total_zero = 0.
    total_weights = 0.
    metrics = {}
    for theta, log_sigma2 in theta_logsigma2:
      log_alpha = vd.common.compute_log_alpha(log_sigma2, theta)
      # Compute the weight mask based on the set threshold
      mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

      is_zero = tf.cast(tf.equal(mask, tf.constant(0., tf.float32)), tf.float32)
      zero = tf.reduce_sum(is_zero)

      layer_weights = tf.cast(tf.size(mask), tf.float32)
      total_zero += zero
      total_weights += layer_weights

      metrics['{0}/sparsity'.format(theta.op.name)] = tf.nn.zero_fraction(mask)

    global_sparsity = total_zero / total_weights
    metrics['global_sparsity'] = global_sparsity

    return metrics


def add_l0_summaries(theta_logalpha=None):
  """Adds summaries for global weight sparsity."""
  if theta_logalpha is None:
    theta_logalpha = tf.get_collection(l0.layers.THETA_LOGALPHA_COLLECTION)

  with tf.name_scope('l0_summaries'):
    total_zero = 0.
    total_weights = 0.
    metrics = {}
    for theta, log_alpha in theta_logalpha:
      # Compute the evaluation time weights
      weight_noise = l0.common.hard_concrete_mean(log_alpha)
      w = theta * weight_noise

      is_zero = tf.cast(tf.equal(w, tf.constant(0., tf.float32)), tf.float32)
      zero = tf.reduce_sum(is_zero)

      layer_weights = tf.cast(tf.size(w), tf.float32)
      total_zero += zero
      total_weights += layer_weights

    global_sparsity = (total_zero / total_weights)
    metrics['global_sparsity'] = global_sparsity
    return metrics
