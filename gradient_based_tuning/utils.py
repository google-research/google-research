# coding=utf-8
# Copyright 2025 The Google Research Authors.
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
"""Shared utility functions for JAX directory."""

import functools
import os
import time
from typing import Any, Callable, Dict, Mapping, MutableMapping, MutableSequence, Optional, Sequence, Tuple

from absl import logging
from flax.deprecated import nn
from flax.training import checkpoints
from flax.training import common_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow.compat.v2 as tf

EOS_ID = 1  # Default <EOS> token. 0 is <pad>. Used in predict.py and data.py.


def log_message(t_program_start, message):
  """Output a timestamp and log a message via logging.info."""
  logging.info('Time: %.2f - ' % (time.time() - t_program_start) + message)


def tohost(x):  # 3D ndarray of floats -> 2D array
  """Collect batches from all devices to host and flatten batch dimensions."""
  n_device, n_batch, *remaining_dims = x.shape
  return np.array(x).reshape((n_device * n_batch,) + tuple(remaining_dims))


def create_learning_rate_scheduler(
    factors = 'constant * linear_warmup * rsqrt_decay',
    base_learning_rate = 0.5,
    warmup_steps = 1000,
    decay_factor = 0.5,
    steps_per_decay = 20000,
    steps_per_cycle = 100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: Interpreted as the constant value.
  * linear_warmup: Interpreted as linear warmup until warmup_steps.
  * rsqrt_decay: Divide by square root of max(step, warmup_steps).
  * rsqrt_normalized_decay: Divide by square root of max(step/warmup_steps, 1)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: Factors separated by '*' that defines the schedule.
    base_learning_rate: The starting constant for the lr schedule.
    warmup_steps: How many steps to warm up for in the warmup schedule.
    decay_factor: float, the amount to decay the learning rate by.
    steps_per_decay: int, how often to decay the learning rate.
    steps_per_cycle: int, steps per cycle when using cosine decay.

  Returns:
    a function learning_rate(step): float -> {'learning_rate': float}, the
    step-dependent lr.
  """
  factors = [n.strip() for n in factors.split('*')]

  def step_fn(step):
    """Step to learning rate function."""
    ret = 1.0
    for name in factors:
      if name == 'constant':
        ret *= base_learning_rate
      elif name == 'linear_warmup':
        ret *= jnp.minimum(1.0, step / warmup_steps)
      elif name == 'rsqrt_decay':
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'rsqrt_normalized_decay':
        ret *= jnp.sqrt(warmup_steps)
        ret /= jnp.sqrt(jnp.maximum(step, warmup_steps))
      elif name == 'decay_every':
        ret *= (decay_factor**(step // steps_per_decay))
      elif name == 'cosine_decay':
        progress = jnp.maximum(0.0,
                               (step - warmup_steps) / float(steps_per_cycle))
        ret *= jnp.maximum(0.0,
                           0.5 * (1.0 + jnp.cos(jnp.pi * (progress % 1.0))))
      else:
        raise ValueError('Unknown factor %s.' % name)
    return jnp.asarray(ret, dtype=jnp.float32)

  return step_fn


def weight_loss(
    loss,  # 2D ndarray of floats
    targets,  # 2D ndarray of ints
    weights = None,  # 2D ndarray of floats
):
  """Helper function for applying token-level loss weights.

  Args:
   loss: [batch, length] float array.
   targets: Categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  # Use jnp.where instead of the faster jnp.prod(targets.shape) because we want
  # to skip positions where targets == 0.
  normalizing_factor = jnp.where(targets > 0, 1, 0).astype(jnp.uint8).sum()
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()
  else:
    loss = loss * normalizing_factor

  return loss.sum(), normalizing_factor


def compute_weighted_cross_entropy(
    logits,  # 3D ndarray of floats
    targets,  # 2D ndarray of ints
    weights = None,  # 2D ndarray of floats
    label_smoothing = 0.0):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: Categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length].
   label_smoothing: Label smoothing constant, used to determine the on and off
     values.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if label_smoothing is None:
    label_smoothing = 0.0
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  vocab_size = logits.shape[-1]
  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = common_utils.onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = -jnp.sum(soft_targets * nn.log_softmax(logits), axis=-1)
  loss = loss - normalizing_constant
  return weight_loss(loss, targets, weights)


def compute_weighted_accuracy(
    logits,  # 3D ndarray of floats
    targets,  # 2D ndarray of ints
    weights = None,  # 2D ndarray of floats
):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: Categorical targets [batch, length] int array.
   weights: None or array of shape [batch, length]

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  return weight_loss(loss, targets, weights)


def compute_metrics(logits,
                    labels,
                    weights = None,
                    label_smoothing = 0.0,
                    metrics = None,
                    tag = None):
  """Compute summary metrics.

  Args:
    logits: [batch, seqlen, vocabsize] jax/numpy Float array
    labels: [batch, seqlen] jax/numpy Int array
    weights: [batch, seqlen] jax/numpy Float array, the relative weight to
      assign to each label, usually used for pad masking
    label_smoothing: How much to smooth all labels
    metrics: dict w/ keys ['loss', 'denominator'], helps to avoid loss recompute
    tag: Optional tag (appended as <METRIC>_tag) to specifically label outputs

  Returns:
    A MutableMapping of metrics, from name to value
  """
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels, weights,
                                                    label_smoothing)
  acc, _ = compute_weighted_accuracy(logits, labels, weights)
  maybe_tag = lambda x: '%s/' % tag + x if tag else x
  metrics = {
      maybe_tag('loss'): loss,
      maybe_tag('accuracy'): acc,
      maybe_tag('denominator'): weight_sum,
  }
  try:
    metrics = jax.lax.psum(metrics, axis_name='batch')
  except NameError:  # if the input is not pmapped, we do not need to psum
    pass
  return metrics


def get_total_regularization_loss(
    guided_vars_dict,
    act_fn_dict,
    raw_ex_vars = None,
    ex_index = None,
    raw_errant_vars = None,
):
  """Calculates regularization loss for example parameters / errant tags."""
  total_regularization_loss = 0.
  if 'ex_index' in guided_vars_dict and raw_ex_vars is not None:
    ex_index_reg_type = guided_vars_dict['ex_index']['regularization_type']
    ex_index_reg_norm = guided_vars_dict['ex_index']['regularization_norm']
    ex_index_reg_alpha = guided_vars_dict['ex_index']['regularization_alpha']
    if ex_index_reg_type is not None and ex_index is not None:
      activated_vars = act_fn_dict['dp-ex_index'](raw_ex_vars)
      # For regularization of individual example weights, we must regularize
      # each batch separately because each weight is only updated once per
      # epoch, so regularizing all together at each step would overwhelm the
      # much rarer actual updates.
      regularized_vals = jnp.take(activated_vars, ex_index.flatten())
      ex_index_reg_loss = apply_regularization_loss(ex_index_reg_type,
                                                    regularized_vals,
                                                    ex_index_reg_norm)
      total_regularization_loss += ex_index_reg_alpha * ex_index_reg_loss

  if 'errant_tag' in guided_vars_dict and raw_errant_vars is not None:
    errant_tag_reg_type = guided_vars_dict['errant_tag']['regularization_type']
    errant_tag_reg_norm = guided_vars_dict['errant_tag']['regularization_norm']
    errant_tag_reg_alpha = guided_vars_dict['errant_tag'][
        'regularization_alpha']
    if errant_tag_reg_type is not None:
      regularized_vals = act_fn_dict['dp-errant_tag'](raw_errant_vars)

      errant_tag_reg_loss = apply_regularization_loss(errant_tag_reg_type,
                                                      regularized_vals,
                                                      errant_tag_reg_norm)
      total_regularization_loss += errant_tag_reg_alpha * errant_tag_reg_loss

  return total_regularization_loss


def apply_regularization_loss(
    reg_type,
    regularization_values,
    regularization_norm = None):
  """Calculates regularization of specified type on supplied values.

  Args:
    reg_type: which regularization type found in regualarization_fn_dict to use
    regularization_values: values subject to regularization
    regularization_norm: <p> value if lp-norm is used

  Returns:
    regularization loss value

  Raises:
    ValueError: if the specified reg_type is not in regualarization_fn_dict
  """
  regualarization_fn_dict = {
      'l1_norm':
          lambda vals, _: sum(jnp.abs(vals)),
      'l2_norm':
          lambda vals, _: jnp.sqrt(sum(jnp.power(vals, 2))),
      'lp_norm':
          lambda vals, norm: jnp.power(  # pylint: disable=g-long-lambda
              sum(jnp.power(jnp.abs(vals), norm)), 1 / norm),
      '1_minus_mean_sqr':
          lambda vals, _: (1 - jnp.mean(vals))**2,
      '1_minus_mean':
          lambda vals, _: (1 - jnp.mean(vals))
  }
  if reg_type not in regualarization_fn_dict:
    raise ValueError(f'Unrecognized regularization type: {reg_type}.')
  reg_fn = regualarization_fn_dict[reg_type]
  return reg_fn(regularization_values, regularization_norm)


def fetch_checkpoints_list(
    model_dir,
    prefix = 'model_',
    specific_ckpts_list = None
):
  """Given a model_dir, extract paths to checkpoints with prefix.

  Args:
    model_dir: The cns dir holding the checkpoints of interest.
    prefix: The prefix for the checkpoints (ie. checkpoint with step count N
      will be at /model_dir/<prefix>N).
    specific_ckpts_list: Optional, if supplied, only returns the intersection of
      present checkpoints and this sequence.

  Returns:
    Checkpoint path files in the model dir, sorted (and possibly filtered).
  """
  glob_path = os.path.join(model_dir, prefix + '*')

  checkpoint_files = checkpoints.natural_sort(tf.gfile.Glob(glob_path))
  checkpoint_files = [f for f in checkpoint_files if 'tmp' not in f]
  if specific_ckpts_list:
    logging.info('All checkpoints found: %s', checkpoint_files)
    checkpoint_files = [
        x for x in checkpoint_files
        if x.rsplit(prefix)[-1] in specific_ckpts_list
    ]
    logging.info('Filtering ckpt list down to: %s', checkpoint_files)
  return checkpoint_files


def compute_entmax_loss(logits,
                        targets,
                        weights,
                        alpha = 1.5,
                        n_iter = 50):
  """Compute weighted cross entropy and entropy for log probs and targets.

  Check paper https://www.aclweb.org/anthology/P19-1146.pdf for details about
  the loss (equation 11).
  Args:
   logits: [batch, length, num_classes] float jax array.
   targets: categorical targets [batch, length] int jax array.
   weights: weights to mask PAD tokens from the loss.
   alpha: the same entmax alpha is used at all time steps.
   n_iter: number of iterations in entmax bisect algorithm.

  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  targets_onehot = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
  alphas = jnp.full((logits.shape[0], logits.shape[1]), alpha)
  entmax_output = entmax(logits, alphas, n_iter, -1)

  diff = (entmax_output - targets_onehot)
  similarity = (diff * logits).sum(axis=-1)
  loss = similarity + h_tsallis_entropy(entmax_output, alphas, -1)
  normalizing_factor = np.prod(logits.shape[:-1])

  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


@functools.partial(jax.jit, static_argnames=['n_iters', 'dim'])
def entmax(logits, alphas, n_iters, dim):
  """Alg.1 from https://www.aclweb.org/anthology/P19-1146.pdf.

  Args:
    logits: (seq_length, n_categorical_parameters) float array,
    alphas: (seq_length,),
    n_iters: int, defines number of iterations in bisection algorithm.
    dim: int

  Returns:
    entmax float array same shape as logits
  """

  def rho(logits):
    return jnp.clip(
        logits, min=0.)**jnp.expand_dims((1. / (alphas - 1.)), axis=dim)

  def loop_fn(tau_tree, _):
    tree_keys = ['tau_min', 'tau_max']
    tau_min, tau_max = [tau_tree.get(k) for k in tree_keys]
    tau = (tau_min + tau_max) / 2.
    normalization_constant = rho(logits -
                                 jnp.expand_dims(tau, axis=dim)).sum(axis=dim)
    tau_min = jnp.where(normalization_constant < 1., tau_min, tau)
    tau_max = jnp.where(normalization_constant < 1., tau, tau_max)

    return {
        'tau_min': tau_min,
        'tau_max': tau_max,
    }, None

  number_of_components = logits.shape[dim]
  # definition row in Alg. 1
  logits = logits * (jnp.expand_dims(alphas, axis=dim) - 1.)
  # line 2 in Alg. 1
  tau_min = logits.max(axis=dim) - 1.
  tau_max = logits.max(axis=dim) - number_of_components**(1 - alphas)

  init_carry = {
      'tau_min': tau_min,
      'tau_max': tau_max,
  }
  tau_tree, _ = jax.lax.scan(loop_fn, init_carry, xs=None, length=n_iters)
  tau = (tau_tree['tau_min'] + tau_tree['tau_max']) / 2.

  final_rho = rho(logits - jnp.expand_dims(tau, axis=dim))

  return final_rho / final_rho.sum(axis=dim, keepdims=True)


@functools.partial(jax.jit, static_argnames=['dim'])
def h_tsallis_entropy(dist_parameters, alphas, dim):
  """Computes Tsallis entropy for entmax loss.

  This loss is defined in Equation 9 of

  https://www.aclweb.org/anthology/P19-1146.pdf.

  Args:
    dist_parameters: entmax parameters to compute entropy.
    alphas: the alpha from corresponding entmax
    dim: dimension over the distribution

  Returns:
    entropy: Tsallis entropy
  """
  summand = (dist_parameters -
             dist_parameters**jnp.expand_dims(alphas, axis=dim)).sum(axis=dim)
  entropy = summand / (alphas * (alphas - 1))
  return entropy
