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

"""Standard loss functions and utilities."""

from typing import Optional, Tuple

import chex
import jax
from jax import lax
import jax.numpy as jnp


@jax.custom_vjp
def _cross_entropy_with_logits(logits, targets):
  """Computes cross entropy loss with custom gradient support.

  Computes a stabilized-gradient version of:
    -jnp.sum(targets * nn.log_softmax(logits), axis=-1)

  Args:
    logits: [..., num_classes] float array.
    targets: categorical targets [..., num_classes] float array.

  Returns:
    per-example cross entropy loss
  """
  assert logits.shape == targets.shape, ("logits and targets must have the same"
                                         " shape (targets must be one-hot or "
                                         "label smoothed).")
  shifted = logits - logits.max(axis=-1, keepdims=True)
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)

  return loss


def _cross_entropy_with_logits_fwd(logits, targets):
  shifted = logits - logits.max(axis=-1, keepdims=True)
  exp_shifted = jnp.exp(shifted)
  sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
  log_softmax = shifted - jnp.log(sum_exp)
  loss = -jnp.sum(targets * log_softmax, axis=-1)
  return loss, (exp_shifted, sum_exp, logits, targets)


def _cross_entropy_with_logits_bwd(res, g):
  exp_shifted, sum_exp, logits, targets = res
  g_logits = jnp.expand_dims(g, axis=-1) * (exp_shifted / sum_exp - targets)
  return jnp.asarray(g_logits, logits.dtype), jnp.asarray(g, targets.dtype)


_cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd,
                                  _cross_entropy_with_logits_bwd)


def onehot(labels,
           num_classes,
           on_value=1.0,
           off_value=0.0):
  """Returns the one-hot encoding of the labels with dimension num_classes.

  Args:
    labels: integer labels to be encoded.
    num_classes: the dimension of the one-hot encoding.
    on_value: the value to use for the "1" values.
    off_value: the value to use for the "0" values.

  Returns:
    an array of shape labels.shape + (num_classes,) containing one-hot encodings
      of labels (as a floating point array).
  """
  labels = jnp.asarray(labels)

  x = (labels[Ellipsis, None] == jnp.arange(num_classes))
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)


def cross_entropy_with_logits(logits,
                              targets,
                              label_smoothing = 0.0):
  """Compute cross entropy and entropy for log probs and targets.

  Cross entropy is taken over the last axis. Remaining axes are unchanged.

  Args:
   logits: [..., length, num_classes] float array.
   targets: categorical targets [..., length] int array.
   label_smoothing: label smoothing constant, used to determine the on and off
     values.

  Returns:
    Array with loss taken over the last axis.
  """
  assert logits.shape[:-1] == targets.shape, (
      "Logits shape must agree with targets, except in the last dimension.")

  chex.assert_type([logits, targets], [jnp.float32, jnp.int32])

  vocab_size = logits.shape[-1]

  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20))
  soft_targets = onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  loss = _cross_entropy_with_logits(logits, soft_targets)
  loss = loss - normalizing_constant

  return loss


def cross_entropy_with_probs(probs,
                             targets,
                             label_smoothing = 0.0,
                             epsilon = 1e-20):
  """Compute cross entropy for a given distribution and targets.

  Cross entropy is taken over the last axis. Remaining axes are unchanged.

  Args:
   probs: [..., length, num_classes] float array.
   targets: categorical targets [..., length] int array.
   label_smoothing: label smoothing constant, used to determine the on and off
     values.
   epsilon: small noise to add to probs when converting to log space.

  Returns:
    Array with loss taken over the last axis.
  """
  assert probs.shape[:-1] == targets.shape, (
      "Logits shape must agree with targets, except in the last dimension.")

  chex.assert_type([probs, targets], [jnp.float32, jnp.int32])

  vocab_size = probs.shape[-1]

  confidence = 1.0 - label_smoothing
  low_confidence = (1.0 - confidence) / (vocab_size - 1)
  normalizing_constant = -(
      confidence * jnp.log(confidence) +
      (vocab_size - 1) * low_confidence * jnp.log(low_confidence + epsilon))
  soft_targets = onehot(
      targets, vocab_size, on_value=confidence, off_value=low_confidence)

  probs = jax.nn.relu(probs)  # help with numerical stability
  loss = -jnp.sum(soft_targets * jnp.log(probs + epsilon), axis=-1)
  loss = loss - normalizing_constant

  return loss


def kl_divergence_with_logits(p_logits = None,
                              q_logits = None,
                              temperature = 1.):
  """Compute the KL between two categorical distributions from their logits.

  Args:
    p_logits: [..., dim] array with logits for the first distribution.
    q_logits: [..., dim] array with logits for the second distribution.
    temperature: the temperature for the softmax distribution, defaults at 1.

  Returns:
    an array of KL divergence terms taken over the last axis.
  """
  chex.assert_type([p_logits, q_logits], float)
  chex.assert_equal_shape([p_logits, q_logits])

  p_logits /= temperature
  q_logits /= temperature

  p = jax.nn.softmax(p_logits)

  log_p = jax.nn.log_softmax(p_logits)
  log_q = jax.nn.log_softmax(q_logits)
  kl = jnp.sum(p * (log_p - log_q), axis=-1)

  ## KL divergence should be positive, this helps with numerical stability
  loss = jax.nn.relu(kl)

  return loss


def kl_divergence_with_probs(p = None,
                             q = None,
                             epsilon = 1e-20):
  """Compute the KL between two categorical distributions from their probabilities.

  Args:
    p: [..., dim] array with probs for the first distribution.
    q: [..., dim] array with probs for the second distribution.
    epsilon: a small float to normalize probabilities with.

  Returns:
    an array of KL divergence terms taken over the last axis.
  """
  chex.assert_type([p, q], float)
  chex.assert_equal_shape([p, q])

  log_p = jnp.log(p + epsilon)
  log_q = jnp.log(q + epsilon)
  kl = jnp.sum(p * (log_p - log_q), axis=-1)

  ## KL divergence should be positive, this helps with numerical stability
  loss = jax.nn.relu(kl)

  return loss


def lp_norm(inputs,
            targets,
            p = 2,
            apply_root = False):
  """Compute the weighted L^p error between inputs and targets.

  Args:
    inputs: the input array.
    targets: the target array.
    p: the norm order to use.
    apply_root: if True, applies the p-norm root. Note that we always assume the
      first axis is a batch axis.

  Returns:
    the L^p error between inputs and targets over the last axis.
  """
  assert inputs.shape == targets.shape, (f"Inputs and target shapes must agree."
                                         f" Found {inputs.shape}, "
                                         f"{targets.shape}.")

  loss = jnp.abs(inputs - targets)**p
  loss = loss.sum()

  if apply_root:
    loss = loss**(1 / float(p))

  return loss


def cosine_distance(inputs,
                    targets,
                    epsilon = 1e-20):
  """Compute the cosine distance along the last axis.

  Args:
    inputs: the input array.
    targets: the target array.
    epsilon: a small float used to normalize the denominator.

  Returns:
    the cosine distance between inputs and targets over the last axis.
  """

  assert inputs.shape == targets.shape, (f"Inputs and target shapes must agree."
                                         f" Found {inputs.shape}, "
                                         f"{targets.shape}.")

  inputs_norm = jnp.linalg.norm(inputs, ord=2, axis=-1)
  targets_norm = jnp.linalg.norm(targets, ord=2, axis=-1)

  loss = 1 - (inputs * targets).sum(axis=-1) / (
      inputs_norm * targets_norm + epsilon)

  return loss


def weighted_accuracy(logits,
                      targets,
                      weights=None):
  """Computes the weighted accuracy of the predicted logits.

  Args:
    logits: [..., num_classes] unnormalized logits.
    targets: [...] categorical tensor with class labels.
    weights: Optional[...] float tensor containing weights to scale targets by.
      Can be used for class weights or masking.

  Returns:
    tuple of sum accuracy across all examples and normalizing factor. To recover
      true accuracy, divide total by weights.
  """
  assert logits.shape[:-1] == targets.shape, (
      f"Logits shape must agree with targets, except "
      f"in the last dimension. Found {logits.shape[:-1]}, {targets.shape}.")

  chex.assert_type([logits, targets], [jnp.float32, jnp.int32])

  if weights is not None:
    assert targets.shape == weights.shape, ("labels and weights must have the "
                                            "same shape.")

  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)

  total_loss, weights = weighted_mean(loss, weights)

  return total_loss, weights


def weighted_mean(array,
                  weights = None):
  """Computes the weighted mean of an array.

  Args:
    array: the array to compute the mean of.
    weights: if supplied, a set of weights which are multiplied with the array
      before average.

  Returns:
    the total loss summed over examples, and the weights to divide by.
  """

  if weights is None:
    weights = jnp.ones_like(array)

  chex.assert_equal_rank([array, weights])

  loss = (array * weights).sum()

  return loss, weights.sum()


def classification_loss_fn(params,
                           inputs,
                           targets,
                           *,
                           model_apply,
                           rng_key,
                           is_eval=False,
                           label_smoothing=0.0):
  """Applies cross entropy loss given a batch and model_apply fn."""
  del is_eval, rng_key

  output = model_apply(params, inputs)

  if isinstance(output, tuple):
    if len(output) != 2:
      raise ValueError("expected model_apply to return logits or (logits, mask)"
                       f", but found a tuple of length {len(output)}.")
    logits, mask = output
  else:
    logits, mask = output, None

  loss = cross_entropy_with_logits(
      logits, targets, label_smoothing=label_smoothing)
  loss, weights = weighted_mean(loss, mask)
  acc, _ = weighted_accuracy(logits, targets, weights=mask)

  metrics = {
      "loss": loss,
      "denominator": weights,
      "accuracy": acc,
  }

  extras = {
      "logits": logits,
      "predictions": logits.argmax(-1),
      "mask": mask,
  }

  return (loss, weights), (metrics, extras)
