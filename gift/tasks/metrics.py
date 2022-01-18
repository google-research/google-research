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

"""Metric functions."""

from flax.deprecated import nn
import jax
import jax.numpy as jnp
import numpy as np


def weighted_accuracy(logits, one_hot_targets, weights=None):
  """Compute weighted accuracy over the given batch.

  This computes the accuracy over a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.
  We assume the trainer will aggregate and divide by number of samples.

  Args:
   logits: float array; Output of model in shape [batch, ..., num_classes].
   one_hot_targets: One hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

  Returns:
    The mean accuracy of the examples in the given batch as a scalar.
  """

  if logits.ndim != one_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(one_hot_targets.shape)))

  preds = jnp.argmax(logits, axis=-1)
  targets = jnp.argmax(one_hot_targets, axis=-1)
  correct = jnp.equal(preds, targets)

  if weights is not None:
    correct = apply_weights(correct, weights)

  if weights is None:
    normalization = np.prod(one_hot_targets.shape[:-1])
  else:
    normalization = weights.sum()
  return jnp.sum(correct), normalization


def weighted_binary_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy over the given batch.

  This computes the accuracy over a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.
  We assume the trainer will aggregate and divide by number of samples.

  Args:
   logits: float array; Output of model in shape [batch, ..., 1].
   targets: float array; Target labels of shape [batch, ..., 1].
   weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

  Returns:
    The mean accuracy of the examples in the given batch as a scalar.
  """

  if logits.ndim != targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(targets.shape)))

  logits = nn.sigmoid(logits)
  preds = logits > 0.5
  correct = jnp.equal(preds, targets)

  if weights is not None:
    correct = apply_weights(correct, weights)

  if weights is None:
    normalization = np.prod(targets.shape[:-1])
  else:
    normalization = weights.sum()
  return jnp.sum(correct), normalization


def weighted_top_one_accuracy(logits, multi_hot_targets, weights=None):
  """Compute weighted number of correctly classified, given top 1 class.

  This computes the weighted number of correctly classified examples/pixels in a
  single, potentially padded minibatch, given top-one prediction. If the
  minibatch/inputs is padded (i.e., it contains null examples/pad pixels) it is
  assumed that weights is a binary mask where 0 indicates that the example/pixel
  is null/padded. We assume the trainer will aggregate and divide by number of
  samples.

  Args:
   logits: float array; Output of model in shape [batch, ..., num_classes].
   multi_hot_targets: Multi hot vector of shape [batch, ..., num_classes].
   weights: None or array of shape [batch, ...] (rank of one_hot_targets -1).

  Returns:
    The number of correctly classified examples in the given batch, given top
    one prediction.
  """
  if logits.ndim != multi_hot_targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s multi_hot_targets' %
        (str(logits.shape), str(multi_hot_targets.shape)))

  top1_idx = jnp.argmax(logits, axis=-1)
  # extracts the label at the highest logit index for each inputs
  top1_correct = jnp.ceil(
      jnp.take_along_axis(multi_hot_targets, top1_idx[:, None], axis=1)[:, 0])
  if weights is not None:
    top1_correct = apply_weights(top1_correct, weights)

  if weights is None:
    normalization = np.prod(multi_hot_targets.shape[:-1])
  else:
    normalization = weights.sum()

  return jnp.sum(top1_correct), normalization


def apply_weights(output, weights):
  """Applies given weights of the inputs in the minibatch to outputs.

  Note that weights can be per example (i.e. of shape `[batch_size,]`) or per
  pixel/token (i.e. of shape `[batch_size, height, width]` or
  `[batch_size, len]`) so we need to broadcast it to the output shape.

  Args:
    output: nd-array; Computed output, which can be loss or the correctly
      classified examples, etc.
    weights: nd-array; Weights of inputs in the batch, which can be None or
      array of shape [batch, ...].

  Returns:
    weighted output.
  """
  desired_weights_shape = weights.shape + (1,) * (output.ndim - weights.ndim)
  weights = jax.lax.broadcast_in_dim(
      weights,
      shape=desired_weights_shape,
      broadcast_dimensions=tuple(range(weights.ndim)))
  # scale the outpus with weights
  return output * weights


def apply_label_smoothing(one_hot_targets, label_smoothing):
  """Apply label smoothing to the one-hot targets.

  Applies label smoothing such that the on-values are transformed from 1.0 to
  `1.0 - label_smoothing + label_smoothing / num_classes`, and the off-values
  are transformed from 0.0 to `label_smoothing / num_classes`.
  https://arxiv.org/abs/1512.00567

  Note that another way of performing label smoothing (which we don't use here)
  is to take `label_smoothing` mass from the on-values and distribute it to the
  off-values; in other words, transform the on-values to `1.0 - label_smoothing`
  and the  off-values to `label_smoothing / (num_classes - 1)`.
  http://jmlr.org/papers/v20/18-789.html


  Args:
    one_hot_targets: One-hot targets for an example, a [batch, ..., num_classes]
      float array.
    label_smoothing: float; A scalar in [0, 1] used to smooth the labels.

  Returns:
    A float array of the same shape as `one_hot_targets` with smoothed label
    values.
  """
  on_value = 1.0 - label_smoothing
  num_classes = one_hot_targets.shape[-1]
  off_value = label_smoothing / num_classes
  one_hot_targets = one_hot_targets * on_value + off_value
  return one_hot_targets


@jax.vmap
def sigmoid_hinge_loss(logits, targets):
  """Computes hinge loss given predictions and labels.

  Args:
    logits: float array; Output of model in shape `[ ..., num_classes]`.
    targets: int array; Labels with shape  `[..., num_classes]`.

  Returns:
    Loss value.
  """
  probs = nn.sigmoid(logits)
  loss = jnp.sum(jnp.maximum(0, 1. - jnp.multiply(probs, targets)), axis=-1)

  return loss


@jax.vmap
def softmax_hinge_loss(logits, targets):
  """Computes hinge loss given predictions and labels.

  Args:
    logits: float array; Output of model in shape `[ ..., num_classes]`.
    targets: int array; Labels with shape  `[..., num_classes]`.

  Returns:
    Loss value.
  """
  probs = nn.softmax(logits, axis=-1)
  loss = jnp.sum(jnp.maximum(0, 1. - jnp.multiply(probs, targets)), axis=-1)

  return loss


@jax.vmap
def categorical_cross_entropy_loss(logits, one_hot_targets):
  return -jnp.sum(one_hot_targets * jax.nn.log_softmax(logits), axis=-1)


@jax.vmap
def sigmoid_cross_entropy_loss(logits, targets):
  """Sigmoid cross entropy (A.K.A.

  binary cross entropy).

  This implementation is numerically stable and this loss function can be used
  in case of binary classification or multi-class classification.

  Args:
    logits: float array; Output of model in shape `[ ..., num_classes]`.
    targets: int array; Labels with shape  `[..., num_classes]`.

  Returns:
    Loss value (its shape depends on the inputs shapes.)
  """

  # Log(p(y)):
  log_p = jax.nn.log_sigmoid(logits)

  # Log(1 - P(y))
  log_not_p = jax.nn.log_sigmoid(-logits)

  return -jnp.sum(targets * log_p + (1. - targets) * log_not_p, axis=-1)


def weighted_cross_entropy_loss(logits,
                                one_hot_targets,
                                weights=None,
                                label_smoothing=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  This computes sum_(x,y) ce(x, y) for a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    one_hot_targets: int array; Target labels of shape [batch, ...,num_classes].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: float scalar to use to smooth the one-hot labels.

  Returns:
    The mean cross entropy of the examples in the given batch as a scalar.
  """

  return weighted_loss(categorical_cross_entropy_loss, logits, one_hot_targets,
                       weights, label_smoothing)


def weighted_sigmoid_cross_entropy_loss(logits,
                                        targets,
                                        weights=None,
                                        label_smoothing=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  This computes sum_(x,y) ce(x, y) for a single, potentially padded minibatch.
  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    targets: int array; Target labels of shape [batch, 1].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: float scalar to use to smooth the one-hot labels.

  Returns:
    The mean cross entropy of the examples in the given batch as a scalar.
  """
  return weighted_loss(sigmoid_cross_entropy_loss, logits, targets, weights,
                       label_smoothing)


def weighted_loss(loss_fn, logits, targets, weights=None, label_smoothing=None):
  """Compute weighted loss for log probs and targets.

  If the minibatch is padded (that is it contains null examples) it is assumed
  that weights is a binary mask where 0 indicates that the example is null.

  Args:
    loss_fn: fn; Main loss function to apply: ` loss_value = loss_fn(logits,
      targets)`
    logits: float array; Output of model in shape [batch, ..., num_classes].
    targets: int array; Target labels of shape [batch, 1].
    weights: None or array of shape [batch x ...] (rank of one_hot_targets -1).
    label_smoothing: float scalar to use to smooth the one-hot labels.

  Returns:
    The mean cross entropy of the examples in the given batch as a scalar.
  """
  if logits.ndim != targets.ndim:
    raise ValueError(
        'Incorrect shapes. Got shape %s logits and %s one_hot_targets' %
        (str(logits.shape), str(targets.shape)))

  # Optionally apply label smoothing.
  if label_smoothing is not None:
    targets = apply_label_smoothing(targets, label_smoothing)

  loss = loss_fn(logits, targets)
  if weights is not None:
    if weights.ndim != targets.ndim - 1:
      raise ValueError(
          'Incorrect shapes. Got shape %s weights and %s one_hot_targets' %
          (str(weights.shape), str(targets.shape)))
    loss = loss * weights

  if weights is None:
    normalization = np.prod(targets.shape[:-1])
  else:
    normalization = weights.sum()

  return jnp.sum(loss), normalization


def l2_regularization(params, include_bias_terms=False):
  """Calculate the L2 loss (square L2 norm), given parameters of the model.

  Args:
    params: pytree; Parameters of the model.
    include_bias_terms: bool; If true apply L2 on bias terms as well.

  Returns:
    l2 norm.

  """
  weight_penalty_params = jax.tree_leaves(params)
  if include_bias_terms:
    dim_th = 0
  else:
    dim_th = 1
  return sum([jnp.sum(x**2) for x in weight_penalty_params if x.ndim > dim_th])


def irm_env_penalty(logits,
                    targets,
                    weights=None,
                    loss_fn=weighted_cross_entropy_loss):
  """Computes penalty term of the IRM loss for a given environment (batch).

  Reference: [Invariant Risk Minimmization](https://arxiv.org/abs/1907.02893)

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    targets: int array; One hot targets.
    weights: float array; Weights of the examples in the given batch.
    loss_fn: fn(logits, targetsm weights); Loss function with respect to which
      the penalty is computed.

  Returns:
    A factor indicating how much the model can improve on the given batch.
  """

  def _loss(scale, logits, one_hot_targets, weights):
    loss_val, loss_norm = loss_fn(logits * scale, one_hot_targets, weights)
    return jax.lax.cond(
        loss_norm > 0,
        lambda _: loss_val / loss_norm,
        lambda _: 0.0,
        operand=None)

  scale = jnp.ones(logits.shape[-1])
  grad = jax.grad(_loss, argnums=0)(scale, logits, targets, weights)

  return jnp.sum(grad**2), 1


def binary_irm_env_penalty(logits,
                           targets,
                           weights=None,
                           loss_fn=weighted_sigmoid_cross_entropy_loss):
  """Computes penalty term of the IRM loss for a given environment (batch).

  Reference: [Invariant Risk Minimmization](https://arxiv.org/abs/1907.02893)

  Args:
    logits: float array; Output of model in shape [batch, ..., num_classes].
    targets: int array; One hot targets.
    weights: float array; Weights of the examples in the given batch.
    loss_fn: fn(logits, targetsm weights); Loss function with respect to which
      the penalty is computed.

  Returns:
    A factor indicating how much the model can improve on the given batch.
  """

  def _loss(scale, logits, one_hot_targets, weights):
    loss_val, loss_norm = loss_fn(logits * scale, one_hot_targets, weights)
    return jax.lax.cond(
        loss_norm > 0,
        lambda _: loss_val / loss_norm,
        lambda _: 0.0,
        operand=None)

  scale = jnp.ones(logits.shape[-1])
  grad = jax.grad(_loss, argnums=0)(scale, logits, targets, weights)

  return jnp.sum(grad**2), 1


def l2_distance(x, y):
  assert x.shape == y.shape, 'x.shape != y.shape.'

  x = x.reshape((-1, x.shape[-1]))
  y = y.reshape((-1, y.shape[-1]))

  dist = x - y
  return jnp.sum(dist**2)


def cosine_distance(x, y):
  """Computes the normalized dot product (cosine) distance between x and y.

  This methods assumes there is a one-one mapping between rows of x and y.

  dot product distance =  1 - dot product similarity

  Args:
    x: float matrix; [..., num_of_features].
    y: float matrix; [..., num_of_features].

  Returns:
    A float scalar that reflects the distance between x and y.
  """
  # TODO(samiraabnar): What is the right way of using dot product similarity
  # to compute the distance between parameters of two models (e.g. what is
  # the distance in case of similarity==-1).

  assert x.shape == y.shape, 'x.shape != y.shape.'

  # [num_examples, num_features]
  x = x.reshape((-1, x.shape[-1]))
  y = y.reshape((-1, y.shape[-1]))

  norm_y = jnp.linalg.norm(y, axis=-1)[Ellipsis, None]
  norm_x = jnp.linalg.norm(x, axis=-1)[Ellipsis, None]
  nnorm_y = jnp.maximum(norm_y, 1e-32)
  nnorm_x = jnp.maximum(norm_x, 1e-32)

  dot_sim = jnp.sum(x / nnorm_x * y / nnorm_y, axis=-1)

  # Similarity is in [-1,1], distance should be in [0, 1] so to compute distance
  # we do (1 - similarity)/2.
  # If similarity==1: distance=0 and if similarity ==-1: distance=1.
  # Also, jnp.maximum(1.0 - jnp.ceil(norm_x + norm_y), 0) will return 1 if
  # norm_x == norm_y == 0 and 0 otherwise. We add this to dot_sim, so if both
  # vectors are zero, their similarity is 1.
  return (1.0 -
          jnp.mean(jnp.maximum(1.0 - jnp.ceil(norm_x + norm_y), 0) + dot_sim))


def parameter_distance(params, base_params, norm_factor, mode):
  """Computes distance between the parameters of two models.

  Assumptions: models have exactly the same architecture.

  Args:
    params: dict; parameters of a model (model 1)
    base_params: dict; parameters of the base model (model 2).
    norm_factor: float; A scalar used to scale the computed distance.
    mode: str; Determines what distance measure to use.

  Returns:
    A scalar reflecting distance between the given models.
  """
  if mode == 'dot':
    params_dists = jax.tree_multimap(cosine_distance, params, base_params)
  elif mode == 'l2':
    params_dists = jax.tree_multimap(l2_distance, params, base_params)
  else:
    raise ValueError('The specified parameter distance mode is not valid.')

  return norm_factor * jnp.mean(jnp.array(jax.tree_leaves(params_dists)))


def mean_logits_entropy(logits, one_hot_targets, weights=None, axis=-1):
  """Computes the entropy of the logits.

  Args:
    logits: jnp.array(float); Logits of shape [batch_size, ..., num_of_classes]
    one_hot_targets: unused (just here to have a similar signature with other
      metrics).
    weights: jnp.array(float); weight of each example with shape [batch_size,
      ...]
    axis: int; Specifies the axis to sum over (default is -1).

  Returns:
    mean entropy of the logits.
  """
  del one_hot_targets

  probs = jax.nn.softmax(logits)
  entropies = jax.numpy.sum(jax.scipy.special.entr(probs), axis=axis)

  if weights is not None:
    entropies = apply_weights(entropies, weights)

  if weights is not None:
    normalization = weights.sum()
  else:
    shape = logits.shape
    normalization = np.prod(shape[:axis] + shape[axis + 1:])

  return jnp.sum(entropies), normalization


def mean_confidence(not_normalized_logits,
                    one_hot_targets,
                    weights=None,
                    axis=-1):
  """Computes the confidence of the logits.

  Args:
    not_normalized_logits: jnp.array(float); Not normalized logits of shape
      [batch_size, ..., num_of_classes]
    one_hot_targets: unused (just here to have a similar signature with other
      metrics).
    weights: jnp.array(float); weight of each example with shape [batch_size,
      ...]
    axis: int; Specifies the axis to sum over (default is -1).

  Returns:
    mean entropy of the logits.
  """
  del one_hot_targets

  probs = jax.nn.softmax(not_normalized_logits, axis=axis)
  max_prob = jnp.max(probs, axis=axis)
  min_prob = jnp.min(probs, axis=axis)
  confidences = max_prob - min_prob

  if weights is not None:
    confidences = apply_weights(confidences, weights)
  if weights is not None:
    normalization = weights.sum()
  else:
    shape = not_normalized_logits.shape
    normalization = np.prod(shape[:axis] + shape[axis + 1:])

  return jnp.sum(confidences), normalization
