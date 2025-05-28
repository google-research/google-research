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

"""Base losses in jax."""

import enum
from typing import Callable, Tuple, Union, Dict, Text, Any, Sequence

import flax
from flax import traverse_util
import gin
import jax
import jax.numpy as jnp
import numpy as np

from utils import gin_utils
from utils.types import Array

ArrayDict = Dict[Text, Array]
FrozenDict = flax.core.frozen_dict.FrozenDict
LossArray = Union[Array, ArrayDict]


@gin.configurable
@enum.unique
class LossReductionType(enum.IntEnum):
  """Reduction type for the loss as defined in TF."""
  MEAN = 0
  SUM = 1
  SUM_BY_NONZERO_WEIGHTS = 2
  NONE = 3
  RETURN_AS_IS = 4


EPSILON = 1e-7
_SIGMOID_EPSILON = 1e-20


def check_shape_equal(pred, labels):
  """Verifies prediction and label shapes are equal.

  Args:
    pred: An array of predictions or logits.
    labels: An array of labels.

  Raises:
    ValueError: Pred and labels do not have the same shape.
  """
  if pred.shape != labels.shape:
    raise ValueError('Prediction and labels shapes must be equal:'
                     f'{pred.shape} vs {labels.shape}.')


def check_dtype_equal(input_dict,
                      target_dtype = jnp.float32,
                      exclude_list = ()):
  """Verifies the nested dictionary dtypes are equal to target dtype.

  Args:
    input_dict: A nested dictionary of jnp.ndarray (pytree) to verify types.
    target_dtype: A target dtype.
    exclude_list: A sequence of top-level key names to exclude type checking.

  Raises:
    TypeError: An element of the dictionary has unexpected type.
  """
  flat_input = traverse_util.flatten_dict(input_dict)
  for key, value in flat_input.items():
    if key[0] in exclude_list:
      continue

    key_name = '_'.join([str(sub_key) for sub_key in key])
    if isinstance(value, jnp.ndarray):
      if value.dtype != target_dtype:
        raise TypeError(f'Input {key_name} has inconsistent type:'
                        f'{value.dtype} vs {target_dtype}')
    else:
      raise TypeError(f'Illegal input type found: {type(value)}.')


def compute_weighted_loss(
    loss,
    weights,
    dtype,
    loss_reduction,
):
  """Weights and reduces the loss.

  We convert to float32 before reducing following TF1 implementation.

  After weighting and reducing the losses, we convert the output back to the
  dtype of the input.

  Args:
    loss: an array of loss.
    weights: An array or scalar which must be broadcastable to logits and labels
      shape.
    dtype: loss output data type.
    loss_reduction: A loss reduction method as in the Tensorflow implementation.
      Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
      NotImplementedError if other values are provided.

  Returns:
    loss: a scalar of weighted and reduced loss.

  Raises:
    NotImplementedError: loss reduction type is undefined.
  """
  if loss_reduction == LossReductionType.RETURN_AS_IS:
    # Handle no loss reduction, by returning tensor as-is.
    return loss
  loss = loss.astype(jnp.float32)
  loss_weight = jnp.broadcast_to(weights, loss.shape).astype(jnp.float32)
  loss *= loss_weight
  total_loss = jnp.sum(loss)

  if loss_reduction == LossReductionType.SUM_BY_NONZERO_WEIGHTS:
    total_loss = safe_divide(total_loss, jnp.sum(loss_weight != 0.0))
  elif loss_reduction == LossReductionType.MEAN:
    total_loss = safe_divide(total_loss, jnp.sum(loss_weight))
  elif loss_reduction != LossReductionType.SUM:
    raise NotImplementedError('LossReductionType not supported for this loss:'
                              f'{loss_reduction}.')

  return total_loss.astype(dtype)


def safe_divide(x,
                y,
                rtol = 1e-5,
                atol = 1e-8):
  """Computes a safe divide which returns 0 if the denominator is zero.

  Reference:
  https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
  Args:
    x: A float of numerator.
    y: A float of denominator.
    rtol: The relative tolerance parameter. See numpy.isclose for more info.
    atol: The absolute tolerance parameter. See numpy.isclose for more info.

  Returns:
    z: output x / y or 0.
  """
  is_zero = jnp.isclose(y, 0.0, rtol=rtol, atol=atol)
  safe_y = jnp.where(is_zero, jnp.ones_like(y), y)
  return jnp.where(is_zero, jnp.zeros_like(x), x / safe_y)


def safe_log(x):
  """Computes a safe log.

  This function returns 0.0 wherever x contains any value <= 0.0.

  Args:
    x: Input array.

  Returns:
    log(x) or 0.0 depending on the value of input x.
  """
  safe_x = jnp.where(x > 0.0, x, jnp.ones_like(x))
  return jnp.where(x > 0.0, jnp.log(safe_x), jnp.zeros_like(x))


@gin.configurable
@gin_utils.allow_remapping
def sigmoid_cross_entropy(logits,
                          labels,
                          weights = 1.0,
                          loss_reduction = LossReductionType
                          .SUM_BY_NONZERO_WEIGHTS,
                          **kwargs):
  """Returns the sigmoid cross entropy loss.

  Implements:
  loss = label * (-1) * log(pred) + (1 — label) * (-1) * log(1 — pred).

  Please note: the default for TF is SUM_BY_NONZERO_WEIGHTS loss reduction.

  Args:
    logits: An array of shape of [batch, ..., num_classes].
    labels: An array of shape of [batch, ..., num_classes].
    weights: An array or scalar which must be broadcastable to logits and labels
      shape.
    loss_reduction: A loss reduction method as in the Tensorflow implementation.
      Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
      NotImplementedError if other values are provided.
    **kwargs: additional keyword arguments.

  Returns:
    A scalar loss
  """
  del kwargs
  check_shape_equal(logits, labels)
  labels = labels.astype(logits.dtype)
  logits = jax.nn.log_sigmoid(logits)
  loss = -labels * logits - (1. - labels) * jnp.log(
      jnp.maximum(-jnp.expm1(logits), _SIGMOID_EPSILON))

  return compute_weighted_loss(loss, weights, logits.dtype, loss_reduction)


@gin.configurable
@gin_utils.allow_remapping
def softmax_cross_entropy(logits,
                          labels,
                          label_smoothing = 0.0,
                          weights = 1.0,
                          loss_reduction = LossReductionType
                          .SUM_BY_NONZERO_WEIGHTS,
                          **kwargs):
  """Returns the softmax cross entropy loss.

  Please note: the default for TF is SUM_BY_NONZERO_WEIGHTS loss reduction.

  Args:
    logits: An array of shape of [batch, ..., num_classes].
    labels: An array of shape of [batch, ..., num_classes] of values betwwen [0,
      1]
    label_smoothing: how much label smoothing to apply, which smoothes out the
      label matrix. The new labels will be (1 - label_smoothing) * labels +
      label_smoothing / num_classes
    weights: A scalar or an array of shape [batch] for weighting the loss per
      example.
    loss_reduction: A loss reduction method as in the Tensorflow implementation.
      Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
      NotImplementedError if other values are provided.
    **kwargs: additional keyword arguments.

  Returns:
    A scalar loss.
  """
  del kwargs
  check_shape_equal(logits, labels)

  labels = labels.astype(logits.dtype)
  if label_smoothing > 0:
    num_classes = labels.shape[-1]
    smooth_weight = label_smoothing / num_classes
    smooth_weight = jnp.array(smooth_weight, dtype=logits.dtype)
    labels = (1. - label_smoothing) * labels + smooth_weight

  logits = jax.nn.log_softmax(logits)
  loss = -labels * logits
  loss = jnp.sum(loss, axis=-1)

  return compute_weighted_loss(loss, weights, logits.dtype, loss_reduction)


@gin.configurable
@gin_utils.allow_remapping
def weighted_softmax_cross_entropy(
    logits,
    labels,
    label_smoothing = 0.0,
    weights = 1.0,
    loss_reduction = LossReductionType
    .SUM_BY_NONZERO_WEIGHTS,
    background_weight = 1.0,
    **kwargs):
  """Returns the softmax cross entropy loss with background loss adjustment.

  Please note: the default for TF is SUM_BY_NONZERO_WEIGHTS loss reduction.

  Args:
    logits: An array of shape of [batch, ..., num_classes].
    labels: An array of shape of [batch, ..., num_classes] of values betwwen [0,
      1]
    label_smoothing: how much label smoothing to apply, which smoothes out the
      label matrix. The new labels will be (1 - label_smoothing) * labels +
      label_smoothing / num_classes
    weights: A scalar or an array of shape [batch] for weighting the loss per
      example.
    loss_reduction: A loss reduction method as in the Tensorflow implementation.
      Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
      NotImplementedError if other values are provided.
    background_weight: A float to adjust the weights of background. Default
      1.0 is a no-op.
    **kwargs: additional keyword arguments.

  Returns:
    A scalar loss.
  """
  del kwargs
  check_shape_equal(logits, labels)

  labels = labels.astype(logits.dtype)
  if label_smoothing > 0:
    num_classes = labels.shape[-1]
    smooth_weight = label_smoothing / num_classes
    smooth_weight = jnp.array(smooth_weight, dtype=logits.dtype)
    labels = (1. - label_smoothing) * labels + smooth_weight

  logits = jax.nn.log_softmax(logits)
  loss = -labels * logits

  # Apply background class weights
  class_weights = np.ones(loss.shape)
  class_weights[Ellipsis, :1] = background_weight  # Background is class 0.
  loss = loss * jnp.array(class_weights)

  loss = jnp.sum(loss, axis=-1)
  return compute_weighted_loss(loss, weights, logits.dtype, loss_reduction)


@gin.configurable
@gin_utils.allow_remapping
def onehot_cross_entropy_loss(
    logits,
    labels,
    loss_reduction = LossReductionType
    .SUM_BY_NONZERO_WEIGHTS,
    **kwargs):
  """Computes the cross entropy loss between logits and the actual labels.

  Converts the labels into one hot and calls softmax_cross_entropy function to
  compute the loss

  Args:
    logits: A float array representing the class prediction for each box with a
      shape of [batch_size, num_tokens, num_classes].
    labels: A float array representing int label for each token [batch_size,
      num_tokens]
    loss_reduction: A loss reduction method as in the Tensorflow implementation.
    **kwargs: additional keyword arguments.

  Returns:
    loss: A scalar representing total loss.
  """
  del kwargs
  vocab_size = logits.shape[-1]
  labels_one_hot = jax.nn.one_hot(labels.astype(jnp.int32), vocab_size)
  weights = jax.numpy.where(labels > 0, 1, 0)
  return softmax_cross_entropy(
      logits,
      labels_one_hot,
      weights=weights,
      loss_reduction=loss_reduction)


def l1_loss(predictions,
            labels,
            weights = 1.0,
            loss_reduction = LossReductionType.SUM_BY_NONZERO_WEIGHTS,
            **kwargs):
  """L1 loss.

  Args:
    predictions: an array of shape [batch, ..., d] containing model predictions.
    labels: an array of shape [batch, ..., d] containing ground truth.
    weights: A scalar or an array of shape [batch, ...] for weighting the loss
      per example.
    loss_reduction: a loss reduction method.
    **kwargs: additional keyword arguments.

  Returns:
    the L1 loss averaged over batch.
  """
  del kwargs  # Unused
  check_shape_equal(predictions, labels)
  l1 = jnp.sum(jnp.abs(predictions - labels), axis=-1)
  return compute_weighted_loss(
      l1,
      weights=weights,
      loss_reduction=loss_reduction,
      dtype=predictions.dtype)


def l2_loss(predictions,
            labels,
            weights = 1.0,
            loss_reduction = LossReductionType.SUM_BY_NONZERO_WEIGHTS,
            **kwargs):
  """L2 loss.

  Args:
    predictions: An array of shape [batch, ..., d] containing model predictions.
    labels: An array of shape [batch, ..., d] containing ground truth.
    weights: A scalar or an array of shape [batch, ...] for weighting the loss
      per example.
    loss_reduction: A loss reduction method.
    **kwargs: additional keyword arguments.

  Returns:
    the L2 loss averaged over batch.
  """
  del kwargs  # Unused
  check_shape_equal(predictions, labels)
  l2 = jnp.sum(jnp.square(predictions - labels), axis=-1)
  return compute_weighted_loss(
      l2,
      weights=weights,
      loss_reduction=loss_reduction,
      dtype=predictions.dtype)


def cosine_loss(predictions,
                labels,
                weights = 1.0,
                loss_reduction = LossReductionType.SUM_BY_NONZERO_WEIGHTS,
                **kwargs):
  """Cosine loss.

  This loss computes the dot product between predictions and labels as loss.
  The value ranges from [0, 2.0] depending on the alignment of prediction and
  label vectors. This loss can be used when we want to optimize the alignment
  of the vectors directly.

  Args:
    predictions: An array of shape [batch, ..., d] containing model predictions.
      The predictions need to be normalized in the last dimension.
    labels: An array of shape [batch, ..., d] containing ground truth.
      The labels need to be normalized in the last dimension.
    weights: A scalar or an array of shape [batch, ...] for weighting the loss
      per example.
    loss_reduction: A loss reduction method.
    **kwargs: additional keyword arguments.

  Returns:
    The cosine loss averaged over batch.
  """
  del kwargs  # Unused
  check_shape_equal(predictions, labels)
  cosine = 1.0 - jnp.sum(predictions * labels, axis=-1)
  return compute_weighted_loss(
      cosine,
      weights=weights,
      loss_reduction=loss_reduction,
      dtype=predictions.dtype)


@gin.configurable
@gin_utils.allow_remapping
def huber_loss(predictions,
               labels,
               weights = 1.0,
               delta = 1.0,
               loss_reduction = LossReductionType.SUM_BY_NONZERO_WEIGHTS,
               **kwargs):
  """Returns the Huber loss.

  Huber loss is computed as:
    0.5 * x^2                  if |x| <= d
    0.5 * d^2 + d * (|x| - d)  if |x| > d
  where x is the difference between labels and predictions.

  Args:
    predictions: An array of shape of [batch, num_channels].
    labels: An array of shape of [batch, num_channels].
    weights: A scalar or an array of shape [batch] for weighting the loss per
      example.
    delta: A range at which the function changes from quadratic to linear.
    loss_reduction: A loss reduction method as in the Tensorflow implementation.
      Currently supports SUM_BY_NONZERO_WEIGHTS, MEAN and SUM. Raises
      NotImplementedError if other values are provided.
    **kwargs: additional keyword arguments.

  Returns:
    A scalar loss.
  """
  del kwargs
  check_shape_equal(predictions, labels)
  labels = labels.astype(predictions.dtype)
  x = labels - predictions

  # Apply the formula above.
  loss = jnp.where(
      jnp.abs(x) <= delta, 0.5 * jax.lax.square(x),
      0.5 * delta * delta + delta * (jnp.abs(x) - delta))

  return compute_weighted_loss(loss, weights, predictions.dtype, loss_reduction)


@gin.configurable
def weight_decay_loss_wrapper(
    loss_fn = gin.REQUIRED,
    factor = gin.REQUIRED,
    exclude = (),
):
  """A wrapper to add weight decay to underlying loss function.

  Use this wrapper if the weight decay in the optimizer is not suitable. For
  example, if you need to exclude some parameters from decay loss.

  Args:
    loss_fn: The underlying loss function which accepts two args: outputs - A
      dictionary of outputs. labels - A dictionary of groundtruth labels.
    factor: A floating point to specify weight decay factor.
    exclude: A sequence of strings to use to filter out parameters.

  Returns:
    The wrapped loss function with weight decay added which accepts three args:
      outputs - A dictionary of outputs.
      labels - A dictionary of groundtruth labels.
      params - A frozen dictionary of parameters (pytree).
      **kwargs - Any additional arguments.
  """
  traversal = traverse_util.ModelParamTraversal(
      lambda path, _: all([e not in path for e in exclude]))

  def wrapped_loss(outputs, *args, params, **kwargs):
    losses = loss_fn(outputs, *args, **kwargs)
    weight_decay_params = list(traversal.iterate(params))
    weight_l2 = sum([jnp.sum(x**2) for x in weight_decay_params])
    weight_penalty = factor * 0.5 * weight_l2

    if isinstance(losses, dict):
      if 'model_loss' not in losses:
        raise ValueError(
            'Losses must contain `model_loss` key as total model loss.')
      losses['pre_weight_penalty_model_loss'] = losses['model_loss']
      losses['model_loss'] += weight_penalty
      losses['l2_regularization_loss'] = weight_penalty
    elif isinstance(losses, jnp.ndarray):
      losses += weight_penalty
    else:
      raise ValueError('Encountered invalid loss type: ', type(losses))

    return losses

  return wrapped_loss
