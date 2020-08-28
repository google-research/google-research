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
"""Utils related to Flax models."""

import collections
import functools
import operator as op
import pprint
import time

from absl import logging
from flax import jax_utils
from flax import nn
from flax import optim
from flax.training import common_utils
import gin
import jax
from jax import lax
import jax.nn
import jax.numpy as jnp
import numpy as np
import tree


def l2_norm(params):
  return jax.tree_util.tree_map(lambda x: jnp.sum(x**2), params)


def l2_regularization(params):
  """Computes l2 regularization term for parameters."""
  return jax.tree_util.tree_reduce(op.add, l2_norm(params))


@functools.partial(jax.jit, static_argnums=(1, 2))
def create_model_and_cache(rng, input_shape, model_def):
  """Create a model and cache definition.

  Args:
    rng: Init RNG.
    input_shape: Input shape.
    model_def: Model definition.

  Returns:
    Tuple of (model, cache_def)
  """
  # Create a cache object for efficient decoding.
  with nn.attention.Cache().mutate() as cache_def:
    _, model = model_def.create_by_shape(
        rng, [(input_shape, jnp.float32)], cache=cache_def)
  return model, cache_def


@functools.partial(jax.jit, static_argnums=(1, 2))
def create_model(rng, input_shape, model_def):
  """Create a model and cache definition.

  Args:
    rng: Init RNG.
    input_shape: Input shape.
    model_def: Model definition.

  Returns:
    Tuple of (model, cache_def)
  """
  _, model = model_def.create_by_shape(
      rng, [(input_shape, jnp.float32)], cache=None)
  return model


def create_adam_optimizer(model,
                          learning_rate,
                          weight_decay=0.0,
                          beta1=0.9,
                          beta2=0.98,
                          eps=1e-9,
                          pmap=True):
  """Create (optionally replicated) Adam optimizer for `model`."""
  optimizer_def = optim.Adam(
      learning_rate,
      beta1=beta1,
      beta2=beta2,
      eps=eps,
      weight_decay=weight_decay)
  optimizer = optimizer_def.create(model)
  if pmap:
    optimizer = jax_utils.replicate(optimizer)
  return optimizer


def compute_weighted_cross_entropy(logits, targets,
                                   token_weights=None,
                                   example_weights=None):
  """Compute weighted cross entropy and entropy for log probs and targets.

  The loss is assumed to be sum_i example_weights[i] * logprob[i], where
  i indexes elements in the batch.

  logprob[i] is the log probability of sequence i, which is a weighted
  average of per-token log probabilities with weights according
  to token_weights. Typically token_weights is a mask for whether tokens are
  padding or not.

  Maximum likelihood training sets example_weights[i] = 1.
  Training with a REINFORCE-style objective may set example_weights[i]
  to any positive or negative number.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   token_weights: None or array of shape [batch x length]
   example_weights: None or array of shape [batch_size]
  Returns:
    Tuple of scalar loss and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  onehot_targets = common_utils.onehot(targets, logits.shape[-1])
  loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
  normalizing_factor = onehot_targets.sum()
  if token_weights is not None:
    loss = loss * token_weights
    normalizing_factor = token_weights.sum()

  if example_weights is not None:
    loss = loss.sum(axis=1)
    loss *= example_weights

  return loss.sum(), normalizing_factor


def compute_weighted_mse(predictions, targets, weights):
  """Compute mean squared error between predictions and targets.

  Args:
   predictions: [batch, length, ...] float array.
   targets: float targets of same size as predictions.
   weights: weights of same shape as predictions.

  Returns:
    Scalar loss.
  """
  if predictions.shape != targets.shape:
    raise ValueError(
        f'Incorrect shapes. Got shape {predictions.shape} predictions'
        f' and {targets.shape} targets')
  per_position_loss = jnp.square(targets - predictions) * weights
  summed_loss = jnp.sum(per_position_loss)
  denominator = jnp.sum(weights)
  return summed_loss, denominator


def compute_weighted_accuracy(logits, targets, weights=None):
  """Compute weighted accuracy for log probs and targets.

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]

  Returns:
    Tuple of scalar accuracy and batch normalizing factor.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError('Incorrect shapes. Got shape %s logits and %s targets' %
                     (str(logits.shape), str(targets.shape)))
  loss = jnp.equal(jnp.argmax(logits, axis=-1), targets)
  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def get_normalized_matrix(domain, freq_dict):
  """Compute the normalized matrix for soft-accuracy computation.

  Args:
    domain: A Sequin domain which provides the ordered list of tokens.
    freq_dict: A dict of dicts containing pairs of frequencies. E.g. for
      computing the normalized matrix based on the Blosum matrix use
      freq_dict=pfam_utils.BLOSUM62_TABLE.to_dict().

  Returns:
    An array of shape (vocab_size, vocab_size) containing the matrix to be
      used for soft-accuracy computation.
  """
  matrix = np.zeros((domain.vocab_size, domain.vocab_size))
  for idx_1, token_1 in enumerate(domain.vocab.tokens):
    for idx_2, token_2 in enumerate(domain.vocab.tokens):
      if token_1 in freq_dict:
        if token_2 in freq_dict[token_1]:
          matrix[idx_1][idx_2] = freq_dict[token_1][token_2]
  matrix -= np.min(matrix)
  matrix /= np.max(matrix)
  return matrix


def compute_weighted_soft_accuracy(logits, targets, weights=None, matrix=None):
  """Compute weighted soft-accuracy for log probs and targets.

  Based on Section 3.4 in
    [ProGen](https://www.biorxiv.org/content/10.1101/2020.03.07.982272v2).

  Args:
   logits: [batch, length, num_classes] float array.
   targets: categorical targets [batch, length] int array.
   weights: None or array of shape [batch x length]
   matrix: [num_classes, num_classes] normalized matrix to use for soft-accuracy
    computation.

  Returns:
    Tuple of scalar soft-accuracy and batch normalizing factor.

  Raises:
    ValueError when the logits and targets have incorrect number of dimensions.
  """
  if logits.ndim != targets.ndim + 1:
    raise ValueError(f'Incorrect shapes. Got shape {logits.shape} for logits '
                     f'and {targets.shape} for targets.')

  # Compute hard accuracy.
  pred = np.argmax(logits, axis=-1)
  loss = np.equal(pred, targets).astype(np.float32)

  # Add matrix-based accuracy for incorrect predictions.
  if matrix is not None:
    matrix = matrix * (np.ones(len(matrix)) - np.eye(len(matrix)))
    loss_matrix = matrix[np.reshape(pred, [-1])]
    loss_matrix = np.transpose(loss_matrix)
    loss_matrix = loss_matrix[np.reshape(targets, [-1])]
    loss_matrix = np.reshape(np.diag(loss_matrix), pred.shape)
    loss += loss_matrix

  normalizing_factor = np.prod(logits.shape[:-1])
  if weights is not None:
    loss = loss * weights
    normalizing_factor = weights.sum()

  return loss.sum(), normalizing_factor


def _psum(target_tree, axis_name='batch'):
  return jax.tree_map(lambda x: lax.psum(x, axis_name), target_tree)


def compute_metrics(logits, labels, token_weights, example_weights=None):
  """Compute summary metrics with loss, accuracy, and normalizing factor."""
  loss, weight_sum = compute_weighted_cross_entropy(logits, labels,
                                                    token_weights,
                                                    example_weights)
  acc, _ = compute_weighted_accuracy(logits, labels, token_weights)
  metrics = {
      'loss': loss,
      'accuracy': acc,
      'denominator': weight_sum,
  }
  try:
    metrics = _psum(metrics)
  except NameError:
    pass  # We are not inside pmap. No need to psum.
  return metrics


def get_params(model):
  """Get model parameters."""
  return model.optimizer.target.params


def param_count(model):
  """Get total parameter count."""
  params = get_params(model)
  num_params = sum(x.size for x in jax.tree_leaves(params))
  return num_params


def param_pprint(model):
  """Pretty print parameter tree to stdout."""
  params = get_params(model)
  x = tree.map_structure(lambda x: x.size / 1024, params)
  as_str = pprint.pformat(x)
  return as_str


def param_reduce(model, log=False):
  """Return a dict containing param counts per submodule."""
  params = get_params(model)
  sizes = collections.defaultdict(int)
  for path, x in tree.flatten_with_path(params):
    size = x.size
    for i in range(len(path)):
      k = path[:i]
      sizes[k] += size
  for k in sorted(sizes):
    if log:
      logging.info('%s: %s', k, sizes[k])
  return sizes


def batchify(inputs, batch_size):
  """Reshapes and pads inputs to include an additional batch dimension.

  The inputs can be of arbitrary length. They length does not need to be a
  multiple of batch_size, in which case padding will be added.

  Args:
    inputs: An np.ndarray or iterable of np.ndarray of shape [input_size, ...].
    batch_size:
      The size of the batches to group the data into.
  Returns:
    batch_inputs: np.ndarray of size [num_batches, batch_size, ...],
    where num_batches is ceil(input_size / batch_size).
    pad_size: Number of examples in the final batch that are padding. We use
      copies of inputs[0] as padding.
  """

  inputs = np.asarray(inputs)

  pad_size = -len(inputs) % batch_size
  if pad_size:
    padding = np.tile(inputs[:1], [pad_size, 1])
    padded_inputs = np.concatenate([inputs, padding], axis=0)
  else:
    padded_inputs = inputs
  batched_shape = (-1, batch_size) + padded_inputs.shape[1:]
  batched_inputs = np.reshape(padded_inputs, batched_shape)
  return batched_inputs, pad_size


def batch_apply(fn, inputs, batch_size):
  """Applies fn() to inputs in batches of size batch_size.

  The inputs can be of arbitrary length. They length does not need to be a
  multiple of batch_size. Padding will be added (and then removed) such that
  fn() is always called on inputs of size exactly batch_size. fn() is assumed
  to operate independently across the batch dimension of its inputs (e.g.
  computing predictions of a model on inputs) instead of performing an operation
  where the batch elements interact (e.g. performing a gradient step of a model
  on a batch of inputs).

  Args:
    fn: The function to map across the inputs.
    inputs: An np.ndarray or iterable of np.ndarray. fn() is mapped
      along the first dimension.
    batch_size:
      The size of the batches to evaluate fn() on.
  Returns:
    np.ndarray where outputs[i] = fn(inputs[i])
  """

  batched_inputs, pad_size = batchify(inputs, batch_size)
  results = np.concatenate([fn(batch) for batch in batched_inputs])
  if pad_size:
    results = results[:-pad_size]
  return results


@gin.configurable
def create_learning_rate_scheduler(
    factors='constant * linear_warmup * rsqrt_decay',
    base_learning_rate=0.5,
    warmup_steps=8000,
    decay_factor=0.5,
    steps_per_decay=20000,
    steps_per_cycle=100000):
  """Creates learning rate schedule.

  Interprets factors in the factors string which can consist of:
  * constant: interpreted as the constant value,
  * linear_warmup: interpreted as linear warmup until warmup_steps,
  * rsqrt_decay: divide by square root of max(step, warmup_steps)
  * decay_every: Every k steps decay the learning rate by decay_factor.
  * cosine_decay: Cyclic cosine decay, uses steps_per_cycle parameter.

  Args:
    factors: A string with factors separated by '*' that defines the schedule.
    base_learning_rate: Float, the starting constant for the lr schedule.
    warmup_steps: How many steps to warm up for in the warmup schedule.
    decay_factor: The amount to decay the learning rate by.
    steps_per_decay: How often to decay the learning rate.
    steps_per_cycle: Steps per cycle when using cosine decay.

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


class Timer(object):
  """Context manager for logging timing.

  Example usage:
    with Timer('my function'):
      my_function(inputs)

  Attributes:
    elapsed: The time in seconds that it took to execute the context.
  """

  def __init__(self, message=None, verbose=True):
    """Creates and instance of this class.

    Args:
      message: The message to be used for logging. If `None`, does not log.
      verbose: Whether to log messages to the console.
    """
    self._message = message
    self._elapsed = None
    self._verbose = verbose

  def _log(self, msg, *args, **kwargs):
    if self._message and self._verbose:
      logging.info(msg, *args, **kwargs)
      logging.flush()

  def __enter__(self):
    self._log('Starting: %s', self._message)
    self._elapsed = None
    self._start = time.time()
    return self

  def __exit__(self, *args):
    self._elapsed = time.time() - self._start
    self._log('Finished: %s. Elapsed seconds: %f', self._message, self._elapsed)

  @property
  def elapsed(self):
    if self._elapsed is None:
      raise ValueError('Timer not executed!')
    return self._elapsed


def get_random_state(seed_or_state):
  """Returns a np.random.RandomState given an integer seed or RandomState."""
  if isinstance(seed_or_state, int):
    return np.random.RandomState(seed_or_state)
  elif seed_or_state is None:
    # This returns the current global np random state.
    return np.random.random.__self__
  elif not isinstance(seed_or_state, np.random.RandomState):
    raise ValueError('Numpy RandomState or integer seed expected! Got: %s' %
                     seed_or_state)
  else:
    return seed_or_state
