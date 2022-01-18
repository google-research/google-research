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

"""Linear eval using L-BFGS."""
import functools

from absl import logging
import frozendict
import jax
from jax import numpy as jnp
import numpy as onp
from tensorflow_probability.substrates import jax as tfp


def reshape_and_pad_data_for_devices(data):
  """Reshape/pad data so that leading dimension equals number of local devices.

  Args:
    data: Tree containing arrays of data.

  Returns:
    data: Tree containing arrays with a new leading dimension that is the number
      of local devices. Data is padded as needed.
    mask: Array containing 1s where examples are not padding.
  """
  num_devices = jax.local_device_count()
  leading_dim = 0

  def _check_dims(v):
    """Make sure leading dim matches leading dims of other elements."""
    nonlocal leading_dim
    if not leading_dim:
      leading_dim = v.shape[0]
    elif leading_dim != v.shape[0]:
      raise ValueError('All values must have the same leading dimension.')
  jax.tree_util.tree_map(_check_dims, data)

  num_extra = leading_dim % num_devices
  padding = 0 if num_extra == 0 else num_devices - num_extra
  examples_per_device = (leading_dim - 1) // num_devices + 1
  mask = onp.ones((leading_dim,))

  def _pad_and_reshape(v):
    """Pad each element if necessary and reshape."""
    if padding:
      v = onp.pad(v, ((0, padding),) + ((0, 0),) * (v.ndim - 1),
                  mode='constant')
    return v.reshape((num_devices, examples_per_device,) + v.shape[1:])
  return jax.tree_util.tree_map(_pad_and_reshape, (data, mask))


def params_to_weights_and_biases(params, embed_dim):
  """Take a vector of parameters and return model weights and biases.

  Args:
    params: Vector of length `(embed_dim + 1) * num_classes`.
    embed_dim: Number of features in embedding.

  Returns:
    weights: `embed_dim x num_classes` matrix of weights.
    biases: Vector of length `num_classes` of biases.
  """
  params = params.reshape((embed_dim + 1), -1)
  return params[1:, :], params[0, :]


def weights_and_biases_to_params(weights, biases):
  """Take a matrix of weights and a vector of biases and return params.

  Args:
    weights: `embed_dim x num_classes` matrix of weights.
    biases: Vector of length `num_classes` of biases.

  Returns:
    params: Vector of length `(embed_dim + 1) * num_classes`.
  """
  params = jnp.concatenate((biases[None, :], weights), 0)
  return params.ravel()


def multinomial_logistic_loss(params, embeddings, labels, mask,
                              num_replicas, l2_regularization):
  """Compute loss for multinomial logistic regression.

  Args:
    params: `(embed_dim + 1) x num_classes` matrix of
      parameters. The first row contains biases for each class. Remaining
      rows contain weights.
    embeddings: `num_examples x embed_dim` matrix of embeddings.
    labels: `num_examples` vector of integer labels.
    mask: Vector of length `num_examples` indicating which embeddings
      should be used to compute the loss (if 1) or ignored (if 0).
    num_replicas: Number of replicas. Note that the loss is summed across
      replicas.
    l2_regularization: Amount of L2 regularization to apply.

  Returns:
    loss: Cross-entropy loss, summed across embeddings.
    gradient: Gradient of loss with respect to `params`.
  """
  weights, biases = params_to_weights_and_biases(params, embeddings.shape[-1])
  sum_squares = jnp.sum(jnp.square(weights))
  logits = embeddings.dot(weights) + biases[None, :]
  return (jnp.sum((-logits[jnp.arange(logits.shape[0]), labels] +
                   jax.nn.logsumexp(logits, -1)) * mask) +
          l2_regularization / num_replicas * sum_squares)


@functools.partial(
    jax.pmap, axis_name='batch', in_axes=(0, 0, 0, None, None, None),
    static_broadcasted_argnums=(4, 5))
def _lbfgs_minimize(embeddings, labels, mask, initial_position, loss,
                    lbfgs_args):
  """Function to be pmapped to minimize LBFGS loss."""
  loss_and_grad_fn = jax.value_and_grad(loss)
  def total_loss_and_grad_fn(params):
    loss, grad = loss_and_grad_fn(params, embeddings, labels, mask)
    return jax.lax.psum(loss, 'batch'), jax.lax.psum(grad, 'batch')
  return tfp.optimizer.lbfgs_minimize(
      total_loss_and_grad_fn, initial_position=initial_position, **lbfgs_args)


def train(embeddings, labels, mask, l2_regularization=0.0,
          initial_weights=None, initial_biases=None,
          loss=multinomial_logistic_loss, **lbfgs_args):
  """Train logistic regression with L-BFGS on distributed embeddings.

  Args:
    embeddings: Array of embeddings of shape
      `[num_devices, examples_per_device, embed_dim]`. See
      `reshape_and_pad_data_for_devices` for an easy way to get data into this
      format.
    labels: Array of integer labels of shape
      `[num_devices, examples_per_device]`.
    mask: Array of indicating which embeddings should be used to compute the
      loss, of shape `[num_devices, examples_per_device]`.
    l2_regularization: Amount of L2 regularization to apply.
    initial_weights: `embed_dim x num_classes` matrix of weights. If not
      given, initialized to zeros.
    initial_biases: Vector of length `num_classes` of biases. If not
      given, initialized to zeros.
    loss: Loss function to use. See `multinomial_logistic_loss` for arguments.
    **lbfgs_args: Additional arguments to pass to
      `tfp.optimizer.lbfgs_minimize`.

  Returns:
    weights: `embed_dim x num_classes` matrix of weights.
    biases: Vector of length `num_classes` of biases.
    optimizer_results: NamedTuple of type LBfgsOptimizerResults, as returned by
      `tfp.optimizer.lbfgs_minimize`.
  """
  # Set some default arguments for optimization. I have no idea if these are
  # any good, but they work and seem to give reasonable results.
  lbfgs_args['tolerance'] = lbfgs_args.get('tolerance', 0)
  lbfgs_args['x_tolerance'] = lbfgs_args.get('x_tolerance', 1e-5)
  lbfgs_args['max_line_search_iterations'] = lbfgs_args.get(
      'max_line_search_iterations', 100)
  lbfgs_args['max_iterations'] = lbfgs_args.get('max_iterations', 1000)
  if l2_regularization is not None:
    loss = functools.partial(loss, l2_regularization=l2_regularization)
  loss = functools.partial(loss, num_replicas=jax.local_device_count())

  # Handle converting inital_weights to flat format.
  embed_dim = embeddings.shape[-1]
  if initial_weights is None:
    num_classes = int(jnp.max(labels)) + 1
    initial_weights = jnp.zeros((embed_dim, num_classes))
  if initial_biases is None:
    initial_biases = jnp.zeros((initial_weights.shape[-1],))
  initial_position = weights_and_biases_to_params(
      initial_weights, initial_biases)

  res = _lbfgs_minimize(
      embeddings, labels, mask, initial_position, loss,
      frozendict.frozendict(lbfgs_args))
  res = jax.tree_util.tree_map(lambda x: x[0], res)
  weights, biases = params_to_weights_and_biases(res.position, embed_dim)
  return weights, biases, res


@functools.partial(
    jax.pmap, axis_name='batch', in_axes=(0, 0, 0, None, None))
def _evaluate(embeddings, labels, mask, weights, biases):
  """Function to be pmapped to perform evaluation."""
  return (
      jnp.sum(
          (jnp.argmax(embeddings.dot(weights) + biases[None, :], -1) ==
           labels) * mask),
      jnp.sum(mask))


def evaluate(embeddings, labels, mask, weights, biases):
  """Compute top-1 accuracy.

  Args:
    embeddings: Array of embeddings of shape
      `[num_devices, examples_per_device, embed_dim]`. See
      `reshape_and_pad_data_for_devices` for an easy way to get data into this
      format.
    labels: Array of integer labels of shape
      `[num_devices, examples_per_device]`.
    mask: Array of indicating which embeddings should be used to compute the
      loss, of shape `[num_devices, examples_per_device]`.
    weights: `embed_dim x num_classes` matrix of weights.
    biases: Vector of length `num_classes` of biases.

  Returns:
    Top-1 accuracy in [0, 1].
  """
  correct, total = _evaluate(embeddings, labels, mask, weights, biases)
  return jnp.sum(correct) / jnp.sum(total)


DEFAULT_HPARAMS = 10.0 ** onp.arange(-6, 7, 2)


def tune_l2_regularization(train_embeddings, train_labels, train_mask,
                           val_embeddings, val_labels, val_mask,
                           loss=multinomial_logistic_loss,
                           initial_range=DEFAULT_HPARAMS, num_steps=4):
  """Tune L2 regularization, following a procedure something like the CLIP.

  This procedure first searches over the initial range, and then iteratively
  searches to the left and right of the best hyperparameter value, halving the
  log step size each at each step.

  See docs for `train` for information about how embeddings, labels, and masks
  should be provided.

  Args:
    train_embeddings: Embeddings of training set.
    train_labels: Labels for training set.
    train_mask: Mask for training set.
    val_embeddings: Embeddings of validation set.
    val_labels: Labels for validation set.
    val_mask: Mask for validation set.
    loss: Loss function to use. See `multinomial_logistic_loss` for arguments.
    initial_range: Initial L2 regularization parameter range. Assumed to be
      logarithmically spaced.
    num_steps: Number of halvings to perform for binary search.

  Returns:
    l2_regularization: Optimal amount of L2 regularization.
    weights: `embed_dim x num_classes` matrix of weights.
    biases: Vector of length `num_classes` of biases.
    accuracy: Validation accuracy obtained.
  """
  weights = None
  biases = None
  accuracies = []
  weights_and_biases = []
  for l2 in initial_range:
    print(l2)
    weights, biases, _ = train(
        train_embeddings, train_labels, train_mask, l2, weights, biases,
        loss=loss)
    accuracy = evaluate(
        val_embeddings, val_labels, val_mask, weights, biases)
    print(f'  {accuracy}')
    logging.info('Initial range: l2 %s, accuracy %s', l2, accuracy)
    accuracies.append(accuracy)
    weights_and_biases.append((weights, biases))

  best_index = onp.argmax(accuracies)
  best_accuracy = accuracies[best_index]
  best_l2 = initial_range[best_index]
  if best_index == 0:
    delta = initial_range[best_index+1] / initial_range[best_index]
  else:
    delta = initial_range[best_index] / initial_range[best_index-1]
  best_weights, best_biases = weights_and_biases[best_index]
  del weights_and_biases

  for _ in range(num_steps):
    delta = onp.sqrt(delta)
    l2 = best_l2 / delta
    print(l2)
    weights, biases, _ = train(
        train_embeddings, train_labels, train_mask,
        l2, best_weights, best_biases, loss=loss)
    accuracy = evaluate(
        val_embeddings, val_labels, val_mask, weights, biases)
    print(f'  {accuracy}')
    logging.info('Fine range: l2 %s, accuracy %s', l2, accuracy)

    if accuracy > best_accuracy:
      best_l2, best_weights, best_biases, best_accuracy = (
          l2, weights, biases, accuracy)
      continue

    l2 = best_l2 * delta
    print(l2)
    weights, biases, _ = train(
        train_embeddings, train_labels, train_mask,
        l2, best_weights, best_biases, loss=loss)
    accuracy = evaluate(
        val_embeddings, val_labels, val_mask, weights, biases)
    print(f'  {accuracy}')
    logging.info('Fine range: l2 %s, accuracy %s', l2, accuracy)

    if accuracy > best_accuracy:
      best_l2, best_weights, best_biases, best_accuracy = (
          l2, weights, biases, accuracy)

  return best_l2, best_weights, best_biases, best_accuracy
