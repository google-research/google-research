# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Training utilities for learning the proxy reward."""

from typing import Dict

from absl import logging
from flax import linen as nn
import jax
from jax import numpy as jnp
import numpy as np
import optax


class LinearReg(nn.Module):
  """JAX implementation of linear regression.

  Attributes:
    kernel_init: Initializer for weights
    bias_init: Initializer for bias
  """

  def setup(self):
    self.kernel_init = nn.initializers.xavier_normal()
    self.bias_init = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    weights = self.param(
        'weights',
        self.kernel_init,  # Initialization function
        (inputs.shape[-1], 1))  # shape info.
    bias = self.param('bias', self.bias_init, (1,))

    return jnp.dot(inputs, weights) + bias


class LogisticReg(nn.Module):
  """JAX implementation of logistic regression as a one-layer NN.

  Attributes:
    kernel_init: Initializer for weights
    bias_init: Initializer for bias
  """

  def setup(self):
    self.kernel_init = nn.initializers.xavier_normal()
    self.bias_init = nn.initializers.zeros

  @nn.compact
  def __call__(self, inputs):
    weights = self.param(
        'weights',
        self.kernel_init,  # Initialization function
        (inputs.shape[-1], 1))  # shape info.
    bias = self.param('bias', self.bias_init, (1,))

    logit = jnp.dot(inputs, weights) + bias

    return nn.sigmoid(logit)


def make_loss_func(model,
                   data,
                   erm_weight = 1.,
                   erm_type='cross_entropy',
                   l2_lamb = 0.,
                   bias_lamb = 0.,
                   bias_norm='max'):
  """Loss Function.

  Args:
    model: nn.Module to be trained
    data: Dictionary containing arrays for surrogates ('m'), outcomes ('y'),
      actions ('a'), and time periods ('t')
    erm_weight: Weight for the erm loss (default 1.0)
    erm_type: one of 'cross_entropy' or 'mse' (mean-sqared error)
    l2_lamb: Weight for L2 regularization (default 0.)
    bias_lamb: Weight for the "policy bias" regularization term (default 0.)
    bias_norm: One of ['l2', 'max']

  Returns:
    Loss Function:
      erm_weight * erm_loss + l2_lamb * l2_reg + bias_lamb * bias_reg
  Raises:
    ValueError: If the data dictionary does not contain the correct keys.
  """

  for k in ['m', 'y', 'a', 't']:
    if k not in data.keys():
      raise ValueError(f'Data is missing entry "{k}"')

  m_batch = data['m']
  y_batch = data['y']
  a_batch = data['a']
  t_batch = data['t']

  actions = jnp.unique(a_batch)
  times = jnp.unique(t_batch)

  def erm_loss(params):

    def cross_entropy_sample(m, y):
      pred = model.apply(params, m)[0]
      label_prob = pred * y + (1 - pred) * (1 - y)
      return -jnp.log(label_prob)

    def mse_sample(m, y):
      pred = model.apply(params, m)[0]
      return jnp.square(y - pred)

    if erm_type == 'cross_entropy':
      return jnp.mean(jax.vmap(cross_entropy_sample)(m_batch, y_batch), axis=0)
    elif erm_type == 'mse':
      return jnp.mean(jax.vmap(mse_sample)(m_batch, y_batch), axis=0)
    else:
      raise ValueError(
          f'Expected erm_type of "cross_entropy" or "mse", got {erm_type}')

  def l2_reg(params):
    weights = params['params']['weights']
    return jnp.vdot(weights, weights)

  def bias_reg(params):
    # get conditional average error for each action / time
    residual = y_batch - model.apply(params, m_batch)[:, 0]
    bias_list = []
    weighted_bias_list = []

    for a in actions:
      for t in times:
        idx = jnp.logical_and(a_batch == a, t_batch == t)
        # Get expected value of the residual given a, t
        total = idx.sum()
        avg_residual = jnp.sum(jnp.where(idx, residual, 0)) / total
        bias_list.append(jnp.square(avg_residual))
        # Proportion of samples belonging to this action/time
        weight = idx.mean()
        weighted_bias_list.append(jnp.square(avg_residual) * weight)

    if bias_norm == 'max':
      return jnp.max(jnp.array(bias_list))
    if bias_norm == 'l2':
      return jnp.sum(jnp.array(weighted_bias_list))

  def loss(params):
    res = erm_weight * erm_loss(params)
    res = res + l2_lamb * l2_reg(params)
    res = res + bias_lamb * bias_reg(params)
    return res

  return jax.jit(loss)


def initialize_params(model, mdim,
                      seed):
  """Initialize parameters.

  Args:
    model: nn.Module
    mdim: Dimensions of the surrogates M
    seed: Random seed (integer).

  Returns:
    Dictionary containing parameters.

  Raises:
    ValueError: Surrogate dimension must be positive.
  """
  if mdim <= 0:
    raise ValueError('Surrogate dimension must be positive')
  k = jax.random.PRNGKey(seed)
  m = jax.random.normal(k, (mdim,))  # Dummy input
  return model.init(k, m)


def train(loss,
          params,
          lr,
          validation_loss=None,
          optimizer='adam',
          nsteps = 1001,
          verbose=False,
          log=False,
          tol=1e-5):
  """Train a model using the provided loss and parameters.

  Args:
    loss: Loss function.
    params: Parameter dictionary.
    lr: Learning Rate.
    validation_loss: Optional validation loss, for checkpoints.
    optimizer: One of 'sgd', 'adam'.
    nsteps: Number of iterations of SGD.
    verbose: If true, log the loss at every 100 iterations.
    log: If true, populate checkpoints every 100 iterations.
    tol: If the l_infinity norm of the gradient is smaller than this value, then
      assume convergence.

  Returns:
    Learned params and checkpoints

  """
  if optimizer == 'adam':
    tx = optax.adam(learning_rate=lr)
  elif optimizer == 'sgd':
    tx = optax.sgd(learning_rate=lr)
  else:
    raise ValueError(f'Expected "adam" or "sgd" for optimizer, got {optimizer}')

  opt_state = tx.init(params)
  loss_grad_fn = jax.value_and_grad(loss)

  checkpoints = []
  for i in range(nsteps):
    loss_val, grads = loss_grad_fn(params)

    if i % 100 == 0:
      if log:
        b, w = jax.tree_leaves(params)
        param_dict = {
            'Step': i,
            'Train Loss': loss_val.item(),
            'b': b.item(),
        }
        if validation_loss is not None:
          iter_validation_loss = validation_loss(params).item()
          param_dict['Validation Loss'] = iter_validation_loss

        for j, this_w in enumerate(w):
          param_dict.update({f'w{j+1}': this_w.item()})

        checkpoints.append(param_dict)

      if verbose:
        logging.info('Step %d: Train Loss %f', i, loss_val)
        if validation_loss is not None:
          logging.info('Step %d: Validation Loss %f', i, iter_validation_loss)

    _, w_grad = jax.tree_leaves(grads)
    if jnp.max(jnp.absolute(w_grad)) < tol:
      logging.info('Converged at Step %d', i)
      break

    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

  # Log the final parameters
  if log:
    loss_val, grads = loss_grad_fn(params)
    b, w = jax.tree_leaves(params)
    param_dict = {
        'Step': i,
        'Train Loss': loss_val.item(),
        'b': b.item(),
    }
    if validation_loss is not None:
      iter_validation_loss = validation_loss(params).item()
      param_dict['Validation Loss'] = iter_validation_loss

    for j, this_w in enumerate(w):
      param_dict.update({f'w{j+1}': this_w.item()})

    checkpoints.append(param_dict)

  return params, checkpoints
