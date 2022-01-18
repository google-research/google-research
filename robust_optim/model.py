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

"""Functions describing a linear model and loss in JAX."""
import re
import jax
from jax import nn
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree

from robust_optim.norm import get_prox_op
from robust_optim.norm import norm_f


def linear_model_init_param(dim):
  """Creates initial parameters for a linear model."""
  return jnp.zeros((dim, 1))


@jax.jit
def linear_model_predict(w, x):
  """Returns predictions of a binary classifier (+/-1 labels)."""
  z = w.T @ x
  return 2. * (z > 0) - 1


@jax.jit
def linear_model_loss(w, x, y):
  """Returns the loss given an input and its ground-truth label."""
  z = w.T @ x
  loss = nn.softplus(-y * z).mean()
  return loss


def linear_model_loss_regularized_w_lp(w, x, y, reg_coeff, norm_type):
  loss = linear_model_loss(w, x, y)
  reg = norm_f(w, norm_type)
  return loss + reg_coeff * reg


def linear_model_loss_regularized_dx_lp(w, x, y, reg_coeff, norm_type):
  """Penalize by ||dL/dx||."""
  loss_and_dloss_dx_f = jax.value_and_grad(linear_model_loss, argnums=1)
  loss, dloss_dx = loss_and_dloss_dx_f(w, x, y)
  reg = norm_f(dloss_dx, norm_type)
  return loss + reg_coeff * reg


def linear_model_loss_regularized_dw_lp(w, x, y, reg_coeff, norm_type):
  """Penalize by ||dL/dx||."""
  loss_and_dloss_dw_f = jax.value_and_grad(linear_model_loss, argnums=0)
  loss, dloss_dw = loss_and_dloss_dw_f(w, x, y)
  reg = norm_f(dloss_dw, norm_type)
  return loss + reg_coeff * reg


def deep_linear_model_init_param(dim, nlayers, r, rng_key):
  """Create initial parameters for a deep linear model with nlayers."""
  params = []
  rng_subkey = jax.random.split(rng_key, nlayers - 1)
  for i in range(nlayers - 1):
    params += [init_normalized_params((dim, dim), r, rng_subkey[i], False)]
  params += [jnp.zeros((dim, 1))]
  return params


def deep_linear_model_linearize_param(params):
  """Computes the product of parameters that is the single layer linear parameters."""
  linear_param = params[0]
  for i in range(1, len(params)):
    linear_param = linear_param @ params[i]
  return linear_param


def deep_linear_model_normalize_param(params, norm_type):
  """Normalizes the last layer weights by the norm of the product of weights."""
  norm_p = norm_f(deep_linear_model_linearize_param(params), norm_type)
  params_new = params
  params_new[-1] = params[-1] / jnp.maximum(1e-7, norm_p)
  return params_new


@jax.jit
def deep_linear_model_predict(w, x):
  """Returns predictions of a binary deep linear classifier (+/-1 labels)."""
  z = x
  for i in range(len(w)):
    z = w[i].T @ z
  return 2. * (z > 0) - 1


@jax.jit
def deep_linear_model_loss(w, x, y):
  """Returns the loss given an input and its ground-truth label."""
  z = x
  for i in range(len(w)):
    z = w[i].T @ z
  loss = nn.softplus(-y * z).mean()
  return loss


def deep_linear_model_loss_regularized_w_lp(w, x, y, reg_coeff, norm_type):
  """Penalize by ||dL/dw||."""
  loss = deep_linear_model_loss(w, x, y)
  w_unravel = ravel_pytree(w)[0]
  reg = norm_f(w_unravel, norm_type)
  return loss + reg_coeff * reg


def deep_linear_model_loss_regularized_dx_lp(w, x, y, reg_coeff, norm_type):
  """Penalize by ||dL/dx||."""
  loss_and_dloss_dx_f = jax.value_and_grad(deep_linear_model_loss, argnums=1)
  loss, dloss_dx = loss_and_dloss_dx_f(w, x, y)
  reg = norm_f(dloss_dx, norm_type)
  return loss + reg_coeff * reg


def deep_linear_model_loss_regularized_dw_lp(w, x, y, reg_coeff, norm_type):
  """Penalize by ||dL/dx||."""
  loss_and_dloss_dw_f = jax.value_and_grad(deep_linear_model_loss, argnums=0)
  loss, dloss_dw = loss_and_dloss_dw_f(w, x, y)
  reg = norm_f(ravel_pytree(dloss_dw)[0], norm_type)
  return loss + reg_coeff * reg


def deep_linear_model_loss_regularized_dw0_lp_da1_lq(w, x, y, reg_coeff,
                                                     norm_type1, norm_type2):
  """Penalize by ||dL/dw0||+||dL/da1||, where w0, a1 are first layer w, o."""
  dloss_da = jax.grad(deep_linear_model_loss, argnums=1)
  loss = deep_linear_model_loss(w, x, y)
  a1 = w[0].T @ x
  reg = norm_f(w[0], norm_type1) + norm_f(dloss_da(w[1:], a1, y), norm_type2)
  return loss + reg_coeff * reg


def two_linear_model_predict(w0, w1, x):
  """Returns the prediction given an input and its ground-truth label.

  Args:
    w0: An array with model parameters for first layer.
    w1: An array with model parameters for second layer.
    x: An array with input data.

  Returns:
    An array of model predictions.
  """
  # TODO(fartash): bias is to be added and analyzed
  z = w1.T @ (w0.T @ x)
  return 2. * (z > 0) - 1


@jax.jit
def two_linear_model_y1_mean(w0, w1, x):
  y = w1.T @ (w0.T @ x)
  return y.mean()  # mean over examples


@jax.jit
def two_linear_model_y0_mean(w0, x):
  y = w0.T @ x
  return y.mean(1).sum()  # mean over examples


@jax.jit
def two_linear_model_loss(w0, w1, x, y):
  """Random features model loss."""
  z = w1.T @ (w0.T @ x)
  loss = nn.softplus(-y * z).mean()
  return loss


def two_linear_model_loss_regularized_dy1dx_lp(w0, w1, x, y, reg_coeff,
                                               norm_type):
  """Penalize by ||dy1/dx||, optimal when first layer is fixed."""
  dy1_dx_f = jax.grad(two_linear_model_y1_mean, argnums=2)
  dy1_dx = dy1_dx_f(w0, w1, x)
  loss = two_linear_model_loss(w0, w1, x, y)
  reg = norm_f(dy1_dx, norm_type)
  return loss + reg_coeff * reg


def two_linear_model_loss_regularized_dy0dx_lp(w0, w1, x, y, reg_coeff,
                                               norm_type):
  """Penalize by ||dy0/dx||, optimal when first layer is fixed."""
  dy0_dx_f = jax.grad(two_linear_model_y0_mean, argnums=1)
  dy0_dx = dy0_dx_f(w0, x)
  loss = two_linear_model_loss(w0, w1, x, y)
  reg = norm_f(dy0_dx, norm_type)
  return loss + reg_coeff * reg


def two_linear_model_loss_regularized_w0w1_lp(w0, w1, x, y, reg_coeff,
                                              norm_type):
  """Penalize by ||w0w1||, optimal when first layer is fixed faster than dydx."""
  loss = two_linear_model_loss(w0, w1, x, y)
  reg = norm_f(w0 @ w1, norm_type)
  return loss + reg_coeff * reg


def two_linear_model_loss_regularized_w0_lp(w0, w1, x, y, reg_coeff, norm_type):
  """Penalize by ||w0||, optimal when first layer is fixed faster than dydx."""
  loss = two_linear_model_loss(w0, w1, x, y)
  reg = norm_f(w0, norm_type)
  return loss + reg_coeff * reg


def two_linear_model_loss_regularized_w1_lp(w0, w1, x, y, reg_coeff, norm_type):
  """Penalize by ||w1||, optimal when first layer is fixed faster than dydx."""
  loss = two_linear_model_loss(w0, w1, x, y)
  reg = norm_f(w1, norm_type)
  return loss + reg_coeff * reg


def init_normalized_params(size, r, rng_key, non_isotropic):
  param0 = jax.random.normal(rng_key, size)
  if non_isotropic:
    param0 = param0.flatten()
    param0 = param0 * jnp.arange(param0.shape)
    param0 = param0.reshape(size)
  param0_norm = jnp.linalg.norm(param0, axis=0, keepdims=True)
  param = param0 / param0_norm * r
  return param


def two_linear_w0fixed_init_param(dim, r, rng_key):
  w0 = init_normalized_params((dim, dim), r, rng_key, False)
  w1 = jnp.zeros((dim, 1))
  return w0, w1


def two_linear_w1fixed_init_param(dim, r, rng_key, non_isotropic):
  w0 = jnp.zeros((dim, dim))
  w1 = init_normalized_params((dim, 1), r, rng_key, non_isotropic)
  return w0, w1


@jax.jit
def circ_1d_conv(w, x):
  """Circular 1D convolution.

  Args:
    w: A vector of size D.
    x: An array with the first size D.

  Returns:
    An array the same shape as x.

  Derivation:
  >>> x = jnp.arange(10)
  >>> w = jnp.arange(10)

  # What we want
  >>> y = []
  >>> for i in range(len(x)):
  >>>   yi = 0
  >>>   for k in range(len(w)):
  >>>     yi += w[k]*x[(i+k) % len(x)]
  >>>   y += [yi]
  >>> print(jnp.array(y))

  # Replace with roll
  >>> y = []
  >>> for i in range(len(x)):
  >>>   z = jnp.roll(x, i)
  >>>   yi = 0
  >>>   for k in range(len(w)):
  >>>     yi += w[k]*z[k]
  >>>   y += [yi]
  >>> print(jnp.array(y))
  >>> y = []

  # Use dot
  >>> for i in range(len(x)):
  >>>   z = jnp.roll(x, i)
  >>>   y += [jnp.vdot(w, z)]
  >>> print(jnp.array(y))

  # Use lambda
  >>> y = []
  >>> for i in range(len(x)):
  >>>   y += [rcvvi(w, x, i)]
  >>> print(jnp.array(y))

  # Test vector matrix product
  >>> X = jnp.arange(10).reshape(-1, 1).repeat(10, axis=1)
  >>> print(rcvv2(w, X))

  """
  rcvvi = lambda w, x, i: w.T @ jnp.roll(x, i, axis=0)  # [a],[a],[] -> []
  rcvv = jax.vmap(rcvvi, (None, None, 0), 0)  # [a],[a],[b] -> [b]
  rcvv2 = lambda w, x: rcvv(w, x, jnp.arange(x.shape[0]))  # [a],[a,b] -> [a,b]
  return rcvv2(w, x) / jnp.sqrt(x.shape[0])


def conv_linear_model_init_param(dim, nlayers, r, rng_key):
  """Creates initial parameters for a linear model."""
  params = []
  rng_subkey = jax.random.split(rng_key, nlayers - 1)
  for i in range(nlayers - 1):
    params += [init_normalized_params((dim,), r, rng_subkey[i], False)]
  params += [jnp.zeros((dim, 1))]
  return params


@jax.jit
def conv_linear_model_predict(w, x):
  """Returns the prediction given an input and its ground-truth label.

  As defined in Gunasekar et al 2018,
  "Implicit Bias of Gradient Descent on Linear Convolutional Networks"
  https://arxiv.org/abs/1806.00468

  Args:
    w: An array with model parameters for first convolution layers then a fully
      connected layer.
    x: An array with input data.

  Returns:
    An array of model predictions.
  """
  z = x
  for i in range(len(w) - 1):
    z = circ_1d_conv(w[i], z)
  z = w[-1].T @ z
  return 2. * (z > 0) - 1


@jax.jit
def conv_linear_model_loss(w, x, y):
  """Returns the loss given an input and its ground-truth label."""
  z = x
  for i in range(len(w) - 1):
    z = circ_1d_conv(w[i], z)
  z = w[-1].T @ z
  loss = nn.softplus(-y * z).mean()
  return loss


@jax.jit
def conv_linear_model_linearize_param(params):
  """Computes the product of parameters that is the single layer linear parameters."""
  # TODO(fartash): prove that flip is not needed
  # wl = jnp.flip(params[-1], 0)
  wl = params[-1]
  for i in range(len(params) - 2, -1, -1):
    wl = circ_1d_conv(params[i], wl)
  # return jnp.flip(wl, 0)
  return wl


def conv_linear_model_normalize_param(params, norm_type):
  """Normalizes the last layer weights by the norm of the product of weights."""
  norm_p = norm_f(conv_linear_model_linearize_param(params), norm_type)
  params_new = params
  params_new[-1] = params[-1] / jnp.maximum(1e-7, norm_p)
  return params_new


def loss_f_with_args(loss_f, *args):

  def loss_f_new(w, x, y):
    return loss_f(w, x, y, *args)

  return loss_f_new


def get_model_functions(rng_key, dim, arch, nlayers, regularizer, reg_coeff, r):
  """Returns model init/predict/loss functions given the model name."""
  loss_and_prox_op = None
  if arch == 'linear':
    init_f, predict_f = linear_model_init_param, linear_model_predict
    if regularizer == 'none':
      loss_f = linear_model_loss
      prox_op = lambda x, _: x
    elif re.match('w_l.*', regularizer) or re.match('w_dft.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = loss_f_with_args(linear_model_loss_regularized_w_lp, reg_coeff,
                                norm_type)
      prox_op = lambda v, lam: get_prox_op(norm_type)(v, lam * reg_coeff)
    elif re.match('dx_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = loss_f_with_args(linear_model_loss_regularized_dx_lp, reg_coeff,
                                norm_type)
    elif re.match('dw_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = loss_f_with_args(linear_model_loss_regularized_dw_lp, reg_coeff,
                                norm_type)
    model_param = init_f(dim)
    loss_adv_f = linear_model_loss
    linearize_param = lambda p: p
    normalize_param = lambda p, n: p / jnp.maximum(1e-7, norm_f(p, n))
    loss_and_prox_op = (linear_model_loss, prox_op)
  elif arch == 'deep_linear':
    init_f = deep_linear_model_init_param
    predict_f = deep_linear_model_predict
    if regularizer == 'none':
      loss_f = deep_linear_model_loss
    elif re.match('w_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = loss_f_with_args(deep_linear_model_loss_regularized_w_lp,
                                reg_coeff, norm_type)
    elif re.match('dx_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = loss_f_with_args(deep_linear_model_loss_regularized_dx_lp,
                                reg_coeff, norm_type)
    elif re.match('dw_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = loss_f_with_args(deep_linear_model_loss_regularized_dw_lp,
                                reg_coeff, norm_type)
    elif re.match('dw0_l.*_da1_l.*', regularizer):
      norm_type = regularizer.split('_')
      norm_type1, norm_type2 = norm_type[1], norm_type[3]
      loss_f = loss_f_with_args(
          deep_linear_model_loss_regularized_dw0_lp_da1_lq, reg_coeff,
          norm_type1, norm_type2)
    model_param = init_f(dim, nlayers, r, rng_key)
    loss_adv_f = deep_linear_model_loss
    linearize_param = deep_linear_model_linearize_param
    normalize_param = deep_linear_model_normalize_param
  elif arch == 'two_linear_fixed_w0':
    w0, model_param = two_linear_w0fixed_init_param(dim, r, rng_key)
    predict_f = lambda w, x: two_linear_model_predict(w0, w, x)
    if regularizer == 'none':
      loss_f = lambda w, x, y: two_linear_model_loss(w0, w, x, y)
    elif re.match('dy1dx_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_dy1dx_lp(  # pylint: disable=g-long-lambda
          w0, w, x, y, reg_coeff, norm_type)
    elif re.match('w0w1_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_w0w1_lp(  # pylint: disable=g-long-lambda
          w0, w, x, y, reg_coeff, norm_type)
    elif re.match('w1_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_w1_lp(  # pylint: disable=g-long-lambda
          w0, w, x, y, reg_coeff, norm_type)
    loss_adv_f = lambda w, x, y: two_linear_model_loss(w0, w, x, y)
    linearize_param = lambda p: w0.T @ p
    normalize_param = lambda p, n: p / jnp.maximum(1e-7, norm_f(w0.T @ p, n))
  elif arch == 'two_linear_fixed_w1' or arch == 'two_linear_fixed_w1_noniso':
    non_isotropic = arch == 'two_linear_fixed_w1_noniso'
    model_param, w1 = two_linear_w1fixed_init_param(dim, r, rng_key,
                                                    non_isotropic)
    predict_f = lambda w, x: two_linear_model_predict(w, w1, x)
    if regularizer == 'none':
      loss_f = lambda w, x, y: two_linear_model_loss(w, w1, x, y)
    elif re.match('dy0dx_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_dy0dx_lp(  # pylint: disable=g-long-lambda
          w, w1, x, y, reg_coeff, norm_type)
    elif re.match('w0w1_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_w0w1_lp(  # pylint: disable=g-long-lambda
          w, w1, x, y, reg_coeff, norm_type)
    elif re.match('w0_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_w0_lp(  # pylint: disable=g-long-lambda
          w, w1, x, y, reg_coeff, norm_type)
    elif re.match('dy1dx_l.*', regularizer):
      norm_type = regularizer.split('_')[1]
      loss_f = lambda w, x, y: two_linear_model_loss_regularized_dy1dx_lp(  # pylint: disable=g-long-lambda
          w, w1, x, y, reg_coeff, norm_type)
    loss_adv_f = lambda w, x, y: two_linear_model_loss(w, w1, x, y)
    linearize_param = lambda p: p.T @ w1
    normalize_param = lambda p, n: p / jnp.maximum(1e-7, norm_f(p.T @ w1, n))
  elif arch == 'conv_linear':
    init_f = conv_linear_model_init_param
    predict_f = conv_linear_model_predict
    if regularizer == 'none':
      loss_f = conv_linear_model_loss
      prox_op = lambda x, _: x
    model_param = init_f(dim, nlayers, r, rng_key)
    loss_adv_f = conv_linear_model_loss
    linearize_param = conv_linear_model_linearize_param
    normalize_param = conv_linear_model_normalize_param
    loss_and_prox_op = (conv_linear_model_loss, prox_op)

  return (model_param, predict_f, loss_f, loss_adv_f, linearize_param,
          normalize_param, loss_and_prox_op)
