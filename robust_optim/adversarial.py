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

"""Functions for generating adversarial attacks."""
import copy

import jax
from jax import numpy as jnp

from robust_optim.norm import norm_projection
from robust_optim.optim import gradient_descent_line_search_step


def find_adversarial_samples(
    data, loss_f, dloss_dx, model_param0, normalize_f, config, rng_key):
  """Generates an adversarial example in the epsilon-ball centered at data.

  Args:
    data: An array of size dim x num, with input vectors as the initialization
      for the adversarial attack.
    loss_f: Loss function for attack.
    dloss_dx: The gradient function of the adversarial loss w.r.t. the input
    model_param0: Current model parameters.
    normalize_f: A function to normalize the weights of the model.
    config: Dictionary of hyperparameters.
    rng_key: JAX random number generator key.

  Returns:
    An array of size dim x num, one adversarial example per input data point.

  - Can gradient wrt parameters be zero but not the gradient wrt inputs? No.
  f = w*x
  df/dw = x (for a linear model)

  dL/dw = dL/df1 df1/dw = G x
  if dL/dw ~= 0, and x != 0 => G~=0

  dL/dx = dL/df1 df1/dx = G w
  G ~= 0 => dL/dx ~= 0
  """

  x0, y = data
  eps_iter, eps_tot = config.eps_iter, config.eps_tot
  norm_type = config.norm_type
  # - Normalize model params, prevents small gradients
  # For both linear and non-linear models the norm of the gradient can be
  # artificially very small. This might be less of an issue for linear models
  # and when we use sign(grad) to adv.
  # For separable data, the norm of the weights of the separating classifier
  # can increase to increase the confidence and decrease the gradient.
  # But adversarial examples within the epsilon ball still exist.
  # For linear models, divide one layer by the norm of the product of weights
  model_param = model_param0
  if config.pre_normalize:
    model_param = normalize_f(model_param0, norm_type)

  # - Reason for starting from a random delta instead of zero:
  # A non-linear model can have zero dL/dx at x0 but huge d^2L/dx^2 which means
  # a gradient-based attack fails if it always starts the optimization from x0
  # but succeed if starts from a point nearby with non-zero gradient.
  # It is not trivial what the distribution for the initial perturbation should
  # be. Uniform within the epsilon ball has its merits but then we have to use
  # different distributions for different norm-balls. We instead config for
  # sampling from a uniform distribution and clipping delta to lie within the
  # norm ball.
  delta = jax.random.normal(rng_key, x0.shape)
  assert eps_iter <= eps_tot, 'eps_iter > eps_tot'
  delta = norm_projection(delta, norm_type, eps_iter)
  options = {'bound_step': True, 'step_size': 1000.}
  for _ in range(config.niters):
    x_adv = x0 + delta
    # Untargeted attack: increases the loss for the correct label
    if config.step_dir == 'sign_grad':
      # Sign grad is optimal for Linf attack, FGSM and PGD attacks use only sign
      grad = dloss_dx(model_param, x_adv, y)
      adv_step = config.lr * jnp.sign(grad)
    elif config.step_dir == 'grad':
      grad = dloss_dx(model_param, x_adv, y)
      adv_step = config.lr * grad
    elif config.step_dir == 'grad_ls':
      # Line search on the negative of the loss to find the ascent direction
      # And switch the order of w and x
      neg_loss_f = lambda x, w, y: -loss_f(w, x, y)
      x_adv_next, _ = gradient_descent_line_search_step(
          (model_param, y), neg_loss_f, x_adv, options)
      adv_step = x_adv_next - x_adv
    # - Reason for having both a per-step epsilon and a total epsilon:
    # Needed for non-linear models. Increases attack success if dL/dx at x0 is
    # huge and f(x) is correct on the entire shell of the norm-ball but wrong
    # inside the norm ball.
    delta_i = norm_projection(adv_step, norm_type, eps_iter)
    delta = norm_projection(delta + delta_i, norm_type, eps_tot)

  x_adv = x0 + delta
  return x_adv


def find_adversarial_samples_multi_attack(
    data, loss_f, dloss_dx, model_param0, normalize_f, config, rng_key):
  """Generates adversarial samples with multiple attacks and returns all samples."""
  config = copy.deepcopy(config)

  # Setting from the config
  rng_key, rng_subkey = jax.random.split(rng_key, 2)
  x_adv_multi = []
  x_adv_multi += [
      find_adversarial_samples(
          data, loss_f, dloss_dx, model_param0, normalize_f, config, rng_subkey)
  ]

  for pre_norm in [True, False]:
    for step_dir in ['grad', 'sign_grad']:
      rng_key, rng_subkey = jax.random.split(rng_key, 2)
      config.pre_normalize = pre_norm
      config.step_dir = step_dir
      x_adv_multi += [
          find_adversarial_samples(
              data, loss_f, dloss_dx, model_param0, normalize_f, config,
              rng_subkey)
      ]

  # TODO(fartash): line search
  return x_adv_multi
