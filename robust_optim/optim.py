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

"""Step functions for optimization methods."""
from absl import logging

import jax
from jax import numpy as jnp
from jax.flatten_util import ravel_pytree


def gradient_descent_step(data, loss_f, model_param, options):
  """Gradient Descent optimization step.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_f: The loss function that takes in model_param, inputs, and labels.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.

  Returns:
    Updated model parameters and options.
  """
  dloss_dw = jax.grad(loss_f, argnums=0)
  inputs, labels = data[0], data[1]
  grad = dloss_dw(model_param, inputs, labels)

  # Handle deep nets
  grad, unravel_fn = ravel_pytree(grad)
  model_param, unravel_fn = ravel_pytree(model_param)

  model_param -= options['lr'] * grad
  return unravel_fn(model_param), options


def backtracking(next_candidate, stop_cond, step_size_init, options, verbose=0):
  """Backtracking line search.

  Args:
    next_candidate: a function generating a candidate from a step size.
    stop_cond: a function determining whether to stop or not from a step size
      and a candidate.
    step_size_init: the initial step size to try.
    options: a dictionary containing line search specific options.
    verbose: whether to enable verbose output or not.

  Returns:
    step_size, next_candidate
  """

  max_iter = options.get('max_linesearch', 20)
  step_factor = options.get('step_factor', 0.5)

  step_size = step_size_init
  next_iter = next_candidate(step_size)

  for it in range(max_iter):
    if stop_cond(step_size, next_iter):
      break

    step_size *= step_factor
    next_iter = next_candidate(step_size)

  if it == max_iter - 1 and verbose:
    print('Line search did not converge.')

  return step_size, next_iter


def gradient_descent_line_search_step(
    data, loss_f, model_param, options):
  """Gradient Descent optimization with line search step.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_f: The loss function that takes in model_param, inputs, and labels.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.

  Returns:
    Updated model parameters and updated step size.
  """
  options = dict(options)
  beta = options.get('beta', 0.9)
  beta_prime = options.get('beta_prime', 1e-4)
  step_size = options.get('step_size', 10000.0)
  verbose = options.get('verbose', False)
  reuse_last_step = options.get('reuse_last_step', False)

  inputs, labels = data[0], data[1]
  loss_with_data_f = lambda param: loss_f(param, inputs, labels)
  value_and_grad_f = jax.value_and_grad(loss_with_data_f)
  value, grad = value_and_grad_f(model_param)

  # Maximum learning rate allowed from Theorem 5 in Gunasekar et al. 2017
  if options['bound_step']:
    # Bound by dual of L2
    b_const = jnp.max(jnp.linalg.norm(inputs, ord=2, axis=0))
    step_size = min(step_size, 1 / (b_const * b_const * value))

  grad, unravel_fn = ravel_pytree(grad)
  x, unravel_fn = ravel_pytree(model_param)

  # If we normalize step_size will be harder to tune.
  direction = -grad

  # TODO(fartash): consider using the condition in FISTA
  def next_candidate(step_size):
    next_iter = x + step_size * direction
    next_value, next_grad = value_and_grad_f(unravel_fn(next_iter))
    next_grad, _ = ravel_pytree(next_grad)
    return next_iter, next_value, next_grad

  def stop_cond(step_size, res):
    _, next_value, next_grad = res
    gd = jnp.sum(grad * direction)

    # Strong Wolfe condition.
    cond1 = next_value <= value + beta_prime * step_size * gd
    cond2 = jnp.sum(jnp.abs(next_grad * direction)) >= beta * gd
    return cond1 and cond2

  step_size, res = backtracking(
      next_candidate, stop_cond, step_size, options=options)
  next_param = res[0]

  if reuse_last_step:
    options['step_size'] = step_size
  if verbose:
    logging.info('Step size: %f', step_size)

  return unravel_fn(next_param), options


def coordinate_descent_step(data, loss_f, model_param, options):
  """Gradient Descent optimization step.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_f: The loss function that takes in model_param, inputs, and labels.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.

  Returns:
    Updated model parameters and options.
  """
  dloss_dw = jax.grad(loss_f, argnums=0)
  inputs, labels = data[0], data[1]
  grad = dloss_dw(model_param, inputs, labels)
  grad_max = grad * (jnp.abs(grad) == jnp.abs(grad).max())

  # Handle deep nets
  grad_max, unravel_fn = ravel_pytree(grad_max)
  model_param, unravel_fn = ravel_pytree(model_param)

  model_param -= options['lr'] * grad_max
  return unravel_fn(model_param), options


def coordinate_descent_line_search_step(data, loss_f, model_param, options):
  """Coordinate Descent with line search optimization step.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_f: The loss function that takes in model_param, inputs, and labels.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.

  Returns:
    Updated model parameters and options.
  """
  options = dict(options)
  beta = options.get('beta', 0.9)
  beta_prime = options.get('beta_prime', 1e-4)
  step_size = options.get('step_size', 10000.0)
  verbose = options.get('verbose', False)
  reuse_last_step = options.get('reuse_last_step', False)

  inputs, labels = data[0], data[1]
  loss_with_data_f = lambda param: loss_f(param, inputs, labels)
  value_and_grad_f = jax.value_and_grad(loss_with_data_f)
  value, grad = value_and_grad_f(model_param)
  grad_max = grad * (jnp.abs(grad) == jnp.abs(grad).max())

  # Maximum learning rate allowed from Theorem 5 in Gunasekar et al. 2017
  if options['bound_step']:
    # Bound by dual of L1
    b_const = jnp.max(jnp.linalg.norm(inputs, ord=jnp.inf, axis=0))
    step_size = min(step_size, 1 / (b_const * b_const * value))

  # Handle deep nets
  grad_max, unravel_fn = ravel_pytree(grad_max)
  x, unravel_fn = ravel_pytree(model_param)

  # If we normalize step_size will be harder to tune.
  direction = -grad_max

  # TODO(fartash): consider using the condition in FISTA
  def next_candidate(step_size):
    next_iter = x + step_size * direction
    next_value, next_grad = value_and_grad_f(unravel_fn(next_iter))
    next_grad, _ = ravel_pytree(next_grad)
    return next_iter, next_value, next_grad

  def stop_cond(step_size, res):
    _, next_value, next_grad = res
    gd = jnp.sum(grad * direction)

    # Strong Wolfe condition.
    cond1 = next_value <= value + beta_prime * step_size * gd
    cond2 = jnp.sum(jnp.abs(next_grad * direction)) >= beta * gd
    return cond1 and cond2

  step_size, res = backtracking(
      next_candidate, stop_cond, step_size, options=options)
  next_param = res[0]

  if reuse_last_step:
    options['step_size'] = step_size
  if verbose:
    logging.info('Step size: %f', step_size)

  return unravel_fn(next_param), options


def coordinate_descent_topk_step(data, loss_f, model_param, options, k=2):
  """Coordinate Descent optimization step.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_f: The loss function that takes in model_param, inputs, and labels.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.
    k: An integere for the number of topk elements.

  Returns:
    Updated model parameters and options.
  """
  # TODO(fartash): add k to config.py to be part of options.
  dloss_dw = jax.grad(loss_f, argnums=0)
  inputs, labels = data[0], data[1]
  grad = dloss_dw(model_param, inputs, labels)
  _, coords = jax.lax.top_k(jnp.abs(grad.T), k)
  grad_max = 0 * grad
  grad_max = jax.ops.index_update(grad_max, coords, grad[coords])

  # Handle deep nets
  grad_max, unravel_fn = ravel_pytree(grad_max)
  model_param, unravel_fn = ravel_pytree(model_param)

  model_param -= options['lr'] * grad_max
  return unravel_fn(model_param), options


def sign_gradient_descent_step(data, loss_f, model_param, options):
  """Sign Gradient Descent optimization step.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_f: The loss function that takes in model_param, inputs, and labels.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.

  Returns:
    Updated model parameters and options.
  """
  dloss_dw = jax.grad(loss_f, argnums=0)
  inputs, labels = data[0], data[1]
  grad = dloss_dw(model_param, inputs, labels)
  grad_sign = jnp.abs(grad).sum() * jnp.sign(grad)

  # Handle deep nets
  grad_sign, unravel_fn = ravel_pytree(grad_sign)
  model_param, unravel_fn = ravel_pytree(model_param)

  model_param -= options['lr'] * grad_sign
  return unravel_fn(model_param), options


def fista_step(data, loss_and_prox_op, model_param, options):
  """Fista optimization step for solving regularized problem.

  Args:
    data: A tuple of inputs and labels passed to the loss function.
    loss_and_prox_op: Tuple of (loss_f, prox_g)
      loss_f is the loss function that takes in model_param, inputs, and labels.
      prox_g is the proximity operator for g.
    model_param: Current model parameters to be passed to loss_f.
    options: A dictionary of optimizer specific hyper-parameters.

  Returns:
    Updated model parameters and updated step size.
  """
  options = dict(options)
  step_size = options.get('step_size', 1.0)
  acceleration = options.get('acceleration', True)
  t = options.get('t', 1.0)
  verbose = options.get('verbose', False)
  reuse_last_step = options.get('reuse_last_step', False)

  loss_f, prox_g = loss_and_prox_op
  inputs, labels = data[0], data[1]
  fun_f = lambda param: loss_f(param, inputs, labels)
  value_and_grad_f = jax.value_and_grad(fun_f)
  x, unravel_fn = ravel_pytree(model_param)
  y = options.get('y', x)
  value_f, grad_f = value_and_grad_f(unravel_fn(y))
  grad_f, unravel_fn = ravel_pytree(grad_f)

  def next_candidate(step_size):
    return prox_g(y - grad_f * step_size, step_size)

  def stop_cond(step_size, next_iter):
    diff = next_iter - y
    sqdist = jnp.sum(diff**2)

    # We do not compute the non-smooth term (g in the paper)
    # as it cancels out from value_F and value_Q.
    value_bigf = fun_f(next_iter)
    value_bigq = value_f + jnp.sum(diff * grad_f) + 0.5 / step_size * sqdist
    return value_bigf <= value_bigq

  x_old = x

  step_size, x = backtracking(next_candidate, stop_cond, step_size, options)

  # Acceleration.
  if acceleration:
    t_next = (1 + jnp.sqrt(1 + 4 * t**2)) / 2.
    y = x + (t - 1) / t_next * (x - x_old)
    t = t_next
    options['y'] = y
    options['t'] = t
  else:
    y = x

  if reuse_last_step:
    options['step_size'] = step_size
  if verbose:
    logging.info('Step size: %f', step_size)

  return unravel_fn(x), options


def get_optimizer_step(options):
  """Return an optimizer given its name."""
  name = options['name']
  if name == 'gd' or name == 'cvxpy':  # TODO(fartash): do cvxpy the right way
    return gradient_descent_step, options
  if name == 'gd_ls':
    return gradient_descent_line_search_step, options
  if name == 'cd':
    return coordinate_descent_step, options
  if name == 'cd_ls':
    return coordinate_descent_line_search_step, options
  if name == 'signgd':
    return sign_gradient_descent_step, options
  if name == 'cdk':
    return coordinate_descent_topk_step, options
  if name == 'fista':
    return fista_step, options
  raise Exception('Invalid optimizer.')
