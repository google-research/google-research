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

"""Probability flow ODE.

See https://github.com/yang-song/score_sde/blob/main/likelihood.py for
Yang Song's original implementation of log-likelihood computation.
"""
import functools
from typing import Any, Callable, Optional, Union, Tuple

import diffrax
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # pylint:disable=g-multiple-import
from score_sde import sde_lib


def draw_epsilon(rng,
                 shape,
                 hutchinson_type = 'rademacher'):
  """Draw an epsilon for Hutchinson-Skilling trace estimation."""
  if hutchinson_type.lower() == 'gaussian':
    epsilon = jax.random.normal(rng, shape)
  elif hutchinson_type.lower() == 'rademacher':
    epsilon = jax.random.randint(
        rng, shape, minval=0, maxval=2).astype(jnp.float32) * 2 - 1
  else:
    raise ValueError(f'Hutchinson type {hutchinson_type} unknown')
  return epsilon


def get_hutchinson_div_fn(fn
                          ):
  """Return the divergence function for `fn(x, t)`.

  Assumes that `fn` takes mini-batched inputs. The returned divergence function
  takes a batch of epsilons and returns a batch of trace estimators.
  Note: we don't use `jax.jvp` or `jax.vjp` here because `x` is mini-batched.

  Args:
    fn: The function to take the divergence of.

  Returns:
    div_fn: The divergence function that takes a batch of `epsilon`s
      for trace estimation.
  """

  def div_fn_one_trace_estimate(x, t, epsilon):
    grad_fn = lambda data: jnp.sum(fn(data, t) * epsilon)
    grad_fn_eps = jax.grad(grad_fn)(x)
    div = jnp.sum(grad_fn_eps * epsilon, axis=tuple(range(1, len(x.shape))))
    return div

  div_fn = jax.vmap(div_fn_one_trace_estimate, in_axes=(None, None, 0))
  # Take the average of the divergence estimates.
  mean_div_fn = lambda x, t, epsilons: jnp.mean(div_fn(x, t, epsilons), axis=0)

  return mean_div_fn


def get_exact_div_fn(fn
                     ):
  """Return the exact divergence function for `fn(x, t)`.

  Assumes that `fn` takes mini-batched inputs. Divergence is calculated
  by computing the trace of the Jacobian.

  Args:
    fn: The function to take the divergence of.

  Returns:
    div_fn: The exact divergence function.
  """
  @functools.partial(jax.vmap, in_axes=(0, 0, None))
  def div_fn(x, t, _):
    # Assume `x` is one sample since this function is vmapped.
    x_dim = jnp.size(x)
    jac = jax.jacrev(fn)(x[None, Ellipsis], jnp.ones(1) * t)
    div = jnp.trace(jac.reshape(x_dim, x_dim))
    return div

  return div_fn


class ProbabilityFlow:
  """A class containing functions related to probability flow ODE.

  All functions are defined on mini-batches.
  """

  def __init__(self,
               sde,
               score_fn,
               solver,
               stepsize_controller,
               adjoint,
               n_trace_estimates = 1,
               hutchinson_type = 'rademacher'):
    """Initialize ProbabilityFlow object.

    Args:
      sde: The forward SDE.
      score_fn: The score function that returns the score of `x` at time `t`.
      solver: The `diffrax.AbstractSolver` to use for main ODEs.
      stepsize_controller: The `diffrax.AbstractStepSizeController` to use for
        main ODEs.
      adjoint: The `diffrax.AbstractAdjoint` to use for adjoint ODEs.
      n_trace_estimates: Number of Hutchinson-Skilling trace estimators
        to average for divergence calculation.
        For exact trace, set `n_trace_estimates` to a value < 0.
      hutchinson_type: Type of vector used for Hutchinson-Skilling.
        ['rademacher' | 'gaussian'].
    """
    self.sde = sde
    self.pf_ode = sde.reverse(score_fn, probability_flow=True)
    self.score_fn = score_fn
    self.hutchinson_type = hutchinson_type
    if n_trace_estimates < 0:
      self._div_fn = get_exact_div_fn(self.drift_fn)
      self.n_trace_estimates = 1
    else:
      self._div_fn = get_hutchinson_div_fn(self.drift_fn)
      self.n_trace_estimates = n_trace_estimates

    # Diffrax objects.
    self.solver = solver
    self.stepsize_controller = stepsize_controller
    self.adjoint = adjoint

  def marginal_dist_params(self, t):
    """The mean coefficient and std. dev. of the marginal distribution at t.

    At time t, x(t) ~ N(alpha(t) * x(0), beta(t)^2 * I).

    Args:
      t: Diffusion time.

    Returns:
      alpha_t: The mean coefficient at time t.
      beta_t: The std. dev. at time t.
    """
    # Since `score_sde.sde_lib.SDE` doesn't have a direct method for getting
    # the mean coefficient, we'll use a hacky way to access alpha(t).
    all_ones = jnp.ones((1, 1))
    t_batch = jnp.ones(1) * t
    mean, std = self.sde.marginal_prob(all_ones, t_batch)
    alpha_t = mean[0][0]
    beta_t = std[0]
    return alpha_t, beta_t

  def drift_fn(self,
               x,
               t_batch):
    """PF-ODE drift function.

    Args:
      x: The mini-batch of x(t), of shape (mini_batch_size, ...).
      t_batch: The diffusion time, repeated along a batch axis,
        of shape (mini_batch_size,).

    Returns:
      drift: The PF-ODE drift of `x`.
    """
    drift, _ = self.pf_ode.sde(x, t_batch)
    return drift

  def div_fn(self,
             rng,
             x,
             t_batch):
    """Divergence of the PF-ODE drift with Hutchinson-Skilling trace estimation.

    Args:
      rng: JAX RNG key for Hutchinson-Skilling trace estimation.
      x: The mini-batch of x(t), of shape (mini_batch_size, ...).
      t_batch: The diffusion time, repeated along a batch axis,
        of shape (mini_batch_size,).

    Returns:
      div: The divergence of the PF-ODE drift for each item in `x`. If
        trace estimation is used, then it is an estimated divergence.
    """
    shape = x[0].shape

    # Draw a batch of epsilons for trace estimation.
    epsilons_shape = (self.n_trace_estimates, *shape)
    epsilons = draw_epsilon(rng, epsilons_shape, self.hutchinson_type)

    div = self._div_fn(x, t_batch, epsilons)

    return div

  def drift_div_fn(self,
                   rng,
                   x,
                   t_batch
                   ):
    """Gradient of `x` and `logp(x)` at time `t`.

    Args:
      rng: A JAX RNG key for Hutchinson-Skilling trace estimation.
      x: The mini-batch of x(t), of shape (mini_batch_size, ...).
      t_batch: The diffusion time, repeated along a batch axis,
        of shape (mini_batch_size,).

    Returns:
      drift: The PF-ODE drift of x(t), of shape (mini_batch_size, ...).
      div: The drift of negative logp(x(t)), of shape (mini_batch_size,).
    """
    # Get `x` drift.
    drift = self.drift_fn(x, t_batch)

    # Get `logp(x)` drift.
    div = self.div_fn(rng, x, t_batch)

    return drift, div

  def ode(self,
          x_init,
          t0,
          t1,
          dt0 = None,
          **kwargs):
    """Run the PF-ODE forward or backward in time using Euler-Maruyama.

    Args:
      x_init: The initial mini-batch of x(0) or x(T).
      t0: Initial time of the ODE.
      t1: Final time of the ODE. Can be > or < t0.
      dt0: Initial step size of the ODE solver with adaptive step size.
      **kwargs: Optional keyword arguments to pass to `diffeqsolve`.

    Returns:
      x_final: The mini-batch at the end of the ODE.
      solution: A `PyTree` containing the ODE solution.
        See https://docs.kidger.site/diffrax/api/solution/ for more info.
    """

    def ode_wrapper(t, y, _):
      dx = self.drift_fn(y, jnp.ones(len(y)) * t)
      return dx

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_wrapper),
        self.solver,
        t0,
        t1,
        dt0,
        y0=x_init,
        stepsize_controller=self.stepsize_controller,
        **kwargs)
    (x_final,) = solution.ys
    return x_final, solution

  def ode_with_logp(self,
                    rng,
                    x_init,
                    t0,
                    t1,
                    dt0 = None,
                    **kwargs
                    ):
    """Run the PF-ODE of `x` and `logp(x)` forward or backward in time.

    Args:
      rng: A JAX pseudo-RNG for Hutchinson-Skilling trace estimation.
      x_init: The initial mini-batch of x(0) or x(T).
      t0: Initial time of the ODE.
      t1: Final time of the ODE. Can be > or < t0.
      dt0: Initial step size of the ODE solver with adaptive step size.
      **kwargs: Optional keyword arguments to pass to `diffeqsolve`.

    Returns:
      x_final: The mini-batch at the end of the ODE.
      logp_final: The total change in logp(x(t)) at the end of the ODE.
      solution: A `PyTree` containing the ODE solution.
        See https://docs.kidger.site/diffrax/api/solution/ for more info.
    """
    mini_batch_shape = x_init.shape

    def ode_wrapper(t, y, args):
      x, _ = y  # y is a tuple of (x, delta -logp).
      *args, rng = args
      dx, div = self.drift_div_fn(rng, x, jnp.ones(len(x)) * t)
      return dx, div

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(ode_wrapper),
        self.solver,
        stepsize_controller=self.stepsize_controller,
        adjoint=self.adjoint,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=(x_init, jnp.zeros(mini_batch_shape[0])),
        args=(rng,),
        **kwargs)
    (x_final,), (delta_logp,) = solution.ys
    return x_final, delta_logp, solution

  def logp_fn(self,
              rng,
              x,
              t0,
              t1,
              dt0):
    """Compute `logp(x)` with the probability flow ODE.

    Args:
      rng: A JAX pseudo-RNG for Hutchinson-Skilling trace estimation.
      x: A mini-batch of data.
      t0: Initial time of the ODE.
      t1: Final time of the ODE. Must be > t0 since we need to integrate
        forward in time.
      dt0: Initial step size of the ODE solver with adaptive step size.

    Returns:
      logp: The logp(x) for each sample in the mini-batch.
    """
    latent_code, delta_logp, _ = self.ode_with_logp(rng, x, t0, t1, dt0)
    logp = self.sde.prior_logp(latent_code) + delta_logp
    return logp
