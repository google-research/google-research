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

"""Diffusions for training and noise scheduling."""

import abc
import dataclasses
import functools
from typing import Any, List, Optional, Sequence, Union

from absl import logging
import chex
import flax
import flax.linen as nn
import gin
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import scipy
import tensorflow as tf

from d3pm.text import losses
from d3pm.text import metrics
from d3pm.text import model_utils
from d3pm.text import models
from d3pm.text import tasks
from d3pm.text import types
from d3pm.text import utils

gin.external_configurable(
    lambda: jax.lax.Precision.HIGHEST, module="Precision", name="HIGHEST")
gin.external_configurable(
    lambda: jax.lax.Precision.DEFAULT, module="Precision", name="DEFAULT")


class DiffusionSchedule:
  """A wrapper around a simple schedule function."""

  def __init__(self, schedule_fn, num_steps, is_constant=False):
    self._schedule_fn = schedule_fn
    self.num_steps = num_steps
    self.is_constant = is_constant

  def __call__(self, step):
    return self._schedule_fn(step)

  def __repr__(self):
    return f"DiffusionSchedule(steps: {self.num_steps}, is_constant: {self.is_constant})"


@dataclasses.dataclass
class MutualInformationSchedule:
  """Mutual information schedule marker, handled by diffusions that support it."""
  num_steps: int
  initial_distribution: chex.Array
  min_exponent: float = 1e-4
  max_exponent: float = 1e5
  interpolation_steps: int = 128


@gin.configurable(module="jump")
class DiscreteDiffusionBase(abc.ABC):
  """Base class for all matrix-noise schedules."""
  num_steps: int
  dim: int
  precision: Any = jax.lax.Precision.HIGHEST

  @abc.abstractmethod
  def stationary_probs(self, shape):
    """Returns probs for the stationary distribution."""

  @abc.abstractmethod
  def sample_stationary(self, key, shape):
    """Draws a sample from the stationary distribution (q(x_T))."""

  @property
  def has_state(self):
    """Indicates if the diffusion has state which needs to be set/updated."""
    return False

  def set_state(self, state):
    pass

  def reset_state(self):
    pass

  def update_state(self, state):
    pass

  def sample_t(self, key, shape=(1,)):
    """Samples batches of time steps to use."""

    num_steps = self.num_steps
    t = jrandom.randint(key, shape, minval=0, maxval=num_steps)
    return t

  @abc.abstractmethod
  def get_qt_given_q0(self,
                      q0,
                      t,
                      return_logits=False,
                      make_one_hot=False,
                      epsilon=1e-20):
    """Get q(x_t), the n-step posterior.

    For example, for t = 0, it returns q0 unchanged.

    Args:
      q0: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_t | x_0).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_t | x_0).
    """

  @abc.abstractmethod
  def sample_and_compute_posterior_q(self,
                                     key,
                                     x_0,
                                     t,
                                     samples=None,
                                     transition_probs=None,
                                     return_logits=True,
                                     return_transition_probs=False,
                                     transition_probs_in_logits=True,
                                     make_one_hot=True,
                                     epsilon=1e-20,
                                     step_size = 1):
    """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0).

    Args:
      key: a jax PRNGKey.
      x_0: an array containing x_0 samples. These are expected to be integral
        unless make_one_hot is False (in which case probabilities can be
        provided).
      t: the timestep to compute (as an int or integer array with shape that
        matches x_0.
      samples: if not None, use these samples to compute the posterior.
      transition_probs: precomputed transition probabilities.
      return_logits: if True, returns the (noisy) log of the probabilities.
      return_transition_probs: if true, returns the transition probs as well.
      transition_probs_in_logits: include transition probs in logits.
      make_one_hot: if True, will convert the input to a one_hot vector.
      epsilon: a small amount of noise to add to logits if needed.
      step_size: if provided, computes q(x_{t + step_size} | x_0), etc. This is
        used to sample fewer steps for ELBO evaluation on a longer trained
        model.

    Returns:
      a list of samples with the same shape as x_0 and the associated posterior
      probabilities (or logits).
    """


@gin.configurable(module="jump")
class DiscreteDiffusionMatrixBase(DiscreteDiffusionBase):
  """Base class for all matrix-noise schedulers."""
  num_steps: int
  dim: int
  precision: Any = jax.lax.Precision.HIGHEST

  def get(self, t):
    """Returns the transition matrix q(x_{t+1} | x_t)."""
    raise NotImplementedError

  def custom_product_fn(self, t):
    """Returns q(x_t | x_0), the product of the first t matrices."""
    raise NotImplementedError

  def supports_efficient_get(self):
    """Returns true if get() is implemented/efficient."""
    return False

  def supports_efficient_inference(self):
    """Returns true if custom_product_fn is implemented.

    The ontology of efficient_get and efficient_inference is this:
      * if efficient_inference is enabled, it is used to return q(x_t | x_0)
        without computing expensive products.
      * if efficient_get is enabled, get(...) is used to get the posterior of
        q(x_{t-1} | x_t, x_0). If not, get_q_given_q0 is called to get
        q(x_{t+1} | x_0), and qt_reverse is called to get the q(x_{t+1} | x_0).
    """
    return False

  def qt_reverse(self,
                 qt_plus_1,
                 t,
                 return_logits=False,
                 make_one_hot=False,
                 epsilon=1e-20):
    """Get q(x_{t+1} | x_t), the one-step posterior efficiently.

    Args:
      qt_plus_1: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_{t+1} | x_t).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_{t+1} | x_t).
    """
    raise NotImplementedError

  def get_qt_matrix(self, t):
    """Returns the matrix Q = q(x_t | x_0) materialized over all x_0."""
    if self.supports_efficient_inference():
      return self.custom_product_fn(t)

    logging.warning("WARNING: using inefficient matrix product.")

    # otherwise, multiply by the ith matrix in a for-loop.
    def product_fn(i, state):
      return jnp.matmul(self.get(i), state, precision=self.precision)

    final_product = jax.lax.fori_loop(0, t, product_fn, jnp.eye(self.dim))

    return final_product

  def get_qt_given_q0(self,
                      q0,
                      t,
                      return_logits=False,
                      make_one_hot=False,
                      epsilon=1e-20):
    """Get q(x_t), the n-step posterior.

    For example, for t = 0, it returns q0 unchanged.

    Args:
      q0: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_t | x_0).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_t | x_0).
    """
    logging.log_first_n(logging.INFO, "matmuls will use precision %s", 1,
                        self.precision)

    if make_one_hot:
      chex.assert_type(q0, jnp.int32)
      q0 = losses.onehot(labels=q0, num_classes=self.dim)

    chex.assert_type(q0, jnp.float32)

    # if efficient inference is supported, just return those matrices.
    if self.supports_efficient_inference():
      prob_at_time_t = jnp.einsum(
          "ij,...j", self.get_qt_matrix(t), q0, precision=self.precision)

      if return_logits:
        return jnp.log(prob_at_time_t + epsilon)
      else:
        return prob_at_time_t

    @chex.dataclass
    class ScanState:
      final_time: int  # target time
      q: Any

    # otherwise, multiply by the ith matrix in a for-loop.
    # TODO(jaaustin) maybe worth doing this in log space?
    @jax.remat
    def product_fn(state, current_time):
      cond = current_time < state.final_time
      transition = self.get(current_time)
      q_t_plus_1 = jnp.einsum(
          "ij,sj->si", transition, state.q, precision=self.precision)

      new_q = jnp.where(cond, q_t_plus_1, state.q)
      return ScanState(final_time=state.final_time, q=new_q), None

    init_val = ScanState(final_time=t, q=q0)
    idx = jnp.arange(self.num_steps)
    final_state, _ = jax.lax.scan(product_fn, init_val, idx)
    prob_at_time_t = final_state.q

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def sample_and_compute_posterior_q(self,
                                     key,
                                     x_0,
                                     t,
                                     samples=None,
                                     transition_probs=None,
                                     return_logits=True,
                                     return_transition_probs=False,
                                     transition_probs_in_logits=True,
                                     make_one_hot=True,
                                     epsilon=1e-20,
                                     step_size = 1):
    """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0).

    Args:
      key: a jax PRNGKey.
      x_0: an array containing x_0 samples. These are expected to be integral
        unless make_one_hot is False (in which case probabilities can be
        provided).
      t: the timestep to compute (as an int or integer array with shape that
        matches x_0.
      samples: if not None, use these samples to compute the posterior.
      transition_probs: precomputed transition probabilities.
      return_logits: if True, returns the (noisy) log of the probabilities.
      return_transition_probs: if true, returns the transition probs as well.
      transition_probs_in_logits: include transition probs in logits.
      make_one_hot: if True, will convert the input to a one_hot vector.
      epsilon: a small amount of noise to add to logits if needed.
      step_size: if provided, computes q(x_{t + step_size} | x_0), etc. This is
        used to sample fewer steps for ELBO evaluation on a longer trained
        model.

    Returns:
      a list of samples with the same shape as x_0 and the associated posterior
      probabilities (or logits).
    """
    logging.info("using precision %s", self.precision)

    chex.assert_rank(key, 1)

    dim = self.dim
    t = jnp.asarray(t)

    if make_one_hot:
      chex.assert_type(x_0, jnp.int32)
      x_0 = losses.onehot(x_0, dim).reshape(x_0.shape + (dim,))

    chex.assert_type(x_0, jnp.float32)
    chex.assert_type(t, jnp.int32)

    prob_at_time_t = self.get_qt_given_q0(q0=x_0, t=t)

    ## most methods support efficiently returning the t-th transition matrix
    ## if so, we use that. Otherwise we recompute the t+1th probability.
    if self.supports_efficient_get():
      if step_size > 1:
        transition_matrix = jnp.eye(self.dim)

        for i in range(step_size):
          transition_matrix = self.get(t + i) @ transition_matrix

      else:
        transition_matrix = self.get(t)

      prob_at_time_t_plus_one = jnp.einsum(
          "ij,...j->...i",
          transition_matrix,
          prob_at_time_t,
          precision=self.precision)

    else:
      prob_at_time_t_plus_one = self.get_qt_given_q0(q0=x_0, t=t + step_size)

    if samples is None and transition_probs is not None:
      raise ValueError("samples were not provided but transition_probs were.")

    ## if samples are provided, we use those. otherwise, we sample more.
    if samples is None:
      logits = jnp.log(prob_at_time_t_plus_one + epsilon)
      samples = jrandom.categorical(key, logits, axis=-1)

    ## we can optionally provide transition probs from another call to this
    ## function. If not, we recompute this. For most methods, we can reuse the
    ## transition matrix. If we didn't compute it, our method must support
    ## qt_reverse which usually computes efficient backwards VJPs.

    if transition_probs is None:
      if self.supports_efficient_get():
        transition_probs = transition_matrix[samples]
      else:
        if step_size > 1:
          transition_probs = losses.onehot(samples, self.dim)
          for i in range(step_size):
            transition_probs = self.qt_reverse(
                qt_plus_1=transition_probs,
                make_one_hot=False,
                t=t + step_size - 1 - i)
        else:
          transition_probs = self.qt_reverse(
              qt_plus_1=samples, make_one_hot=True, t=t)

    if not transition_probs_in_logits and not return_logits:
      raise ValueError(
          "Cannot exclude transition probs from logits if return_logits is false."
      )

    if return_logits:
      # for numerical stability, we can compute log(a*b) = log(a) + log(b)
      posterior_logits = jnp.log(prob_at_time_t + epsilon)

      if transition_probs_in_logits:
        posterior_logits += jnp.log(transition_probs + epsilon)

      if return_transition_probs:
        return posterior_logits, samples, transition_probs
      else:
        return posterior_logits, samples
    else:

      ## here we hope this never actually sums to zero. There's a chance
      ## this will produce NaN gradients, but that's OK because they'll be
      ## skipped.
      posterior = transition_probs * prob_at_time_t
      denominator = jnp.sum(posterior, axis=-1, keepdims=True)
      posterior = posterior / denominator
      # posterior = jnp.where(
      #     denominator == 0., 1 / dim, posterior / denominator)

      if return_transition_probs:
        return posterior, samples, transition_probs
      else:
        return posterior, samples


@gin.configurable
class BetaDiagonalDiffusion(DiscreteDiffusionMatrixBase):
  """A simple diffusion that diffuses away from the identity matrix."""

  def __init__(self,
               dim,
               schedule,
               precision=jax.lax.Precision.HIGHEST,
               use_fast_inference=True):
    """A simple diffusion for beta-diagonal policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiffusionSchedule object to use for rate information
      precision: matmul precision.
      use_fast_inference: if False, uses a slower, brute force approach.
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.precision = precision
    self.use_fast_inference = use_fast_inference
    self.dim = dim

    self.state = self._create_state()

    logging.info(
        "Creating BetaDiagonal scheduler with supports_efficient_inference: %s and schedule %s.",
        self.supports_efficient_inference(), self.schedule)

  def _create_state(self):
    """Creates state to use for computing products efficiently."""

    def poly_for_loop_body(i, p):
      beta = self.schedule(i)
      p = np.polymul(p, np.array([beta, (1 - beta)], np.float64))
      p[-2] += p[:-2].sum()
      p = p[-2:]
      return p

    p = np.array([0., 1.], dtype=np.float64)

    state = np.zeros((self.num_steps + 1, 2), np.float64)
    for i in range(self.num_steps):
      state[i] = p
      p = poly_for_loop_body(i, p)

    state[-1] = np.array([1.0, 0.0])

    return jnp.asarray(state, dtype=jnp.float32)

  def stationary_probs(self, shape):
    """Returns probabilities over the stationary distribution (uniform)."""
    return jnp.ones(shape + (self.dim,)) / self.dim

  def sample_stationary(self, key, shape):
    return jrandom.randint(key, shape, 0, self.dim)

  def supports_efficient_inference(self):
    return self.use_fast_inference

  def custom_product_fn(self, t):
    """Returns product of first n matrices. Only supported for beta constant."""
    dim = self.dim

    ## this is also fun magic. (beta * I + (1 - beta) / D 1 1^T)^T is solvable
    ## analytically for fixed betas.
    if self.schedule.is_constant:
      beta = self.schedule(0)
      return (1 - beta)**t * jnp.eye(dim) + (1 -
                                             (1 - beta)**t) / dim * jnp.ones(
                                                 (dim, dim))

    ## fun polynomial magic (over R[n] / X^2 - X). If X = 1 1^T / D, X^2 = X,
    ## and I*A = A, so \prod (1 - \beta) * I + \beta 1 1^T / D can be seen as
    ## multiplication of (1 - \beta)  + \beta X over a quotient field.
    else:
      p = self.state[t]
      return p[1] * jnp.eye(dim) + p[0] * jnp.ones((dim, dim)) / dim

  def get(self, t):
    """Returns the transition matrix at time t.

    Note that this is not exactly 1 - beta probability of staying the same. We
    instead have (1 - beta + beta / D) probability of staying the same.

    Args:
      t: the timestep.

    Returns:
      a transition matrix at time t.
    """

    beta = self.schedule(t)
    dim = self.dim
    return (1 - beta) * jnp.eye(dim) + beta * jnp.ones((dim, dim)) / dim

  def qt_reverse(self,
                 qt_plus_1,
                 t,
                 return_logits=False,
                 make_one_hot=False,
                 epsilon=1e-20):
    """Get q(x_{t+1} | x_t), the one-step posterior efficiently.

    Args:
      qt_plus_1: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_{t+1} | x_t).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_{t+1} | x_t).
    """

    if make_one_hot:
      chex.assert_type(qt_plus_1, jnp.int32)
      qt_plus_1 = losses.onehot(labels=qt_plus_1, num_classes=self.dim)

    chex.assert_type(qt_plus_1, jnp.float32)

    beta = self.schedule(t)

    # p[1] is same, p[0] is change
    def reverse_fn(qt_plus_1):
      uniform = jnp.ones_like(qt_plus_1) / qt_plus_1.shape[-1]
      return beta * uniform + (1 - beta) * qt_plus_1

    prob_at_time_t = jax.vmap(reverse_fn)(qt_plus_1)

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def get_qt_given_q0(self,
                      q0,
                      t,
                      return_logits = False,
                      make_one_hot = False,
                      epsilon = 1e-20):
    """Get q(x_t), the n-step posterior.

    Can do efficiently for masks.

    For example, for t = 0, it returns q0 unchanged.

    Args:
      q0: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_t | x_0).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_t | x_0).
    """
    if not self.supports_efficient_inference():
      return super().get_qt_given_q0(
          q0,
          t,
          return_logits=return_logits,
          make_one_hot=make_one_hot,
          epsilon=epsilon)

    logging.log_first_n(logging.INFO, "matmuls will use precision %s", 1,
                        self.precision)

    if make_one_hot:
      chex.assert_type(q0, jnp.int32)
      q0 = losses.onehot(labels=q0, num_classes=self.dim)

    chex.assert_type(q0, jnp.float32)
    chex.assert_rank(q0, 2)

    # p[1] is probability of staying the same. p[0] is prob of switching.
    p = self.state[t]

    def get_qt_fn(q0):
      uniform = jnp.ones_like(q0) / q0.shape[-1]
      return p[1] * q0 + p[0] * uniform

    prob_at_time_t = jnp.where(t == 0, q0, jax.vmap(get_qt_fn)(q0))

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def supports_efficient_get(self):
    return not self.use_fast_inference


@gin.configurable
class MaskBetaDiagonalDiffusion(DiscreteDiffusionMatrixBase):
  """A simple diffusion that diffuses away from the identity matrix."""

  def __init__(
      self,
      dim,
      schedule,
      mask_fraction=0.66666,
      precision=jax.lax.Precision.HIGHEST,
      use_fast_inference=True,
  ):
    """A beta + mask diffusion schedule.

    This uses the standard schedule to converge to masks (1 / T - 1), but uses
    the provided schedule to control how rapidly probability mass diffuses
    randomly.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: ignored.
      mask_fraction: rate at which we transition to masks.
      precision: matmul precision.
      use_fast_inference: if False, uses a slower, brute force approach.
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.mask_fraction = mask_fraction
    self.scale = 1 / mask_fraction
    self.precision = precision
    self.use_fast_inference = use_fast_inference
    self.dim = dim

    if self.use_fast_inference:
      self._create_state()

    logging.info(
        "Creating MaskBetaDiagonal scheduler with supports_efficient_inference: %s and schedule %s.",
        self.supports_efficient_inference(), self.schedule)

  def _create_state(self):
    """Creates state to use for computing products efficiently."""

    def poly_for_loop_body(i, p):
      beta = 1 / (self.num_steps - i)
      p = np.polymul(p, np.array([beta, (1 - beta)], np.float64))
      p[-2] += p[:-2].sum()
      p = p[-2:]
      return p

    p = np.array([0., 1.], dtype=np.float64)

    state = np.zeros((self.num_steps + 1, 2), np.float64)
    for i in range(self.num_steps):
      state[i] = p
      p = poly_for_loop_body(i, p)

    state[-1] = np.array([1.0, 0.0])

    betas = 1 / (self.scale * self.num_steps - np.arange(self.num_steps))
    betas = jnp.concatenate([jnp.array([0.]), betas])
    self.alphas = np.cumprod(1 - betas)

    self.state = jnp.asarray(state, dtype=jnp.float32)

  def stationary_probs(self, shape):
    """Stationary distribution is one-hot at mask token."""
    mask = (1 - self.mask_fraction) * jnp.ones(shape + (self.dim,)) / (
        self.dim - 1)
    return mask.at[Ellipsis, -1].set(self.mask_fraction)

  def sample_stationary(self, key, shape):
    """Stationary distribution is one-hot at mask token."""
    probs = self.stationary_probs(shape)
    return jrandom.categorical(key, jnp.log(probs + 1e-20))

  def supports_efficient_inference(self):
    return self.use_fast_inference

  def custom_product_fn(self, t):
    """Returns product of first n matrices. Only supported for beta constant."""
    dim = self.dim

    p = self.state[t]
    alpha = self.alphas[t]

    mat = alpha * (
        p[1] * jnp.eye(dim) + p[0] * jnp.ones((dim, dim)) / (dim - 1))

    mat = mat.at[-1].set(1 - alpha)
    mat = mat.at[:, -1].set(0.0)
    mat = mat.at[-1, -1].set(1.0)

    return mat

  def get(self, t):
    """Returns the transition matrix at time t.

    Args:
      t: the timestep.

    Returns:
      a transition matrix at time t.
    """

    beta = self.schedule(t)
    mask_beta = 1 / (self.scale * self.num_steps - t)

    dim = self.dim
    beta_matrix = (1 - beta) * jnp.eye(dim) + beta * jnp.ones((dim, dim)) / (
        dim - 1)

    mat = (1 - mask_beta) * beta_matrix
    mat = mat.at[-1].set(mask_beta)
    mat = mat.at[:, -1].set(0.0)
    mat = mat.at[-1, -1].set(1.0)

    return mat

  def supports_efficient_get(self):
    return True


@gin.configurable
class MaskDiffusion(DiscreteDiffusionMatrixBase):
  """A simple schedule that diffuses away from the identity matrix."""

  def __init__(self,
               dim,
               schedule,
               precision=jax.lax.Precision.HIGHEST,
               use_fast_inference=True):
    """A simple scheduler for masking policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiffusionSchedule object for scheduling rates.
      precision: matmul precision.
      use_fast_inference: if False, uses a slower, brute force approach.
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.use_fast_inference = use_fast_inference
    self.precision = precision
    self.dim = dim  # allow mask
    self.state = self._create_state()

    logging.info(
        "Creating MaskDiffusion with supports_efficient_inference: %s.",
        self.supports_efficient_inference())

  def _create_state(self):
    """Initializes values used by the get function."""
    betas = np.concatenate(
        [np.array([0.0]),
         self.schedule(np.arange(self.num_steps))]).astype(np.float64)
    alphas = 1 - betas
    state = np.cumprod(alphas)
    state[-1] = 0.0

    return jnp.asarray(state, dtype=jnp.float32)

  def supports_efficient_inference(self):
    return self.use_fast_inference

  def stationary_probs(self, shape):
    """Stationary distribution is one-hot at mask token."""
    sample = jnp.full(shape, self.dim - 1)
    probs = losses.onehot(sample, self.dim)
    return probs

  def sample_stationary(self, key, shape):
    """Stationary distribution is one-hot at mask token."""
    return jnp.full(shape, self.dim - 1)

  def custom_product_fn(self, t):
    """Returns product of first n matrices. Only supported for beta constant."""
    dim = self.dim

    if self.schedule.is_constant:
      beta = self.schedule(0)
      return (1 - beta)**t * jnp.eye(dim) + (1 -
                                             (1 - beta)**t) * self._get_mask()

    else:
      p = self.state[t]
      return p * jnp.eye(dim) + (1 - p) * self._get_mask()

  def _get_mask(self):
    dim = self.dim
    return jnp.ones((dim, dim)) * (jnp.arange(0, dim)[:, None]
                                   == (dim - 1)).astype(jnp.float32)

  def get(self, t):
    beta = self.schedule(t)
    dim = self.dim

    return (1 - beta) * jnp.eye(dim) + beta * self._get_mask()

  def qt_reverse(self,
                 qt_plus_1,
                 t,
                 return_logits=False,
                 make_one_hot=False,
                 epsilon=1e-20):
    """Get q(x_{t+1} | x_t), the one-step posterior efficiently.

    Args:
      qt_plus_1: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_{t+1} | x_t).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_{t+1} | x_t).
    """

    if make_one_hot:
      chex.assert_type(qt_plus_1, jnp.int32)
      qt_plus_1 = losses.onehot(labels=qt_plus_1, num_classes=self.dim)

    chex.assert_type(qt_plus_1, jnp.float32)

    beta = self.schedule(t)

    def reverse_fn(qt_plus_1):
      non_mask_prob = (1 - beta) * qt_plus_1[:-1] + beta * qt_plus_1[-1:]
      prob_at_time_t = losses.onehot(jnp.array(self.dim - 1),
                                     self.dim) * qt_plus_1[-1:]
      prob_at_time_t = prob_at_time_t.at[:-1].set(non_mask_prob)
      return prob_at_time_t

    prob_at_time_t = jax.vmap(reverse_fn)(qt_plus_1)

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def get_qt_given_q0(self,
                      q0,
                      t,
                      return_logits = False,
                      make_one_hot = False,
                      epsilon = 1e-20):
    """Get q(x_t), the n-step posterior.

    Can do efficiently for masks.

    For example, for t = 0, it returns q0 unchanged.

    Args:
      q0: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_t | x_0).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_t | x_0).
    """
    if not self.supports_efficient_inference():
      return super().get_qt_given_q0(
          q0,
          t,
          return_logits=return_logits,
          make_one_hot=make_one_hot,
          epsilon=epsilon)

    logging.log_first_n(logging.INFO, "matmuls will use precision %s", 1,
                        self.precision)

    if make_one_hot:
      chex.assert_type(q0, jnp.int32)
      q0 = losses.onehot(labels=q0, num_classes=self.dim)

    chex.assert_type(q0, jnp.float32)
    chex.assert_rank(q0, 2)

    # p is probability of staying the same. (1 - p) is prob of masking.
    p = self.state[t]

    def get_qt_fn(q0):
      non_mask_prob = p * q0[:-1]
      mask_prob = (1 - non_mask_prob.sum())

      prob_at_time_t = mask_prob * losses.onehot(
          jnp.array(self.dim - 1), self.dim)
      prob_at_time_t = prob_at_time_t.at[:-1].set(non_mask_prob)
      return prob_at_time_t

    prob_at_time_t = jnp.where(t == 0, q0, jax.vmap(get_qt_fn)(q0))

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def supports_efficient_get(self):
    return not self.use_fast_inference


@gin.configurable
class BandDiagonalDiffusion(DiscreteDiffusionMatrixBase):
  """A simple diffusion that diffuses away from the identity matrix."""

  def __init__(
      self,
      dim,
      schedule,
      width=5,
      precision=jax.lax.Precision.HIGHEST,
  ):
    """A simple diffusion for band-diagonal policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiffusionSchedule to use for rate information.
      width: width of the band diagonal.
      precision: matmul precision.
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.precision = precision
    self.dim = dim
    self.width = width

    logging.info("Creating BandDiagonal scheduler with %s.", self.schedule)

  def stationary_probs(self, shape):
    """Stationary probs are still uniform (for band wide enough)."""
    return jnp.ones(shape + (self.dim,)) / self.dim

  def sample_stationary(self, key, shape):
    """Samples uniformly from the vocab."""
    return jrandom.randint(key, shape, 0, self.dim)

  def supports_efficient_inference(self):
    return False

  def get(self, t):
    """We compute a band diagonal matrix with width width."""
    beta = self.schedule(t)
    dim = self.dim
    width = self.width

    band = jnp.tri(
        dim, dim, width // 2, dtype=jnp.int32) - jnp.tri(
            dim, dim, -width // 2, dtype=jnp.int32)

    arr = band / band.sum(0, keepdims=True)

    return beta * arr + (1 - beta) * jnp.eye(dim)

  def supports_efficient_get(self):
    return True


@gin.configurable
class GaussianPrecomputedDiffusion(DiscreteDiffusionMatrixBase):
  """A simple diffusion that diffuses away from the identity matrix."""

  def __init__(
      self,
      dim,
      schedule,
      precision=jax.lax.Precision.HIGHEST,
  ):
    """A simple diffusion for band-diagonal policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiffusionSchedule to use for rate information.
      precision: matmul precision.
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.precision = precision
    self.dim = dim

    logging.info("Creating GaussianBandDiagonal scheduler with %s.",
                 self.schedule)

    transitions = np.zeros((self.num_steps, self.dim, self.dim), np.float64)
    products = np.zeros((self.num_steps, self.dim, self.dim), np.float64)

    mat = np.eye(self.dim, dtype=np.float64)
    for t in range(self.num_steps):
      products[t] = mat
      transitions[t] = self._get_gaussian_transition_mat(t)
      mat = transitions[t] @ mat

    self.transitions = jnp.asarray(transitions, dtype=jnp.float32)
    self.products = jnp.asarray(products, dtype=jnp.float32)

  def stationary_probs(self, shape):
    """Stationary probs are still uniform (for band wide enough)."""
    return jnp.ones(shape + (self.dim,)) / self.dim

  def sample_stationary(self, key, shape):
    """Samples uniformly from the vocab."""
    return jrandom.randint(key, shape, 0, self.dim)

  def _get_gaussian_transition_mat(self, t):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_classes, num_classes).
    """
    transition_bands = self.dim - 1
    beta = self.schedule(t)

    mat = np.zeros((self.dim, self.dim), dtype=np.float64)

    values = np.linspace(start=0., stop=self.dim, num=self.dim, endpoint=True)
    values = values * 2. / (self.dim - 1.)
    values = values[:transition_bands + 1]
    values = -values * values / beta

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values, axis=0)
    values = values[transition_bands:]

    for k in range(1, transition_bands + 1):
      off_diag = np.full(
          shape=(self.dim - k,), fill_value=values[k], dtype=np.float64)

      mat += np.diag(off_diag, k=k)
      mat += np.diag(off_diag, k=-k)

    diag = 1. - mat.sum(1)
    mat += np.diag(diag, k=0)

    return mat

  def get(self, t):
    """We compute a band diagonal matrix with width width."""
    return self.transitions[t]

  def custom_product_fn(self, t):
    return self.products[t]

  def supports_efficient_inference(self):
    return True

  def supports_efficient_get(self):
    return True


@gin.configurable
class MaskGaussianPrecomputedDiffusion(DiscreteDiffusionMatrixBase):
  """A simple diffusion that diffuses away from the identity matrix."""

  def __init__(
      self,
      dim,
      schedule,
      precision=jax.lax.Precision.HIGHEST,
  ):
    """A simple diffusion for band-diagonal policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiffusionSchedule to use for rate information.
      precision: matmul precision.
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.precision = precision
    self.dim = dim

    logging.info("Creating GaussianBandDiagonal scheduler with %s.",
                 self.schedule)

    transitions = np.zeros((self.num_steps, self.dim, self.dim), np.float64)
    products = np.zeros((self.num_steps + 1, self.dim, self.dim), np.float64)

    mat = np.eye(self.dim, dtype=np.float64)
    for t in range(self.num_steps):
      beta = 1 / (self.num_steps - t)
      products[t] = mat
      transition = self._get_gaussian_transition_mat(t, dim=self.dim - 1)
      transitions[t, :self.dim - 1, :self.dim - 1] = (1 - beta) * transition
      transitions[t, -1] = beta
      transitions[t, -1, -1] = 1.
      mat = transitions[t] @ mat

    products[-1] = self.stationary_probs((self.dim,)).T

    self.transitions = jnp.asarray(transitions, dtype=jnp.float32)
    self.products = jnp.asarray(products, dtype=jnp.float32)

  def stationary_probs(self, shape):
    """Stationary distribution is one-hot at mask token."""
    sample = jnp.full(shape, self.dim - 1)
    probs = losses.onehot(sample, self.dim)
    return probs

  def sample_stationary(self, key, shape):
    """Stationary distribution is one-hot at mask token."""
    return jnp.full(shape, self.dim - 1)

  def _get_gaussian_transition_mat(self, t, dim):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    Args:
      t: timestep. integer scalar (or numpy array?)
      dim: dimension of transition to create.

    Returns:
      Q_t: transition matrix. shape = (num_classes, num_classes).
    """
    transition_bands = dim - 1
    beta = self.schedule(t)

    mat = np.zeros((dim, dim), dtype=np.float64)

    values = np.linspace(start=0., stop=dim, num=dim, endpoint=True)
    values = values * 2. / (dim - 1.)
    values = values[:transition_bands + 1]
    values = -values * values / beta

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = scipy.special.softmax(values, axis=0)
    values = values[transition_bands:]

    for k in range(1, transition_bands + 1):
      off_diag = np.full(
          shape=(dim - k,), fill_value=values[k], dtype=np.float64)

      mat += np.diag(off_diag, k=k)
      mat += np.diag(off_diag, k=-k)

    diag = 1. - mat.sum(1)
    mat += np.diag(diag, k=0)

    return mat

  def get(self, t):
    """We compute a band diagonal matrix with width width."""
    return self.transitions[t]

  def custom_product_fn(self, t):
    return self.products[t]

  def supports_efficient_inference(self):
    return True

  def supports_efficient_get(self):
    return True


@gin.configurable
def mask_matrix_exponential(exponent, dim, mask_token=3):
  """Mask diffusion as matrix exponential.

  A mask diffusion can be written as a matrix exponential of a particular
  rate matrix

      [-1  0  0  0 ...  0  0]
      [ 0 -1  0  0 ...  0  0]
      [ 0  0 -1  0 ...  0  0]
      ...
      [ 0  0  0  0 ... -1  0]
      [ 1  1  1  1 ...  1  0]

  where here we've written the mask token as the last token (not always the
  case), and each column of this matrix is the rates of transitions away from
  each state (and so sums to zero in each column).

  Eigenvectors:

      [  1, 1, 1, 1, ..., 1, 1 ]  with eigenvalue 0
      [ ... 0, 1, 0, ..., 0, 0 ]  with eigenvalue -1 for each non-mask index

  We can then derive the matrix by exponentiating the eigenvalues, and scaling
  the appropriate subspaces: we scale down the diagonal, then compensate with
  the mask so that the result sums to 1.

  Args:
    exponent: diagonal value for matrix exponential.
    dim: dimension of matrix.
    mask_token: which token to use as the mask transition.

  Returns:
    an exponentiated transition matrix.
  """
  one_hot_on_mask = jax.nn.one_hot(mask_token, dim)
  if exponent == "stationary":
    return one_hot_on_mask
  diagonal_value = jnp.exp(-exponent)
  result = (
      diagonal_value * jnp.eye(dim) +
      (1 - diagonal_value) * one_hot_on_mask[:, None])
  return result


@gin.configurable
def uniform_diagonal_matrix_exponential(exponent, dim):
  """Uniform/diagonal (beta diagonal) diffusion as matrix exponential.

  Uniform/diagonal diffusion can be written as a matrix exponential of a
  particular rate matrix

      [(x-1)  x  x  x ...  x  x]
      [ x (x-1)  x  x ...  x  x]
      [ x  x (x-1)  x ...  x  x]
      ...
      [ x  x  x  x ... (x-1)  x]
      [ x  x  x  x ...  x (x-1)]

  where x = 1/dim.

  Eigenvectors:

      [  1, 1, 1, 1, ..., 1, 1 ] with eigenvalue 0
      The entire subspace of vectors that sum to zero, with eigenvalue -1.

  We can then derive the matrix by exponentiating the eigenvalues, and scaling
  the appropriate subspaces: we scale down the subspace of vectors that sum
  to zero by shifting an identity matrix so that it sums to zero and scaling
  that.

  Args:
    exponent: diagonal value for matrix exponential.
    dim: dimension of matrix.

  Returns:
    an exponentiated transition matrix.
  """
  uniform_vector = jnp.full([dim], 1 / dim)
  if exponent == "stationary":
    return uniform_vector
  subspace_decay = jnp.exp(-exponent)
  result = (subspace_decay * (jnp.eye(dim) - 1 / dim) + 1 / dim)
  return result


@gin.configurable
def hierarchical_matrix_exponential(exponent, dim, level_rates=gin.REQUIRED):
  """Hierarchical clustering transition matrix.

  Suppose each token is assigned a binary string such that tokens that are more
  similar are numerically closer (differ only in least-significant digits.)
  Consider a transition operator defined as follows: given a bit string
  (b0, b1, b2, ..., bk),
    - with log prob l0, flip to (~b0, {random suffix})
    - with log prob l1, flip to (b0, ~b1, {random suffix})
    - with log prob l2, flip to (b0, b1, ~b2, {random suffix})
    - ... etc

  If l0 is small and lk is large, we are more likely to make small jumps than
  we are to make large ones.

  Args:
    exponent: Exponent to raise the base rate matrix to. At time 0 nothing
      changes.
    dim: Dimension.
    level_rates: List of floats giving rates that transitions happen for each
      bit. First entry is the rate of global transitions to

  Returns:
    a transition matrix.
  """
  uniform_vector = jnp.full([dim], 1 / dim)
  if exponent == "stationary":
    return uniform_vector

  N = len(level_rates)  # pylint: disable=invalid-name
  assert dim == 2**N

  def build(i):
    if i == N:
      return jnp.array(1.)

    # Let i be some bit index. Consider the two groups of states
    # A: (b0, ..., b{i-1}, 0, [anything])
    # B: (b0, ..., b{i-1}, 1, [anything])
    # Note that the rate of transitioning between each of these two groups
    # is determined only by level_rates[i]. Furthermore, we know that
    # if a transition occurs between A and B, we end up at a uniformly random
    # position wherever we ended up.
    # Now let's imagine we start in A, and we track whether we end up:
    # - never leaving A, and maybe having lower correlations at lower levels of
    #   detail
    # - leaving A and going to a uniform in B
    # - leaving A, going to B, then returning to a uniform in A that's
    #   uncorrelated with original position.

    r = exponent * level_rates[i]
    # not the most efficient way but the easiest to do and should be super
    # cheap regardless
    # pyformat:disable
    # pylint:disable=bad-whitespace
    local_mat = jnp.array([
        # A1, A2, B
        [-r,  0,  r],  # A1
        [ 0, -r,  r],  # A2
        [ 0,  r, -r],  # B
    ])
    # pylint:enable=bad-whitespace
    # pyformat:enable

    local_probs = jax.scipy.linalg.expm(local_mat)

    stay_prob = local_probs[0, 0]
    leave_prob = local_probs[0, 2]
    leave_and_return_prob = local_probs[0, 1]

    inner_block = build(i + 1)

    blockdim = 2**(N - i - 1)
    block1 = stay_prob * inner_block + leave_and_return_prob / blockdim
    block2 = jnp.broadcast_to(leave_prob / blockdim, (blockdim, blockdim))
    return jnp.block([
        [block1, block2],
        [block2, block1],
    ])

  return build(0)


@gin.configurable
class ClosedFormMatrixExponentialDiffusion(DiscreteDiffusionMatrixBase):
  """Base class for matrices with a closed-form matrix exponential representation."""

  def __init__(self,
               dim,
               schedule,
               builder_fn,
               use_fast_inference=True,
               override_last_step_to_mask=True,
               mask_token=None):
    """A simple scheduler for masking policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: The schedule to use. Only mutual information schedules are
        supported.
      builder_fn: Function that computes a matrix exponential.
      use_fast_inference: if False, uses a slower, brute force approach.
      override_last_step_to_mask: Whether to add an extra step that always
        masks.
      mask_token: Token ID of the mask token.
    """
    self.use_fast_inference = use_fast_inference
    self.builder_fn = builder_fn
    self.schedule = schedule
    self.dim = dim
    self.num_steps = schedule.num_steps

    assert isinstance(schedule, MutualInformationSchedule)

    if mask_token is None:
      mask_token = dim - 1
    self.override_last_step_to_mask = override_last_step_to_mask
    self.mask_token = mask_token

    query_exponents, query_info_removals = (
        model_utils.compute_information_removal_samples_closed_form(
            lambda t: self.builder_fn(t, self.dim),
            initial_distribution=self.schedule.initial_distribution,
            min_exponent=self.schedule.min_exponent,
            max_exponent=self.schedule.max_exponent,
            interpolation_steps=self.schedule.interpolation_steps))
    if self.override_last_step_to_mask:
      num_interpolated_steps = self.num_steps - 1
    else:
      num_interpolated_steps = self.num_steps
    _, middle_exponents = (
        model_utils.build_mutual_information_schedule(num_interpolated_steps,
                                                      query_exponents,
                                                      query_info_removals))
    exponents = jnp.concatenate([jnp.zeros([1]), middle_exponents])
    if self.override_last_step_to_mask:
      exponents = jnp.concatenate([exponents, exponents[-1][None]])

    self.exponents = exponents

    logging.info(
        "Creating ClosedFormMatrixExponentialDiffusion with "
        "supports_efficient_inference: %s, builder_fn: %s, exponents: %s",
        self.supports_efficient_inference(), self.builder_fn, self.exponents)

  def supports_efficient_inference(self):
    return self.use_fast_inference

  def stationary_probs(self, shape):
    """Look up stationary distribution."""
    if self.override_last_step_to_mask:
      result = jax.nn.one_hot(self.mask_token, self.dim)
    else:
      result = self.builder_fn("stationary", self.dim)
    return jnp.broadcast_to(result, shape + (self.dim,))

  def sample_stationary(self, key, shape):
    return jax.random.categorical(key, jnp.log(self.stationary_probs(shape)))

  def _mask_mat(self):
    return jnp.broadcast_to(
        jax.nn.one_hot(self.mask_token, self.dim)[:, None],
        (self.dim, self.dim))

  def custom_product_fn(self, t):
    """Returns product of first n matrices."""
    the_product = self.builder_fn(self.exponents[t], self.dim)
    if self.override_last_step_to_mask:
      # not sure if this actually happens?
      return jnp.where(t == self.num_steps, self._mask_mat(), the_product)
    else:
      return the_product

  def get(self, t):
    beta = self.exponents[t + 1] - self.exponents[t]
    the_matrix = self.builder_fn(beta, self.dim)
    if self.override_last_step_to_mask:
      return jnp.where(t == self.num_steps - 1, self._mask_mat(), the_matrix)
    else:
      return the_matrix

  def supports_efficient_get(self):
    return True


@gin.configurable
class AutoRegressiveDiffusion(DiscreteDiffusionBase):
  """A diffusion model that should be equivalent to an autoregressive model."""

  def __init__(self, dim, schedule, precision=jax.lax.Precision.HIGHEST):
    """A simple scheduler for masking policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: unused (except to determine sequence length)
      precision: the precision of the matrix multiplies (ignored).
    """

    self.num_steps = schedule.num_steps
    self.schedule = schedule
    self.precision = precision
    self.dim = dim  # allow mask

    logging.info("Creating AutoRegressiveDiffusion model.")

  def stationary_probs(self, shape):
    """Stationary distribution is one-hot at mask token."""
    sample = jnp.full(shape, self.dim - 1)
    probs = losses.onehot(sample, self.dim)
    return probs

  def sample_stationary(self, key, shape):
    return jnp.full(shape, self.dim - 1)

  def get_qt_given_q0(self,
                      q0,
                      t,
                      return_logits=False,
                      make_one_hot=False,
                      epsilon=1e-20):
    """Get q(x_t | x_0), the t-step posterior."""
    if make_one_hot:
      chex.assert_type(q0, jnp.int32)
      q0 = losses.onehot(labels=q0, num_classes=self.dim)

    chex.assert_type(q0, jnp.float32)
    assert q0.ndim >= 2, "AutoRegresssiveDiffusion requires a sequence."
    assert q0.shape[-2] == self.num_steps, (
        "sequence must have same shape as number of "
        "diffusion steps.")

    to_mask = self.num_steps - t

    mask = jnp.zeros_like(q0)
    mask = mask.at[Ellipsis, :, -1].set(1.0)
    index_mask = jnp.arange(mask.shape[-2]) >= to_mask
    prob_at_time_t = jnp.where(index_mask[:, None], mask, q0)

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def sample_and_compute_posterior_q(self,
                                     key,
                                     x_0,
                                     t,
                                     samples=None,
                                     transition_probs=None,
                                     return_logits=True,
                                     return_transition_probs=False,
                                     transition_probs_in_logits=True,
                                     make_one_hot=True,
                                     epsilon=1e-20):
    """Samples from q(x_{t+1} | x_0), then computes q(x_t | x_{t+1}, x_0)."""
    del transition_probs, transition_probs_in_logits

    chex.assert_rank(key, 1)

    dim = self.dim
    t = jnp.asarray(t)

    if make_one_hot:
      chex.assert_type(x_0, jnp.int32)
      x_0 = losses.onehot(x_0, dim).reshape(x_0.shape + (dim,))

    chex.assert_type(x_0, jnp.float32)
    chex.assert_type(t, jnp.int32)

    if samples is None:
      logits = self.get_qt_given_q0(q0=x_0, t=t + 1, return_logits=True)
      samples = jrandom.categorical(key, logits, axis=-1)

    posterior = self.get_qt_given_q0(q0=x_0, t=t)
    sample_probs = losses.onehot(samples, dim)

    # what am I doing here? take prefix from sample, suffix is all mask, and
    # position t is taken from x_0. [0, 1, 2] -> [0, 1, MASK] -> [0, MASK, MASK]
    # -> [MASK, MASK, MASK]
    # for instance, if t == 0, to_mask is 40 (let's say), and we want everything
    # but the very last position from the sample, so we want to mask up to 38,
    # then 39 is taken from x0.
    to_mask = self.num_steps - t
    mask = jnp.arange(posterior.shape[-2]) >= to_mask - 1
    posterior = jnp.where(mask[:, None], posterior, sample_probs)

    if return_logits:
      posterior_logits = jnp.log(posterior + epsilon)
      if return_transition_probs:
        return posterior_logits, samples, None
      else:
        return posterior_logits, samples
    else:
      if return_transition_probs:
        return posterior, samples, None
      else:
        return posterior, samples


@gin.configurable
class NearestNeighborDiffusion(DiscreteDiffusionMatrixBase):
  """A simple schedule that diffuses away from the identity matrix."""

  def __init__(
      self,
      dim,
      schedule,
      knn=10,
      num_chunks=32,
      num_sinkhorn_iterations = 100,
      use_matrix_exponential = False,
      expm_type = "jax",
      precision=jax.lax.Precision.HIGHEST,
  ):
    """A simple scheduler for beta-diagonal policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiscreteDiffusion schedule object to use.
      knn: number of nearest neighbors to use.
      num_chunks: the number of chunks to use to compute nearest neighbors
      num_sinkhorn_iterations: the number of sinkhorn steps to use to normalize
        the adjacency matrix.
      use_matrix_exponential: use the matrix exponential.
      expm_type: how to compute matrix exponential.
      precision: matmul precision.
    """
    self.schedule = schedule
    self.num_steps = schedule.num_steps
    self.precision = precision
    self.num_sinkhorn_iterations = num_sinkhorn_iterations
    self.num_chunks = num_chunks
    self.dim = dim
    self.initialized = False
    self.knn = knn
    self.expm_type = expm_type
    self.use_matrix_exponential = use_matrix_exponential

    logging.info("Creating NearestNeighbor diffusion with schedule %s.",
                 self.schedule)

  def supports_efficient_inference(self):
    return False

  @property
  def has_state(self):
    return True

  def stationary_probs(self, shape):
    return jnp.ones(shape + (self.dim,)) / self.dim

  def sample_stationary(self, key, shape):
    return jrandom.randint(key, shape, 0, self.dim)

  def update_state(self, state):
    neighbors = model_utils.get_nearest_neighbors(
        state,
        k=self.knn,
        include_self=False,
        num_chunks=min(state.shape[0] - 1, self.num_chunks))  # |V| x k
    matrix = jnp.zeros((self.dim, self.dim), jnp.float32)
    matrix = matrix.at[neighbors, jnp.arange(self.dim)[:, None]].set(1.)

    matrix = matrix + matrix.T

    if self.use_matrix_exponential:
      transition_rate = matrix - jnp.diagflat(jnp.sum(matrix, axis=1))
      beta = self.schedule(0)
      if self.expm_type == "naive":
        matrix = utils.naive_expm(
            beta * transition_rate, iterations=self.num_expm_iterations)
      elif self.expm_type == "scipy":
        matrix = scipy.linalg.expm(
            np.array(beta * transition_rate, dtype=np.float64))
      else:
        matrix = jax.scipy.linalg.expm(beta * transition_rate)
    else:
      for _ in range(self.num_sinkhorn_iterations):
        matrix = matrix / matrix.sum(1, keepdims=True)
        matrix = matrix / matrix.sum(0, keepdims=True)

    return matrix / matrix.sum(0, keepdims=True)

  def set_state(self, state):
    self.matrix = state
    self.initialized = True

  def reset_state(self):
    self.matrix = None
    self.initialized = False

  def get(self, t):
    if not self.initialized:
      raise ValueError(
          "Cannot call get on NearestNeighborDiffusion before initialization.")

    beta = self.schedule(t)
    dim = self.dim
    return (1 - beta) * jnp.eye(dim) + beta * self.matrix

  def supports_efficient_get(self):
    return True


@gin.configurable
class CachedDiffusion(DiscreteDiffusionMatrixBase):
  """A simple schedule that diffuses away from the identity matrix."""

  def __init__(
      self,
      dim,
      schedule,
      use_slow_get=False,
      use_numpy=False,
      powers=None,
      precision=jax.lax.Precision.HIGHEST,
      override_last_step_to_mask=False,
      mask_token=None,
      num_expm_iterations=15,
      expm_type="jax",
      lazy=False,
  ):
    """A simple diffusion schedule for beta-diagonal policies.

    Args:
      dim: int, the dimensionality of the state space.
      schedule: a DiscreteDiffusion schedule object to use.
      use_slow_get: disables reverse pullback probability.
      use_numpy: uses high precision numpy matmuls.
      powers: if specified, overrides powers computed from the schedule.
      precision: matmul precision.
      override_last_step_to_mask: Replace the last step with one that goes to
        all masks.
      mask_token: if specified, the mask token to use.
      num_expm_iterations: number of iterations in simple Taylor expm
        implementation.
      expm_type: type of expm to use (naive, less_naive, scipy, jax).
      lazy: If True, then don't actually precompute matrix powers; instead,
        compute them on demand in JAX.
    """
    self.dim = dim
    self.schedule = schedule
    self.num_steps = schedule.num_steps
    self.precision = precision
    self.use_numpy = use_numpy
    self.use_slow_get = use_slow_get
    self.override_last_step_to_mask = override_last_step_to_mask
    if mask_token is None:
      mask_token = dim - 1
    self.mask_token = mask_token
    self.num_expm_iterations = num_expm_iterations
    self.expm_type = expm_type
    self.lazy = lazy

    self.initialized = False

    if powers is None:
      self.override_powers = None
    else:
      # Override powers!
      logging.warning(
          "CachedDiffusion overriding computed powers with hardcoded powers")
      self.override_powers = jnp.asarray(powers, dtype=jnp.int32)

    logging.info("Creating CachedDiffusion with schedule %s.", self.schedule)

  def supports_efficient_inference(self):
    return False

  def get_base_matrix(self, beta_min, *args, **kwargs):
    """Returns the base matrix to be raised to powers."""
    transition_rate = self.get_rate_matrix(*args, **kwargs)
    if self.expm_type == "naive":
      matrix = utils.naive_expm(
          beta_min * transition_rate, iterations=self.num_expm_iterations)
    elif self.expm_type == "less_naive":
      matrix = utils.transition_rate_expm(beta_min * transition_rate)
    elif self.expm_type == "scipy":
      matrix = scipy.linalg.expm(
          np.array(beta_min * transition_rate, dtype=np.float64))
    else:
      matrix = jax.scipy.linalg.expm(beta_min * transition_rate)
    return matrix

  def get_rate_matrix(self, *args, **kwargs):
    """Get a transition rate matrix. Must not depend on min_exponent/powers."""
    raise NotImplementedError

  def update_state(self, *args, **kwargs):

    if isinstance(self.schedule, DiffusionSchedule):
      # Set powers based on cumulative sums of steps from schedule.
      betas = self.schedule(np.arange(self.num_steps))
      min_exponent = jnp.min(betas)
      shifted_betas = jnp.concatenate([jnp.array([0.]), betas])
      exponents = jnp.cumsum(shifted_betas)

    elif isinstance(self.schedule, MutualInformationSchedule):
      assert not self.override_powers
      # Set schedule based on mutual information. Must support get_rate_matrix!
      transition_rate = self.get_rate_matrix(*args, **kwargs)
      if transition_rate.shape[-1] == len(
          self.schedule.initial_distribution) + 1:
        logging.info("padding distribution for masks")
        self.schedule.initial_distribution = jnp.concatenate(
            [self.schedule.initial_distribution,
             jnp.array([0.])])

      query_exponents, query_info_removals = (
          model_utils.compute_information_removal_samples_by_squaring(
              transition_rate,
              initial_distribution=self.schedule.initial_distribution,
              min_exponent=self.schedule.min_exponent,
              max_exponent=self.schedule.max_exponent,
              interpolation_steps=self.schedule.interpolation_steps))
      if self.override_last_step_to_mask:
        num_interpolated_steps = self.num_steps - 1
      else:
        num_interpolated_steps = self.num_steps
      _, middle_exponents = (
          model_utils.build_mutual_information_schedule(num_interpolated_steps,
                                                        query_exponents,
                                                        query_info_removals))
      min_exponent = middle_exponents[0]
      exponents = jnp.concatenate([jnp.zeros([1]), middle_exponents])
      if self.override_last_step_to_mask:
        exponents = jnp.concatenate([exponents, exponents[-1][None]])

    if self.override_powers:
      powers = self.override_powers
    else:
      powers = jnp.round(exponents / min_exponent).astype(jnp.int32)

    logging.info(
        "CachedDiffusion computed schedule with min exponent %s and powers %s",
        min_exponent, list(powers))

    matrix = self.get_base_matrix(min_exponent, *args, **kwargs)
    if self.lazy:
      matrix_power_state = model_utils.LazyMatrixPowerState(matrix)
    else:
      matrix_power_state = model_utils.CachedMatrixPowerState.precompute(
          matrix,
          max_power=powers[-1],
          precision=self.precision,
          use_numpy=self.use_numpy)
    return {
        "min_exponent": min_exponent,
        "powers": powers,
        "matrix_power_state": matrix_power_state,
    }

  def set_state(self, state):
    self.min_exponent = state["min_exponent"]
    self.powers = state["powers"]
    self.matrix_power_state = state["matrix_power_state"]
    self.initialized = True

  def reset_state(self):
    self.min_exponent = None
    self.powers = None
    self.matrix_power_state = None
    self.initialized = False

  def _mask_mat(self):
    return jnp.broadcast_to(
        jax.nn.one_hot(self.mask_token, self.dim)[:, None],
        (self.dim, self.dim))

  def get(self, t):
    if not self.initialized:
      raise ValueError(
          "Cannot call get on NearestNeighborCachedDiffusion before initialization."
      )

    power = self.powers[t + 1] - self.powers[t]

    ## TODO(jaaustin) this is expensive and can be amortized.
    ## 1. can combine with the get_qt_given_q0 scan to only scan once.
    ## 2. can pass the posterior sample and compute this as an MVP.
    result = self.matrix_power_state.matrix_power(power)
    if self.override_last_step_to_mask:
      return jnp.where(t == self.num_steps - 1, self._mask_mat(), result)
    else:
      return result

  def supports_efficient_get(self):
    return self.use_slow_get

  def qt_reverse(self,
                 qt_plus_1,
                 t,
                 return_logits=False,
                 make_one_hot=False,
                 epsilon=1e-20):
    """Get q(x_{t+1} | x_t), the one-step posterior efficiently.

    Args:
      qt_plus_1: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_{t+1} | x_t).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_{t+1} | x_t).
    """
    logging.info("using precision %s", self.precision)

    if not self.initialized:
      raise ValueError("Diffusion object must be initialized.")

    if make_one_hot:
      chex.assert_type(qt_plus_1, jnp.int32)
      qt_plus_1 = losses.onehot(labels=qt_plus_1, num_classes=self.dim)

    chex.assert_type(qt_plus_1, jnp.float32)

    power = self.powers[t + 1] - self.powers[t]
    prob_at_time_t = jax.vmap(
        functools.partial(
            self.matrix_power_state.matrix_power_multiply, transpose=True),
        in_axes=(0, None))(qt_plus_1, power)

    if self.override_last_step_to_mask:
      prob_at_time_t = jnp.where(t == self.num_steps - 1,
                                 jnp.ones_like(prob_at_time_t), prob_at_time_t)

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t

  def get_qt_given_q0(self,
                      q0,
                      t,
                      return_logits=False,
                      make_one_hot=False,
                      epsilon=1e-20):
    """Get q(x_t), the n-step posterior.

    For example, for t = 0, it returns q0 unchanged.

    Args:
      q0: an array of floats specifying a distribution over p(x_0).
      t: t in q(x_t | x_0).
      return_logits: if True, return the output logits
      make_one_hot: if True, will convert q0 to floats if needed.
      epsilon: a small number to normalize logits conversion with, if needed.

    Returns:
      q(x_t | x_0).
    """
    logging.info("using precision %s", self.precision)

    if not self.initialized:
      raise ValueError("Diffusion object must be initialized.")

    if make_one_hot:
      chex.assert_type(q0, jnp.int32)
      q0 = losses.onehot(labels=q0, num_classes=self.dim)

    chex.assert_type(q0, jnp.float32)

    power = self.powers[t]
    prob_at_time_t = jax.vmap(
        self.matrix_power_state.matrix_power_multiply,
        in_axes=(0, None),
    )(q0, power)

    if self.override_last_step_to_mask:
      prob_at_time_t = jnp.where(t == self.num_steps,
                                 jax.nn.one_hot(self.mask_token, self.dim),
                                 prob_at_time_t)

    if return_logits:
      return jnp.log(prob_at_time_t + epsilon)
    else:
      return prob_at_time_t


class NearestNeighborCachedDiffusion(CachedDiffusion):
  """NearestNeighbor diffusion with cache."""

  def __init__(self,
               dim,
               schedule,
               knn=50,
               use_matrix_exponential=True,
               use_gaussian_transitions=False,
               num_sinkhorn_iterations=5,
               num_chunks=32,
               **kwargs):
    """Nearest Neighbor schedule with cache.

    Args:
      dim: dimension of diffusion
      schedule: noise schedule.
      knn: number of nearest neighbors.
      use_matrix_exponential: whether to use matrix exponential.
      use_gaussian_transitions: Gaussian structured transition matrix.
      num_sinkhorn_iterations: number of sinkhorn steps to normalize matrix
        with. Only relevant if use_matrix_exponential is false.
      num_chunks: number of chunks for nearest-neighbor calculation.
      **kwargs: kwargs for cache init
    """

    super().__init__(dim=dim, schedule=schedule, **kwargs)

    self.use_matrix_exponential = use_matrix_exponential
    self.num_sinkhorn_iterations = num_sinkhorn_iterations
    self.use_gaussian_transitions = use_gaussian_transitions
    self.knn = knn
    self.num_chunks = num_chunks

    logging.info(
        "CachedNearestNeighborDiffusion with use_matrix_exponential: "
        "%s, knn: %d, num_expm_iterations: %d, expm_type: %s",
        self.use_matrix_exponential, self.knn, self.num_expm_iterations,
        self.expm_type)

  def _get_similarity_matrix(self, embeddings):
    """Returns a matrix describing how similar embeddings are."""
    embeddings = embeddings / jnp.linalg.norm(
        embeddings, axis=-1, keepdims=True)
    neighbors = model_utils.get_nearest_neighbors(
        embeddings,
        k=self.knn,
        include_self=False,
        num_chunks=min(embeddings.shape[0] - 1, self.num_chunks))  # |V| x k
    matrix = jnp.zeros((self.dim, self.dim), jnp.float32)
    if self.use_gaussian_transitions:
      transitions = self.knn * jax.nn.softmax(
          -2 * jnp.arange(self.knn)**2 / self.knn)
      matrix = matrix.at[neighbors, jnp.arange(self.dim)[:,
                                                         None]].set(transitions)
    else:
      matrix = matrix.at[neighbors, jnp.arange(self.dim)[:, None]].set(1.0)

    matrix = matrix + matrix.T
    matrix = matrix / (2 * self.knn)
    return matrix

  def get_rate_matrix(self, embeddings):
    if not self.use_matrix_exponential:
      raise NotImplementedError(
          "get_rate_matrix not allowed when use_matrix_exponential=False")
    matrix = self._get_similarity_matrix(embeddings)
    transition_rate = matrix - jnp.diagflat(jnp.sum(matrix, axis=0))
    return transition_rate

  def get_base_matrix(self, beta_min, embeddings):
    if self.use_matrix_exponential:
      return super().get_base_matrix(beta_min, embeddings)

    matrix = self._get_similarity_matrix(embeddings)
    for _ in range(self.num_sinkhorn_iterations):
      matrix = matrix / matrix.sum(1, keepdims=True)
      matrix = matrix / matrix.sum(0, keepdims=True)

    matrix = matrix / matrix.sum(0, keepdims=True)
    matrix = (1 - beta_min) * jnp.eye(self.dim) + beta_min * matrix
    return matrix

  @property
  def has_state(self):
    return True

  def stationary_probs(self, shape):
    """Look up stationary distribution."""
    if self.override_last_step_to_mask:
      result = jax.nn.one_hot(self.mask_token, self.dim)
      return jnp.broadcast_to(result, shape + (self.dim,))
    else:
      return jnp.ones(shape + (self.dim,)) / self.dim

  def sample_stationary(self, key, shape):
    return jax.random.categorical(key, jnp.log(self.stationary_probs(shape)))


@gin.configurable
class MaskNearestNeighborCachedDiffusion(CachedDiffusion):
  """NearestNeighbor diffusion with cache."""

  def __init__(self,
               dim,
               schedule,
               knn=10,
               num_chunks=32,
               mask_rate=4,
               **kwargs):
    """Nearest Neighbor schedule with cache.

    Args:
      dim: dimension of diffusion
      schedule: noise schedule.
      knn: number of nearest neighbors.
      num_chunks: number of chunks for nearest-neighbor calculation.
      mask_rate: rate of transition to a mask. If equal to 1, transitions to
        masks equally often as it transitions to any neighbor (summed across
        neighbors).
      **kwargs: kwargs for cache init
    """

    super().__init__(dim=dim, schedule=schedule, mask_token=dim - 1, **kwargs)

    self.knn = knn
    self.num_chunks = num_chunks
    self.mask_rate = mask_rate

    logging.info("MaskNearestNeighborCachedDiffusion with knn: %d", self.knn)

  def get_rate_matrix(self, embeddings):
    """Returns the rate matrix for mask + NN diffusion.

    Pseudocode:
    * build the NN matrix, symmetrize it (ignoring the mask token, which is
      handled separately).
    * add in an extra rate at which we transition to masks

    Args:
      embeddings: the embedding graph to use.

    Returns:
      the base matrix to take powers of.
    """

    # Build and symmetrize without the mask token.
    if embeddings.shape[0] == self.dim:
      embeddings = embeddings[:-1, :-1]

    neighbors = model_utils.get_nearest_neighbors(
        embeddings,
        k=self.knn,
        include_self=False,
        num_chunks=min(embeddings.shape[0] - 1, self.num_chunks))  # |V| x k
    matrix = jnp.zeros((self.dim - 1, self.dim - 1), jnp.float32)
    matrix = matrix.at[neighbors, jnp.arange(self.dim - 1)[:, None]].set(1.)
    matrix = matrix + matrix.T
    matrix = matrix / (2 * self.knn)

    # Add transitions for the mask token.
    # Recall: matrix[i, j] gives transitions from j to i.
    matrix = jnp.pad(matrix, [(0, 1), (0, 1)])
    matrix = matrix.at[-1, :].set(self.mask_rate)

    # Normalize so it is a valid rate matrix.
    transition_rate = matrix - jnp.diagflat(jnp.sum(matrix, axis=0))
    return transition_rate

  @property
  def has_state(self):
    return True

  def stationary_probs(self, shape):
    """Stationary distribution is one-hot at mask token."""
    sample = jnp.full(shape, self.dim - 1)
    probs = losses.onehot(sample, self.dim)
    return probs

  def sample_stationary(self, key, shape):
    """Stationary distribution is one-hot at mask token."""
    return jnp.full(shape, self.dim - 1)


def create_discrete_diffusion_schedule(
    kind = "linear",
    beta_min = 1e-3,
    beta_max = 1e-1,
    num_steps = 100,
    scale=1.0,
):
  """Creates a callable schedule object to use for diffusion rates.

  Args:
    kind: str, one of 'standard', 'linear', 'cosine', 'mutual_information'. If
      standard, performs standard binomial diffusion taken from Sohl-Dicksteein
      et al, ignoring betas. Otherwise, linear schedule between beta_min and
      beta_max.
    beta_min: the minimum beta. Ignored if kind == standard.
    beta_max: the maximum beta.
    num_steps: int, the number of steps to take.
    scale: for standard schedule, rescales num_steps by this amount.

  Returns:
    a DiffusionSchedule object.
  """

  assert beta_min <= beta_max
  assert num_steps > 0
  assert scale >= 1

  if kind == "standard":
    logging.info("using standard schedule with num_steps: %d.", num_steps)

    def schedule_fn(step):
      return 1 / (scale * num_steps - step)

    return DiffusionSchedule(schedule_fn, num_steps, is_constant=False)

  elif kind == "linear":
    logging.info("using provided beta_min %f and beta_max %f.", beta_min,
                 beta_max)

    is_constant = beta_min == beta_max

    schedule_fn = utils.create_learning_rate_scheduler(
        "constant * linear_warmup_from",
        warmup_steps=num_steps,
        min_learning_rate=beta_min,
        base_learning_rate=beta_max)

    return DiffusionSchedule(schedule_fn, num_steps, is_constant=is_constant)
  elif kind == "cosine":
    logging.info("using cosine schedule inspired by OpenAI I-DDPM paper.")

    s = 0.008

    def cosine_fn(step):
      return jnp.cos((step / num_steps + s) / (1 + s) * jnp.pi / 2)

    def schedule_fn(step):
      return jnp.clip(1 - (cosine_fn(step + 1) / cosine_fn(step)), 0, 0.999)

    return DiffusionSchedule(schedule_fn, num_steps, is_constant=False)
  else:
    raise ValueError(f"kind {kind} is not supported.")


@gin.configurable
def create_discrete_diffusion(
    dataset_info,
    task,
    num_steps = 40,
    kind = "beta-diagonal",
    schedule_type = "standard",
    beta_min = 1e-3,
    beta_max = 0.2,
    scale = 1.0,
    width = 7,
    knn = 5,
    dim = None,
    update_every = 50,
    use_matrix_exponential = True,
    num_sinkhorn_iterations = 50,
    expm_type="scipy",
    use_numpy = True,
    num_chunks=32,
    pretrained_weights_path=None,
    precision=jax.lax.Precision.HIGHEST,
    mutual_info_initial_distribution = None,
    mutual_info_min_exponent = 1e-4,
    mutual_info_max_exponent = 1e5,
    mutual_info_interpolation_steps = 128,
    override_last_step_to_mask = True,
):
  """Creates a noise schedule for a diffusion model.

  Args:
    dataset_info: a DatasetInfo object.
    task: a task object.
    num_steps: the number of steps to use for the diffusion.
    kind: the diffusion type.
    schedule_type: the type of beta diffusion rate schedule to use.
    beta_min: the minimum value of beta.
    beta_max: the maximum value of beta.
    scale: rescales schedule by this amount.
    width: for BandDiagonal schedule, the width to use.
    knn: for NearestNeighbors, the number of neighbors to use.
    dim: if provided, specifies the size of the diffusion. Otherwise defaults to
      vocab size.
    update_every: updates the schedule ever n steps.
    use_matrix_exponential: if True, will use expm for nn diffusion.
    num_sinkhorn_iterations: if not using an expm transition, use sinkhorn to
      get a doubly-stochastic matrix.
    expm_type: type of expm to use.
    use_numpy: if True, will use numpy to compute powers of transition matrices
      at high precision. Only supported if we are not updating state under jit
    num_chunks: number of nearest neighbor chunks (increase to reduce memory
      overhead).
    pretrained_weights_path: path for pretrained embeddings.
    precision: matmul precision.
    mutual_info_initial_distribution: Initial distribution if using a mutual
      info schedule. This will be permuted automatically if the dataset includes
      a permutation vocab.
    mutual_info_min_exponent: Minimum exponent for the rate matrix when solving
      for the MI schedule.
    mutual_info_max_exponent: Maximum exponent for the rate matrix when solving
      for the MI schedule.
    mutual_info_interpolation_steps: Number of query points to make when solving
      for the MI schedule
    override_last_step_to_mask: Whether to override the last step to a mask
      token.

  Returns:
    a DiscreteDiffusionBase object.
  """
  del task
  dim = dim or dataset_info.vocab.vocab_size

  if kind == "autoregressive":
    logging.info(
        "Setting num_steps to the sequence length for autoregressive model.")
    num_steps = dataset_info.shapes["targets"].shape[-1]

  if schedule_type == "mutual_information":
    assert mutual_info_initial_distribution is not None
    if hasattr(dataset_info.vocab, "permutation"):
      # Permute our initial distribution as well.
      logging.info("Applying permutation to initial distribution.")
      permutation = dataset_info.vocab.permutation.numpy()
      mutual_info_initial_distribution = (
          mutual_info_initial_distribution[permutation])

    schedule = MutualInformationSchedule(
        num_steps, np.array(list(mutual_info_initial_distribution)),
        mutual_info_min_exponent, mutual_info_max_exponent,
        mutual_info_interpolation_steps)
  else:
    schedule = create_discrete_diffusion_schedule(
        kind=schedule_type,
        beta_min=beta_min,
        beta_max=beta_max,
        num_steps=num_steps,
        scale=scale,
    )

  if kind == "beta-diagonal":
    diffusion = BetaDiagonalDiffusion(
        dim=dim, schedule=schedule, precision=precision)

  elif kind == "band-diagonal":
    diffusion = BandDiagonalDiffusion(
        dim=dim,
        schedule=schedule,
        width=width,
        precision=precision,
    )

  elif kind == "band-diagonal-precomputed":
    diffusion = GaussianPrecomputedDiffusion(
        dim=dim,
        schedule=schedule,
        precision=precision,
    )

  elif kind == "mask-band-diagonal-precomputed":
    diffusion = MaskGaussianPrecomputedDiffusion(
        dim=dim,
        schedule=schedule,
        precision=precision,
    )

  elif kind == "mask":
    diffusion = MaskDiffusion(
        dim=dim,
        schedule=schedule,
        precision=precision,
    )

  elif kind == "mask-beta-diagonal":
    diffusion = MaskBetaDiagonalDiffusion(
        dim=dim,
        schedule=schedule,
        precision=precision,
    )

  elif kind == "nearest-neighbor":
    diffusion = NearestNeighborDiffusion(
        dim=dim,
        schedule=schedule,
        knn=knn,
        precision=precision,
        num_chunks=num_chunks,
        num_sinkhorn_iterations=num_sinkhorn_iterations)

  elif kind == "nearest-neighbor-cached":
    diffusion = NearestNeighborCachedDiffusion(
        dim=dim,
        schedule=schedule,
        knn=knn,
        precision=precision,
        use_matrix_exponential=use_matrix_exponential,
        expm_type=expm_type,
        use_numpy=use_numpy,
        num_chunks=num_chunks,
        num_sinkhorn_iterations=num_sinkhorn_iterations,
        override_last_step_to_mask=override_last_step_to_mask)

  elif kind == "nearest-neighbor-mask":
    diffusion = MaskNearestNeighborCachedDiffusion(
        dim=dim,
        schedule=schedule,
        knn=knn,
        precision=precision,
        use_numpy=use_numpy,
        num_chunks=num_chunks,
        override_last_step_to_mask=override_last_step_to_mask)

  elif kind == "autoregressive":
    diffusion = AutoRegressiveDiffusion(
        dim=dim,
        schedule=schedule,
        precision=precision,
    )
  elif kind == "closed-form-expm":
    diffusion = ClosedFormMatrixExponentialDiffusion(  # pylint: disable=no-value-for-parameter
        schedule=schedule,
        dim=dim,
        override_last_step_to_mask=override_last_step_to_mask)
  else:
    raise ValueError(
        f"Diffusion diffusion schedule of kind {kind} is not supported.")

  if not diffusion.has_state:
    return types.State(static_state={"diffusion": diffusion})

  if pretrained_weights_path is not None:
    logging.info("using pretrained weights.")

    with tf.io.gfile.Open(pretrained_weights_path, "rb") as f:
      embeddings = np.load(f)

    embeddings = embeddings / jnp.linalg.norm(
        embeddings, axis=-1, keepdims=True)
    state = diffusion.update_state(embeddings)
    state = flax.jax_utils.replicate(state)

    return types.State(
        static_state={"diffusion": diffusion},
        dynamic_state={"diffusion_state": state},
    )

  def dynamic_update_fn(dynamic_state, params, static_state):
    logging.info("Rejitting the diffusion state update function.")
    del dynamic_state

    diffusion = static_state["diffusion"]
    embeddings = params["params"]["module"]["decoder"]["token_embedder"][
        "embedding"]
    new_diffusion_state = diffusion.update_state(embeddings)
    return {"diffusion_state": new_diffusion_state}

  return types.State(
      static_state={"diffusion": diffusion},
      dynamic_state={"diffusion_state": None},
      dynamic_update_fn=dynamic_update_fn,
      dynamic_update_freq=update_every,
      jit_update=True,
  )


@gin.configurable(allowlist=["special_case_x0", "transition_probs_in_logits"])
def p_forward(
    denoise_fn,
    x_t,
    t,
    diffusion,
    predict_x0=True,
    return_x0=False,
    return_logits=False,
    special_case_x0=False,
    transition_probs = None,
    transition_probs_in_logits = True,
    maximum_likelihood = False,
    epsilon=1e-20,
    step_size = 1,
):
  """Returns probabilities from the reverse process p(x_{t-1} | x_t).

  Args:
    denoise_fn: the reverse process. Must support embed, call, and attend.
    x_t: the current value of x_t to condition on.
    t: the timestep t.
    diffusion: the Diffusion object to use for noise.
    predict_x0: if True, assumes the model output corresponds to its prediction
      for p(x_0 | x_t). Otherwise assumes model predicts p(x_{t-1} | x_t).
    return_x0: if True, will return probs for x_0 as well as x_{t-1}.
    return_logits: if True, will return logits instead of probabilities.
    special_case_x0: if True, will directly predict x0 instead of using the
      forward process probabilities.
    transition_probs: if provided, q(x_{t+1} | x_t) probs to reuse.
    transition_probs_in_logits: if False, will ignore transition probs in logits
      (only allowed if return_logits is True). This is because this term is
      independent of theta.
    maximum_likelihood: if true, will draw the most likely x0 before applying
      the forward process.
    epsilon: a small number.
    step_size: step size to compute posterior from.

  Returns:
    probabilities for q(x_{t-1} | x_t) (and probabilities for x0 if predict_x0
    is True)
  """
  assert not (step_size > 1 and not predict_x0)

  logits = denoise_fn(targets=x_t, timestep=t)
  probs = nn.softmax(logits, axis=-1)

  if not predict_x0:
    retval = logits if return_logits else probs
    if return_x0:
      return retval, None
    else:
      return retval

  if maximum_likelihood:
    probs = probs.argmax(-1)

  # we use this to compute p(x_{t-1} | x_t) = sum_x0 q(x_{t-1} | x_t, x_0)
  # p(x_0 | x_t).
  fake_key = jrandom.PRNGKey(0)
  qt_probs, _ = diffusion.sample_and_compute_posterior_q(
      key=fake_key,
      x_0=probs,
      t=t - step_size,
      make_one_hot=maximum_likelihood,
      return_logits=return_logits,
      transition_probs_in_logits=transition_probs_in_logits,
      transition_probs=transition_probs,
      samples=x_t,
      epsilon=epsilon,
      step_size=step_size,
  )

  retval_x0 = logits if return_logits else probs
  retval = qt_probs

  # we can special case t = 1 to just use the raw logits outputs.
  mask = (t == step_size) & special_case_x0
  retval = mask * retval_x0 + (1 - mask) * retval

  if return_x0:
    return retval, retval_x0
  else:
    return retval


def q_sample(x_start, t, diffusion, key):
  """Draws a sample from the posterior q(x_t | x_start)."""

  chex.assert_type(x_start, jnp.int32)

  dim = diffusion.dim
  x_start = losses.onehot(x_start, dim)

  logits = diffusion.get_qt_given_q0(q0=x_start, t=t, return_logits=True)
  sample = jrandom.categorical(key, logits=logits)
  return sample


def compute_prior_kl(x_start, diffusion, target_mask=None):
  """Computes KL divergence between q(x_T) and the true distribution."""
  chex.assert_type(x_start, jnp.int32)

  num_steps = diffusion.num_steps

  q_probs = diffusion.get_qt_given_q0(
      q0=x_start, t=num_steps, return_logits=False,
      make_one_hot=True)  # get end step
  p_probs = diffusion.stationary_probs(q_probs.shape[:-1])

  loss = losses.kl_divergence_with_probs(q_probs, p_probs)

  if target_mask is not None:
    loss = (loss * target_mask).sum()
  else:
    loss = loss.sum()

  return loss, 1


@gin.configurable
def compute_kl_reverse_process(rng_key,
                               x_start,
                               t,
                               *,
                               diffusion,
                               denoise_fn,
                               predict_x0=True,
                               log_space=False,
                               label_smoothing = 0.0,
                               hybrid_lambda = 0.0,
                               use_cached_transition = True,
                               target_mask=None,
                               step_size=1):
  """Returns the KL for one term in the ELBO (time t) (loss L_t).

  This assumes x_start is a sample from x_0, from which we draw samples from
  q(x_t | x_0) and then compute q(x_{t-1} | x_t, x_0) following the LaTeX. This
  is the KL divergence for terms L_1 through L_{T-1}.

  Args:
    rng_key: the Jax PRNGKey to use.
    x_start: a sample from p(data) (or q(x_0)).
    t: the loss term to compute.
    diffusion: the diffusion object to use.
    denoise_fn: a functool.partial-ed version of the model_apply function which
      takes a set of targets (x_t) and noise level and returns q(x_{t-1} | x_t,
      x_0).
    predict_x0: if True, will predict a distribution over x0 instead of x_{t-1}.
    log_space: if True, will perform the loss calculations in log space.
    label_smoothing: label smoothing for cross entropy.
    hybrid_lambda: coefficient for hybrid cross-entropy loss.
    use_cached_transition: if True, will reuse q(x_{t+1} | x_t) computation.
    target_mask: mask for target sequence.
    step_size: the step size over which the ELBO is computed.

  Returns:
    the KL divergence and denominator.
  """
  chex.assert_type(x_start, jnp.int32)
  chex.assert_rank(t, 0)

  if step_size > 1 and not predict_x0:
    raise ValueError("cannot skip steps when not predicting x0.")

  # sample from q(x_{t+1} | x_start), then compute q(x_t | x_{t+1}, x_start)
  # q_t and p_t can be logits or probs depending on log_space.
  q_t, x_t_plus_1, transition_probs = diffusion.sample_and_compute_posterior_q(
      rng_key,
      x_start,
      t,
      return_logits=log_space,
      return_transition_probs=True,
      step_size=step_size)

  transition_probs = transition_probs if use_cached_transition else None

  p_t = p_forward(
      denoise_fn,
      x_t_plus_1,
      t + step_size,
      diffusion,
      predict_x0=predict_x0,
      return_x0=predict_x0 and hybrid_lambda > 0.0,
      return_logits=log_space,
      transition_probs=transition_probs,
      step_size=step_size)

  if predict_x0 and hybrid_lambda > 0.0:
    p_t, p_0 = p_t
    if log_space:
      cross_entropy = losses.cross_entropy_with_logits(
          logits=p_0, targets=x_start, label_smoothing=label_smoothing)
    else:
      cross_entropy = losses.cross_entropy_with_probs(
          probs=p_0, targets=x_start, label_smoothing=label_smoothing)

    hybrid_loss = hybrid_lambda * cross_entropy
  else:
    hybrid_loss = jnp.asarray([0.0])

  if log_space:
    kl = losses.kl_divergence_with_logits(q_t, p_t)
    cross_entropy = losses.cross_entropy_with_logits(
        logits=p_t, targets=x_start, label_smoothing=label_smoothing)
  else:
    kl = losses.kl_divergence_with_probs(q_t, p_t)
    cross_entropy = losses.cross_entropy_with_probs(
        probs=p_t, targets=x_start, label_smoothing=label_smoothing)

  if target_mask is not None:
    kl = (kl * target_mask).sum()
    cross_entropy = (cross_entropy * target_mask).sum()
    hybrid_loss = (hybrid_loss * target_mask).sum()
  else:
    kl = kl.sum()
    cross_entropy = cross_entropy.sum()
    hybrid_loss = hybrid_loss.sum()

  mask = t == 0
  base_loss = mask * cross_entropy + (1 - mask) * kl
  loss = base_loss + hybrid_loss
  denominator = 1

  metrics_dict = {
      "loss": loss,
      "denominator": denominator,
      "kl/hybrid_loss": hybrid_loss,
      "kl/base_loss": base_loss,
      "kl/cross_entropy_loss": cross_entropy,
      "kl/t0_loss": mask * cross_entropy,
      "kl/kl_loss": kl,
  }

  return metrics_dict


def discrete_diffusion_elbo(
    x_start,
    *,
    denoise_fn,
    rng_key,
    diffusion,
    target_mask,
    predict_x0=True,
    length_probs=None,
    normalize_without_padding=True,
    eval_step_size = 1,
    return_all_likelihoods=False,
):
  """Computes the ELBO likelihood bound for discrete diffusion models.

  Pseudocode:
    1. starting at t = T and going towards t = 0:
    2. sample P(x_t | x_0)
    3. use NN to compute P(x_{t-1} | x_t)
    4. get q(x_{t-1} | x_t, x_0)
    5. compute KL divergence
    6. At T = 0, get discrete log likelihoods

  Args:
    x_start: data point.
    denoise_fn: the denoise_fn function (including params).
    rng_key: Jax RNGKey.
    diffusion: the noise schedule object.
    target_mask: mask for padding targets
    predict_x0: if True, assumes the neural net predicts x0.
    length_probs: list of probabilities for each sequence length.
    normalize_without_padding: if True, ignore padding when normalizing.
    eval_step_size: step size for evaluation.
    return_all_likelihoods: if True, will return all likelihoods for all
      timesteps.

  Returns:
    the full ELBO bound.
  """
  assert diffusion.num_steps % eval_step_size == 0
  assert diffusion.num_steps > eval_step_size

  @flax.struct.dataclass
  class State:
    t: Any
    log_likelihood: Any
    rng_key: Any

  def elbo_body_fn(state, _):
    sampling_key, rng_key = jrandom.split(state.rng_key)

    metrics_dict = compute_kl_reverse_process(
        sampling_key,
        x_start,
        state.t,
        denoise_fn=denoise_fn,
        diffusion=diffusion,
        predict_x0=predict_x0,
        target_mask=target_mask,
        hybrid_lambda=0.0,
        step_size=eval_step_size,
    )

    log_likelihood = metrics_dict["loss"] / metrics_dict["denominator"]

    return State(
        t=state.t - eval_step_size,
        log_likelihood=state.log_likelihood + log_likelihood,
        rng_key=rng_key,
    ), None

  init_state = State(
      t=diffusion.num_steps - eval_step_size,
      log_likelihood=jnp.array(0.0),
      rng_key=rng_key,
  )

  metrics_dict = {}

  num_steps = diffusion.num_steps // eval_step_size

  if return_all_likelihoods:

    def scan_wrapper(state, x):
      new_state, _ = elbo_body_fn(state, x)
      return new_state, new_state.log_likelihood

    final_state, likelihoods = jax.lax.scan(
        scan_wrapper, init_state, None, num_steps, reverse=True)

    metrics_dict["all_likelihoods"] = likelihoods
  else:
    final_state, _ = jax.lax.scan(
        elbo_body_fn, init_state, None, num_steps, reverse=True)

  log_likelihood = final_state.log_likelihood

  prior, denominator = compute_prior_kl(
      x_start, diffusion, target_mask=target_mask)

  target_length = jnp.count_nonzero(target_mask)

  if length_probs is not None:
    length_probs = jnp.asarray(length_probs)
    length_log_likelihood = -jnp.log(length_probs[target_length])
  else:
    length_log_likelihood = 0.0

  elbo = log_likelihood + length_log_likelihood + prior / denominator

  elbo_length = target_length if normalize_without_padding else x_start.shape[-1]

  return {
      "elbo": elbo,
      "elbo_in_bits_per_dim": elbo / (jnp.log(2) * elbo_length),
      "likelihood": log_likelihood,
      "prior": prior,
      "length_likelihood": length_log_likelihood,
      "nn/num_steps": num_steps,
  }


def _build_denoise_fn(model_apply, params, encoded, encoder_mask, decoder_mask):
  """Creates a (conditional) denoise function."""
  model_apply = functools.partial(model_apply, params)

  def denoise_fn(targets, timestep):
    logits = model_apply(
        encoded=encoded,
        decoder_input_tokens=targets,
        encoder_padding_mask=encoder_mask,
        decoder_padding_mask=decoder_mask,
        timestep=timestep,
        method="decode")

    return logits

  return denoise_fn


@gin.configurable
def discrete_diffusion_loss_fn(
    params,
    targets,
    *,
    model_apply,
    rng_key,
    diffusion,
    diffusion_state=None,
    inputs=None,
    is_eval = False,
    predict_x0 = False,
    hybrid_lambda = 0.0,
    compute_elbo = True,
    mask_padding = False,
    normalize_without_padding = False,
    length_probs = None,
    eval_step_size = 1,
    return_all_likelihoods = False,
):
  """Loss function for the discrete-time, discrete-space diffusion.

  Args:
    params: parameters of neural network (passed to model_apply).
    targets: the targets to use for prediction/corruption.
    model_apply: a function which takes the parameters and inputs and applies
      the neural network.
    rng_key: a jax PRNGKey.
    diffusion: a matrix noise schedule.
    diffusion_state: state used by the diffusion object.
    inputs: if provided, used as conditioning inputs.
    is_eval: if True, performing evaluation.
    predict_x0: if True, predict P(x_0 | x_t) instead of P(x_{t-1} | x_t).
    hybrid_lambda: cross entropy hybrid loss coefficient. 0 disables hybrid
      loss.
    compute_elbo: if True, computes the full ELBO during evaluation.
    mask_padding: mask out padding tokens and exclude them from ELBO.
    normalize_without_padding: if True, reports ELBO in bits/dim according to
      non padding tokens.
    length_probs: if provided, a list of empirical probabilities for each length
      in the training set.
    eval_step_size: step size for evaluation (only used to compute ELBO).
    return_all_likelihoods: returns all individual likelihoods from ELBO.

  Returns:
    loss, metrics, and extras.
  """
  logging.info(
      "Re-jitting discrete_diffusion_loss_fn (with predict_x0 %s, is_eval %s, mask_padding %s, normalize_without_padding %s, eval_step_size: %d)",
      predict_x0, is_eval, mask_padding, normalize_without_padding,
      eval_step_size)

  if diffusion.has_state:
    diffusion.set_state(diffusion_state)

  ## use a fixed noise schedule for eval to allow for comparisons.
  time_key, rng_key = jrandom.split(rng_key)
  t = diffusion.sample_t(time_key, shape=())
  target_length = jnp.count_nonzero(
      targets) if mask_padding else targets.shape[-1]

  if inputs is not None:
    encoder_mask = inputs > 0 if mask_padding else jnp.ones_like(inputs) > 0

    encoded_inputs = model_apply(
        params,
        encoder_input_tokens=inputs,
        encoder_padding_mask=encoder_mask,
        method="encode")
  else:
    encoded_inputs, encoder_mask = None, None

  if encoded_inputs is not None and mask_padding:
    length_logits = model_apply(
        params,
        encoded_inputs=encoded_inputs,
        targets=targets,
        method="predict_length")

    length_loss = losses.cross_entropy_with_logits(length_logits,
                                                   target_length - 1)

    length_acc, _ = losses.weighted_accuracy(length_logits, target_length - 1)
    length_pred = length_logits.argmax(-1) + 1
  else:
    length_loss = 0.0
    length_acc = 0.0
    length_pred = 0.0

  target_mask = targets > 0 if mask_padding else jnp.ones_like(targets) > 0

  denoise_fn = _build_denoise_fn(
      model_apply,
      params,
      encoded=encoded_inputs,
      encoder_mask=encoder_mask,
      decoder_mask=target_mask)

  # import pdb; pdb.set_trace()
  ## sample from the posterior
  metrics_dict = compute_kl_reverse_process(
      rng_key,
      targets,
      t,
      diffusion=diffusion,
      predict_x0=predict_x0,
      denoise_fn=denoise_fn,
      target_mask=target_mask,
      hybrid_lambda=hybrid_lambda,
  )

  loss = metrics_dict.pop("loss") + length_loss
  denominator = metrics_dict.pop("denominator")

  if mask_padding and length_probs is not None:
    length_probs = jnp.asarray(length_probs)
    length_log_likelihood = jnp.log(length_probs[target_length])
  else:
    length_log_likelihood = 0.0

  kl_loss = metrics_dict["kl/base_loss"]
  if normalize_without_padding:
    estimated_elbo = (kl_loss * diffusion.num_steps - length_log_likelihood) / (
        jnp.count_nonzero(targets) * jnp.log(2))
  else:
    estimated_elbo = (kl_loss * diffusion.num_steps) / (
        targets.shape[-1] * jnp.log(2))

  metrics_dict.update({
      "loss": loss,
      "denominator": denominator,
      "length_loss": length_loss,
      "elbo_estimate_bits_per_dim": estimated_elbo,
      "t0_loss": loss * (t == 0),
      "nn/seq_len_not_masked": (target_mask > 0).sum(),
      "nn/length": target_length,
      "nn/predicted_length": length_pred,
      "nn/length_acc": length_acc * 100,
      "nn/length_log_likelihood": length_log_likelihood,
  })

  if is_eval and compute_elbo:
    elbo_dict = discrete_diffusion_elbo(
        targets,
        denoise_fn=denoise_fn,
        rng_key=rng_key,
        diffusion=diffusion,
        predict_x0=predict_x0,
        target_mask=target_mask,
        length_probs=length_probs,
        normalize_without_padding=normalize_without_padding,
        eval_step_size=eval_step_size,
        return_all_likelihoods=return_all_likelihoods,
    )

    metrics_dict.update(elbo_dict)

  if is_eval:
    prior, denominator = compute_prior_kl(
        targets, diffusion, target_mask=target_mask)
    metrics_dict["prior"] = prior

  extras = {
      "t": t,
  }

  if diffusion.has_state:
    diffusion.reset_state()

  return (loss, denominator), (metrics_dict, extras)


def discrete_diffusion_predict_completions_fn(
    *,
    params,
    rng_key,
    targets,
    model,
    dataset_info,
    diffusion,
    diffusion_state,
    inputs,
    predict_x0,
    use_maximum_likelihood_decoding,
    mask_padding,
):
  """Predict a series of completions for rate-distortion measures."""
  del dataset_info

  forward_key, reverse_key, model_apply_key = jax.random.split(rng_key, 3)

  model_apply = utils.make_model_apply(model, model_apply_key)

  if inputs is not None:
    # note that we always mask input padding (because we always want to).
    encoder_mask = inputs > 0
    encoded_inputs = model_apply(
        params,
        encoder_input_tokens=inputs,
        encoder_padding_mask=encoder_mask,
        method="encode")
  else:
    encoder_mask, encoded_inputs = None, None
    ##  if we have an encoder, we use that to predict the target length.
    ##  Otherwise, we just use the length of the batch.

  if diffusion.has_state:
    diffusion.set_state(diffusion_state)

  # Always use the target length.
  if mask_padding:
    length = jnp.count_nonzero(targets > 0)
  else:
    length = targets.shape[-1]

  target_mask = jnp.arange(targets.shape[-1]) < length

  denoise_fn = _build_denoise_fn(  # okay that outside of loop because no RNG
      model_apply,
      params,
      encoded=encoded_inputs,
      encoder_mask=encoder_mask,
      decoder_mask=target_mask)

  # FORWARD PROCESS: Add noise to the target.
  def forward_step(xt, t):
    transition = diffusion.get(t)
    xtplus1_distn = jax.vmap(lambda tok: transition[:, tok])(xt)
    key = jax.random.fold_in(forward_key, t)
    xtplus1 = jax.random.categorical(key, jnp.log(xtplus1_distn))
    return xtplus1, xtplus1

  _, noisy_sequences = jax.lax.scan(
      forward_step, init=targets, xs=jnp.arange(diffusion.num_steps))

  # (TRUNCATED) REVERSE PROCESS: Denoise starting at each noisy version.
  def reverse_cond(state):
    _, t = state
    return t > 0

  def reverse_step(state):
    xt, t = state
    # note: reuses reverse_key for each starting point, but different values
    # at each timestep. Might reduce variance, probably harmless even if not.
    key = jax.random.fold_in(reverse_key, t)
    logits = p_forward(
        denoise_fn,
        x_t=xt,
        t=t,
        diffusion=diffusion,
        predict_x0=predict_x0,
        return_x0=False,
        return_logits=True,
        maximum_likelihood=use_maximum_likelihood_decoding)
    xtminus1 = jrandom.categorical(key, logits, axis=-1)
    return xtminus1, t - 1

  def truncated_reverse(t):
    xt = noisy_sequences[t - 1]
    x0, _ = jax.lax.while_loop(reverse_cond, reverse_step, (xt, t))
    return x0

  reconstructions = jax.lax.map(truncated_reverse,
                                1 + jnp.arange(diffusion.num_steps))

  # Add endpoints
  noisy_sequences = jnp.concatenate([targets[None], noisy_sequences])
  reconstructions = jnp.concatenate([targets[None], reconstructions])

  if diffusion.has_state:
    diffusion.reset_state()

  return {
      "noisy_sequences": noisy_sequences,
      "reconstructions": reconstructions,
  }


@gin.configurable
def discrete_diffusion_predict_fn(
    params,
    rng_key,
    targets,
    *,
    model,
    dataset_info,
    diffusion,
    diffusion_state=None,
    inputs=None,
    return_intermediates=False,
    predict_x0=False,
    use_maximum_likelihood_decoding=False,
    mask_padding = False,
    predict_completions = False,
    step_size = 1,
):
  """Predict an image or text from a diffusion model.

  Args:
    params: a PyTree of parameters for the model.
    rng_key: an RNG key.
    targets: ignored, used for shape info.
    model: the Flax model to use.
    dataset_info: the Problem object for the current task.
    diffusion: the noise schedule to use to condition the prediction steps.
    diffusion_state: if provided, a state object used by the diffusion class.
    inputs: if provided, used to condition the prediction.
    return_intermediates: if True, uses lax.scan to return all intermediate
      steps in the reverse process.
    predict_x0: if True, will predict a distribution over x_0 instead of x_{t-1}
      which allows for the number of inference steps to be varied after
      training.
    use_maximum_likelihood_decoding: if True, will take the maximum likelihood
      sample instead of sampling from the posterior. Will tend to produce very
      trivial results, unless predict_x0 is True.
    mask_padding: if True, mask out padding tokens.
    predict_completions: if True, instead of predicting from x_T, predict from
      other points x_t for each possible t. Returns different metrics and
      shapes.
    step_size: tne size of each inference step (step_size > 1 skips steps).

  Returns:
    a dictionary containing metrics and information about the prediction
      process.
  """
  if predict_completions:
    # slight hack to make this integrate nicely with the trainer: delegate to
    # separate completion helper.
    return discrete_diffusion_predict_completions_fn(
        params=params,
        rng_key=rng_key,
        targets=targets,
        model=model,
        dataset_info=dataset_info,
        diffusion=diffusion,
        diffusion_state=diffusion_state,
        inputs=inputs,
        predict_x0=predict_x0,
        use_maximum_likelihood_decoding=use_maximum_likelihood_decoding,
        mask_padding=mask_padding)

  num_steps = diffusion.num_steps
  assert num_steps % step_size == 0
  assert step_size < num_steps

  @flax.struct.dataclass
  class SamplingState:
    x: jnp.ndarray  # current predicted seqeunce
    x0: Any  # only used if predict_x0 is true
    key: jnp.ndarray  # PRNGKey
    t: int  # current step

  if diffusion.has_state:
    diffusion.set_state(diffusion_state)

  extra_key, rng_key = jrandom.split(rng_key)
  model_apply = utils.make_model_apply(model, extra_key)
  length_key, rng_key = jrandom.split(rng_key)

  if inputs is not None:
    # note that we always mask input padding (because we always want to).
    encoder_mask = inputs > 0
    encoded_inputs = model_apply(
        params,
        encoder_input_tokens=inputs,
        encoder_padding_mask=encoder_mask,
        method="encode")
  else:
    encoder_mask, encoded_inputs = None, None
    ##  if we have an encoder, we use that to predict the target length.
    ##  Otherwise, we just use the length of the batch.

  if encoded_inputs is not None and mask_padding:
    length_logits = model_apply(
        params,
        encoded_inputs=encoded_inputs,
        targets=targets,
        method="predict_length")
    length = jrandom.categorical(length_key, length_logits) + 1
  elif mask_padding:
    length = jnp.count_nonzero(targets > 0)
  else:
    length = targets.shape[-1]

  target_mask = jnp.arange(targets.shape[-1]) < length

  denoise_fn = _build_denoise_fn(  # okay that outside of loop because no RNG
      model_apply,
      params,
      encoded=encoded_inputs,
      encoder_mask=encoder_mask,
      decoder_mask=target_mask)

  def sampling_step(step, state):
    del step

    t = state.t  # initially, num_steps, and decreases from there.
    key = state.key

    logits, x0_logits = p_forward(
        denoise_fn,
        x_t=state.x,
        t=t,
        diffusion=diffusion,
        predict_x0=predict_x0,
        return_x0=True,
        return_logits=True,
        maximum_likelihood=use_maximum_likelihood_decoding,
        step_size=step_size)

    if x0_logits is not None:
      x0 = x0_logits.argmax(-1)
    else:
      x0 = None

    sampling_key, key = jrandom.split(state.key)
    sample = jrandom.categorical(sampling_key, logits, axis=-1)

    mask = (t == step_size)
    sample = mask * logits.argmax(-1) + (1 - mask) * sample

    return SamplingState(x=sample, key=key, x0=x0, t=t - step_size)

  x = diffusion.sample_stationary(rng_key, targets.shape)

  if predict_x0:
    init_state = SamplingState(x, x, rng_key, num_steps)
  else:
    init_state = SamplingState(x, None, rng_key, num_steps)

  total_steps = num_steps // step_size
  if return_intermediates:

    def scan_fn(state, step):
      state = sampling_step(step, state)
      return state, (state.x, state.x0)

    final_state, intermediates = jax.lax.scan(
        scan_fn, init_state, xs=None, length=total_steps)
  else:
    final_state = jax.lax.fori_loop(0, total_steps, sampling_step, init_state)

  def crop(x, length):
    x = x.at[length].set(1)
    x = jnp.where(jnp.arange(x.shape[0]) > length, 0, x)
    return x

  if dataset_info.vocab is not None:  # this is the text example

    final_text = crop(final_state.x, length)

    predictions = {
        "text/final_text": final_text,
        "text/initial_text": init_state.x,
        "scalar/num_steps": num_steps,
        "scalar/length": length,
        "scalar/total_steps": total_steps,
    }

    if inputs is not None:
      predictions["text/inputs"] = inputs

  else:
    predictions = {
        "other/final_state": final_state.x,
        "other/initial_state": init_state.x,
        "scalar/num_steps": num_steps,
        "scalar/length": length,
        "scalar/total_steps": total_steps,
    }

  if return_intermediates:
    intermediates, x0_intermediates = intermediates

    predictions["other/intermediates"] = jax.vmap(
        crop, in_axes=(0, None))(intermediates, length)

    if predict_x0:
      predictions["other/x0_intermediates"] = jax.vmap(
          crop, in_axes=(0, None))(x0_intermediates, length)

  if diffusion.has_state:
    diffusion.reset_state()

  return predictions


LM_DIFFUSION_TASK = tasks.register(
    name="diffusion",
    task=types.Task(
        loss_fn=discrete_diffusion_loss_fn,
        predict_fn=discrete_diffusion_predict_fn,
        input_features=["targets"],
        init_features=["targets"],
        predict_features=["targets"],
        state_init_fn=create_discrete_diffusion,
        metric_fns=[metrics.take_n],
        model_init_fn=models.transformer_init,
        vmap_batch=True,
    ))
