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

"""Scoring functions for a search space given an iteration budget."""

import functools
import operator
from typing import Any, Callable, Dict, List

import jax
import jax.numpy as jnp


class UtilityMeasure:
  """Class for utility measures used in the scoring functions."""

  def __init__(self,
               params,
               x_locations_for_p_min = None,
               fantasize_y_values_for_p_min = None,
               y_fantasized_num_for_p_min = 500,
               initial_entropy_key=None,
               eps = 1e-16,
               incumbent = None):
    """Set the required arguments for calculating the utility measures.

    Args:
      params: a dictionary from names to values to specify GP hyperparameters.
      x_locations_for_p_min: (n, d) shaped array of n x-locations in d
        dimensions. We estimate the pmf of the global min over
        x_locations_for_p_min.
      fantasize_y_values_for_p_min: a function generating y value draws at
        x_locations_for_p_min. Inputs of this function include a PRNG key, the
        observed data, params, x_locations and the number of desired draws.
      y_fantasized_num_for_p_min: number of desired draws of y. This parameter
        will be passed to fantasize_y_values_for_p_min function.
      initial_entropy_key : PRNG key for jax.random. We use this key to evaluate
        the initial entropy of p_min given the observed data.
      eps: optional float tolerance to avoid numerical issues of entropy.
      incumbent: float y value for improvement-based utility measures. One
        typical choice under noise-less evals is the best observed y value.
    """
    if (x_locations_for_p_min is not None) * (fantasize_y_values_for_p_min
                                              is not None):

      self.fantasize_y_values_for_p_min = functools.partial(
          fantasize_y_values_for_p_min,
          x_locations_for_p_min=x_locations_for_p_min,
          params=params,
          y_fantasized_num_for_p_min=y_fantasized_num_for_p_min)
    self.initial_entropy_key = initial_entropy_key
    self.eps = eps
    self.incumbent = incumbent

  def is_improvement(self, y_batch):
    """Return whether a batch of y values can improve over the incumbent.

    Args:
      y_batch: (q, t) shaped array of t y values corresponding to a batch of
      size q of x-locations. Each column of this array corresponds to one
        realization of y values at q x-locations evaluated/predicted t times.
    Returns:
      (t,) shaped array of boolean values indicating whether the best of
      q y value within each of t realizations have improved over the incumbent.
    """
    return jnp.min(y_batch, axis=0) < self.incumbent

  def improvement(self, y_batch):
    """Return how much a batch of y values can improve over the incumbent.

    Args:
      y_batch: (q, t) shaped array of t y values corresponding to a batch of
      size q of x-locations. Each column of this array corresponds to one
        realization of y values at q x-locations evaluated/predicted t times.
    Returns:
      (t,) shaped array of non-negative float values indicating the
      improvement the best of q y value within each of t realizations achieves
      over the incumbent.
    """
    difference = self.incumbent - jnp.min(y_batch, axis=0)
    return jnp.maximum(0.0, difference)

  def _p_x_min(self, y_fantasized):
    """Estimate a probablity mass function over the x-location of global min.

    Args:
      y_fantasized: (n, m) shaped array of m fantasized y values over a common
      set of n x-locations.

    Returns:
      Estimated (n,) shaped array of pmf of the global min over x-location where
      the domain of the pmf is the common set of previous x-locations.

    """
    counts = jnp.bincount(
        jnp.argmin(y_fantasized, axis=0), length=y_fantasized.shape[0])
    return counts / jnp.sum(counts)

  def _entropy(self, p, eps = 1e-16):
    """Evaluate the entropy of an empirical probablity distribution.

    Args:
      p: (n,) shaped array of probability values over n x_lodations.
      eps: optional float tolerance to avoid numerical issues.

    Returns:
      estimated entropy of p.
    """
    return -jnp.sum(
        jnp.where(p < eps, 0., (p + eps) * jnp.log(p + eps)), axis=0)

  @functools.partial(jax.jit, static_argnums=(0,))
  def _entropy_of_p_x_min_given_data(
      self,
      for_i_index,
      key,
      x_obs,
      y_obs,
      x_batch = None,
      y_batch = None):
    """Compute information nats for locating global min given 1 batch data pair.

    Args:
      for_i_index: int used to fold in over the key and to slice y_batch.
      key: PRNG key for jax.random.
      x_obs: (k, d) shaped array of k observed x-locations in d dimensions.
      y_obs: (k, 1) shaped array of observed y values at x_obs.
      x_batch: (q, d) shaped array of q candidate x-locations in d dimensions.
      y_batch: (q, 1) shaped array of y values corresponding to x_batch.

    Returns:
      Non-negative float value of the nats of gained information from
      x_batch and y_batch for locating the global min.
    """
    if operator.xor(x_batch is None, y_batch is None):
      raise ValueError("Both x_batch and y_batch need to be provided.")

    if (x_batch is not None) and (y_batch is not None):
      if y_batch.ndim == 1:
        y_batch = y_batch[:, None]
      x_obs = jnp.vstack((x_obs, x_batch))
      y_obs = jnp.vstack((y_obs, y_batch[:, for_i_index][:, None]))
      key = jax.random.fold_in(key, for_i_index)

    fantasized_y_values = self.fantasize_y_values_for_p_min(
        key=key, x_obs=x_obs, y_obs=y_obs)
    p_x_min = self._p_x_min(fantasized_y_values)
    entropy = self._entropy(p_x_min, self.eps)
    return entropy

  def information_gain(self, key, x_obs,
                       y_obs, x_batch,
                       y_batch,
                       include_initial_entropy = True,
                       vectorized = False):
    """Compute information nats for locating the global min given batch data.

    In the below function, k refers to the number of observations,
    d is the data dimension, q is the size of the batch of interest,
    and t refers to the number of predictions/evaluations of y perfomed at the
    batch of x-locations.

    Args:
      key: PRNG key for jax.random.
      x_obs: (k, d) shaped array of k observed x-locations in d dimensions.
      y_obs: (k, 1) shaped array of observed y values at x_obs.
      x_batch: (q, d) shaped array of q candidate x-locations in d dimensions.
      y_batch: (q, t) shaped array of y values corresponding to x_batch. Each
        column of this array corresponds to one realization of q y values at
        x_batch evaluated/predicted for t times.
      include_initial_entropy: bool which decides whether initial entropy should
        be included in the calculations.
      vectorized: bool to set whether to evaluate conditional entropy over
        y_batch in a vectorized manner. Note: vectorizing requires significantly
          more memory.
    Returns:
      (t,) shaped array of non-negative float values indicating the nats
      of gained information from the x_batch and y_batch for locating the
      x-location of the global min.
    """
    for_i_index_all = jnp.arange(y_batch.shape[1])
    partial_conditional_entropy = functools.partial(
        self._entropy_of_p_x_min_given_data,
        key=key,
        x_obs=x_obs,
        y_obs=y_obs,
        x_batch=x_batch,
        y_batch=y_batch)

    if vectorized:
      conditional_entropy = jax.vmap(partial_conditional_entropy)(
          for_i_index_all)
    else:
      conditional_entropy = jax.lax.map(partial_conditional_entropy,
                                        for_i_index_all)
    if include_initial_entropy:
      if self.initial_entropy_key is None:
        raise ValueError("initial_entropy_key needs to be initialized.")
      initial_entropy = self._entropy_of_p_x_min_given_data(
          for_i_index=0,  # dummy int
          key=self.initial_entropy_key,
          x_obs=x_obs,
          y_obs=y_obs)
      return initial_entropy - conditional_entropy
    else:
      return - conditional_entropy


def _mean_utility(for_i_index,
                  key_xs,
                  key_ys,
                  key_utilities,
                  search_space,
                  budget,
                  utility_measure_fn,
                  draw_x_location_fn,
                  draw_y_values_fn,
                  y_drawn_num = 1000):
  """Evaluate the mean utility of a search space at a budget."""
  key_x = key_xs[for_i_index, :]
  key_y = key_ys[for_i_index, :]
  key_utility = key_utilities[for_i_index, :]

  x_locations = draw_x_location_fn(key_x, search_space, budget)
  y_values = draw_y_values_fn(key_y, x_locations, y_drawn_num)
  utility_values = utility_measure_fn(key_utility, x_locations, y_values)
  return jnp.mean(utility_values, axis=0)


def uniform(key_x, search_space, budget):
  return jax.random.uniform(
      key_x,
      shape=(budget, search_space.shape[0]),
      minval=search_space[:, 0],
      maxval=search_space[:, 1])


def mean_utility(key,
                 search_space,
                 budget,
                 utility_measure_fn,
                 draw_y_values_fn,
                 x_drawn_num,
                 y_drawn_num,
                 draw_x_location_fn = None,
                 vectorized = False):
  """Evaluate the mean utility of budget # of x-locations from a search space.

  This function evaluates the utility (e.g., improvement) of budget # of
  x-locations drawn from the search space via sampling fn draw_x_location_fn,
  where the utility is averaged over the y values of the x-locations. The y
  values are calculated via draw_y_values_fn.

  Args:
    key: PRNG key for jax.random.
    search_space: (d, 2) shaped array of lower and upper bounds for x-locations.
    budget: integer number of x-locations one can afford to query.
    utility_measure_fn: a function of an instance of the UtilityMeasure class.
    draw_y_values_fn: a function which inputs the x-locations and evaluates or
      predicts their corresponding y-values. This function can be either the
      ground-truth function or a predictive function (e.g., fantasize_y_values)
      of a probablistic model (e.g., GP).
    x_drawn_num: number of MC iterations for x-location.
    y_drawn_num: number of MC iterations for y-values of each x-location draws.
    draw_x_location_fn: a function which inputs the search space and the budget
      and draws budget of x-locations within the search space. Default uniform.
    vectorized: bool to set whether to evaluate mean utility over keys in a
      vectorized manner. Note: vectorizing requires significantly more memory.
  Returns:
    (x_drawn_num,) shaped array of utilities at x-locations averaged over
    their y-values.

  Examples of how to set the utility_measure: ``` utility_measure =
    scores.UtilityMeasure() # if utility is is-improvement and improvement
  utility_measure_fn = lambda key, x_batch, y_batch:
  utility_measure.is_improvement(y_batch)
  utility_measure_fn= lambda key, x_batch, y_batch:
  utility_measure.improvement(y_batch)
  # if utility is information gain
  utility_measure_fn= partial(utility_measure.information_gain,
  x_obs=x_obs, y_obs=y_obs)
  ```
  """
  key_x, key_y, key_utility = jax.random.split(key, 3)

  key_x_mc = jax.random.split(key_x, x_drawn_num)
  key_y_mc = jax.random.split(key_y, x_drawn_num)
  key_utility_mc = jax.random.split(key_utility, x_drawn_num)

  if draw_x_location_fn is None:
    draw_x_location_fn = uniform
  partial_mean_utility = functools.partial(
      _mean_utility,
      key_xs=key_x_mc,
      key_ys=key_y_mc,
      key_utilities=key_utility_mc,
      search_space=search_space,
      budget=budget,
      utility_measure_fn=utility_measure_fn,
      draw_x_location_fn=draw_x_location_fn,
      draw_y_values_fn=draw_y_values_fn,
      y_drawn_num=y_drawn_num)
  for_i_index_all = jnp.arange(x_drawn_num)

  if vectorized:
    mean_utility_arr = jax.vmap(partial_mean_utility, for_i_index_all)
  else:
    mean_utility_arr = jax.lax.map(partial_mean_utility, for_i_index_all)

  return mean_utility_arr


def scores(
    mean_utility_arr,
    statistic_fns):
  """Evaluate the statisctics of mean utility of a search space and a budget.

  Args:
    mean_utility_arr: (x_drawn_num,) shaped array of utilities at x-locations
      averaged over their y-values. This array is the output of score function.
    statistic_fns: a list of statistics to be evaluated across the mean_utlity
      including generic functions (e.g., jnp.mean & jnp.median) and user-defined
      functions.

  Returns:
    result: a dict containing statistics of the mean_utility, i.e., the scores.

  """
  results = {}
  for statistic_fn in statistic_fns:
    results[statistic_fn.__name__] = statistic_fn(mean_utility_arr)
  return results
# pylint: enable=g-doc-return-or-yield
