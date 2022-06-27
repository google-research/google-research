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

"""Running a sampling method on the base and the reduced search spaces."""
import functools
import operator
from typing import Any, Callable


import jax
import jax.numpy as jnp
import spaceopt.bo_utils as bo
import spaceopt.gp_utils as gp
import spaceopt.scores as scores
import spaceopt.search_spaces as search_spaces

# pylint: disable=g-long-lambda


def point_in_search_space(points, search_space):
  assert points.shape[1] == search_space.shape[0]
  mask = jnp.all(
      search_space[:, 0] <= points, axis=1) * jnp.all(
          search_space[:, 1] >= points, axis=1)
  return mask


def prep_utility_fn(params,
                    x_b1,
                    y_b1,
                    utility_type,
                    steps=1000,
                    percentile=0,
                    y_incumb=None):
  """Prepare the utility functions for the score."""
  gp_util = gp.GPUtils()
  params = gp_util.fit_gp(x_b1, y_b1, params, steps=steps)
  if y_incumb is None:
    y_incumb = jnp.percentile(y_b1, percentile, axis=0)
  utility_measure = scores.UtilityMeasure(incumbent=y_incumb, params=params)
  utility_measure_fn = lambda key, x_batch, y_batch: getattr(
      utility_measure, utility_type)(y_batch)
  return utility_measure_fn, params


def draw_y_values(key,
                  params,
                  x_obs,
                  y_obs,
                  x_test,
                  num_samples,
                  sampling_gp_method):
  """Draw y samples from the GP posterior."""
  gp_util = gp.GPUtils()
  mu, cov = gp_util.posterior_mean_cov(params, x_obs, y_obs, x_test)
  samples = gp_util.draw_gp_samples(
      key, mu, cov, num_samples=num_samples, method=sampling_gp_method)
  return samples


def eval_score(key,
               search_space,
               budget,
               utility_measure_fn,
               draw_y_values_fn,
               centrality_over_x,
               x_drawn_num=500,
               y_drawn_num=500):
  """Evaluate the (is)improvement scores."""
  mean_utility_val = scores.mean_utility(
      key,
      search_space,
      budget,
      utility_measure_fn,
      draw_y_values_fn,
      x_drawn_num=x_drawn_num,
      y_drawn_num=y_drawn_num)
  mean_utility_is = mean_utility_val > 0
  imp = scores.scores(mean_utility_val, centrality_over_x)
  is_imp = scores.scores(mean_utility_is, centrality_over_x)
  imp_method_to_stats = {}
  imp_method_to_stats['imp'] = imp
  imp_method_to_stats['is_imp'] = is_imp
  return imp_method_to_stats


def extract_best_search_space(scores_dict, centrality_key,
                              search_spaces_reduced):
  """Select the arg max score search space."""
  return search_spaces_reduced[jnp.argmax(scores_dict[centrality_key]), :, :]


class SamplingMethod:
  """Class for the sampling method to spend a budget over a search space."""

  def __init__(self,
               search_space,
               objective_fn,
               x_precollect=None,
               y_precollect=None,
               additional_info_precollect=None):
    self.search_space = search_space
    self.objective_fn = objective_fn
    self.x_precollect = x_precollect
    self.y_precollect = y_precollect
    self.additional_info_precollect = additional_info_precollect

  def rs(self, key, budget):
    """Random search sampling method with uniform distribution."""
    if operator.xor(self.x_precollect is None, self.y_precollect is None):
      raise ValueError(
          'Both x_precollect and y_precollect need to be provided.')

    if self.x_precollect is not None:
      ind_pre = point_in_search_space(self.x_precollect, self.search_space)
      x_precollect_in_search_space = self.x_precollect[ind_pre, :]
      y_precollect_in_search_space = self.y_precollect[ind_pre, :]

      if y_precollect_in_search_space.shape[0] < budget:
        raise ValueError(
            'budget is larger than the precollected data in the search space.')
      ind_chosen = jax.random.choice(
          key,
          y_precollect_in_search_space.shape[0],
          shape=(budget,),
          replace=False)
      ind_chosen = jnp.sort(ind_chosen)
      x = x_precollect_in_search_space[ind_chosen, :]
      y = y_precollect_in_search_space[ind_chosen, :]
      additional_info_dict = {}
    else:
      x = jax.random.uniform(
          key,
          shape=(budget, self.search_space.shape[0]),
          minval=self.search_space[:, 0],
          maxval=self.search_space[:, 1])
      y, additional_info_dict = self.objective_fn(x)
      ind_chosen = None
    return x, y, additional_info_dict, ind_chosen

  def bo(self,
         key,
         budget,
         params,
         x_obs,
         y_obs,
         batch_size=1,
         num_points=500,
         num_steps=1000,
         sampling_gp_method='tfp'):
    """Bayesian optimization sampling method."""
    rest_budget = budget - x_obs.shape[0]
    x, y, additional_info_dict = bo.bo(
        key,
        x_obs,
        y_obs,
        self.objective_fn,
        params,
        self.search_space,
        rest_budget,
        batch_size=batch_size,
        num_points=num_points,
        num_steps=num_steps,
        method=sampling_gp_method)
    ind_chosen = None
    return x, y, additional_info_dict, ind_chosen


def run_sampling_method(key,
                        objective_fn,
                        search_space,
                        budget,
                        sampling_method='RS',
                        params=None,
                        x_init=None,
                        y_init=None,
                        num_init_for_bo=1,
                        batch_size_for_bo=1,
                        num_pnts_for_af=500,
                        num_steps_for_gp=1000,
                        sampling_gp_method='tfp',
                        x_precollect=None,
                        y_precollect=None,
                        additional_info_precollect=None):
  """Run the sampling method on a search space given a budget."""
  sampling = SamplingMethod(search_space, objective_fn, x_precollect,
                            y_precollect, additional_info_precollect)
  if sampling_method == 'RS':
    x_sampled, y_sampled, _, ind_sampled_in_precollected = sampling.rs(
        key, budget)
  elif sampling_method == 'BO':
    if (x_init is None) and (y_init is None):
      key_init, key_rest = jax.random.split(key)
      x_init, y_init, _, _ = sampling.rs(key_init, num_init_for_bo)
    else:
      key_rest = key
    x_sampled, y_sampled, _, ind_sampled_in_precollected = sampling.bo(
        key_rest,
        budget,
        params,
        x_init,
        y_init,
        batch_size=batch_size_for_bo,
        num_points=num_pnts_for_af,
        num_steps=num_steps_for_gp,
        sampling_gp_method=sampling_gp_method)
  else:
    raise ValueError('Sampling method should be either RS or BO.')
  return x_sampled, y_sampled, ind_sampled_in_precollected


def run_method_on_base_and_reduced_search_spaces(
    key,
    objective_fn,
    search_space,
    budget,
    budget_b1,
    params,
    centrality_over_x,
    sampling_method_primary='RS',
    num_init_for_bo=1,
    batch_size_for_bo=1,
    num_pnts_for_af=500,
    num_steps_for_gp=1000,
    num_x_for_score=500,
    num_y_for_score=500,
    reduce_rates=None,
    num_ss_per_rate=50,
    sampling_method_secondary=None,
    acquisition_method='improvement',
    sampling_gp_method='tfp',
    percentile=0,
    y_incumb=None,
    x_precollect=None,
    y_precollect=None,
    additional_info_precollect=None):
  """Run the sampling method on base and best reduced search space."""
  key_sampling_prim, key_sampling_sec, key_ss, key_score = jax.random.split(
      key, 4)
  x_base, y_base, _ = run_sampling_method(
      key_sampling_prim,
      objective_fn,
      search_space,
      budget,
      sampling_method=sampling_method_primary,
      params=params,
      x_init=None,
      y_init=None,
      num_init_for_bo=num_init_for_bo,
      batch_size_for_bo=batch_size_for_bo,
      num_pnts_for_af=num_pnts_for_af,
      num_steps_for_gp=num_steps_for_gp,
      x_precollect=x_precollect,
      y_precollect=y_precollect,
      additional_info_precollect=additional_info_precollect)

  reduce_rates_repeated = jnp.repeat(reduce_rates, num_ss_per_rate)
  keys_ss = jax.random.split(key_ss, reduce_rates_repeated.shape[0])
  search_spaces_reduced = jax.vmap(
      search_spaces.generate_search_space_reduce_vol,
      in_axes=(0, None, 0))(keys_ss, search_space, reduce_rates_repeated)
  vols_reduced = jax.vmap(search_spaces.eval_vol)(search_spaces_reduced)

  x_b1 = x_base[:budget_b1, :]
  y_b1 = y_base[:budget_b1, :]

  utility_measure_fn, params_optimized = prep_utility_fn(
      params,
      x_b1,
      y_b1,
      acquisition_method,
      steps=num_steps_for_gp,
      percentile=percentile,
      y_incumb=y_incumb)

  # preparing the inputs for eval_score function to evaluate the score of
  # randomly generated reduced-volume search spaces at the remaining budget.
  draw_y_values_fn = lambda key, x_test, num_samples: draw_y_values(
      key=key,
      params=params_optimized,
      x_obs=x_b1,
      y_obs=y_b1,
      x_test=x_test,
      num_samples=num_samples,
      sampling_gp_method=sampling_gp_method)
  budget_b2 = budget - budget_b1
  partial_eval_score = functools.partial(
      eval_score,
      budget=budget_b2,
      utility_measure_fn=utility_measure_fn,
      draw_y_values_fn=draw_y_values_fn,
      centrality_over_x=centrality_over_x,
      x_drawn_num=num_x_for_score,
      y_drawn_num=num_y_for_score)
  keys_score = jax.random.split(key_score, search_spaces_reduced.shape[0])
  # the vmap evaluates the score for the generated search spaces.
  imp_method_to_stats = jax.vmap(partial_eval_score)(keys_score,
                                                     search_spaces_reduced)

  if sampling_method_secondary is None:
    sampling_method_secondary = sampling_method_primary

  result_secondary = {}
  result_secondary['search_spaces_reduced'] = search_spaces_reduced
  result_secondary['vols_reduced'] = vols_reduced

  for imp in imp_method_to_stats.keys():
    imp_dict = imp_method_to_stats[imp]
    for centrality in imp_dict.keys():
      secondary_dict = {}
      search_space_best = extract_best_search_space(imp_dict, centrality,
                                                    search_spaces_reduced)
      x_secondary, y_secondary, _ = run_sampling_method(
          key_sampling_sec,
          objective_fn,
          search_space_best,
          budget_b2,
          sampling_method=sampling_method_secondary,
          params=params,
          x_init=x_b1,
          y_init=y_b1,
          num_init_for_bo=num_init_for_bo,
          batch_size_for_bo=batch_size_for_bo,
          num_pnts_for_af=num_pnts_for_af,
          num_steps_for_gp=num_steps_for_gp,
          x_precollect=x_precollect,
          y_precollect=y_precollect,
          additional_info_precollect=additional_info_precollect)
      if sampling_method_secondary == 'RS':
        x_secondary = jnp.vstack((x_b1, x_secondary))
        y_secondary = jnp.vstack((y_b1, y_secondary))
      secondary_dict['x_secondary'] = x_secondary
      secondary_dict['y_secondary'] = y_secondary
      secondary_dict['search_space_best'] = search_space_best
      secondary_dict['vol_best'] = search_spaces.eval_vol(
          search_space_best) / search_spaces.eval_vol(search_space)
      result_secondary[(imp, centrality)] = secondary_dict

  results = {}
  results['x_base'] = x_base
  results['y_base'] = y_base
  results['secondary'] = result_secondary

  return results
