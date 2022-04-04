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

"""Launch template for comparison between baseline and boosted sampling methods.
"""
import jax
import jax.numpy as jnp
import numpy as np
import spaceopt.run_comparison as runcomp


def launch_comparison(key,
                      objective_fn,
                      search_space,
                      budget,
                      budget_b1,
                      params,
                      num_rounds,
                      sampling_method_primary,
                      sampling_method_secondary=None,
                      num_init=1,
                      batch_size=1,
                      reduce_rates=None,
                      num_pnts_for_af=500,
                      num_steps_for_gp=1000,
                      num_x_for_score=1000,
                      num_y_for_score=1000,
                      num_ss_per_rate=100,
                      sampling_gp_method='tfp',
                      x_precollect=None,
                      y_precollect=None,
                      additional_info_precollect=None):
  """Launches a comparison run."""
  if reduce_rates is None:
    reduce_rates = jnp.array([[.2], [.4], [.6], [.8], [1]]).flatten()
  vols_best = dict()
  search_spaces_best = dict()
  keys_vols = [('is_imp', 'mean'), ('is_imp', 'median'), ('imp', 'mean'),
               ('imp', 'median')]
  for key_vol in keys_vols:
    vols_best[key_vol] = []
    search_spaces_best[key_vol] = []
  y_secondary_is_imp_mean = jnp.empty(shape=((budget), 0))
  y_secondary_is_imp_median = jnp.empty(shape=((budget), 0))
  y_secondary_imp_mean = jnp.empty(shape=((budget), 0))
  y_secondary_imp_median = jnp.empty(shape=((budget), 0))
  y_base = jnp.empty(shape=((budget), 0))

  for i in range(num_rounds):
    key_loop = jax.random.fold_in(key, i)
    results = runcomp.run_method_on_base_and_reduced_search_spaces(
        key_loop,
        objective_fn,
        search_space,
        budget,
        budget_b1=budget_b1,
        params=params,
        centrality_over_x=[jnp.mean, jnp.median],
        sampling_method_primary=sampling_method_primary,
        num_init_for_bo=num_init,
        batch_size_for_bo=batch_size,
        num_x_for_score=num_x_for_score,
        num_y_for_score=num_y_for_score,
        reduce_rates=reduce_rates,
        num_pnts_for_af=num_pnts_for_af,
        num_steps_for_gp=num_steps_for_gp,
        num_ss_per_rate=num_ss_per_rate,
        sampling_method_secondary=sampling_method_secondary,
        sampling_gp_method=sampling_gp_method,
        x_precollect=x_precollect,
        y_precollect=y_precollect,
        additional_info_precollect=additional_info_precollect)

    y_secondary_is_imp_mean = jnp.hstack(
        (y_secondary_is_imp_mean,
         np.minimum.accumulate(results['secondary'][('is_imp',
                                                     'mean')]['y_secondary'])))
    y_secondary_is_imp_median = jnp.hstack(
        (y_secondary_is_imp_median,
         np.minimum.accumulate(
             results['secondary'][('is_imp', 'median')]['y_secondary'])))
    y_secondary_imp_mean = jnp.hstack(
        (y_secondary_imp_mean,
         np.minimum.accumulate(results['secondary'][('imp',
                                                     'mean')]['y_secondary'])))
    y_secondary_imp_median = jnp.hstack(
        (y_secondary_imp_median,
         np.minimum.accumulate(
             results['secondary'][('imp', 'median')]['y_secondary'])))
    y_base = jnp.hstack((y_base, np.minimum.accumulate(results['y_base'])))
    for key_vol in keys_vols:
      vols_best[key_vol].append(results['secondary'][key_vol]['vol_best'])
      search_spaces_best[key_vol].append(
          results['secondary'][key_vol]['search_space_best'])
  results_all = {}
  results_all['search_spaces_best'] = search_spaces_best
  results_all['vols_best'] = vols_best
  results_all['y_base'] = y_base
  results_all['y_secondary_is_imp_mean'] = y_secondary_is_imp_mean
  results_all['y_secondary_is_imp_median'] = y_secondary_is_imp_median
  results_all['y_secondary_imp_mean'] = y_secondary_imp_mean
  results_all['y_secondary_imp_median'] = y_secondary_imp_median
  return results_all

