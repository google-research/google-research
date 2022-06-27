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

"""Tests for scores."""

import jax
import jax.numpy as jnp

from spaceopt import gp_utils as gp


def thompson_sampling(key,
                      x_obs,
                      y_obs,
                      gaussian_process,
                      gp_util,
                      search_space,
                      batch_size=1,
                      num_points=2000,
                      method='tfp'):
  """Run one round of Thompson sampling for seq/batch BO ."""
  key_x, key_gp_sample = jax.random.split(key)

  x_test = jax.random.uniform(key_x,
                              shape=(num_points, search_space.shape[0]),
                              minval=search_space[:, 0],
                              maxval=search_space[:, 1])
  mu, cov = gp_util.posterior_mean_cov(gaussian_process, x_obs, y_obs, x_test)
  gp_samples = gp_util.draw_gp_samples(
      key_gp_sample, mu, cov, num_samples=batch_size, method=method)
  min_ind = jnp.argmin(gp_samples, axis=1)
  x_best = x_test[min_ind, :]
  return x_best


def bo(key,
       x_obs,
       y_obs,
       obj,
       params_init,
       params_bounds,
       search_space,
       num_bo_rounds,
       acquisition_fn=thompson_sampling,
       batch_size=1,
       num_points=2000,
       num_steps=1000,
       method='tfp'):
  """Run seq-batch Bayesian Optimization (BO)."""
  gp_util = gp.GPUtils()
  additional_info_dict = {}
  for i in range(num_bo_rounds//batch_size):
    key_loop = jax.random.fold_in(key, i)
    gaussian_process = gp_util.fit_gp(
        x_obs, y_obs, params_init, params_bounds, steps=num_steps)

    x_new = acquisition_fn(key_loop,
                           x_obs,
                           y_obs,
                           gaussian_process,
                           gp_util,
                           search_space,
                           batch_size=batch_size,
                           num_points=num_points,
                           method=method)
    y_new, additional_info = obj(x_new)
    additional_info_dict[i] = additional_info
    x_obs = jnp.vstack((x_obs, x_new))
    y_obs = jnp.vstack((y_obs, y_new))
  return x_obs, y_obs, additional_info_dict
