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

"""Sample and compute logprobs for a mixture of rings.
"""
import functools

import jax
import jax.numpy as jnp
import jax.scipy as jscipy


def sample_1d_gaussian_params(
    key, mean_mean, mean_scale, scale_shape, scale_scale):
  k1, k2 = jax.random.split(key)
  scale = jax.random.gamma(k1, scale_shape)*scale_scale
  mean = jax.random.normal(k2)*mean_scale + mean_mean
  return mean, scale


def sample_ring_mixture(
    key, n_points, radius_means, radius_scales, centers, log_weights):
  k1, k2, k3 = jax.random.split(key, num=3)
  cs = jax.random.categorical(k1, log_weights, shape=(n_points,))
  rs = jax.random.normal(
      k2, shape=(n_points,)) * radius_scales[cs] + radius_means[cs]
  thetas = jax.random.uniform(
      k3, shape=[n_points], minval=0., maxval=2 * jnp.pi)
  xs = jnp.stack([jnp.sin(thetas), jnp.cos(thetas)],
                 axis=1) * rs[:, jnp.newaxis] + centers[cs]
  return xs, cs


def sample_ring_params(
    key, n_rings,
    radius_mean_mean, radius_mean_scale, radius_scale_shape, radius_scale_scale,
    center_mean, center_cov, raw_weights_scale):
  keys = jax.random.split(key, num=5)
  # Sample the radius means and scales
  r_means, r_scales = jax.vmap(
      sample_1d_gaussian_params,
      in_axes=(0, None, None, None,
               None))(jax.random.split(keys[0], num=n_rings), radius_mean_mean,
                      radius_mean_scale, radius_scale_shape, radius_scale_scale)
  r_means = jnp.abs(r_means)
  # Sample the centers
  centers = jax.random.multivariate_normal(
      keys[3], center_mean, center_cov, shape=[n_rings])
  # Sample the log weights
  raw_weights = jax.random.normal(keys[4], shape=[n_rings])*raw_weights_scale
  log_weights = jax.nn.log_softmax(raw_weights)
  return r_means, r_scales, centers, log_weights


def sample_params_and_points(
    key, num_points, num_rings,
    radius_mean_prior_mean, radius_mean_prior_scale,
    radius_scale_prior_shape, radius_scale_prior_scale,
    center_mean, center_cov, raw_weights_scale):
  k1, k2 = jax.random.split(key)
  r_means, r_scales, centers, log_weights = sample_ring_params(
      k1, num_rings, radius_mean_prior_mean, radius_mean_prior_scale,
      radius_scale_prior_shape, radius_scale_prior_scale,
      center_mean, center_cov, raw_weights_scale)
  xs, cs = sample_ring_mixture(
      k2, num_points, r_means, r_scales, centers, log_weights)
  return xs, cs, (r_means, r_scales, centers, log_weights)


def ring_log_p(x, radius_mean, radius_scale, center):
  norm = jnp.linalg.norm(x - center)
  r_log_p = jscipy.stats.norm.logpdf(norm, loc=radius_mean, scale=radius_scale)
  theta_log_p = - jnp.log(2*jnp.pi)
  return theta_log_p + r_log_p - jnp.log(norm)


@functools.partial(jax.jit)
def ring_mixture_log_p(x, radius_means, radius_scales, centers, log_weights):
  log_ps = jax.vmap(ring_log_p, in_axes=(None, 0, 0, 0))(
      x, radius_means, radius_scales, centers)
  return jscipy.special.logsumexp(log_ps + log_weights)
