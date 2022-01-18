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

"""Regularization functions."""
import jax.numpy as jnp

from cold_posterior_flax.cifar10.models import new_initializers


def he_normal_prior_regularizer(kernel, scale=1):
  regularization = variance_scaling_prior(kernel, 'fan_in', 1.0, scale)
  return regularization


def normal_prior_regularizer(kernel, scale=1, mean=0.0):
  regularization = variance_scaling_prior(kernel, 'standard', 1.0, scale, mean)
  return regularization


def variance_scaling_prior(kernel, mode, he_scale, scale, mean=0.0):
  """Prior based on variance scaling rules used in initialization."""
  if len(kernel.shape) > 1:
    fan_in, fan_out = new_initializers.compute_fans(kernel.shape)
  else:
    fan_in, fan_out = 1, kernel.shape[0]
  if mode == 'fan_in':
    denominator = fan_in
  elif mode == 'fan_out':
    denominator = fan_out
  elif mode == 'fan_avg':
    denominator = (fan_in + fan_out) / 2
  elif mode == 'standard':
    denominator = 1.
  else:
    raise ValueError(
        'invalid mode for variance scaling initializer: {}'.format(mode))
  variance = jnp.array(he_scale / denominator)
  reg_lambda = 0.5 * scale / variance
  regularization = reg_lambda * jnp.sum(jnp.square(kernel - mean))
  return regularization
