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

"""Mostly utilities for creating Batch Ensembles."""

from typing import Optional, Sequence

import chex
import haiku as hk
# from haiku._src import data_structures
import jax
import jax.numpy as jnp

data_structures = hk._src.data_structures
FlatMap = data_structures.FlatMap
ENSEMBLE_PARAM_KEYWORD = '__ensemble__'


def prune_empty_flatmaps(params):
  d = {}
  for k, v in params.items():
    if isinstance(v, FlatMap):
      if len(v) == 0:
        pass
      else:
        d[k] = prune_empty_flatmaps(v)
    else:
      d[k] = v
  return FlatMap(d)


def _split_ensemble_params(params):
  shared = {}
  ensemble = {}
  for k, v in params.items():
    if ENSEMBLE_PARAM_KEYWORD in k:
      ensemble[k] = v
    else:
      if isinstance(v, FlatMap):
        only_shared, ensemble_under_shared = _split_ensemble_params(v)
        shared[k] = only_shared
        ensemble[k] = ensemble_under_shared
      else:
        shared[k] = v

  shared = FlatMap(shared)
  ensemble = FlatMap(ensemble)
  return shared, ensemble


def split_ensemble_params(params):
  shared, ensemble = _split_ensemble_params(params)
  return prune_empty_flatmaps(shared), prune_empty_flatmaps(ensemble)


def merge_ensemble_params(shared, ensemble):
  if len(shared) == 0:
    return ensemble
  if len(ensemble) == 0:
    return shared

  d = {}
  shared_keys = shared.keys()
  ensemble_keys = ensemble.keys()
  all_keys = set(iter(shared_keys)).union(set(iter(ensemble_keys)))
  for k in all_keys:
    if k in shared_keys:
      shared_subtree = shared[k]
    else:
      shared_subtree = FlatMap()

    if k in ensemble_keys:
      ensemble_subtree = ensemble[k]
    else:
      ensemble_subtree = FlatMap()

    d[k] = merge_ensemble_params(shared_subtree, ensemble_subtree)

  return FlatMap(d)


def transform_init_for_ensemble(init_fn, init_same=False):
  def _new_init_fn(ensemble_size, rng, *args):
    num_devices = jax.device_count()
    assert ensemble_size % num_devices == 0
    num_models_per_device = ensemble_size // num_devices

    # first get the shared params on the host device
    rng, sub_rng = jax.random.split(rng)
    params = init_fn(sub_rng, *args)
    shared_params, ensemble_params = split_ensemble_params(params)
    del ensemble_params

    # get the ensemble params through pmapping (and vmapping) the init
    # so the params are initialized on the respective devices
    num_args = len(args)
    if init_same:
      in_axes_shape = tuple([None] + [None]*num_args)
    else:
      in_axes_shape = tuple([0] + [None]*num_args)
    pv_init_fn = jax.pmap(
        jax.vmap(init_fn, in_axes=in_axes_shape, out_axes=0),
        in_axes=in_axes_shape, out_axes=0
    )
    if init_same:
      all_rngs = rng
    else:
      all_rngs = jax.random.split(rng, num=ensemble_size)
      all_rngs = jnp.reshape(all_rngs, [num_devices, num_models_per_device, 2])
    pv_params = pv_init_fn(all_rngs, *args)
    pv_shared_params, pv_ensemble_params = split_ensemble_params(pv_params)
    del pv_shared_params

    return shared_params, pv_ensemble_params
  return _new_init_fn


def transform_apply_for_ensemble_member(apply_fn):
  def _new_apply_fn(shared_params, ensemble_params, *args):
    params = merge_ensemble_params(shared_params, ensemble_params)
    return apply_fn(params, *args)
  return _new_apply_fn


def build_same_batch_ensemble_apply_fn(ensemble_member_apply_fn, num_args):
  """Build a function that applies the entire ensemble to the same batch of data.

  The first two arguments are:
  1) device replicated shared params
  2) all ensemble params"""
  in_axes_shape = tuple([None, 0] + [None]*num_args)
  vmapped_apply = jax.vmap(
      ensemble_member_apply_fn,
      in_axes=in_axes_shape,
      out_axes=0)
  in_axes_shape = tuple([0, 0] + [None]*num_args)
  pvmapped_apply = jax.pmap(
      vmapped_apply,
      in_axes=in_axes_shape,
      out_axes=0)
  return pvmapped_apply


def build_different_batch_ensemble_apply_fn(ensemble_member_apply_fn, num_args):
  """Build a function that applies the entire ensemble to a different batch of
  data for each ensemble member.

  The first two arguments are:
  1) device replicated shared params
  2) all ensemble params"""
  in_axes_shape = tuple([None, 0] + [0]*num_args)
  vmapped_apply = jax.vmap(
      ensemble_member_apply_fn,
      in_axes=in_axes_shape,
      out_axes=0)
  in_axes_shape = tuple([0, 0] + [0]*num_args)
  pvmapped_apply = jax.pmap(
      vmapped_apply,
      in_axes=in_axes_shape,
      out_axes=0)
  return pvmapped_apply


class RandomSign(hk.initializers.Initializer):
  """Initializes to +/-1 each element sampled from bernoulli with given prob."""

  def __init__(self, random_sign_prob = 0.5):
    """Constructs a :class:`RandomSign` initializer.

    Args:
      random_sign_prob: probability of +1 for the bernoulli distribution.
    """
    self.random_sign_prob = random_sign_prob

  def __call__(self, shape, dtype):
    w = jax.random.bernoulli(
        hk.next_rng_key(), p=self.random_sign_prob, shape=shape)
    w = jax.lax.convert_element_type(w, dtype)
    return 2.*w - 1.


class DenseBatchEnsembleLayer(hk.Module):
  """Dense Batch Ensemble Layer."""

  def __init__(
      self,
      hidden_dim,
      activation=None,
      random_sign_prob = 0.5,
      name = None):
    super().__init__(name)
    self._alpha_gamma_init = RandomSign(random_sign_prob=random_sign_prob)
    self._activation = activation
    self._hidden_dim = hidden_dim

  def __call__(self, x):
    chex.assert_rank(x, 2)
    x_shape = list(x.shape)

    alpha = hk.get_parameter(
        ENSEMBLE_PARAM_KEYWORD + '_be_alpha',
        [1] + x_shape[1:],
        x.dtype,
        self._alpha_gamma_init)

    w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")
    b_init = jnp.zeros
    fc = hk.Linear(
        self._hidden_dim,
        with_bias=False,
        # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'), # default what SAC implementation uses
        # w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'), # what I had used in the tf implementation
        w_init=w_init,
        # b_init=None, # default what SAC implementation uses
        # b_init=hk.initializers.RandomNormal(stddev=1e-6),
        # b_init=b_init,
        name='shared_linear')

    gamma = hk.get_parameter(
        ENSEMBLE_PARAM_KEYWORD + '_be_gamma',
        [1, self._hidden_dim],
        x.dtype,
        self._alpha_gamma_init)

    bias = hk.get_parameter(
        ENSEMBLE_PARAM_KEYWORD + '_be_bias',
        [1, self._hidden_dim],
        x.dtype,
        b_init)

    y = alpha * x
    y = fc(y)
    y = gamma * y
    y = y + bias
    if self._activation is not None:
      y = self._activation(y)

    return y


# def build_hk_dense_batch_ensemble_layer(
#     hidden_dim,
#     random_sign_prob=0.5,
#     activation=None):
#   def hk_dense_batch_ensemble_layer(x):
#     chex.assert_rank(x, 2)

#     x_shape = list(x.shape)
#     alpha_gamma_init = RandomSign(random_sign_prob=random_sign_prob)

#     alpha = hk.get_parameter(
#         ENSEMBLE_PARAM_KEYWORD + '_be_alpha',
#         [1] + x_shape[1:],
#         x.dtype,
#         alpha_gamma_init)
#     fc = hk.Linear(
#         hidden_dim,
#         with_bias=True,
#         # w_init=hk.initializers.VarianceScaling(1.0, 'fan_in', 'uniform'), # default what SAC implementation uses
#         w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'), # what I had used in the tf implementation
#         # b_init=None, # default what SAC implementation uses
#         b_init=hk.initializers.RandomNormal(stddev=1e-6),
#         name='fc')
#     gamma = hk.get_parameter(
#         ENSEMBLE_PARAM_KEYWORD + '_be_gamma',
#         [1, hidden_dim],
#         x.dtype,
#         alpha_gamma_init)

#     y = alpha * x
#     y = fc(y)
#     if activation is not None:
#       y = activation(y)
#     y = gamma * y

#     return y


def build_ensemble_optimizer(
    ensemble_size,
    shared_params,
    ensemble_params,
    optimizer,
    optimizer_kwargs):
  num_devices = jax.device_count()
  assert ensemble_size % num_devices == 0
  num_models_per_device = ensemble_size // num_devices

  optim = optimizer(**optimizer_kwargs)
  shared_params_optim_state = optim.init(shared_params)

  ensemble_params_optim_init = jax.pmap(
      jax.vmap(optim.init, in_axes=0, out_axes=0),
      in_axes=0, out_axes=0)
  ensemble_params_optim_state = ensemble_params_optim_init(ensemble_params)

  return optim, shared_params_optim_state, ensemble_params_optim_state
