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

"""Collection of the supported optimizers."""

from typing import Any, Callable, NamedTuple

from flax import traverse_util
import jax
import jax.numpy as jnp
import optax

from imp.max.core import constants
from imp.max.optimization import config as opt_config
from imp.max.optimization import schedules
from imp.max.utils import tree as mtu
from imp.max.utils import typing


Optimizer = constants.Optimizer


class ScaleByLowRankAdamState(NamedTuple):
  """State for the Adam algorithm."""
  count: jax.Array  # shape=(), dtype=jnp.int32.
  low_rank_projectors: optax.Updates
  mu: optax.Updates
  nu: optax.Updates


def scale_by_galore_adam(
    b1 = 0.9,
    b2 = 0.999,
    eps = 1e-8,
    eps_root = 0.0,
    rank = 128,
    svd_frequency = 250,
    mu_dtype = None,
    precision = None,
    *,
    nesterov = False
):
  """Rescale updates according to GaLore-Adam as in arxiv.org/abs/2403.03507."""

  adam_scale = optax.scale_by_adam(
      b1=b1,
      b2=b2,
      eps=eps,
      eps_root=eps_root,
      mu_dtype=mu_dtype,
      nesterov=nesterov,
  )

  def init_fn(params):
    low_rank_projectors = mtu.tree_low_rank_projector(
        array_tree=params,
        rank=rank,
        method='svd',
    )
    low_rank_params = mtu.tree_project_array(
        array_tree=params,
        projection_state_tree=low_rank_projectors,
        back_projection=False,
        precision=precision,
    )
    adam_state = adam_scale.init(low_rank_params)
    # pytype: disable=attribute-error
    return ScaleByLowRankAdamState(
        count=adam_state.count,
        low_rank_projectors=low_rank_projectors,
        mu=adam_state.mu,
        nu=adam_state.nu,
    )
    # pytype: enable=attribute-error

  def update_fn(updates, state, params=None):
    del params
    # We only perform SVD if svd_frequency is met. Otherwise, we re-use
    # previous projectors that are already included in 'state'.
    low_rank_projectors = jax.lax.cond(
        jnp.mod(state.count, svd_frequency) == 0,
        true_fun=lambda: mtu.tree_low_rank_projector(updates, rank, 'svd'),
        false_fun=lambda: state.low_rank_projectors,
    )
    # We need to project 'updates' to low-rank and fetch new updates based on
    # Adam update algorithm in the low-rank space.
    low_rank_updates = mtu.tree_project_array(
        array_tree=updates,
        projection_state_tree=low_rank_projectors,
        back_projection=False,
        precision=precision,
    )
    # Note that 'state' is already low-rank, hence no need for reprojection
    low_rank_updates, low_rank_adam_state = adam_scale.update(
        low_rank_updates, state)
    # We project back the low-rank updates to the original space so that params
    # are updated.
    updates = mtu.tree_project_array(
        array_tree=low_rank_updates,
        projection_state_tree=low_rank_projectors,
        back_projection=True,
        precision=precision,
    )
    # pytype: disable=attribute-error
    state = ScaleByLowRankAdamState(count=low_rank_adam_state.count,
                                    low_rank_projectors=low_rank_projectors,
                                    mu=low_rank_adam_state.mu,
                                    nu=low_rank_adam_state.nu)
    # pytype: enable=attribute-error
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)


def galore_adamw(
    learning_rate,
    b1 = 0.9,
    b2 = 0.999,
    eps = 1e-8,
    eps_root = 0.0,
    rank = 128,
    svd_frequency = 250,
    mu_dtype = None,
    weight_decay = 1e-4,
    mask = None,
    *,
    nesterov = False,
):
  """Adam with weight decay regularization as in arxiv.org/abs/2403.03507."""
  return optax.chain(
      scale_by_galore_adam(
          b1=b1,
          b2=b2,
          eps=eps,
          eps_root=eps_root,
          rank=rank,
          svd_frequency=svd_frequency,
          mu_dtype=mu_dtype,
          nesterov=nesterov,
      ),
      optax.add_decayed_weights(weight_decay, mask),
      optax.scale_by_learning_rate(learning_rate),
  )


_OPTIMIZERS = {
    Optimizer.SGD: optax.sgd,
    Optimizer.ADAM: optax.adam,
    Optimizer.ADAM_W: optax.adamw,
    Optimizer.ADAFACTOR: optax.adafactor,
    Optimizer.GALORE_ADAM_W: galore_adamw,
}


def _weight_decay_mask_fn(
    params,
):
  """Specifies which parameters should be used for weight decay."""
  flat_params = traverse_util.flatten_dict(params, sep='/')
  flat_mask = {name: name.endswith('/kernel') for name in flat_params}
  return traverse_util.unflatten_dict(flat_mask, sep='/')


def get_optimizer(
    config):
  """Returns optimizer object with scheduled LR according to config."""

  learning_rate = schedules.get_schedule(config.learning_rate)

  config = config.as_dict()
  config['learning_rate'] = learning_rate
  name = config.pop('name')
  optimizer = _OPTIMIZERS[name]

  if name in (Optimizer.ADAM_W, Optimizer.GALORE_ADAM_W):
    config['mask'] = _weight_decay_mask_fn
  elif name == Optimizer.ADAFACTOR:
    config['weight_decay_mask'] = _weight_decay_mask_fn

  return optimizer(**config)
