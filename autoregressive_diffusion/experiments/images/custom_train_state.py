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

"""Copyright 2021 The Flax Authors (Public).

Adapted from
https://flax.readthedocs.io/en/latest/_modules/flax/
training/train_state.html#TrainState.
"""

import copy
from typing import Union

from flax import core
from flax import struct
import jax
from jax import numpy as jnp
import optax

ParamDict = core.FrozenDict[str, Union[jnp.ndarray, core.FrozenDict]]


class TrainState(struct.PyTreeNode):
  """Simple train state for the common case with a single Optax optimizer.
  """
  # New with respect to Flax definition:
  ema_params: ParamDict

  # Same as in Flax definition of train_state:
  step: int
  params: ParamDict
  tx: optax.GradientTransformation = struct.field(pytree_node=False)
  opt_state: optax.OptState

  def apply_gradients(self, *, grads, ema_momentum=0.9999, **kwargs):
    """Updates `step`, `params`, `opt_state` and `**kwargs` in return value.

    Note that internally this function calls `.tx.update()` followed by a call
    to `optax.apply_updates()` to update `params` and `opt_state`.

    Args:
      grads: Gradients that have the same pytree structure as `.params`.
      ema_momentum: momentum for EMA updates.
      **kwargs: Additional dataclass attributes that should be `.replace()`-ed.

    Returns:
      An updated instance of `self` with `step` incremented by one, `params`
      and `opt_state` updated by applying `grads`, and additional attributes
      replaced as specified by `kwargs`.
    """
    updates, new_opt_state = self.tx.update(
        grads, self.opt_state, self.params)
    new_params = optax.apply_updates(self.params, updates)

    def update_ema(ema, p):
      return ema_momentum * ema + (1. - ema_momentum) * p

    new_ema_params = jax.tree_multimap(update_ema, self.ema_params, new_params)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        ema_params=new_ema_params,
        opt_state=new_opt_state,
        **kwargs,
    )

  @classmethod
  def create(cls, *, params, tx, **kwargs):
    """Creates a new instance with `step=0` and initialized `opt_state`."""
    opt_state = tx.init(params)
    return cls(
        step=0,
        ema_params=params,
        # Deepcopy to avoid donating the same object twice in train_step.
        params=copy.deepcopy(params),
        tx=tx,
        opt_state=opt_state,
        **kwargs,
    )
