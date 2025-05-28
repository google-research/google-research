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

"""Data structures used by EV3 methods."""

from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple, TypeVar, Union

import chex
import flax
import optax

from ev3 import base


def update_model_history(
    model_history,
    update_logs,
    max_entries = 5,
):
  """Updates the model history to append the current results.

  Args:
    model_history: The model history (expected to be a list) to be updated. If
      None, an empty list will be created.
    update_logs: Updates/results to be added to the history.
    max_entries: An int that specifies how many results to be kept in history,
      default is 5. If negative value, keep all the history.

  Returns:
    A list as the updated model history.
  """
  if max_entries == 0:
    return None
  if update_logs is None:
    return model_history
  if model_history is None:
    return [update_logs]
  if max_entries < 0:
    return model_history + [update_logs]
  else:
    return (model_history + [update_logs])[-max_entries:]


class ModelExpandFn(Protocol):

  def __call__(
      self,
      nn_model,
      params,
      init_key,
      batch,
      **expand_kwargs,
  ):
    """The signature of the function that expands the model."""


class ModelGraph(base.ModelGraph):
  """The model graph data structure."""

  nn_model: Any = flax.struct.field(pytree_node=False)
  expand_fn: ModelExpandFn = flax.struct.field(pytree_node=False, default=None)
  expand_kwargs: Dict[str, Any] = flax.struct.field(
      pytree_node=False, default=None
  )

  def expand(self, params, stable_params, init_key, batch):
    if self.expand_fn is None:
      return self, params, stable_params

    if self.expand_kwargs is None:
      expand_kwargs = {}
    else:
      expand_kwargs = self.expand_kwargs

    _, new_params = self.expand_fn(
        self.nn_model, params, init_key, batch, **expand_kwargs
    )
    new_nn_model, new_stable_params = self.expand_fn(
        self.nn_model, stable_params, init_key, batch, **expand_kwargs
    )

    return (
        self.replace(apply_fn=new_nn_model.apply, nn_model=new_nn_model),
        new_params,
        new_stable_params,
    )


class ModelUpdates(base.ModelUpdates):
  """The update data structure.

  Attributes:
    params_list: List of update parameters to be added to model parameters.
    graph: If not None, then this replaces the model graph.
    stable_params: The new best-performing model parameters.
    logs: The logs to be added to model history.
  """

  params_list: Sequence[base.Params]
  rand_key: chex.PRNGKey = None
  graph: ModelGraph = None
  stable_params: base.Params = None
  logs: Dict[str, Any] = flax.struct.field(pytree_node=False, default=None)

  def __getitem__(self, key):
    return self.replace(params_list=(self.params_list[key],))

  def __len__(self):
    return len(self.params_list)


ModelUpdatesT = TypeVar('ModelUpdatesT', bound=ModelUpdates)


class Model(base.Models):
  """The model data structure containing both the graph and the parameters.

  Attributes:
    graph: Model graph is inherited from base.Model.
    params: Model parameters.
    stable_params: The best-performing model parameters encountered so far.
    history: The model history object to keep track of changes to the model.
    history_max_entries: The max number of entries to be saved into history to
      avoid infinite growth. Default is 5.
    update_model_history_fn: A function that takes inputs [history, logs,
      history_max_entries] and returns the updated history. The default is
      update_model_history(), which format history as a list and append new logs
      to it at each update.
  """

  graph: ModelGraph
  rand_key: chex.PRNGKey
  params: base.Params = None
  stable_params: base.Params = None
  history: Any = flax.struct.field(pytree_node=False, default=None)
  history_max_entries: int = 5
  update_model_history_fn: Callable[[Any, Any, int], Any] = flax.struct.field(
      pytree_node=False, default=update_model_history
  )
  just_expanded: bool = False

  def __add__(self, update):
    assert len(update.params_list) == 1, (
        'A ModelUpdates object can only be added to a Model object when it has '
        f'a single parameter, not {len(update.params_list)}.'
    )
    if update.graph is not None:
      new_model = self.replace(
          graph=update.graph,
          params=update.params_list[0],
          stable_params=update.stable_params,
          history=None,
          rand_key=update.rand_key,
          just_expanded=True,
      )
      return new_model
    else:
      output = self.replace(
          params=update.params_list[0],
          history=self.update_model_history_fn(
              self.history, update.logs, self.history_max_entries
          ),
          just_expanded=False,
      )
      if update.stable_params is not None:
        output = output.replace(stable_params=update.stable_params)
      return output


ModelT = TypeVar('ModelT', bound=Model)


class ProposeStateBase(base.ProposeState):
  data_iter: Iterator[base.Batch]
  # Losses
  loss_fn_list: Sequence[base.LossFn]
  loss_states: Sequence[base.EvalState]


class ProposeStateBaseWithDefaultValues(base.ProposeState):
  trajectory_length: int = 1
  # Optax optimizers
  tx_list: Sequence[Any] = (optax.sgd(learning_rate=0.1),)
  # Variables added during init_fn.
  grad_fn_list: Sequence[base.GradFn] = None
  opt_states: Sequence[Sequence[optax.OptState]] = None
  # A factor to increase the trajectory length wrt insignificant updates.
  traj_mul_factor: float = 1
  # Whether there is auxiliary info when calculating gradients using jax.grad.
  has_aux: bool = False


class ProposeState(ProposeStateBaseWithDefaultValues, ProposeStateBase):
  """Holds variables needed to propose updates.

  Cf.
  https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
  # pylint: disable=line-too-long
  """


ProposeStateT = TypeVar('ProposeStateT', bound=ProposeState)


class OptimizeStateBase(base.OptimizeState):
  data_iter: Iterator[base.Batch]
  # Metrics
  metric_fn_list: Sequence[base.VectorizedMetricFn]


class OptimizeStateBaseWithDefaultValues(base.OptimizeState):
  ucb_alpha: float = 1.0


class OptimizeState(OptimizeStateBaseWithDefaultValues, OptimizeStateBase):
  """Holds variables needed to optimize updates.

  Cf.
  https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
  # pylint: disable=line-too-long
  """


OptimizeStateT = TypeVar('OptimizeStateT', bound=OptimizeState)


class DecideStateBase(base.DecideState):
  data_iter: Iterator[base.Batch]
  # Metrics
  metric_fn_list: Sequence[base.VectorizedMetricFn]


class DecideStateBaseWithDefaultValues(base.DecideState):
  ucb_alpha: float = 1.0


class DecideState(DecideStateBaseWithDefaultValues, DecideStateBase):
  """Holds variables needed to decide updates.

  Cf.
  https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
  # pylint: disable=line-too-long
  """


DecideStateT = TypeVar('DecideStateT', bound=DecideState)


class GetBatchFn(Protocol):

  def __call__(
      self,
      state,
  ):
    """The signature of the function that evaluates updates."""


class EvaluateUpdatesFn(Protocol):

  def __call__(
      self,
      model_updates,
      state,
      model,
      add_model_params,
      get_batch_fn,
  ):
    """The signature of the function that evaluates updates."""
