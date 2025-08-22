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

"""Base interfaces and datatypes for EV3 components."""

from typing import Dict, Protocol, Sequence, Tuple, TypeVar, Union

import chex
import flax


BatchFeatures = chex.ArrayTree
ModelOutput = chex.ArrayTree
Batch = chex.ArrayTree
Params = Union[chex.ArrayTree, Dict[str, chex.ArrayTree]]
ParamList = Sequence[Params]


class ModelGraphApplyFn(Protocol):

  def __call__(
      self,
      params,
      batch_features,
      **kwargs,
  ):
    """The signature of the init function for the propose step of EV3."""


class ModelGraph(flax.struct.PyTreeNode):
  apply_fn: ModelGraphApplyFn = flax.struct.field(pytree_node=False)


ModelGraphT = TypeVar('ModelGraphT', bound=ModelGraph)


class ModelUpdates(flax.struct.PyTreeNode):
  """The update data structure."""


ModelUpdatesT = TypeVar('ModelUpdatesT', bound=ModelUpdates)


class Models(flax.struct.PyTreeNode):
  """The model data structure containing both the graph and the parameters."""

  graph: ModelGraph

  def __add__(self, update):
    """Uses an update to modify the model.

    Args:
      update: An object that encodes a change to the model. The intended use
        case is for 'update' to be an object of a subclass of ModelUpdates, but
        this might not always be the best choice.

    Returns:
      An object of type Models that is obtained from 'self' by applying the
      modification encoded in 'update'.

    Raises:
      NotImplementedError: If this method has not be overloaded in the subclass
      of Models.
    """
    raise NotImplementedError(
        f'The method to add an update of type {update.__class__.__name__} to a '
        f'model of type {self.__class__.__name__} has not been implemented.'
    )


ModelT = TypeVar('ModelT', bound=Models)


class EvalState(flax.struct.PyTreeNode):
  pass


EvalStateT = TypeVar('EvalStateT', bound=EvalState)


class LossFn(Protocol):

  def __call__(
      self,
      model_params,
      model_graph,
      loss_state,
      batch,
  ):
    """Loss function signature."""


class GradFn(Protocol):

  def __call__(
      self,
      model_params,
      model_graph,
      loss_state,
      batch,
  ):
    """Grad function signature."""


VectorizedMetricOutput = chex.ArrayTree


class VectorizedMetricFn(Protocol):

  def __call__(
      self,
      model_params,
      model_graph,
      batch,
  ):
    """Metric function signature."""


class DataIterator(object):

  def next(self, num_batches=1):
    raise NotImplementedError

  def __next__(self):
    raise NotImplementedError

  def __iter__(self):
    return self


class ProposeState(flax.struct.PyTreeNode):
  pass


ProposeStateT = TypeVar('ProposeStateT', bound=ProposeState)


class ProposeInitFn(Protocol):

  def __call__(
      self,
      state,
      model,
  ):
    """The signature of the init function for the propose step of EV3."""


class ProposeUpdateFn(Protocol):

  def __call__(
      self,
      state,
      model,
  ):
    """The signature of the update function for the propose step of EV3."""


class OptimizeState(flax.struct.PyTreeNode):
  pass


OptimizeStateT = TypeVar('OptimizeStateT', bound=OptimizeState)


class OptimizeInitFn(Protocol):

  def __call__(
      self,
      state,
      model,
  ):
    """The signature of the init function for the optimize step of EV3."""


class OptimizeUpdateFn(Protocol):

  def __call__(
      self,
      updates,
      state,
      model,
  ):
    """The signature of the update function for the optimize step of EV3."""


class DecideState(flax.struct.PyTreeNode):
  pass


DecideStateT = TypeVar('DecideStateT', bound=DecideState)


class DecideInitFn(Protocol):

  def __call__(
      self,
      state,
      model,
  ):
    """The signature of the init function for the decide step of EV3."""


class DecideUpdateFn(Protocol):

  def __call__(
      self,
      updates,
      state,
      model,
  ):
    """The signature of the update function for the decide step of EV3."""
