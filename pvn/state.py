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

"""State."""

import functools
import operator
from typing import Any, Callable, Mapping, Optional, Text, Tuple, Union

from absl import logging
import chex
from flax import core
from flax import struct
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
import optax
from orbax import checkpoint

from pvn.utils import mesh_utils
from pvn.utils import tree_utils

TargetParamsUpdateFunction = Callable[
    [chex.ArrayTree, chex.ArrayTree, int], chex.ArrayTree
]
ElementSpec = Mapping[Text, jax.ShapeDtypeStruct]


def identity(buffer, dummy_variable):
  """Identity fn. with non-trivial computation to prevent jit optimization."""
  return buffer, jnp.sin(dummy_variable**2)


class TrainState(struct.PyTreeNode):  # pytype: disable=invalid-function-definition  # dataclass_transform
  """Train State. This resembles Flax's train state."""

  step: int
  apply_fn: Callable[Ellipsis, Any] = struct.field(pytree_node=False)
  params: chex.ArrayTree

  optim: optax.GradientTransformation = struct.field(pytree_node=False)
  optim_state: optax.OptState

  def apply_gradients(self, *, grads, **kwargs):
    updates, new_optim_state = self.optim.update(
        grads, self.optim_state, self.params
    )
    new_params = optax.apply_updates(self.params, updates)

    return self.replace(
        step=self.step + 1,
        params=new_params,
        optim_state=new_optim_state,
        **kwargs,
    )

  @classmethod
  def create(
      cls,
      *,
      apply_fn,
      params,
      optim,
      **kwargs,
  ):
    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        optim=optim,
        optim_state=optim.init(params),
    )


class FittedValueTrainState(TrainState):  # pytype: disable=invalid-function-definition  # dataclass_transform
  """Train State for fitted value iteration methods."""

  target_params: core.FrozenDict[str, Any]
  target_params_update_fn: TargetParamsUpdateFunction = struct.field(
      pytree_node=False
  )

  def apply_gradients(
      self, *, grads, **kwargs
  ):
    updates, new_optim_state = self.optim.update(
        grads, self.optim_state, self.params
    )
    new_params = optax.apply_updates(self.params, updates)
    new_target_params = self.target_params_update_fn(
        self.params, self.target_params, self.step
    )

    return self.replace(
        step=self.step + 1,
        params=new_params,
        target_params=new_target_params,
        optim_state=new_optim_state,
        **kwargs,
    )

  @classmethod
  def create(
      cls,
      *,
      apply_fn,
      params,
      target_params_update_fn,
      optim,
  ):
    target_params = operator.getitem(jax.jit(identity)(params, 2), 0)

    return cls(
        step=0,
        apply_fn=apply_fn,
        params=params,
        optim=optim,
        optim_state=optim.init(params),
        target_params=target_params,
        target_params_update_fn=target_params_update_fn,
    )


def create_indicator_state(
    element_spec,
    *,
    model,
    optim,
    rng,
):
  """Create a train state for the indicator network."""
  # Initialize parameters
  dummy_observation = jnp.zeros(
      element_spec['observation'].shape[:-1],
      dtype=element_spec['observation'].dtype,
  )
  params = model.init(rng, dummy_observation)
  # We have to unfreeze the parameters or else optax has some issues
  # with optax.masked
  params = params.unfreeze()

  return TrainState.create(apply_fn=model.apply, params=params, optim=optim)


def create_train_state(
    element_spec,
    *,
    model,
    optim,
    target_params_update_fn,
    rng,
):
  """Initializes the TrainState."""

  # Initialize parameters
  dummy_observation = jnp.zeros(
      element_spec['observation'].shape[:-1],
      dtype=element_spec['observation'].dtype,
  )
  params = model.init(rng, dummy_observation)

  return FittedValueTrainState.create(
      apply_fn=model.apply,
      params=params,
      target_params_update_fn=target_params_update_fn,
      optim=optim,
  )


def create_train_state_partition_spec_from_shape(
    state_shape
):
  # Map leading axis
  return tree_utils.tree_map_with_regex(
      mesh_utils.map_leading_axis_to_pspec,
      state_shape,
      [(r'.*params/aux_tasks/.*', 'model')],
      lambda _: None,
  )


def create_train_state_partition_spec(
    element_spec,
    *,
    model,
    optim,
    target_params_update_fn,
):
  """Create train state partition spec without performing any operations."""
  # Create partition specs
  # Start with evaluating the shape of the states
  train_state_shape = jax.eval_shape(
      functools.partial(
          create_train_state,
          model=model,
          optim=optim,
          target_params_update_fn=target_params_update_fn,
          rng=jax.random.PRNGKey(0),
      ),
      element_spec,
  )

  return create_train_state_partition_spec_from_shape(train_state_shape)


def create_indicator_state_partition_spec_from_shape(
    state_shape
):
  return tree_utils.tree_map_with_regex(
      mesh_utils.map_leading_axis_to_pspec,
      state_shape,
      [(r'.*params/reward_bias.*', 'model')],
      lambda _: None,
  )


def create_indicator_state_partition_spec(
    element_spec,
    *,
    model,
    optim,
):
  """Create indicator state partition spec without performing any operations."""
  indicator_state_shape = jax.eval_shape(
      functools.partial(
          create_indicator_state,
          model=model,
          optim=optim,
          rng=jax.random.PRNGKey(0),
      ),
      element_spec,
  )

  return create_indicator_state_partition_spec_from_shape(indicator_state_shape)


def maybe_restore_train_and_indicator_state(
    train_state,
    indicator_state,
    *,
    ckpt_manager,
    mesh,
):
  """Maybe restores the train and indicator state given a checkpoint manager."""
  latest_step = ckpt_manager.latest_step()
  if latest_step is None:
    return train_state, indicator_state
  logging.info('Restoring from step %d', latest_step)

  # Check if the directory is empty, Orbax could have failed to save
  # the checkpoint alltogether
  save_dir = checkpoint.utils.get_save_directory(
      latest_step,
      ckpt_manager.directory,
  )
  # If there's no files in the directory we should remove it and try
  # again with the checkpoint before that.
  if not any(save_dir.iterdir()):
    logging.info(
        'Save directory %s is empty, removing and recursing restore',
        save_dir,
    )
    save_dir.rmdir()
    return maybe_restore_train_and_indicator_state(
        train_state,
        indicator_state,
        ckpt_manager=ckpt_manager,
        mesh=mesh,
    )

  def restore_arguments_with_mesh_axes(
      mesh_axes,
  ):
    if not mesh:
      mesh_axes = None

    def closure(_):
      return checkpoint.ArrayRestoreArgs(
          restore_type=jax.Array,
          mesh=mesh,
          mesh_axes=mesh_axes,
      )

    return closure

  # Evaluate the shape and filter empty nodes
  # We save the entire PyTree so there's no need to further filter
  train_state_shape = jax.eval_shape(lambda x: x, train_state)
  train_state_shape = tree_utils.filter_empty_nodes(
      train_state_shape, train_state_shape
  )
  train_state_pspec = create_train_state_partition_spec_from_shape(
      train_state_shape
  )
  train_state_restore_args = jax.tree_util.tree_map(
      restore_arguments_with_mesh_axes(train_state_pspec), train_state_shape
  )

  indicator_state_shape = jax.eval_shape(lambda x: x, indicator_state)
  indicator_state_shape = tree_utils.tree_map_with_regex(
      lambda _: None,
      indicator_state_shape,
      [(r'.*params/encoder/.*',)],
      lambda leaf: leaf,
  )
  indicator_state_shape = tree_utils.filter_empty_nodes(
      indicator_state_shape, indicator_state_shape
  )
  indicator_state_pspec = create_indicator_state_partition_spec_from_shape(
      indicator_state_shape
  )
  indicator_state_restore_args = jax.tree_util.tree_map(
      restore_arguments_with_mesh_axes(indicator_state_pspec),
      indicator_state_shape,
  )

  restored_state = ckpt_manager.restore(
      latest_step,
      items={'train': train_state_shape, 'indicator': indicator_state_shape},
      restore_kwargs={
          'train': {'restore_args': train_state_restore_args},
          'indicator': {'restore_args': indicator_state_restore_args},
      },
  )

  restored_state = checkpoint.apply_transformations(
      original_tree=restored_state,
      transformations=dict(),
      new_tree={'train': train_state, 'indicator': indicator_state},
      default_to_original=False,
  )
  logging.info('Restore finished')

  return operator.itemgetter('train', 'indicator')(restored_state)


def save_train_and_indicator_state(
    step,
    train_state,
    indicator_state,
    *,
    ckpt_manager,
    wait_until_finished = False,
):
  """Save train and indicator state using ckpt_manager."""

  # We just need to filter out empty nodes for the train state.
  transformed_train_state = tree_utils.filter_empty_nodes(
      train_state, train_state
  )  # pytype: disable=wrong-arg-types

  # The indicator state requires a little more care.
  # We must filter out the random network parameters.
  # We already masked the optimizer state and this will be handled
  # by tree_utils.filter_empty_nodes.
  transformed_indicator_state = tree_utils.tree_map_with_regex(
      lambda _: None,
      indicator_state,
      [(r'.*params/encoder/.*',)],
      lambda leaf: leaf,
  )
  transformed_indicator_state = tree_utils.filter_empty_nodes(
      transformed_indicator_state, transformed_indicator_state
  )  # pytype: disable=wrong-arg-types

  items = {
      'train': transformed_train_state,
      'indicator': transformed_indicator_state,
  }
  ckpt_manager.save(step, items=items)
  if wait_until_finished:
    ckpt_manager.wait_until_finished()
