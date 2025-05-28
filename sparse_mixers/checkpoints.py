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

"""Functions for checkpointing models with sharded parameters.

The functions in this library wrap Flax's checkpoint library to handle saving
and restoring models with a mixture of replicated and sharded parameters.
"""

from typing import Any, Callable, Optional, Tuple

from absl import logging
from flax.training import checkpoints
import jax
import numpy as np

from sparse_mixers import core_utils

flax_checkpoint_path = checkpoints._checkpoint_path  # pylint: disable=protected-access

# Type Stubs
PyTree = Any


def save_checkpoint(ckpt_dir,
                    target,
                    sharded_match_fn,
                    step,
                    keep = 1,
                    process_id = None,
                    process_count = None):
  """Saves a checkpoint of the model.

  Important: Like Flax's save_checkpoint(), this function doesn't handle
  "unreplication" of the replicated parameters. Use
  core_utils.tree_unreplicate_by_name() before calling this function.

  Args:
    ckpt_dir: Directory to store checkpoint files in.
    target: Serializable Flax object, usually a Flax optimizer.
    sharded_match_fn: Function that returns true if a given parameter name
      corresponds to that of a sharded parameter. If no sharded match function
      is given, all parameters are treated as replicated parameters.
    step: Training step number.
    keep: Number of past checkpoint files to keep.
    process_id: Identifier for process saving the checkpoint. If None (default),
      uses jax.process_index().
    process_count: Total number of processes in the system. If None (default),
      uses jax.process_count().

  Returns:
    - replicated_filepath: Filepath containing replicated parameters.
    - sharded_filepath: Filepath containing the sharded parameters corresponding
      to the current process.
  """
  process_id = process_id or jax.process_index()
  process_count = process_count or jax.process_count()

  if sharded_match_fn is None:
    # Treat all parameters as replicated; assume that no parameters are sharded.
    sharded_match_fn = lambda name: False
    sharded_filepath = ""
  else:
    logging.info("Saving sharded checkpoint from process: %d", process_id)

    # Function to match parameters that are replicated, not sharded.
    not_sharded_match_fn = lambda name: not sharded_match_fn(name)

    # For all processes, save any sharded parameters they have.
    sharded_filepath = checkpoints.save_checkpoint(
        ckpt_dir,
        target=core_utils.tree_map_with_names(lambda _: np.array([]), target,
                                              not_sharded_match_fn),
        step=step,
        prefix=_sharded_checkpoint_pattern(process_id, process_count),
        keep=keep)

  if process_id == 0:
    # We only save the replicated parameters from one process.
    replicated_filepath = checkpoints.save_checkpoint(
        ckpt_dir,
        target=core_utils.tree_map_with_names(lambda _: np.array([]), target,
                                              sharded_match_fn),
        step=step,
        prefix=_replicated_checkpoint_pattern(),
        keep=keep)
  else:
    # This is the filepath of the replicated parameters.
    replicated_filepath = flax_checkpoint_path(
        ckpt_dir, step=step, prefix=_replicated_checkpoint_pattern())

  return replicated_filepath, sharded_filepath


def restore_checkpoint(ckpt_dir,
                       target,
                       sharded_match_fn,
                       step = None,
                       process_id = None,
                       process_count = None):
  """Restores the last checkpoint from checkpoints in path, or a specific one.

  Sorts the checkpoint files naturally, returning the highest-valued file, e.g.:
    ckpt_1, ckpt_2, ckpt_3 --> ckpt_3.

  Important: Like Flax's restore_checkpoint(), this function doesn't handle
  replication or sharding of parameters. Once a checkpoint is loaded, call:
  target = core_utils.tree_replicate_by_name(target, not_sharded_match_fn)
  target = core_utils.tree_shard_by_name(target, sharded_match_fn)
  to replicated and sharded the relevant parameters.

  Args:
    ckpt_dir: Directory of checkpoints to restore from.
    target: Serializable Flax object, usually a Flax optimizer.
    sharded_match_fn: Function that returns true if a given parameter name
      corresponds to that of a sharded parameter. If no sharded match function
      is given, we only attempt to load replicated parameters from the ckpt_dir.
    step: Training step number. If None, restores the last one.
    process_id: Identifier for process saving the checkpoint. If None (default),
      uses jax.process_index().
    process_count: Total number of processes in the system. If None (default),
      uses jax.process_count().

  Returns:
    Restored target updated from checkpoint file. If no step is given and no
    checkpoints can be found, returns None.
  """
  process_id = process_id or jax.process_index()
  process_count = process_count or jax.process_count()

  # Restore parameters to replicate across all devices.
  target_to_replicate = checkpoints.restore_checkpoint(
      ckpt_dir=ckpt_dir,
      target=target,
      step=step,
      prefix=_replicated_checkpoint_pattern())
  if target_to_replicate is target:
    logging.info("No replicate checkpoint found: returning None.")
    return None

  if sharded_match_fn is None:
    # Treat all parameters as replicated; don't attempt to restore any sharded
    # parameters.
    return target_to_replicate

  target_to_shard = checkpoints.restore_checkpoint(
      ckpt_dir=ckpt_dir,
      target=target,
      step=step,
      prefix=_sharded_checkpoint_pattern(process_id, process_count))
  if target_to_shard is target:
    logging.info("No sharded checkpoint found: returning None.")
    return None

  if target is None:
    target = target_to_replicate
  treedef = jax.tree.structure(target)
  names = [name for name, _ in core_utils.tree_flatten_with_names(target)[0]]
  values_to_replicate = jax.tree.leaves(target_to_replicate)
  values_to_shard = jax.tree.leaves(target_to_shard)
  target = jax.tree.unflatten(treedef, [
      vs if sharded_match_fn(name) else vr
      for name, vr, vs in zip(names, values_to_replicate, values_to_shard)
  ])

  target = jax.tree.map(_recover_bfloat16_dtype, target)

  return target


def _sharded_checkpoint_pattern(process_index, process_count):
  """Returns the sharded checkpoint prefix."""
  return f"shard-{process_index:05d}-of-{process_count:05d}_checkpoint_"


def _replicated_checkpoint_pattern():
  """Returns the replicated checkpoint prefix."""
  return "replicated_checkpoint_"


def _recover_bfloat16_dtype(a):
  """Restores bfloat16 dtype, from np.void (Numpy does not have bfloat16)."""
  if hasattr(a, "dtype") and a.dtype.type is np.void:
    assert a.itemsize == 2, f"Unknown dtype with itemsize = {a.itemsize}!"
    return a.view(jax.numpy.bfloat16)
  else:
    return a
