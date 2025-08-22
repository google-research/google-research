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

"""Utilities for dynamic programming in Jax."""

import abc
import collections
import dataclasses
import functools
import itertools
from typing import Any, Callable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

NDArray = Any
PyTree = Any

SLOW_PYTHON_LOOP = False


class DynamicProgrammingSchedule(abc.ABC):
  """Schedule for computing outputs of a dynamic program."""

  @property
  @abc.abstractmethod
  def num_steps(self):
    """Number of steps to run."""
    Ellipsis

  @abc.abstractmethod
  def indices(self, t):
    """Return a tuple of indices to update in parallel for the current step.

    Args:
      t: Timestep

    Returns:
      Tuple of ndarrays, such that taking the nth value from each ndarray
      produces an index into the table of interest. May include values "past
      the end" and thus out of bounds to indicate updates to ignore.
    """
    Ellipsis


@dataclasses.dataclass
class SerialSchedule(DynamicProgrammingSchedule):
  """Schedule that simply iterates through the array in order."""
  shape: Tuple[int, Ellipsis]

  @property
  def num_steps(self):
    return np.prod(self.shape)

  def indices(self, t):
    idxs = jnp.unravel_index(t, self.shape)
    return tuple(jnp.array([i]) for i in idxs)


@dataclasses.dataclass
class LookupSchedule(DynamicProgrammingSchedule):
  """Schedule based on a precomputed lookup table."""
  indices_table: Tuple[NDArray, Ellipsis]

  @property
  def num_steps(self):
    return self.indices_table[0].shape[0]

  def indices(self, t):
    return tuple(ixs[t, :] for ixs in self.indices_table)


def packed_block_schedule(
    shape, block_size,
    dependency_fn
):
  """Constructs a packed schedule that computes elements in parallel.

  Arguments:
    shape: The shape of the table.
    block_size: Number of elements to try to compute in parallel. Every loop
      iteration will run the kernel function this many times in parallel, but
      some of those values may be ignored if fewer than `block_size` entries in
      the table are ready to evaluate.
    dependency_fn: Function from position indices to lists of the other
      positions that must be computed first. This is used to sequence
      computations that depend on each other. Should have the same access
      pattern as the eventual kernel function, but instead of actually doing
      computation, just return the list of indices read.

  Returns:
    Lookup table schedule that executes `block_size` indices in parallel,
    sequencing accesses according to `dependency_fn`.
  """
  # Algorithm: Iterate backward through time, scheduling elements that
  # no one depends on as close to the end as possible. Once each element is
  # scheduled, it then means we have accounted for the dependencies of that
  # element, so we can schedule the predecessors as well. This algorithm should
  # be optimal because, given any optimal schedule, we are free to reorder it
  # so that events happen as late as possible.

  # Count how many other elements depend on each element.
  consumer_countdowns = collections.Counter()

  for position in itertools.product(*(range(sz) for sz in shape)):
    deps = dependency_fn(position)
    for dep in deps:
      consumer_countdowns[dep] += 1

  # Identify elements that nobody else depends on; these will be packed first
  # into the end of the schedule and will execute last.
  request_fifo = collections.deque()
  for position in itertools.product(*(range(sz) for sz in shape)):
    if position not in consumer_countdowns:
      request_fifo.append(position)

  blocks = []
  while request_fifo:
    # Initialize with a one-past-the-end index, which JAX will treat as padding
    # and ignore updates for.
    block = tuple(
        np.full(shape=(block_size,), fill_value=sz + 1) for sz in shape)

    # Copy up to block_size elements into a single execution block.
    positions = []
    for i in range(min(block_size, len(request_fifo))):
      position = request_fifo.popleft()
      positions.append(position)
      for b, p in zip(block, position):
        b[i] = p

    # Decrease countdown for all dependencies, checking if it's now safe to
    # schedule them as well.
    for position in positions:
      for previous in dependency_fn(position):
        consumer_countdowns[previous] -= 1
        if consumer_countdowns[previous] == 0:
          del consumer_countdowns[previous]
          request_fifo.append(previous)

    blocks.append(block)

  if consumer_countdowns:
    raise ValueError(
        "Couldn't fill due to unsatisfied dependencies and nothing ready")

  # We packed backwards in time, so flip them to obtain a proper schedule.
  blocks.reverse()
  indices = tuple(
      jnp.array(np.stack([blocks[i][j]
                          for i in range(len(blocks))]))
      for j in range(len(shape)))

  return LookupSchedule(indices)


def _read_lookbacks(tables, lookbacks):

  def go(table, lookback_dict):
    return {k: table[indices] for k, indices in lookback_dict.items()}

  return jax.tree.map(go, tables, lookbacks)


def _add_lookbacks(tables, lookbacks, updates):

  def go(table, lookback_dict, updates):
    for k, indices in lookback_dict.items():
      table = table.at[indices].add(updates[k])
    return table

  return jax.tree.map(go, tables, lookbacks, updates)


def dynamic_program(destination,
                    lookback_fn,
                    kernel_fn,
                    schedule,
                    rewrite_vjp = True):
  """Helper to run a dynamic programming problem.

  Args:
    destination: Empty array(s) to write results into.
    lookback_fn: Function (current_index) -> (previous_access_indices) that
      returns a PyTree structured like destination, but where each leaf of
      destination instead contains a dictionary of tuples of indexers. For
      instance, if we want to access destination["foo"][i, j] we would return
      {"foo": {"ij":(i, j)}} here.
    kernel_fn: Function (accesses, index) that determines value for the given
      index. `accesses` will be a pytree that holds the values of the table at
      positions determined by lookback_fn.
    schedule: Schedule of indices to visit.
    rewrite_vjp: Whether to rewrite the VJP of this operation so that it stores
      no extra memory, instead rematerializing in reverse. This is highly
      experimental and may not work in all cases.

  Returns:
    Populated version of destination.
  """
  if not rewrite_vjp:
    return _dynamic_program(destination, lookback_fn, kernel_fn, schedule)
  else:
    # If rewriting our VJP, we need to closure-convert our kernel_fn, so that
    # we can take gradients with respect to the ndarrays accessed while running
    # kernel_fn. Those args will be captured into `kernel_args` and threaded
    # through custom gradient logic.
    example_indices = jax.tree.map(lambda b: b[0], schedule.indices(0))
    # TODO(ddjohnson): this may not allow gradients when under batch tracers.
    lookback_closure, lookback_args = jax.closure_convert(
        lookback_fn, example_indices)

    example_lookback_ixs = jax.eval_shape(lookback_fn, example_indices)
    example_lookback_ixs = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype),
                                        example_lookback_ixs)
    example_lookback_reads = _read_lookbacks(destination, example_lookback_ixs)

    kernel_closure, kernel_args = jax.closure_convert(kernel_fn,
                                                      example_lookback_reads,
                                                      example_indices)
    return _dynamic_program_custom_vjp(schedule, lookback_closure,
                                       kernel_closure, destination,
                                       lookback_args, kernel_args)


def _dynamic_program(destination, lookback_fn, kernel_fn, schedule):
  """Internal runner for dynamic programs."""

  def step(t, table):
    # Extract indices from the schedule.
    indices = schedule.indices(t)

    # Run our kernel.
    def go(indices_slice):
      lookback_ixs = lookback_fn(indices_slice)
      lookback_values = _read_lookbacks(table, lookback_ixs)
      return kernel_fn(lookback_values, indices_slice)

    outs = jax.vmap(go)(indices)
    # Scatter into our table in parallel.
    return jax.tree.map(lambda st, sv: st.at[indices].set(sv), table, outs)

  return jax.lax.fori_loop(0, schedule.num_steps, step, destination)


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _dynamic_program_custom_vjp(schedule, lookback_closure, kernel_closure,
                                destination, lookback_args, kernel_args):
  """Custom VJP wrapper for dynamic program."""
  # If we are executing a dynamic program without autodiff, just undo the
  # closure conversion.
  lookback_fn = lambda i: lookback_closure(i, *lookback_args)
  kernel_fn = lambda t, i: kernel_closure(t, i, *kernel_args)
  return _dynamic_program(destination, lookback_fn, kernel_fn, schedule)


def _dynamic_program_fwd(schedule, lookback_closure, kernel_closure,
                         destination, lookback_args, kernel_args):
  """Forward pass for dynamic program."""
  # In forward mode, save the table for the backward pass, along with the
  # values of the closed-over ndarrays we might want derivatives for.
  table = _dynamic_program_custom_vjp(schedule, lookback_closure,
                                      kernel_closure, destination,
                                      lookback_args, kernel_args)
  return table, (table, lookback_args, kernel_args)


def zero_cotangents_for_primal(primal):
  """Compute a zero cotangent for a given primal."""
  # The zero of the cotangent vector space is given by the the VJP of a
  # function from the primal to a trivial (zero-dimensional) vector space.
  # Unlike zeros_like, this handles differing primal/tangent types correctly.
  return jax.vjp(lambda _: None, primal)[1](None)[0]


def _dynamic_program_bwd(schedule, lookback_closure, kernel_closure, saved,
                         table_bar):
  """Backward pass for dynamic program."""
  # In reverse mode, loop through our schedule in REVERSE, writing cotangents
  # into a cotangents table. Since we know that the table was only sparsely
  # read and updated, we can just use the final table, leading to constant
  # memory autodiff.
  table, lookback_args, kernel_args = saved

  def step(s, state):
    table_bar, kernel_args_bar = state
    t = schedule.num_steps - s - 1
    indices = schedule.indices(t)

    # inner_fn is just the original logic from the dynamic program update, but
    # threading through the table and aux args explicitly.
    def inner_fn(lookback_values, kernel_args):
      outs = jax.vmap(lambda vs, i: kernel_closure(vs, i, *kernel_args))(
          lookback_values, indices
      )
      new_table = jax.tree.map(
          lambda st, sv: st.at[indices].set(sv), table, outs
      )
      return new_table, kernel_args

    # Use VJP to compute the updates for our cotangents. But, since we know
    # each entry of the table was never read until after it was written, we can
    # just pass in the final table, which will still produce the same results
    # that the partially-computed table did in the forward pass.
    lookback_ixs = jax.vmap(lambda i: lookback_closure(i, *lookback_args))(
        indices)
    lookback_values = _read_lookbacks(table, lookback_ixs)

    _, pullback = jax.vjp(inner_fn, lookback_values, kernel_args)
    lookback_bar, kernel_args_bar = pullback((table_bar, kernel_args_bar))
    table_bar = _add_lookbacks(table_bar, lookback_ixs, lookback_bar)
    return table_bar, kernel_args_bar

  # Initialize our cotangents to zero.
  init_kernel_args_bar = zero_cotangents_for_primal(kernel_args)
  lookback_args_bar = zero_cotangents_for_primal(lookback_args)

  # Propagate our cotangents backward.
  destination_bar, kernel_args_bar = jax.lax.fori_loop(
      0, schedule.num_steps, step, (table_bar, init_kernel_args_bar))

  # destination_bar should be zero here, but return it anyway just in case we
  # need the flexibility for some reason.
  return destination_bar, lookback_args_bar, kernel_args_bar


_dynamic_program_custom_vjp.defvjp(_dynamic_program_fwd, _dynamic_program_bwd)
