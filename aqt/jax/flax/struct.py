# coding=utf-8
# Copyright 2021 The Google Research Authors.
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


# This file is forked from third_party/py/flax/struct.py to optionally make
# `flax.struct.dataclass` mutable. That is useful in our codebase since
# we construct a 'base model configuration' as an HParams dataclass and then
# want to mutate it in the axes handlers for running experiment sweeps.
#
# We default unsafe_hash to True in our fork since the Jax compiler caches
# static parameters based on their hash. The hash of a mutable dataclass is
# by default its object id, and so mutating its fields won't change its
# hash and will cause Jax to incorrectly use a cached version of JITed
# functions that don't reflect new mutations. Setting unsafe_hash causes the
# hash to instead be based on the value of the dataclass, and so Jax will
# correctly recompile when a field has changed.
"""Utilities for defining custom classes that can be used with jax transformations.
"""

import dataclasses
from flax import serialization
import jax


def dataclass(clz=None, *, frozen=False, unsafe_hash=True):
  """Create a class which can be passed to functional transformations.

  Jax transformations such as `jax.jit` and `jax.grad` require objects that are
  immutable and can be mapped over using the `jax.tree_util` methods.
  The `dataclass` decorator makes it easy to define custom classes that can be
  passed safely to Jax. For example::
    from flax import struct
    @struct.dataclass
    class Model():
      params: Any
      # use pytree_node=False to indicate an attribute should not be touched
      # by Jax transformations.
      apply_fn: FunctionType = struct.field(pytree_node=False)
      def __apply__(self, *args):
        return self.apply_fn(*args)
    model = Model(params, apply_fn)
    model.params = params_b  # Model is immutable. This will raise an error.
    model_b = model.replace(params=params_b)  # Use the replace method instead.
    # This class can now be used safely in Jax to for example to compute
    # gradients w.r.t. the parameters.
    model = Model(params, apply_fn)
    model_grad = jax.grad(some_loss_fn)(model)
  Args:
    clz: the class that will be transformed by the decorator.
    frozen: whether to freeze the dataclass (default=False). WARNING: mutations
      that occur to an unfrozen dataclass within a function that has been
      transformed by Jax (including Flax modules) are not visible outside of
      that function.
    unsafe_hash: see help for corresponding argument in dataclasses.dataclass.
  Returns:
    The new class.
  """

  def wrapped(clz):
    data_clz = dataclasses.dataclass(
        frozen=frozen, unsafe_hash=unsafe_hash)(
            clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
      is_pytree_node = field_info.metadata.get('pytree_node', True)
      if is_pytree_node:
        data_fields.append(name)
      else:
        meta_fields.append(name)

    def replace(self, **updates):
      """"Returns a new object replacing the specified fields with new values."""
      return dataclasses.replace(self, **updates)

    data_clz.replace = replace

    def iterate_clz(x):
      meta = tuple(getattr(x, name) for name in meta_fields)
      data = tuple(getattr(x, name) for name in data_fields)
      return data, meta

    def clz_from_iterable(meta, data):
      meta_args = tuple(zip(meta_fields, meta))
      data_args = tuple(zip(data_fields, data))
      kwargs = dict(meta_args + data_args)
      return data_clz(**kwargs)

    jax.tree_util.register_pytree_node(data_clz, iterate_clz, clz_from_iterable)

    def to_state_dict(x):
      state_dict = {
          name: serialization.to_state_dict(getattr(x, name))
          for name in data_fields
      }
      return state_dict

    def from_state_dict(x, state):
      """Restore the state of a data class."""
      state = state.copy()  # copy the state so we can pop the restored fields.
      updates = {}
      for name in data_fields:
        if name not in state:
          raise ValueError(f'Missing field {name} in state dict while restoring'
                           f' an instance of {clz.__name__}')
        value = getattr(x, name)
        value_state = state.pop(name)
        updates[name] = serialization.from_state_dict(value, value_state)
      if state:
        names = ','.join(state.keys())
        raise ValueError(f'Unknown field(s) "{names}" in state dict while'
                         f' restoring an instance of {clz.__name__}')
      return x.replace(**updates)

    serialization.register_serialization_state(data_clz, to_state_dict,
                                               from_state_dict)

    return data_clz

  # If this is called as @dataclass(frozen=...), clz is will be None and
  # and we return a function that will be later called with clz.
  if clz is None:
    return wrapped
  # Otherwise, this was called directly as @dataclass so we apply the
  # wrapper to the clz immediately.
  else:
    return wrapped(clz)


def field(pytree_node=True, **kwargs):
  return dataclasses.field(metadata={'pytree_node': pytree_node}, **kwargs)
