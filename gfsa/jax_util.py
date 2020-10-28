# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python3
"""Miscellaneous JAX helper functions."""

import functools
from typing import Any, Callable, Type, TypeVar, Union

import dataclasses
import flax
import jax
import jax.numpy as jnp
import numpy as np

jax.config.enable_omnistaging()

# Type alias for functions that handle NDArrays
NDArray = Union[np.ndarray, jnp.DeviceArray]

T = TypeVar("T")


@dataclasses.dataclass
class LeafPlaceholder:
  """Represents a dataclass tree leaf of a particular type.

  The main purpose for a LeafPlaceholder object is to be a jax pytree leaf
  that we can replace with some other concrete value of the appropriate type.

  Attributes:
    ty: The type annotation for the leaf.
  """
  ty: Union[Type[Any], str]

  # Support pickling, since types can't be pickled directly.
  def __getstate__(self):
    if isinstance(self.ty, str):
      return self.ty
    else:
      return repr(self.ty)

  def __setstate__(self, state):
    self.ty = state


def synthesize_dataclass(ty):
  """Synthesize an instance of a dataclass.

  Any fields of the dataclass that are also dataclasses will be recursively
  synthesized as well. Types with a "default constructor" ty() will be
  instantiated with that type, and other types (those for which ty() is a
  TypeError, such as typing.Any or jax_util.NDArray) will be instantiated with a
  LeafPlaceholder.

  Args:
    ty: Type to synthesize, usually a dataclass.

  Returns:
    Instance of the type, or a leaf placeholder.
  """
  if dataclasses.is_dataclass(ty):
    return ty(
        **{
            field.name: synthesize_dataclass(field.type)
            for field in dataclasses.fields(ty)
        })
  else:
    try:
      return ty()
    except TypeError:
      return LeafPlaceholder(ty)


def vmap_with_kwargs(fun,
                     positional_axes=0,
                     out_axes=0,
                     **kwargs_axes):
  """Wrapper around jax.vmap that supports specifying axes for kwargs.

  For instance, if we have

    def foo(w, x, y, z):
      ...

  then instead of writing `jax.vmap(foo, in_axes=(0, 1, (2, 3), None))` and
  calling with only positional arguments, you can instead write

    vmap_with_kwargs(foo, positional_axes=(0,), x_axis=1, y_axes=(2, 3))

  and call it with x, y, and z as keyword arguments.

  Args:
    fun: Function to vmap.
    positional_axes: Input axes for positional arguments; like `in_axes` for
      jax.vmap. If not provided, all positional arguments will be vmapped across
      their first dimension.
    out_axes: Output axes; see jax.vmap.
    **kwargs_axes: Input axes for keyword arguments, which works the same way as
      in_axes does for positional arguments. Each keyword argument should have
      the suffix "_axis" or "_axes", corresponding to the axis to vectorize the
      keyword argument along. Any missing kwargs will be assumed to be
      broadcasted (i.e. it is as if they were given an axis of None).

  Returns:
    Batched/vectorized version of `fun`; see jax.vmap. Note that the positional
    and keyword arguments used to call this batched version must match the
    positional and keyword axis specifications passed in to vmap_with_kwargs.
  """

  known_kw_axes = {}
  for keyword, value in kwargs_axes.items():
    if not (keyword.endswith("_axis") or keyword.endswith("_axes")):
      raise ValueError(
          f"Keyword argument {keyword} does not end in '_axis' or '_axes'")
    known_kw_axes[keyword[:-5]] = value

  @functools.wraps(fun)
  def apply(args, known_kw_axes, extra_kwargs):
    return fun(*args, **known_kw_axes, **extra_kwargs)

  mapped = jax.vmap(
      apply, in_axes=(positional_axes, known_kw_axes, None), out_axes=out_axes)

  @functools.wraps(mapped)
  def wrapper(*args, **kwargs):
    return mapped(args, {k: v for k, v in kwargs.items() if k in known_kw_axes},
                  {k: v for k, v in kwargs.items() if k not in known_kw_axes})

  return wrapper


def np_or_jnp(arr):
  """Return either numpy or jax.numpy based on the type of arr."""
  # See also https://numpy.org/neps/nep-0037-array-module.html
  if isinstance(arr,
                (jnp.DeviceArray, jax.core.UnshapedArray, jax.core.Tracer)):
    return jnp
  else:
    return np


def pad_to(arr, size, axis = 0):
  """Pad one axis of an array to a specific size by adding zeros at the end.

  Args:
    arr: Array to pad.
    size: Requested size of the axis to pad.
    axis: Axis to pad.

  Returns:
    Version of arr padded with zeros along the requested axis.
  """
  pad_widths = [[0, 0] if i != axis else [0, size - arr.shape[axis]]
                for i in range(arr.ndim)]
  return np_or_jnp(arr).pad(arr, pad_widths, mode="constant")


def register_dataclass_pytree(cls):
  """Register a dataclass as a JAX pytree and a flax serializable object.

  This makes it so that wrapped dataclasses can be used as parameters inside
  a JAX/flax model, and handled by core jax/flax functions. We assume that every
  parameter of the dataclass is a JAX datatype that should be mapped over.

  Differences from flax.struct.dataclass:

  - Assumes dataclass wrapper has already been applied. This allows customizing
    the creation of the dataclass before registering the object.
  - Does not support `pytree_node` fields.

  Args:
    cls: Class to register as a pytree.

  Returns:
    The input argument (so that this can be used as a decorator).

  Raises:
    ValueError: If cls is not a dataclass.
  """
  if not dataclasses.is_dataclass(cls):
    raise ValueError(f"{cls} is not a dataclass. Perhaps you need to call "
                     "dataclasses.dataclass first?")

  def to_shallow_dict(instance):
    """Returns a shallow-dict view of instance, with keys in field order."""
    fields = dataclasses.fields(instance)
    return {field.name: getattr(instance, field.name) for field in fields}

  jax.tree_util.register_pytree_node(
      cls,
      lambda instance: (to_shallow_dict(instance).values(), None),
      lambda _, values: cls(*values),
  )

  def to_state_dict(instance):
    """Returns a flax state dict for this instance."""
    # Convert object to a shallow dict, then let flax do the rest.
    return flax.serialization.to_state_dict(to_shallow_dict(instance))

  def from_state_dict(representative, state_dict):
    """Returns an instance of the object with the given state dict."""
    # Tell flax to restore to a shallow dict, then construct the object.
    old_shallow = to_shallow_dict(representative)
    new_shallow = flax.serialization.from_state_dict(old_shallow, state_dict)
    return cls(**new_shallow)

  flax.serialization.register_serialization_state(cls, to_state_dict,
                                                  from_state_dict)

  return cls


@flax.nn.module
def flax_tag(arr):
  """Wraps a value in a flax module, to inspect intermediate values."""
  return arr


def force_physical_layout(operand):
  """Force the physical layout of `operand` to match its logical layout.

  The return value of this function is identical to the argument, but is
  guaranteed to have its physical layout match the order of dimensions in the
  shape. The last dimension will be the minormost dimension (the one that
  changes the fastest, and should be a multiple of 128 on TPU) and the first
  dimension will be the majormost dimension (the one that changes the slowest).

  Note that XLA may still insert copies before or after this operation, so it
  doesn't guarantee that this layout will persist. However, it should serve as
  a hint to encourage XLA to choose a good layout instead of a bad one, and
  can be used to prevent a bad but required choice from propagating to other
  values.

  Args:
    operand: Array to constrain.

  Returns:
    Copy of operand whose physical layout matches its shape.
  """
  return force_physical_layout_p.bind(operand)


def _force_physical_layout_impl(operand):
  """Implementation for force_physical_layout_p."""
  # Flatten the operand.
  flat = jnp.reshape(operand, (-1,))
  # Do something XLA can't simplify, but is actually a no-op.
  # Since the false branch depends on the linearized order of the elements,
  # this means the reshapes must actually happen. On TPU, all reshapes are
  # implemented as bitcasts, which implies that the order of the dimensions is
  # in major-to-minor order (i.e. the physical layout matches the logical one).
  flat = jax.lax.cond(
      jax.lax.rng_uniform(jax.lax.tie_in(operand, 0), 1, ()) < 2, flat,
      lambda f: f, flat, lambda f: f[::-1])
  # Restore the operand.
  return jnp.reshape(flat, operand.shape)


force_physical_layout_p = jax.core.Primitive("force_physical_layout")
force_physical_layout_p.def_impl(_force_physical_layout_impl)
force_physical_layout_p.def_abstract_eval(
    lambda operand, **_: jax.abstract_arrays.raise_to_shaped(operand))
jax.interpreters.xla.translations[
    force_physical_layout_p] = jax.interpreters.xla.lower_fun(
        _force_physical_layout_impl, multiple_results=False)
jax.interpreters.ad.deflinear(force_physical_layout_p,
                              lambda ct: [force_physical_layout(ct)])
jax.interpreters.batching.primitive_batchers[force_physical_layout_p] = (
    lambda args, dims: (force_physical_layout(args[0]), dims[0]))
