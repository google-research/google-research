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

"""An API for embedding a MapReduce programming model into JAX.

This is done by patching a global module during decorated-function invocation to
give the appearance a single, uniform API. This could certainly be changed
by altering the method of dispatch--e.g., by letting the data we carry through
function invocations drive which concrete function is called.
"""

from collections.abc import Callable, Mapping
import functools
from typing import Any

from absl import logging
import jax

from . import impls
from . import primitives

# We define type aliases to make specifying the API below easier.
_NestedJTensor = Any
_NestedUnplacedTensor = Any
_NestedPlacedTensor = Any
_NestedPSpec = Any


# We don't want to make the degenerate case of directly calling these symbols
# outside of a DrJAX decorator part of the API.
# pylint: disable=g-doc-exception
_MAPREDUCE_PRIMITIVES = {}


class OperatorUndefinedError(Exception):

  def __init__(self, name):
    self.message = (
        f'The DrJAX operator {name} is only defined in the context of a DrJAX '
        'program decorator. Attempted to call without using this decorator.'
    )
    super().__init__(self.message)


# The following functions which 'do nothing but raise' can be considered
# something like a specification of DrJAX's API.


# The 'transport' symbols (broadcast, sum, mean, etc) can be considered the
# core of DrJAX: everything else is 'just helpers'. DrJAX's main job is t
# capture the transport of arrays at a high level, and provide implementations
# which manage the triplet of 'MapReduce semantics / logical representation in
# program / physical layout'.
def broadcast(
    x,
):
  """Broadcasts its input to the active placement.

  The broadcast tensor will be partitioned along the placement dimension of
  the mesh. Other dimensions will be unconstrained, and are expected to be laid
  out well by the XLA compiler (e.g., preserving the sharding of input tensors).

  Args:
    x: A structure of arrays to be broadcast.

  Returns:
    A structure of arrays broadcast to the placement dimension (i.e., each array
    has a placement-dimension inserted on the left, of size configured in the
    DrJAX decorator), laid out as described above.
  """
  raise OperatorUndefinedError('broadcast')


def reduce_mean(x):
  """Computes an unweighted mean across placed values.

  At runtime in the datacenter, this function simply runs as a `jnp.mean`
  across the placement dimension.

  Args:
    x: The (potentially nested) placed tensor to be reduce_mean'ed.

  Returns:
    The result of an unweighted mean across the placement dimension.
  """
  raise OperatorUndefinedError('reduce_mean')


def reduce_sum(x):
  """Computes a sum across placed values.

  At runtime in the datacenter, this function simply runs as a `jnp.sum`
  across the placement dimension.

  Args:
    x: The (potentially nested) placed tensor to be reduce_sum'ed.

  Returns:
    The result of sum across the placed dimension.
  """
  raise OperatorUndefinedError('reduce_sum')


def reduce_weighted_mean(
    x, w
):
  """Computes a weighted mean across placed values.

  At runtime in the datacenter, this function delegates internally to two
  `reduce_sum`s, to compute numerator and denominator.

  Args:
    x: The (potentially nested) placed tensor to be weighted-mean'ed.
    w: The weight to use for this average.

  Returns:
    The result of the weighted mean across the placed dimension.
  """
  raise OperatorUndefinedError('reduce_weighted_mean')


# The mapping functions here are provided simply for convenience. Users can
# write *any* processing across their arrays; but we provide our
# mapping functions to ease the burden of writing sharded + distributed jax.
# In particular, `map_fn` is intended as a location to centralize and codify
# knowledge on writing code which enables weak scaling with respect to the size
# of the placement axis, a more subtle question than it might seem.


def map_fn(
    fn, arg
):
  """Maps `fn` across the placed dimension of `arg`.

  This function will direct the GSPMD compiler that the mapped axis
  corresponds to the placement dimension of the mesh installed at tracing time,
  if such an axis exists.

  Args:
    fn: A callable accepting slices along the placed dimension of `arg`.
    arg: A placed structure of tensors to be mapped. If `arg` is a tuple, this
      tuple will be unpacked upon invoking `fn`.

  Returns:
    The result of mapping the function `fn` as described above.
  """
  raise OperatorUndefinedError('map_fn')


def _implement_api(api_fn, impl):
  """Wrap an abstract API symbol with a concrete implementation."""

  @functools.wraps(api_fn)
  def wrapper(*args, **kwargs):
    try:
      # This function will raise an exception. However, by calling it we can
      # propagate things like type errors to the user.
      api_fn(*args, **kwargs)
    except OperatorUndefinedError:
      return impl(*args, **kwargs)

  return wrapper


# pylint: disable=g-long-lambda
def _replace_api(
    api, placed_computations, prim_computations, *, placement
):
  # importlib.util.module_from_spec()
  """A binding of MapReduce primitive implementations in DrJAX."""
  map_fn_impl = lambda fn, arg: placed_computations.map_to_placement(
      fn, arg, placement
  )
  api.map_fn = _implement_api(map_fn, map_fn_impl)

  reduce_mean_impl = lambda x: jax.tree_util.tree_map(
      prim_computations[f'mean_from_{placement}'], x
  )
  api.reduce_mean = _implement_api(reduce_mean, reduce_mean_impl)

  reduce_sum_impl = lambda x: jax.tree_util.tree_map(
      prim_computations[f'sum_from_{placement}'], x
  )
  api.reduce_sum = _implement_api(reduce_sum, reduce_sum_impl)

  def reduce_weighted_mean_impl(x, w):
    mult_at_placement = api.map_fn(
        lambda arg1, arg2: jax.tree_util.tree_map(
            lambda x, y: x * y, arg1, arg2
        ),
        (x, w),
    )
    sum_mult = api.reduce_sum(mult_at_placement)
    denom_sum = api.reduce_sum(w)
    return jax.tree_util.tree_map(lambda x, y: x / y, sum_mult, denom_sum)

  api.reduce_weighted_mean = _implement_api(
      reduce_weighted_mean, reduce_weighted_mean_impl
  )

  def broadcast_impl(x, *, input_pspecs=None):
    del input_pspecs  # Unused

    return jax.tree_util.tree_map(
        prim_computations[f'broadcast_{placement}'], x
    )

  api.broadcast = _implement_api(broadcast, broadcast_impl)
  return api


# pylint: enable=g-long-lambda


def drjax_program(
    *,
    placements,
    self_module,
    use_spmd_axis_name = True,
):
  """Patches symbols into current module and call `jax.jit` on the result.

  This decorator enables calling:

  * map_fn
  * reduce_mean
  * reduce_sum
  * reduce_weighted_mean
  * broadcast

  The functions returned by this decorator are compatible with a broad set of
  jax transformations--ideally, all of them. In particular, they can be
  jit-compiled, lowered wholesale to HLO, batched, and differentiated through
  forwards and backwards.

  Args:
    placements: Dictionary defining placement cardinality. Must contain a single
      key-value pair, string to int, which determines the name of the mesh
      across which the DrJAX map-reduce operations will be sharded along with
      the (logical) cardinality of arrays 'placed' at this dimension (e.g., the
      size of the placed dimension after a `drjax.broadcast`). If this placement
      name is not present in the runtime mesh, DrJAX will not inject sharding
      constraints. Any functions which are mapped in the body of the
      `drjax_program` must be agnostic to the placement name (e.g., using
      collectives referencing this name results in undefined behavior).
    self_module: The Python module to patch the API when performing DrJAX
      tracing.
    use_spmd_axis_name: Boolean controlling whether DrJAX will inject
      `spmd_axis_name` corresponding to `placements` to JAX's `vmap` during its
      `map` operations. Semantically this injection should be safe.

  Returns:
    A decorated function enabling the calling of the DrJAX API. Interoperable
    with other JAX code.
  """
  try:
    [placement] = placements.keys()
  except ValueError as exc:
    raise ValueError(
        f'Expected a single-element dict for placements. Got {placements=}'
    ) from exc

  placed_computations = impls.PlacedComputations(
      placements_to_n_elements=placements,
      use_spmd_axis_name=use_spmd_axis_name,
  )
  prim_computations, primdefs = primitives.register_primitives(
      placements=placements
  )
  global _MAPREDUCE_PRIMITIVES
  if not _MAPREDUCE_PRIMITIVES:
    # First time these primitives are defined; attach them to the global.
    _MAPREDUCE_PRIMITIVES = primdefs

  def fn_decorator(fn):
    @functools.wraps(fn)
    def jax_callable(*args, **kwargs):
      old_api_symbols = {
          attribute_name: getattr(self_module, attribute_name)
          for attribute_name in dir(self_module)
      }
      # We need to patch down here so that the values that get traced at
      # 'runtime' are the expected ones.
      try:
        _replace_api(
            self_module,
            placed_computations,
            prim_computations,
            placement=placement,
        )
        logging.info('lib patched!')
        return fn(*args, **kwargs)
      finally:
        for symbol_name, symbol in old_api_symbols.items():
          setattr(self_module, symbol_name, symbol)
        logging.info('lib patch reverted!')

    return jax_callable

  return fn_decorator
