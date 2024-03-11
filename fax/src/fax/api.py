# coding=utf-8
# Copyright 2024 The Google Research Authors.
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

"""An API for embedding a federated programming model into JAX.

This is done by patching a global module during decorated-function invocation to
give the appearance a single, uniform API. This could certainly be changed
by altering the method of dispatch--e.g., by letting the data we carry through
function invocations drive which concrete function is called.
"""

from collections.abc import Callable
import functools
from typing import Any

from absl import logging
import jax

from . import impls
from . import primitives

# We define type aliases to make specifying the API below easier.
_NestedJTensor = Any
_NestedServerTensor = Any
_NestedClientsTensor = Any
_NestedPSpec = Any


# We don't want to make the degenerate case of directly calling these symbols
# outside of a fax decorator part of the API.
# pylint: disable=g-doc-exception
_FEDERATED_PRIMITIVES = {}


class OperatorUndefinedError(Exception):

  def __init__(self, name):
    self.message = (
        f'The FAX operator {name} is only defined in the context of a fax '
        'program decorator. Attempted to call without using this decorator.'
    )
    super().__init__(self.message)


# The following functions which 'do nothing but raise' can be considered
# something like a specification of FAX's API.


# The 'transport' symbols (broadcast, sum, mean, etc) can be considered the
# core of fax: everything else is 'just helpers'. FAX's main job is to capture
# the transport of arrays at a high level, and provide implementations which
# manage the triplet of 'federated semantics / logical representation in
# program / physical layout'.
def federated_broadcast(
    x,
):
  """Broadcasts its input to the clients.

  The broadcast tensor will be partitioned along the 'clients' dimension of
  the
  mesh. Other dimensions will be unconstrained, and are expected to be laid
  out
  well by the XLA compiler (e.g., preserving the sharding of input tensors).

  Args:
    x: A structure of arrays to be broadcast.

  Returns:
    A structure of arrays broadcast to the clients dimension (i.e., each array
    has a clients-dimension inserted on the left, of size configured in the
    FAX
    decorator), laid out as described above.
  """
  raise OperatorUndefinedError('federated_broadcast')


def federated_mean(x):
  """Computes a federated (unweighted) mean across clients.

  At runtime in the datacenter, this function simply runs as a `jnp.mean`
  across
  the client dimension.

  Args:
    x: The (potentially nested) clients-placed tensor to be federated_mean'ed.

  Returns:
    The result of an unweighted mean across the clients dimension.
  """
  raise OperatorUndefinedError('federated_mean')


def federated_sum(x):
  """Computes a federated sum across clients.

  At runtime in the datacenter, this function simply runs as a `jnp.sum`
  across
  the client dimension.

  Args:
    x: The (potentially nested) clients-placed tensor to be federated_sum'ed.

  Returns:
    The result of sum across the clients dimension.
  """
  raise OperatorUndefinedError('federated_sum')


def federated_weighted_mean(
    x, w
):
  """Computes a federated weighted mean across clients.

  At runtime in the datacenter, this function delegates internally to two
  `federated_sum`s, to compute numerator and denominator.

  Args:
    x: The (potentially nested) clients-placed tensor to be weighted-mean'ed.
    w: The weight to use for this average.

  Returns:
    The result of the weighted mean across the clients dimension.
  """
  raise OperatorUndefinedError('federated_weighted_mean')


# The mapping functions here are provided simply for convenience. Users can
# write *any* processing across their federated arrays; but we provide our
# mapping functions to ease the burden of writing sharded + distributed jax.
# In particular, `federated_map_clients` is intended as a location to
# centralize and codify knowledge on writing code which enables weak scaling
# with respect to clients, a more subtle question than it might seem.


def federated_map_clients(
    fn, arg
):
  """Maps `fn` across the clients dimension of `arg`.

  This function will direct the GSPMD compiler that the mapped axis
  corresponds
  to the 'clients' dimension of the mesh installed at tracing time, if such an
  axis exists.

  Args:
    fn: A callable accepting slices along the clients dimension of `arg`.
    arg: A clients-placed structure of tensors to be mapped. If `arg` is a
      tuple, this tuple will be unpacked upon invoking `fn`.

  Returns:
    The result of mapping the function `fn` as described above.
  """
  raise OperatorUndefinedError('federated_map_clients')


def federated_map_server(
    fn, arg
):
  """Calls `fn` on `arg`.

  Since fax represents server-placed arrays as *identical* to unplaced
  arrays, this function simply calls `fn` on `arg`, and users are free to
  replace this function with its body.

  Args:
    fn: The function to apply.
    arg: The (singleton) argument to pass to `fn`. If `arg` is a tuple, this
      tuple will be unpacked while invoking `fn`.

  Returns: The result of calling `fn` on `arg`.
  """
  raise OperatorUndefinedError('federated_map_server')


# pylint: disable=g-long-lambda
def _replace_api(api, placed_computations, prim_computations):
  # importlib.util.module_from_spec()
  """A binding of the implementations in FAX to represent FL."""
  api.federated_map_clients = (
      lambda fn, arg: placed_computations.map_to_placement(fn, arg, 'clients')
  )
  api.federated_map_server = (
      lambda fn, arg: placed_computations.map_to_placement(fn, arg, 'server')
  )

  api.federated_mean = lambda x: jax.tree_util.tree_map(
      prim_computations['mean_from_clients'], x
  )
  api.federated_sum = lambda x: jax.tree_util.tree_map(
      prim_computations['sum_from_clients'], x
  )

  def _weighted_mean(x, w):
    mult_at_clients = api.federated_map_clients(
        lambda arg1, arg2: jax.tree_util.tree_map(
            lambda x, y: x * y, arg1, arg2
        ),
        (x, w),
    )
    sum_mult = api.federated_sum(mult_at_clients)
    denom_sum = api.federated_sum(w)
    return jax.tree_util.tree_map(lambda x, y: x / y, sum_mult, denom_sum)

  api.federated_weighted_mean = _weighted_mean

  def _broadcast_fn(x, *, input_pspecs=None):
    # TODO: b/308448854 - Remove this parameter and update callsites.
    del input_pspecs  # Unused

    return jax.tree_util.tree_map(prim_computations['broadcast_clients'], x)

  api.federated_broadcast = _broadcast_fn
  return api


# pylint: enable=g-long-lambda


def fax_program(*, placements, self_module):
  """Patches symbols into current module and call `jax.jit` on the result.

  This decorator enables calling:

  * federated_map_at_clients
  * federated_map_at_server
  * federated_mean
  * federated_sum
  * federated_weighted_mean
  * federated_broadcast

  The functions returned by this decorator has been jit compiled, and are
  compatible with `jax.jacfwd` and `jax.jacrev`.


  Args:
    placements: Dictionary defining placements. Must contain specification of
      number of clients. Other placements may be ignored inside this decorator.
    self_module: The Python module to patch the API when performing FAX tracing.

  Returns:
    A decorated function enabling the calling of the FAX API. Interoperable
    with other JAX code.
  """
  if 'clients' not in placements:
    raise ValueError(
        'Need a specification for cardinalities of clients placement.'
    )

  placed_computations = impls.PlacedComputations(
      placements_to_n_elements=placements,
  )
  prim_computations, primdefs = primitives.register_primitives(
      placements=placements
  )
  global _FEDERATED_PRIMITIVES
  if not _FEDERATED_PRIMITIVES:
    # First time these primitives are defined; attach them to the global.
    _FEDERATED_PRIMITIVES = primdefs

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
        _replace_api(self_module, placed_computations, prim_computations)
        logging.info('lib patched!')
        return fn(*args, **kwargs)
      finally:
        for symbol_name, symbol in old_api_symbols.items():
          setattr(self_module, symbol_name, symbol)
        logging.info('lib patch reverted!')

    return jax_callable

  return fn_decorator


# pylint: enable=g-doc-exception
