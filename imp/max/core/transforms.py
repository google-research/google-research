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

"""Transform methods developed for modeling modules."""

import functools
from typing import Any

import flax.core.lift
import flax.linen as nn
from flax.linen import partitioning
from flax.linen import transforms
import jax

from imp.max.core import constants
from imp.max.utils import typing


PARAMS = constants.FlaxCollection.PARAMS
BATCH_STATS = constants.FlaxCollection.BATCH_STATS
CACHE = constants.FlaxCollection.CACHE
INTERMEDIATES = constants.FlaxCollection.INTERMEDIATES
PROBES = constants.FlaxCollection.PROBES
AUX_LOSS = constants.FlaxCollection.AUX_LOSS


def remat(module,
          level = 'zero',
          scanned = False,
          static_argnums = ()):
  """Maybe apply lifted jax.lax.remat to a Flax module.

  Args:
    module: The to-be-rematted flax module.
    level: The checkpointing level.
    scanned: Whether the `module` is scanned.
    static_argnums: The static args of the rematted module.
  Returns:
    The rematted flax module.
  """

  if level == 'zero':
    return module

  else:
    remat_transform = functools.partial(
        # TODO(hassanak): Replace with nn.remat once Flax bug is resolved
        partitioning.remat,
        target=module,
        prevent_cse=not scanned,
        static_argnums=static_argnums,
    )

    if level == 'minimal':
      return remat_transform(
          policy=jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      )

    elif level == 'full':
      return remat_transform(policy=None)

    else:
      raise ValueError(
          f'Remat level must be `zero`, `minimal`, or `full`, got {level}.'
      )


def scan(
    module,
    length,
    scan_axis = 0,
    in_axes = nn.broadcast,
    out_axes = 0,
    sharding_axis = None,
    rng_keys = (),
):
  """Scans a module with same config and returns the configurable stack.

  jax.lax.scan and its lifted versions under Flax have strict assumptions.
  The (to-be-scanned) module should only accept an init followed by carry
  arguments: inputs, carry. It should also output the transformed init and its
  carry: outputs, carry. We use the following wrapper to resolve the issue for
  inputs:
    class ScannedOutputModule(module):
        def __call__(self, inputs, *args):
          return super().__call__(inputs, *args), None
  This enables us to scan any layer without touching its input/output signature.
  However, this limits the way we can call the scanned module. The call should
  only be ordered positional arguments: outputs = module(arg1, arg2). Calling
  the module using outputs = module(arg1=value1, arg2=value2) would result in
  two types of error:
    ValueError: Tuple arity mismatch: 0 != 1; tuple: ()
        scan does not understand which argument in carry to broadcast or map.
    TypeError: inner() missing 1 required positional argument: 'init'.
        scan does not understand which argument is inputs and which is carry
  This wrapper is generic enough to accommodate with most usecases. However, it
  assumes that the to-be-scanned module has this input structure:
  inputs, carry1, carry2, ... and user should call it with same order.

  Args:
    module: The to-be-instantiated flax module.
    length: The number of layer instantiations.
    scan_axis: The weight stacking axis.
    in_axes: The axis along which the inputs are split (to be fed to the scanned
      layers).
    out_axes: The axis along which the outputs are stacked.
    sharding_axis: The spmd sharding axis.
    rng_keys: The rng array mappings.

  Returns:
    The scanned layer.
  """

  class ScannedOutputModule(module):
    """A simple wrapper for changing module's output to (out, None)."""

    def __call__(self, inputs, *args):
      return super().__call__(inputs, *args), None

  def _lift_keep_carry(fn):
    def inner(scope_fn, repack_fn, variable_groups, rng_groups, *args):
      scope = scope_fn(variable_groups, rng_groups)
      y, _ = fn(scope, *args)
      return y, repack_fn(scope)
    return flax.core.lift.pack(inner, (True,), (True,), (True,))

  variable_axes = {
      PARAMS: scan_axis,
      BATCH_STATS: 0,
      CACHE: 0,
      INTERMEDIATES: 0,
      PROBES: 0,
      AUX_LOSS: 0
  }
  split_rngs = {k: True for k in rng_keys + (PARAMS,)}
  scanned = nn.scan(
      ScannedOutputModule,
      length=length,
      variable_axes=variable_axes,
      in_axes=in_axes,
      out_axes=out_axes,
      split_rngs=split_rngs,
      metadata_params={nn.meta.PARTITION_NAME: sharding_axis},
  )
  scanned_carry = transforms.lift_transform(_lift_keep_carry, scanned)
  return scanned_carry
