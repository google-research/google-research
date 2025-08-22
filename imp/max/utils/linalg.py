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

"""Utilities related to Linear Algebra."""

from typing import Any, NamedTuple

from absl import logging
from flax import struct
import jax
import jax.numpy as jnp


def _derive_projection_params(array, rank):
  """Derives the parameters for the projection.

  According to GaLore (https://arxiv.org/abs/2403.03507), in the SVD projection,
  if an array with shape (M, N) is given and M <= N we peform U^T * A to store
  smaller projection states (since U is [M, R] and M <= N) and if M > N, we
  perform A * V to store smaller projection states.

  Args:
    array: The array to be projected.
    rank: The SVD projection rank.
  Returns:
    A NamedTuple that contains the shape of projector, after-projection-array,
    contract axes for performing dot-general op, and the SVD side.
  """

  class _ProjectionParams(NamedTuple):
    projector_shape: tuple[int, Ellipsis] | None = None
    projected_shape: tuple[int, Ellipsis] | None = None
    array_contract_axis: tuple[int, Ellipsis] | None = None
    projector_contract_axis: tuple[int, Ellipsis] | None = None
    back_projection_contract_axis: tuple[int, Ellipsis] | None = None
    svd_projector_index: int | None = None

  if array.ndim < 2:
    return _ProjectionParams(projected_shape=array.shape)

  elif array.ndim == 2:
    m, n = array.shape
    if min(m, n) <= rank:
      return _ProjectionParams(projected_shape=array.shape)

    elif m <= n:
      # Projection would be: U^T * A
      return _ProjectionParams(
          projector_shape=(m, rank),
          projected_shape=(rank, n),
          array_contract_axis=(0,),
          projector_contract_axis=(0,),
          back_projection_contract_axis=(array.ndim - 1,),
          svd_projector_index=0,
      )
    else:
      # Projection would be: A * V
      return _ProjectionParams(
          projector_shape=(rank, n),
          projected_shape=(m, rank),
          array_contract_axis=(array.ndim - 1,),
          projector_contract_axis=(array.ndim - 1,),
          back_projection_contract_axis=(0,),
          svd_projector_index=2,
      )
  else:
    raise NotImplementedError(
        'Projection on arrays with more than 2 dimensions is not supported. '
        f'Instead, received `{array.shape=}`.')


@struct.dataclass
class LowRankProjectionState:
  """State for the low-rank projection."""
  projector: jax.Array
  array_contract_axis: tuple[int, Ellipsis] = struct.field(pytree_node=False)
  projector_contract_axis: tuple[int, Ellipsis] = struct.field(pytree_node=False)
  back_projection_contract_axis: tuple[int, Ellipsis] = struct.field(
      pytree_node=False)


@struct.dataclass
class EmptyLowRankProjectionState:
  """State for the empty low-rank projection."""


# TODO(hassanak): Add sharded_svd, where the input to the core svd impl is
# the jax.Array shards on the device that svd impl would be executed.
# Any remaining leading dim would also be parallelized using vmap.
def svd_projector(
    array,
    rank,
):
  """Performs SVD on a given array and returns the projection setting.

  Inspired by GaLore (https://arxiv.org/abs/2403.03507), if A is an MxN matrix,
  we perform A = USV^T decompositino (SVD) and return U[:, :r] if M <= N and
  V^T[:r, :] if M > N. The choice of returning U or V^T is incentivised by
  saving memory by carrying smaller projectors in the optimization states.

  Args:
    array: An MxN array.
    rank: The number of first eigen vectors to return as the low-rank projector.

  Returns:
    A LowRankProjectionState that contains the projector and contraction axes
    necessary to perform a dot-general between the array and the projector for
    low-rank projection and back projection (to the original space).
  """
  if array.ndim < 2:
    # SVD is not applicable if array is a vector or scalar.
    logging.debug(
        'Received array with less than 2 dimensions, skipping SVD.')
    return EmptyLowRankProjectionState()

  if min(array.shape[0], array.shape[-1]) <= rank:
    # SVD is not applicable if array is already lower rank.
    logging.debug('Array is already lower rank, skipping SVD.')
    return EmptyLowRankProjectionState()

  # Derive efficient projection parameters
  proj_params = _derive_projection_params(array, rank)

  if jax.devices()[0].platform in ('cpu', 'gpu'):
    # `subset_by_index` is not supported under cpu/gpu yet.
    projectors = jnp.linalg.svd(array, full_matrices=False)
    projector_index = proj_params.svd_projector_index
    projector = projectors[projector_index]
    if projector_index == 0:
      projector = projector[:, :rank]
    elif projector_index == 2:
      projector = projector[:rank, :]
  else:
    projectors = jnp.linalg.svd(
        array,
        subset_by_index=(0, rank),
        full_matrices=False,
    )
    projector = projectors[proj_params.svd_projector_index]

  array_contract_axis = proj_params.array_contract_axis
  projector_contract_axis = proj_params.projector_contract_axis
  back_projection_contract_axis = proj_params.back_projection_contract_axis

  return LowRankProjectionState(
      projector=projector,
      array_contract_axis=array_contract_axis,
      projector_contract_axis=projector_contract_axis,
      back_projection_contract_axis=back_projection_contract_axis,
  )


def project_array(
    array,
    projection_state,
    back_projection = False,
    precision = None,
):
  """Projects an array to a low-rank space using a projection state.

  This function takes an array and its corresponding projection state (e.g. the
  output of the 'svd_projector' above) and projects it to a low-rank space OR
  takes a low-rank array and projects it back to the original space.
  The projection dimension is determined by the contract axes in the projection
  state. Whether the array should be projected to a lower space or the original
  is determined by the 'back_projection' flag.

  Args:
    array: An array with one of the shapes: {MxN, RxN or MxR}.
    projection_state: States required for the projection.
    back_projection: Whether to project back to original space. If True, array
      should be low-rank.
    precision: The dot-general computation precision for the projection.
  Returns:
    The projected array.
  """
  if isinstance(projection_state, EmptyLowRankProjectionState):
    return array

  # Determine contraction axes based on forward/backward projection
  array_contract_axis = projection_state.array_contract_axis
  if back_projection:
    projector_contract_axis = projection_state.back_projection_contract_axis
  else:
    projector_contract_axis = projection_state.projector_contract_axis

  # Determine order of projection based on left/right projection
  if projection_state.array_contract_axis == (0,):
    # Perform left projection
    return jax.lax.dot_general(
        projection_state.projector,
        array,
        ((projector_contract_axis,
          array_contract_axis),
         ((), ())),
        precision=precision,
    )
  else:
    # perform right projection
    return jax.lax.dot_general(
        array,
        projection_state.projector,
        ((array_contract_axis,
          projector_contract_axis),
         ((), ())),
        precision=precision,
    )

