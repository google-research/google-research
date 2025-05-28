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

"""Random or greedy sampling from the output logits of a model."""
from typing import Any, Optional

from flax import struct
import jax
from jax import lax
from jax.experimental.shard_map import shard_map
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
import typing_extensions

from scaling_transformer_inference_efficiency import partitioning

# from t5x import binary_search


@struct.dataclass
class SamplingHyperParams:
  temperature: Any
  top_k: Optional[Any] = struct.field(pytree_node=False, default=None)
  top_p: Optional[Any] = struct.field(pytree_node=False, default=None)

  @classmethod
  def physical_axes(cls):
    return SamplingHyperParams(temperature=P(), top_k=P(), top_p=P())


def sample(
    logits,
    step_rngs,
    hyper_params,
    mesh):
  """Samples from the output logits of a model.

  Args:
    logits: The output logits to sample from. float32[batch, vocab_size].
    step_rngs: For each batch element, the RNG state for sampling.
      jax.random.PRNGKey[batch]
    hyper_params: -
    mesh: For manual compat

  Returns:
    The selected samples, as token IDs. int32[batch].
  """
  del mesh  # used for manual mode compat
  # Ensure it is unsharded along vocab dimension
  # pylint: disable = protected-access
  logits = partitioning._with_sharding_constraint(
      logits, P('logit_batch', None)
  )
  # logits = binary_search.topp_mask(logits, hyper_params.top_p, -1e10)

  if hyper_params.top_k is not None:
    logits, top_k_indices = lax.approx_max_k(
        logits, hyper_params.top_k, recall_target=1.0
    )

  def sample_nonzero():
    # jax.random.categorical expects just one rng. We use vmap to extend it to
    # support a batch of rngs.

    return jnp.int32(
        jax.vmap(jax.random.categorical)(
            step_rngs, logits / hyper_params.temperature
        )
    )

  def sample_zero():
    return jnp.int32(jnp.argmax(logits, -1))

  # To avoid numerical instability when dividing by very small temperatures,
  # we sample deterministically (greedily) when the temperature is
  # sufficiently close to zero.
  sampled_logits = lax.cond(
      hyper_params.temperature > 1e-4, sample_nonzero, sample_zero
  )

  if hyper_params.top_k is not None:
    sampled_logits = jax.vmap(lambda indices, sampled: indices[sampled])(
        top_k_indices, sampled_logits  # pylint: disable=undefined-variable
    )

  return partitioning._with_sharding_constraint(
      sampled_logits, P('logit_batch')
  )


def sample_manual(
    logits,
    step_rngs,
    hyper_params,
    mesh,
    batch_unsharded = False,
):
  """Samples from the output logits when within xmap."""

  def lowered_fn(logits, step_rngs):
    y_axis = lax.psum(1, 'y')
    z_axis = lax.psum(1, 'z')
    yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
    batch, _ = logits.shape

    with jax.named_scope('sample'):
      # logits: float32[batch, vocab.YZ] || float32[batch.X, vocab.YZ]

      if batch < z_axis:
        # float32[batch, vocab.YZ] -> float32[batch, vocab]
        # || float32[batch.X, vocab.YZ] -> float32[batch.X, vocab]
        logits = lax.all_gather(logits, ('y', 'z'), axis=1, tiled=True)
        all_gather_tokens = None if batch_unsharded else 'x'
      elif batch >= z_axis and batch < y_axis * z_axis:
        # float32[batch, vocab.YZ] -> float32[batch.Z, vocab]
        # || float32[batch.X, vocab.YZ] -> float32[batch.XZ, vocab]
        logits = lax.all_to_all(
            logits, 'z', split_axis=0, concat_axis=1, tiled=True
        )
        logits = lax.all_gather(logits, 'y', axis=1, tiled=True)
        split_size = batch // z_axis
        step_rngs = lax.dynamic_slice_in_dim(
            step_rngs, lax.axis_index('z') * split_size, (split_size), axis=0
        )
        all_gather_tokens = 'z' if batch_unsharded else ('x', 'z')
      elif batch >= y_axis * z_axis:
        # float32[batch, vocab.YZ] -> float32[batch.YZ, vocab]
        # || float32[batch.X, vocab.YZ] -> float32[batch.XYZ, vocab]
        logits = lax.all_to_all(
            logits, ('y', 'z'), split_axis=0, concat_axis=1, tiled=True
        )
        split_size = batch // y_axis // z_axis
        step_rngs = lax.dynamic_slice_in_dim(
            step_rngs, yz_index * split_size, (split_size), axis=0
        )
        all_gather_tokens = ('y', 'z') if batch_unsharded else ('x', 'y', 'z')
      else:
        raise NotImplementedError

      assert logits.shape[0] == step_rngs.shape[0]
      # TODO(sholto): Confirm this is the best way of doing it
      # logits = binary_search.topp_mask(logits, 0.9, -1e10)
      # TODO(sholto): maybe put t5x binary search back in
      sample_result = jnp.int32(
          jax.vmap(jax.random.categorical)(
              step_rngs, logits / hyper_params.temperature
          )
      )
      if all_gather_tokens is not None:
        # sample: int32[batch]
        sample_result = lax.all_gather(
            sample_result, all_gather_tokens, axis=0, tiled=True
        )
      return sample_result

  logit_specs = partitioning.logical_to_physical(P('logit_batch', 'vocab'))
  rng_specs = partitioning.logical_to_physical(P('logit_batch', None))
  # if it cannot be sharded as such, then do not
  # rng_specs = partitioning.safe_sharding(step_rngs, P(('x', 'y', 'z')), mesh)
  sample_result = shard_map(
      lowered_fn,
      mesh=mesh,
      in_specs=(logit_specs, rng_specs),
      out_specs=P(None),
      check_rep=False,
  )(logits, step_rngs)

  return sample_result


def sample_manual_batch_unsharded(
    logits,
    step_rngs,
    hyper_params,
    mesh):
  """Samples from output logits within xmap, with batch unshardedable.

  Args:
    logits: [batch, vocab.YZX]
    step_rngs: [batch]
    hyper_params: -
    mesh: for manual collectives

  Returns:
    sample" int32[batch]
  """
  def lowered_fn(logits, step_rngs):
    with jax.named_scope('sample'):
      # multi-part all gather not implemented for xmap in jit see lax.parallel
      logits = lax.all_gather(logits, 'x', axis=1, tiled=True)
      logits = lax.all_gather(logits, 'z', axis=1, tiled=True)
      logits = lax.all_gather(logits, 'y', axis=1, tiled=True)
      assert logits.shape[0] == step_rngs.shape[0]
      sample_result = jnp.int32(
          jax.vmap(jax.random.categorical)(
              step_rngs, logits / hyper_params.temperature
          )
      )
      return sample_result
  logit_specs = partitioning.logical_to_physical(P('logit_batch', 'vocab'))
  sample_result = shard_map(
      lowered_fn,
      mesh=mesh,
      in_specs=(logit_specs, P(None)),
      out_specs=P(None),
      check_rep=False,
  )(logits, step_rngs)
  return sample_result


class SampleFn(typing_extensions.Protocol):
  """A function providing a forwards pass through a model."""

  def __call__(
      self,
      logits,
      step_rngs,
      hyper_params,
      mesh,
  ):
    Ellipsis
