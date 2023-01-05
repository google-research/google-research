# coding=utf-8
# Copyright 2022 The Google Research Authors.
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
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp


@struct.dataclass
class Sampling:
  """Hyperparameters controlling sampling from a model."""
  temperature: float

  # TODO(reinerp): topk/topp support.

  def sample(self, logits, step_rngs):
    """Samples from the output logits of a model.

    Args:
      logits: The output logits to sample from. float32[batch, vocab_size].
      step_rngs: For each batch element, the RNG state for sampling.
        jax.random.PRNGKey[batch]

    Returns:
      The selected samples, as token IDs. int32[batch].
    """

    def sample_nonzero():
      # jax.random.categorical expects just one rng. We use vmap to extend it to
      # support a batch of rngs.
      return jnp.int32(
          jax.vmap(jax.random.categorical)(step_rngs,
                                           logits / self.temperature))

    def sample_zero():
      return jnp.int32(jnp.argmax(logits, -1))

    # To avoid numerical instability when dividing by very small temperatures,
    # we sample deterministically (greedily) when the temperature is
    # sufficiently close to zero.
    return lax.cond(self.temperature > 1e-4, sample_nonzero, sample_zero)

  def sample_manual(self, logits,
                    step_rngs):
    """Samples from the output logits when within xmap."""

    with jax.named_scope('sample'):
      # logits:
      # float32[batch.X, vocab.YZ]
      #   -> float32[batch.XYZ, vocab]
      y_axis = lax.psum(1, 'y')
      z_axis = lax.psum(1, 'z')
      yz_index = lax.axis_index('y') * z_axis + lax.axis_index('z')
      batch_x, _ = logits.shape
      padded_batch_x = max(batch_x, y_axis * z_axis)
      if padded_batch_x > batch_x:
        logits = jnp.pad(
            logits,
            pad_width=((0, padded_batch_x - batch_x), (0, 0), (0, 0)),
            mode='constant')
      # We all to all so that we get the full logit on each, but shard batch
      # as much as possible
      logits = lax.all_to_all(
          logits, ('y', 'z'), split_axis=0, concat_axis=1, tiled=True)
      # need to only take the relevant part of this
      split_size = (batch_x // y_axis // z_axis)
      step_rngs = lax.dynamic_slice_in_dim(
          step_rngs,
          yz_index * split_size, (batch_x // y_axis // z_axis),
          axis=0)
      # TODO(sholto): Confirm this is the best way of doing it
      # logits = binary_search.topp_mask(logits, 0.9, -1e10)
      # TODO(sholto): maybe put t5x binary search back in
      sample = jnp.int32(
          jax.vmap(jax.random.categorical)(step_rngs,
                                           logits / self.temperature))
      # sample: int32[batch]
      sample = lax.all_gather(sample, ('x', 'y', 'z'), axis=0, tiled=True)
    return sample

  def sample_manual_batch_unsharded(self, logits,
                                    step_rngs):
    """Samples from output logits within xmap, with batch unshardedable.

    Args:
      logits: [batch, vocab.YZX]
      step_rngs: [batch]
    Returns:
      sample" int32[batch]
    """

    with jax.named_scope('sample'):
      # multi-part all gather not implemented for xmap in jit see lax.parallel
      logits = lax.all_gather(logits, ('x'), axis=1, tiled=True)
      logits = lax.all_gather(logits, ('z'), axis=1, tiled=True)
      logits = lax.all_gather(logits, ('y'), axis=1, tiled=True)
      assert logits.shape[0] == step_rngs.shape[0]
      sample = jnp.int32(
          jax.vmap(jax.random.categorical)(step_rngs,
                                           logits / self.temperature))
    return sample
