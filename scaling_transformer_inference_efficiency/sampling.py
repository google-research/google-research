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
