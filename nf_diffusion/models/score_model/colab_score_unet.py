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

"""ScoreNet from colab."""
import flax.linen as nn
import jax.numpy as jnp
from nf_diffusion.models.layers import ldm_utils


class ScoreNet(nn.Module):
  """ScoreNet."""

  embedding_dim: int = 64
  n_layers: int = 10

  @nn.compact
  def __call__(self, z, t=None, cond=None, train=True, **kwargs):
    n_embd = self.embedding_dim

    assert t is not None
    assert jnp.isscalar(t) or len(t.shape) == 0 or len(t.shape) == 1
    t = t * jnp.ones(z.shape[0])  # ensure t is a vector

    temb = ldm_utils.get_timestep_embedding(t, n_embd)
    if cond is not None:
      cond = jnp.concatenate([temb, cond], axis=1)
    else:
      cond = temb
    cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense0")(cond))
    cond = nn.swish(nn.Dense(features=n_embd * 4, name="dense1")(cond))
    cond = nn.Dense(n_embd)(cond)

    h = nn.Dense(n_embd)(z)
    h = ldm_utils.ResNet2D(n_embd, self.n_layers)(h, cond)
    return z + h
