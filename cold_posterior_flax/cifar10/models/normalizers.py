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

"""Normalization modules for Flax."""
from flax.deprecated import nn
import jax.numpy as jnp


class FRN(nn.Module):
  """Filter Response Normalization as per [1].

  - [1] Singh, S., & Krishnan, S. (2019, November 21). Filter Response
  Normalization Layer: Eliminating Batch Dependence in the Training of Deep
  Neural Networks. arXiv [cs.LG]. http://arxiv.org/abs/1911.09737
  """

  def apply(self,
            x,
            eps=1e-6,
            learn_eps=False,
            scale_init=nn.initializers.ones,
            bias=True,
            scale=True):
    features = x.shape[-1]
    axis = tuple(range(1, x.ndim - 1))
    mag = jnp.square(x).mean(axis, keepdims=True)
    if learn_eps:
      eps = self.param('eps', (features,),
                       lambda _, shape: jnp.full(shape, eps))
      eps = jnp.abs(eps)
    x = x / jnp.sqrt(mag + eps)

    if scale:
      scale = self.param('scale', (features,), scale_init)
    else:
      scale = 1
    if bias:
      beta = self.param('beta', (features,), nn.initializers.zeros)
    else:
      beta = 0

    return scale * x + beta


class ReScale(nn.Module):
  """Multiplies x with a learned scalar weight for each input feature."""

  def apply(
      self,
      x,
      scale_init=None,
      scale=True,
  ):
    if scale_init is not None and scale:
      features = x.shape[-1]
      scale = self.param('scale', (features,), scale_init)
    else:
      scale = 1

    return scale * x
