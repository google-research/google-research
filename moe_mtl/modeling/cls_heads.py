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

"""Classes to build various prediction heads in JAX detection models.

The implementation follows cloud TPU detection heads.
"""
from flax import linen as nn
import gin
import jax.numpy as jnp

Array = jnp.ndarray


@gin.register
class ClassificationHead(nn.Module):
  """A standard Classification head."""
  num_classes: int = 1000
  train: bool = True
  dtype: jnp.dtype = jnp.float32
  batch_norm_group_size: int = 0

  @nn.compact
  def __call__(self, backbone_features):
    x = nn.Dense(
        features=self.num_classes,
        name='classification_fc',
        kernel_init=nn.initializers.xavier_uniform(),
        dtype=jnp.float32,
    )(
        backbone_features)

    return x
