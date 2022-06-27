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

"""Utilities for computing the coherence loss."""

import jax
import jax.numpy as jnp


EPSILON = 1e-9


def orthogonal_features_coherence(matrix, option='max'):
  n, k = matrix.shape
  if option == 'max':
    return n / k * jnp.max(jnp.sum(jnp.square(matrix), axis=1))
  elif option == 'logsumexp':
    return n / k * jax.nn.logsumexp(jnp.sum(jnp.square(matrix), axis=1))
  if option == 'mean':
    return n / k * jnp.mean(jnp.sum(jnp.square(matrix), axis=1))
  else:
    raise ValueError(f'{option} is not valid.')


def orthogonality(matrix):
  """Computes cosine similarity between all pairs of vectors in x and y."""
  norm = jnp.sqrt(jnp.sum(matrix**2, axis=1))
  matrix_normalized = matrix / (norm[:, None] + EPSILON)
  return matrix_normalized @ matrix_normalized.transpose()
