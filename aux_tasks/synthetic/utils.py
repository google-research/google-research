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

"""Several objective functions."""
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds


def sample_discrete_states(
    key,
    num_samples,
    *,
    num_states,
    sample_with_replacement = False):
  """Samples indices of discrete states.

  Args:
    key: the PRNGKey to use for sampling.
    num_samples: how many samples to draw.
    num_states: the number of states to sample from.
    sample_with_replacement: whether to sample with replacement.

  Returns:
    A tuple of (sampled_state_indices, rng), where sampled_state_indices
      has shape (num_samples,)
  """
  sample_key, key = jax.random.split(key)
  states = jax.random.choice(
      sample_key, num_states, (num_samples,), replace=sample_with_replacement)
  return states, key


def compute_max_feature_norm(features):
  """Computes the maximum norm (squared) of a collection of feature vectors."""
  feature_norm = jnp.linalg.norm(features, axis=1)
  feature_norm = jnp.max(feature_norm)**2

  return feature_norm


# pylint: disable=invalid-name
def inner_objective_mc(
    Phi, Psi, W):
  # Inner Objective function: $\|\Phi W - \Psi \|^2_F$
  return jnp.linalg.norm(Phi @ W - Psi, ord='fro')**2


def outer_objective_mc(Phi, Psi):
  # Outer objective function: $J(\Phi) =\min_W \|\Phi W - \Psi \|^2_F$
  W_star, _, _, _ = jnp.linalg.lstsq(Phi, Psi, rcond=1e-5)
  return inner_objective_mc(Phi, Psi, W_star)


def generate_psi_linear(Psi):
  U, S, V = np.linalg.svd(Psi)
  S_new = np.linspace(1, 1_000, S.shape[0])
  return U @ np.diag(S_new) @ V


def generate_psi_exp(Psi):
  U, S, V = np.linalg.svd(Psi)
  S_new = np.logspace(0, 3, S.shape[0])
  return U @ np.diag(S_new) @ V


def get_mnist_data(num_samples = 60000):
  """Returns MNIST data as a matrix."""
  ds = tfds.load('mnist:3.*.*', split='train')
  ds = ds.batch(num_samples)
  data = next(ds.as_numpy_iterator())
  X = np.reshape(data['image'], (num_samples, -1)) / 255.
  # Returns a matrix of size `784 x num_samples`
  return X.T


# pylint: enable=invalid-name
