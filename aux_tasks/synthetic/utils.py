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

"""Several objective functions."""
import dataclasses
import functools
import json
from typing import Callable, Optional, Union

from absl import logging
from etils import epath
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
from shapely import geometry
import tensorflow_datasets as tfds

from aux_tasks.puddle_world import arenas
from aux_tasks.puddle_world import puddle_world
from aux_tasks.puddle_world import utils as pw_utils


Parameters = Union[flax.core.FrozenDict, jnp.ndarray]


@dataclasses.dataclass
class SyntheticExperiment:
  compute_phi: Callable[[Parameters, jnp.ndarray], jnp.ndarray]
  compute_psi: Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]
  sample_states: Callable[[jnp.ndarray, int], tuple[jnp.ndarray, jnp.ndarray]]
  eval_states: jnp.ndarray
  optimal_subspace: jnp.ndarray
  params: Parameters
  key: jnp.ndarray


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
  U, S, V = jnp.linalg.svd(Psi)
  S_new = jnp.linspace(1, 1_000, S.shape[0])
  return U @ jnp.diag(S_new) @ V


def generate_psi_exp(Psi):
  U, S, V = jnp.linalg.svd(Psi)
  S_new = jnp.logspace(0, 3, S.shape[0])
  return U @ jnp.diag(S_new) @ V


def compute_optimal_subspace(Psi, d):
  left_svd, _, _ = jnp.linalg.svd(Psi)
  return left_svd[:, :d]


def get_mnist_data(num_samples = 60000):
  """Returns MNIST data as a matrix."""
  ds = tfds.load('mnist:3.*.*', split='train')
  ds = ds.batch(num_samples)
  data = next(ds.as_numpy_iterator())
  X = np.reshape(data['image'], (num_samples, -1)) / 255.
  # Returns a matrix of size `784 x num_samples`
  data = X.T
  # Subtract the mean
  data -= np.mean(data, axis=1, keepdims=True)
  return jnp.asarray(data)


def create_synthetic_experiment(
    config):
  """Creates a synthetic experiment with finite matrices."""
  key = jax.random.PRNGKey(config.seed)
  phi_key, psi_key, key = jax.random.split(key, 3)

  Phi = jax.random.normal(phi_key, (config.S, config.d), dtype=jnp.float32)

  if config.use_mnist:
    Psi = get_mnist_data()
  else:
    Psi = jax.random.normal(psi_key, (config.S, config.T), dtype=jnp.float32)
    if config.rescale_psi == 'linear':
      Psi = generate_psi_linear(Psi)
    elif config.rescale_psi == 'exp':
      Psi = generate_psi_exp(Psi)

  sample_states = functools.partial(
      sample_discrete_states,
      num_states=config.S,
      sample_with_replacement=config.sample_with_replacement)
  eval_states = jnp.arange(config.S)

  compute_phi = lambda phi, states: phi[states, :]
  params = Phi

  def compute_psi(
      states, tasks = None):
    if tasks is None:
      return Psi[states, :]
    return Psi[states, tasks]

  if config.svd_path:
    logging.info('Loading SVD from %s', config.svd_path)
    with epath.Path(config.svd_path).open('rb') as f:
      left_svd = np.load(f)
      optimal_subspace = left_svd[:, :config.d]
  else:
    Psi = compute_psi(eval_states, None)
    optimal_subspace = compute_optimal_subspace(Psi, config.d)

  return SyntheticExperiment(
      compute_phi=compute_phi,
      compute_psi=compute_psi,
      sample_states=sample_states,
      eval_states=eval_states,
      optimal_subspace=optimal_subspace,
      params=params,
      key=key
  )


def create_puddle_world_experiment(
    config):
  """Creates a Puddle World experiment."""
  key = jax.random.PRNGKey(config.seed)
  network_key, key = jax.random.split(key)

  all_arenas = {'sutton_10', 'sutton_20', 'sutton_100'}
  if config.puddle_world_arena not in all_arenas:
    raise ValueError(f'Unknown arena {config.puddle_world_arena}.')

  if not config.puddle_world_path:
    raise ValueError('puddle_world_path wasn\'t supplied. Please pass a path.')

  path = epath.Path(config.puddle_world_path)
  path = path / config.puddle_world_arena

  with (path / 'metadata.json').open('r') as f:
    metadata = json.load(f)

  puddles = arenas.get_arena(metadata['arena_name'])
  pw = puddle_world.PuddleWorld(
      puddles, goal_position=geometry.Point((1.0, 1.0)))
  dpw = pw_utils.DiscretizedPuddleWorld(pw, metadata['num_bins'])

  with (path / 'sr.np').open('rb') as f:
    Psi = np.load(f)
  Psi = jnp.asarray(Psi, dtype=jnp.float32)

  with (path / 'svd.np').open('rb') as f:
    optimal_subspace = np.load(f)
  optimal_subspace = jnp.asarray(optimal_subspace[:, :config.d])

  if config.T != Psi.shape[1]:
    logging.warning(
        'Num tasks T (%d) does not match columns of Psi (%d). Overwriting.',
        config.T,
        Psi.shape[1])
    config.T = Psi.shape[1]

  # Ensure we don't use the tabular gradient, since we will always use
  # neural networks with puddle world experiments.
  config.use_tabular_gradient = False

  eval_states = []
  for i in range(dpw.num_states):
    bottom_left, top_right = dpw.get_bin_corners_by_bin_idx(i)
    mid_x = (bottom_left.x + top_right.x) / 2
    mid_y = (bottom_left.y + top_right.y) / 2
    eval_states.append([mid_x, mid_y])

  eval_states = jnp.asarray(eval_states, dtype=jnp.float32)

  def sample_states_continuous(
      key, num_samples):
    """Samples a random (x, y) coordinate."""
    sample_key, key = jax.random.split(key)
    samples = jax.random.uniform(
        sample_key, (num_samples, 2), dtype=jnp.float32)
    return samples, key

  def sample_states_discrete(
      key, num_samples):
    """Samples from the (x, y) coordinates at the center of a bin."""
    sample_key, key = jax.random.split(key)
    samples = jax.random.choice(
        sample_key, eval_states, (num_samples,))
    return samples, key

  if config.use_center_states_only:
    sample_states = sample_states_discrete
  else:
    sample_states = sample_states_continuous

  def compute_psi(
      states, tasks = None):
    # First, we get which column and row the x and y falls into, and then
    # clip to make sure the edge cases when x=1.0 or y=1.0 falls into a
    # valid bin.
    cols_and_rows = jnp.clip(
        jnp.floor(states * metadata['num_bins']),
        min=0,
        max=metadata['num_bins'] - 1)

    # Bin indices are assigned starting in the bottom left moving right, and
    # then advancing upwards after finishing each row.
    # e.g. in a 10x10 grid, row 2 col 3 corresponds to bin 13.
    bin_indices = (
        cols_and_rows[:, 0] + cols_and_rows[:, 1] * metadata['num_bins'])
    bin_indices = bin_indices.astype(jnp.int32)

    if tasks is None:
      return Psi[bin_indices, :]
    return Psi[bin_indices, tasks]

  class Module(nn.Module):

    @nn.compact
    def __call__(self, x):
      for _ in range(config.phi_hidden_layers):
        x = jax.nn.relu(nn.Dense(config.phi_hidden_layer_width)(x))
      return nn.Dense(config.d)(x)

  module = Module()
  params = module.init(
      network_key, jnp.zeros((10, 2), dtype=jnp.float32))
  compute_phi = module.apply

  return SyntheticExperiment(  # pytype: disable=wrong-arg-types  # jax-ndarray
      compute_phi=compute_phi,
      compute_psi=compute_psi,
      sample_states=sample_states,
      eval_states=eval_states,
      optimal_subspace=optimal_subspace,
      params=params,
      key=key,
  )

# pylint: enable=invalid-name
