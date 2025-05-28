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

"""Tests for train."""

import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from differentially_private_gnns import dataset_readers
from differentially_private_gnns import input_pipeline
from differentially_private_gnns import train
from differentially_private_gnns.configs import dpgcn
from differentially_private_gnns.configs import dpmlp
from differentially_private_gnns.configs import gcn
from differentially_private_gnns.configs import mlp


_ALL_CONFIGS = {
    'dpgcn': dpgcn.get_config(),
    'dpmlp': dpmlp.get_config(),
    'gcn': gcn.get_config(),
    'mlp': mlp.get_config(),
}


def update_dummy_config(config):
  """Updates the dummy config."""
  config.dataset = 'dummy'
  config.batch_size = dataset_readers.DummyDataset.NUM_DUMMY_TRAINING_SAMPLES // 2
  config.max_degree = 2
  config.num_training_steps = 10
  config.num_classes = dataset_readers.DummyDataset.NUM_DUMMY_CLASSES


class TrainTest(parameterized.TestCase):

  @parameterized.product(
      config_name=['dpmlp', 'dpgcn'], rng_key=[0, 1], max_degree=[0, 1, 2])
  def test_per_example_gradients(self, config_name, rng_key,
                                 max_degree):
    # Load dummy config.
    config = _ALL_CONFIGS[config_name]
    update_dummy_config(config)
    config.max_degree = max_degree

    # Load dummy dataset.
    rng = jax.random.PRNGKey(rng_key)
    rng, dataset_rng = jax.random.split(rng)
    dataset = input_pipeline.get_dataset(config, dataset_rng)
    graph, labels, _ = jax.tree.map(jnp.asarray, dataset)
    labels = jax.nn.one_hot(labels, config.num_classes)
    num_nodes = labels.shape[0]

    # Create subgraphs.
    graph = jax.tree.map(np.asarray, graph)
    subgraphs = train.get_subgraphs(graph, config.pad_subgraphs_to)
    graph = jax.tree.map(jnp.asarray, graph)

    # Initialize model.
    rng, init_rng = jax.random.split(rng)
    estimation_indices = jnp.asarray([0])
    state = train.create_train_state(init_rng, config, graph, labels, subgraphs,
                                     estimation_indices)

    # Choose indices for batch.
    rng, train_rng = jax.random.split(rng)
    indices = jax.random.choice(train_rng, num_nodes, (config.batch_size,))

    # Compute per-example gradients.
    per_example_grads = train.compute_updates_for_dp(
        state, graph, labels, subgraphs, indices,
        config.adjacency_normalization)
    per_example_grads_summed = jax.tree.map(lambda grad: jnp.sum(grad, axis=0),
                                            per_example_grads)

    # Compute batched gradients.
    batched_grads = train.compute_updates(state, graph, labels, indices)

    # Check that these gradients match.
    chex.assert_trees_all_close(
        batched_grads, per_example_grads_summed, atol=1e-3, rtol=1e-3)

  @parameterized.parameters('gcn', 'mlp', 'dpgcn', 'dpmlp')
  def test_train_and_evaluate(self, config_name):

    # Load config for dummy dataset.
    config = _ALL_CONFIGS[config_name]
    update_dummy_config(config)

    # Create a temporary directory where metrics are written.
    workdir = tempfile.mkdtemp()

    # Training should proceed without any errors.
    train.train_and_evaluate(config, workdir)


if __name__ == '__main__':
  absltest.main()
