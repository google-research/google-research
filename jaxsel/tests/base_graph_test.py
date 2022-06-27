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

"""Base test classes for graph API."""
from absl.testing import absltest

import jax
import jax.numpy as jnp

from jaxsel import agents
from jaxsel import graph_models
from jaxsel import subgraph_extractors
from jaxsel import synthetic_data


class BaseGraphTest(absltest.TestCase):
  """Tests basic functionnality of our Graph and Agent APIs."""

  def setUp(self):
    """Sets up an example image and associated graph and agent."""
    super().setUp()

    grid_size = 20
    num_paths = 2
    num_classes = 3

    agent_hidden_dim = 4
    max_graph_size = 75
    max_subgraph_size = 11
    num_steps_extractor = 50
    rho = 1e-6
    alpha = 1e-3
    ridge = 1e-7
    num_heads = 2
    n_encoder_layers = 2
    qkv_dim = 32
    mlp_dim = 64
    embedding_dim = 4
    hidden_dim = 16

    graph, start_node_id, label = synthetic_data.generate(
        grid_size, num_paths, num_classes)

    self.graph = graph
    self.start_node_id = start_node_id
    self.label = label

    agent_config = agents.AgentConfig(graph.graph_parameters(),
                                      agent_hidden_dim, agent_hidden_dim)

    extractor_config = subgraph_extractors.ExtractorConfig(
        max_graph_size, max_subgraph_size, rho, alpha, num_steps_extractor,
        ridge, agent_config)

    graph_classifier_config = graph_models.TransformerConfig(
        graph.graph_parameters(),
        num_heads=num_heads,
        num_layers=n_encoder_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        image_size=grid_size**2,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes)

    extractor = subgraph_extractors.SparseISTAExtractor(extractor_config)
    agent = agents.SimpleFiLMedAgentModel(agent_config)

    self.rng = jax.random.PRNGKey(2)
    self.agent = agent
    self.extractor = extractor
    self.graph_classifier = graph_models.TransformerClassifier(
        graph_classifier_config)

  def test_random_walk_on_graph(self):
    """Tests ability to perform a random walk on a graph built from an image."""

    rng_agent, self.rng = jax.random.split(self.rng)
    self.agent.init_with_output(rng_agent, self.graph, method=self.agent.walk)

  def test_out_of_bounds_pixel_neighbors(self):
    """The out of bounds pixel should only be linked to the start node."""
    relation_ids, neighbor_node_ids = self.graph.outgoing_edges(-1)
    del relation_ids
    assert jnp.all(neighbor_node_ids == self.graph._start_node_id)
