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

# Lint as: python3
"""Tests for gfsa.training.train_edge_supervision_lib."""

from absl.testing import absltest
import dataclasses
import flax
import jax
import jax.numpy as jnp
import numpy as np
from gfsa import automaton_builder
from gfsa import sparse_operator
from gfsa.datasets import graph_bundle
from gfsa.model import model_util
from gfsa.model import side_outputs
from gfsa.training import train_edge_supervision_lib


class TrainEdgeSupervisionLibTest(absltest.TestCase):

  def _make_example(self):
    example = graph_bundle.zeros_like_padded_example(
        graph_bundle.PaddingConfig(
            static_max_metadata=automaton_builder.EncodedGraphMetadata(
                num_nodes=5, num_input_tagged_nodes=0),
            max_initial_transitions=0,
            max_in_tagged_transitions=0,
            max_edges=8))
    example = dataclasses.replace(
        example,
        graph_metadata=automaton_builder.EncodedGraphMetadata(
            num_nodes=4, num_input_tagged_nodes=0),
        edges=sparse_operator.SparseCoordOperator(
            input_indices=jnp.array([[0], [0], [0], [0], [1], [2], [2], [0]]),
            output_indices=jnp.array([[1, 2], [2, 3], [2, 2], [3, 0], [0, 2],
                                      [0, 3], [0, 0], [0, 0]]),
            values=jnp.array([1, 1, 1, 1, 1, 1, 0, 0])))
    return example

  def test_loss_fn(self):
    example = self._make_example()

    def mock_model(example):
      del example
      return model_util.safe_logit(
          jnp.array([
              [0.2, 0.0, 0.1, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.3, 0.7, 0.0],
              [0.9, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0],
          ]))

    loss, metrics = train_edge_supervision_lib.loss_fn(
        *train_edge_supervision_lib.extract_outputs_and_targets(
            mock_model, (example, None), target_edge_index=0, num_edge_types=3))

    np.testing.assert_allclose(
        loss,
        -np.log(0.9) - np.log(0.8) - np.log(0.9) - np.log(0.3) - np.log(0.7),
        rtol=1e-5)
    np.testing.assert_allclose(
        metrics["avg_per_target"],
        (-np.log(0.9) - np.log(0.3) - np.log(0.7)) / 4,
        rtol=1e-5)
    np.testing.assert_allclose(
        metrics["avg_per_non_target"], (-np.log(0.8) - np.log(0.9)) / 12,
        rtol=1e-5)

  def test_sample_loss_fn(self):
    example = self._make_example()
    example = dataclasses.replace(
        example,
        edges=sparse_operator.SparseCoordOperator(
            input_indices=jnp.array([[0], [0], [0], [0], [1], [2], [0], [0]]),
            output_indices=jnp.array([[1, 2], [2, 3], [2, 2], [3, 0], [0, 2],
                                      [0, 3], [0, 0], [0, 0]]),
            values=jnp.array([1, 1, 1, 1, 1, 1, 0, 0])))

    @flax.nn.module
    def mock_model_def(example):
      del example
      side_outputs.SideOutput(
          -jnp.arange(5).astype("float32").reshape((1, 5)),
          name="one_sample_log_prob_per_edge_per_node")
      side_outputs.SideOutput(0.3, name="one_sample_reward_baseline")

      return model_util.safe_logit(
          jnp.array([
              [0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0],
          ]))

    _, params = mock_model_def.init(jax.random.PRNGKey(0), example)
    mock_model = flax.nn.Model(mock_model_def, params)

    _, _, _, loss, metrics = train_edge_supervision_lib.sample_loss_fn(
        mock_model, (example, jax.random.PRNGKey(0)),
        target_edge_index=0,
        num_edge_types=3,
        num_rollouts=1,
        leave_one_out_baseline=False)

    np.testing.assert_allclose(metrics["reward"], 0.75, rtol=1e-5)
    np.testing.assert_allclose(metrics["shifted_reward"], 0.75 - 0.3, rtol=1e-5)
    np.testing.assert_allclose(metrics["policy_log_prob"], -1.5, rtol=1e-5)
    np.testing.assert_allclose(metrics["learned_baseline"], 0.3, rtol=1e-5)
    np.testing.assert_allclose(
        metrics["baseline_penalty"],
        0.001 * (0.75 * (0.7 * 0.7) + 0.25 * (0.3 * 0.3)),
        rtol=1e-5)
    np.testing.assert_allclose(
        metrics["reinforce_term"], (0 * 0.7 + 1 * 0.7 + 2 * 0.7 + 3 * -0.3) / 4,
        rtol=1e-5)

    np.testing.assert_allclose(
        loss,
        metrics["reinforce_term"] + metrics["baseline_penalty"],
        rtol=1e-5)

    (output_logits, targets, valid_mask, loss,
     metrics) = train_edge_supervision_lib.sample_loss_fn(
         mock_model, (example, jax.random.PRNGKey(0)),
         target_edge_index=0,
         num_edge_types=3,
         num_rollouts=20,
         leave_one_out_baseline=True)

    self.assertEqual(output_logits.shape, (5, 5))
    self.assertEqual(targets.shape, (5, 5))
    self.assertEqual(valid_mask.shape, (5, 5))

    np.testing.assert_allclose(metrics["reward"], 0.75, rtol=1e-5)
    np.testing.assert_allclose(metrics["shifted_reward"], 0, rtol=1e-5)
    np.testing.assert_allclose(metrics["learned_baseline"], 0.3, rtol=1e-5)
    np.testing.assert_allclose(metrics["baseline_penalty"], 0.0, rtol=1e-5)


if __name__ == "__main__":
  absltest.main()
