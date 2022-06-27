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

"""Tests for gfsa.training.train_var_misuse_lib."""

from absl.testing import absltest
import flax
import jax
import jax.numpy as jnp
import numpy as np
from gfsa import sparse_operator
from gfsa.datasets.var_misuse import example_definition
from gfsa.model import side_outputs
from gfsa.training import train_var_misuse_lib


class TrainVarMisuseLibTest(absltest.TestCase):

  def test_loss_fn(self):
    mock_example = example_definition.VarMisuseExample(
        input_graph=None,
        bug_node_index=2,
        repair_node_mask=jnp.array([0., 1., 1., 0.5, 0.]),
        candidate_node_mask=None,
        unique_candidate_operator=sparse_operator.SparseCoordOperator(
            input_indices=jnp.array([0, 1, 2, 3, 3, 4])[:, None],
            output_indices=jnp.array([0, 1, 1, 1, 2, 3])[:, None],
            values=jnp.array([1, 1, 1, 0.5, 0.5, 1])),
        repair_id=1)

    mock_metadata = object()

    @flax.deprecated.nn.module
    def mock_model_def(example, metadata):
      # Check that we get the right inputs.
      self.assertIs(example, mock_example)
      self.assertIs(metadata, mock_metadata)

      # Register a side output
      side_outputs.SideOutput(jnp.array(.1234), name="test_penalty")

      # Make sure we can generate an rng key with flax.
      _ = flax.deprecated.nn.make_rng()

      return jnp.log(
          jnp.array([
              [.0, .0, .0, .0, .0],
              [.1, .0, .0, .2, .0],
              [.0, .1, .2, .2, .1],  # <- This row is the "correct" bug index.
              [.0, .0, .0, .0, .0],
              [.1, .0, .0, .0, .0],
          ]))

    with flax.deprecated.nn.stochastic(jax.random.PRNGKey(0)):
      _, params = mock_model_def.init(
          jax.random.PRNGKey(0), mock_example, mock_metadata)

    mock_model = flax.deprecated.nn.Model(mock_model_def, params)

    loss, metrics = train_var_misuse_lib.loss_fn(
        mock_model, (mock_example, jax.random.PRNGKey(0)),
        mock_metadata,
        regularization_weights={"penalty": 2})

    np.testing.assert_allclose(metrics["nll/joint"], -np.log(0.4), atol=1e-7)
    np.testing.assert_allclose(metrics["side/test_penalty"], .1234, atol=1e-7)
    np.testing.assert_allclose(loss, -np.log(0.4) + 2 * .1234, atol=1e-7)

    np.testing.assert_allclose(
        metrics["nll/marginal_bug"], -np.log(0.6), atol=1e-7)
    np.testing.assert_allclose(
        metrics["nll/marginal_repair"], -np.log(0.5), atol=1e-7)
    np.testing.assert_allclose(
        metrics["nll/repair_given_bug"], -np.log(0.4 / 0.6), atol=1e-7)
    np.testing.assert_allclose(
        metrics["nll/bug_given_repair"], -np.log(0.4 / 0.5), atol=1e-7)
    np.testing.assert_allclose(metrics["inaccuracy/classification_overall"], 0)
    np.testing.assert_allclose(
        metrics["inaccuracy/classification_given_nobug"].numerator, 0)
    np.testing.assert_allclose(
        metrics["inaccuracy/classification_given_nobug"].denominator, 0)
    np.testing.assert_allclose(
        metrics["inaccuracy/classification_given_bug"].numerator, 0)
    np.testing.assert_allclose(
        metrics["inaccuracy/classification_given_bug"].denominator, 1)


if __name__ == "__main__":
  absltest.main()
