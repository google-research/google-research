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

"""Tests for checkpoints."""

import os

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from sparse_mixers import checkpoints

# Parse absl flags test_tmpdir.
jax.config.parse_flags_with_absl()

NUM_LOCAL_DEVICES = 8


class CheckpointsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.ckpt_dir = self.create_tempdir().full_path
    replicated = np.ones((NUM_LOCAL_DEVICES, 10))
    sharded = np.arange(NUM_LOCAL_DEVICES * 4).reshape((NUM_LOCAL_DEVICES, 4))
    params = [
        # Parameters that would be stored in process 0.
        {
            "sharded": 1.0 * sharded,
            "replicated": replicated,
        },
        # Parameters that would be stored in process 1.
        {
            "sharded": 2.0 * sharded,
            "replicated": replicated,
        },
    ]
    self.params = jax.tree.map(jnp.asarray, params)

  def test_save_and_restore_parameters(self):
    # Process 1 only saves sharded parameters, not the replicated ones.
    replicated_filepath, sharded_filepath = checkpoints.save_checkpoint(
        self.ckpt_dir,
        self.params[1],
        sharded_match_fn=lambda name: "sharded" in name,
        step=1,
        process_id=1,
        process_count=2)
    self.assertFalse(os.path.exists(replicated_filepath))
    self.assertTrue(os.path.exists(sharded_filepath))

    # Process 0 saves its own parameters and the replicated ones.
    replicated_filepath, sharded_filepath = checkpoints.save_checkpoint(
        self.ckpt_dir,
        self.params[0],
        sharded_match_fn=lambda name: "sharded" in name,
        step=1,
        process_id=0,
        process_count=2)
    self.assertTrue(os.path.exists(replicated_filepath))
    self.assertTrue(os.path.exists(sharded_filepath))

    # Restore all parameters for both processes.
    restored_params = []
    for process_id in [0, 1]:
      restored_params.append(
          checkpoints.restore_checkpoint(
              self.ckpt_dir, {
                  "sharded": 0,
                  "replicated": 0
              },
              sharded_match_fn=lambda name: "sharded" in name,
              process_id=process_id,
              process_count=2))
    restored_params = jax.tree.map(np.asarray, restored_params)

    # Ensure that the restored parameters match the original values.
    np.testing.assert_allclose(restored_params[0]["sharded"],
                               self.params[0]["sharded"])
    np.testing.assert_allclose(restored_params[1]["sharded"],
                               self.params[1]["sharded"])
    np.testing.assert_allclose(restored_params[0]["replicated"],
                               self.params[0]["replicated"])
    np.testing.assert_allclose(restored_params[1]["replicated"],
                               self.params[1]["replicated"])

  def test_save_and_restore_parameters_keep_2(self):
    # Save checkpoints for 3 steps, but we only keep 2 checkpoints.
    for step in [1, 2, 3]:
      for process_id in [0, 1]:
        # Update parameters, simulating SGD steps.
        for name, incr in zip(["sharded", "replicated"], [1.0, -1.0]):
          self.params[process_id][name] = self.params[process_id][name] + incr
        # Save checkpoint for current step and process.
        checkpoints.save_checkpoint(
            self.ckpt_dir,
            self.params[process_id],
            sharded_match_fn=lambda name: "sharded" in name,
            step=step,
            keep=2,
            process_id=process_id,
            process_count=2)

    # Check that checkpoints for step=1 are gone.
    self.assertFalse(
        os.path.exists(os.path.join(self.ckpt_dir, "checkpoint_replicated_1")))
    self.assertFalse(
        os.path.exists(
            os.path.join(self.ckpt_dir, "checkpoint_sharded_1-00000-of-00002")))
    self.assertFalse(
        os.path.exists(
            os.path.join(self.ckpt_dir, "checkpoint_sharded_1-00001-of-00002")))

    # Restore all parameters for both processes. This should restore the last
    # checkpoint, corresponding to step=3.
    restored_params = []
    for process_id in [0, 1]:
      restored_params.append(
          checkpoints.restore_checkpoint(
              self.ckpt_dir, {
                  "sharded": 0,
                  "replicated": 0
              },
              sharded_match_fn=lambda name: "sharded" in name,
              process_id=process_id,
              process_count=2))
    restored_params = jax.tree.map(np.asarray, restored_params)

    # Ensure that the restored parameters match the last values.
    np.testing.assert_allclose(restored_params[0]["sharded"],
                               self.params[0]["sharded"])
    np.testing.assert_allclose(restored_params[1]["sharded"],
                               self.params[1]["sharded"])
    np.testing.assert_allclose(restored_params[0]["replicated"],
                               self.params[0]["replicated"])
    np.testing.assert_allclose(restored_params[1]["replicated"],
                               self.params[1]["replicated"])

  def test_save_and_restore_bfloat16(self):
    params = {
        "a": jax.random.normal(jax.random.PRNGKey(0), (4, 3, 2)),
        "b": jax.random.normal(jax.random.PRNGKey(1), (1, 2)),
    }
    params = jax.tree.map(lambda x: jnp.asarray(x, jnp.bfloat16), params)
    sharded_match_fn = lambda name: name == "b"
    checkpoints.save_checkpoint(
        self.ckpt_dir,
        params,
        sharded_match_fn=sharded_match_fn,
        step=0,
        process_id=0,
        process_count=1)
    restored_params = checkpoints.restore_checkpoint(
        self.ckpt_dir,
        {
            "a": 1,
            "b": 2
        },  # Some pytree with same structure as params.
        sharded_match_fn=sharded_match_fn,
        process_id=0,
        process_count=1)
    self.assertEqual(restored_params["a"].dtype, jnp.bfloat16)
    self.assertEqual(restored_params["b"].dtype, jnp.bfloat16)
    # Bitcast bfloat16 to int16 and compare values. Numpy does not have bfloat16
    # type.
    bitcast_int16_fn = lambda x: jax.lax.bitcast_convert_type(x, jnp.int16)
    params_int16 = jax.tree.map(bitcast_int16_fn, params)
    restored_params_int16 = jax.tree.map(bitcast_int16_fn, restored_params)
    np.testing.assert_array_equal(restored_params_int16["a"], params_int16["a"])
    np.testing.assert_array_equal(restored_params_int16["b"], params_int16["b"])


if __name__ == "__main__":
  absltest.main()
