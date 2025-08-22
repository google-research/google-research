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

"""Test NDP Model."""

from absl import logging
from absl.testing import absltest
import flax.jax_utils as flax_utils
import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from hct import ndp
from hct.common import utils


class NDPTest(absltest.TestCase):
  """Test the NDP model."""

  def setUp(self):
    super().setUp()

    self.num_devices = jax.local_device_count()

    # Create a small model
    loss = lambda u_true, u_pred: jnp.sum(jnp.square(u_true - u_pred))
    self.action_dim = 2
    self.num_actions = 4
    self.ndp_model = ndp.NDP(action_dim=self.action_dim,
                             num_actions=self.num_actions,
                             loss_fnc=loss,
                             activation=nn.relu,
                             zs_dim=32,
                             zs_width=64,
                             zo_dim=16)

    # Initialize model
    self.model_prng = hk.PRNGSequence(jax.random.PRNGKey(123456))
    self.data_prng = hk.PRNGSequence(jax.random.PRNGKey(654321))

    self.sample_images = jax.random.uniform(next(self.data_prng),
                                            shape=(2, 64, 64, 3))
    self.sample_hf_obs = jax.random.normal(next(self.data_prng), shape=(2, 16))
    self.sample_u_true = jax.random.normal(
        next(self.data_prng), shape=(2, self.num_actions, self.action_dim))

    params_init = self.ndp_model.init(next(self.model_prng),
                                      self.sample_images, self.sample_hf_obs)

    self.params = params_init

  def test_flow_computation(self):
    """Compute flows."""

    # Compute for 'prediction' all at once
    u_pred = self.ndp_model.apply(self.params, self.sample_images,
                                  self.sample_hf_obs)
    aug_pred = self.ndp_model.apply(
        self.params, self.sample_images, self.sample_hf_obs, self.sample_u_true,
        method=self.ndp_model.compute_augmented_flow)

    self.assertEqual(u_pred.shape, (2, self.num_actions, self.action_dim))
    self.assertEqual(aug_pred[0].shape, (2, self.num_actions, self.action_dim))
    self.assertEqual(aug_pred[1].shape, (2,))

    self.assertTrue(np.allclose(u_pred, aug_pred[0]))

    # Compute 'densely'
    pred_times = jnp.linspace(0, 1 - (1/self.num_actions), 50)
    u_traj = self.ndp_model.apply(self.params, self.sample_images,
                                  self.sample_hf_obs, pred_times,
                                  method=self.ndp_model.compute_ndp_flow)
    self.assertEqual(u_traj.shape, (2, 50, self.action_dim))

    # Compute 'step-by-step'
    re_init, step_fwd = self.ndp_model.step_functions

    # First: the 0th step
    ndp_state, ndp_args = re_init(self.params, self.sample_images[0],
                                  self.sample_hf_obs[0])
    self.assertTrue(np.allclose(ndp_state[:self.action_dim], u_pred[0][0]))

    # Next: the rest
    tau = 0.
    for i in range(1, self.num_actions):
      ndp_state, tau = step_fwd(self.params, ndp_state, tau, ndp_args)
      self.assertTrue(np.allclose(ndp_state[:self.action_dim], u_pred[0][i]))

  def test_training(self):
    """Create train state and try training for one-step."""

    state = ndp.create_ndp_train_state(
        self.ndp_model, next(self.model_prng), 1e-3, 1e-4,
        self.sample_images, self.sample_hf_obs)
    logging.info('# of params: %d', utils.param_count(state.params))

    # Replicate state across devices
    state = flax_utils.replicate(state)

    # Create a batch of data, split across devices
    batch_images = utils.split_across_devices(
        self.num_devices,
        jax.random.uniform(next(self.data_prng), shape=(2, 64, 64, 3)))
    batch_hf_obs = utils.split_across_devices(
        self.num_devices,
        jax.random.normal(next(self.data_prng), shape=(2, 16)))
    batch_u_true = utils.split_across_devices(
        self.num_devices,
        jax.random.normal(next(self.data_prng), shape=(2, 4, 2)))

    # Do train step
    loss, state = ndp.optimize_ndp(state,
                                   batch_images,
                                   batch_hf_obs,
                                   batch_u_true)

    # Check finiteness
    self.assertTrue(jnp.isfinite(loss))
    self.assertTrue(
        utils.check_params_finite(flax_utils.unreplicate(state).params))


if __name__ == '__main__':
  absltest.main()
