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

"""Test that DPI training for posterior sampling works."""
import unittest

import diffrax
import flax
import jax
import jax.numpy as jnp
import optax
from score_sde import sde_lib

from score_prior import inference_utils
from score_prior import probability_flow
from score_prior.configs import posterior_sampling_config
from score_prior.posterior_sampling import losses
from score_prior.posterior_sampling import model_utils

# Parameters for shape of dummy data:
_IMAGE_SIZE = 8
_N_CHANNELS = 1
_IMAGE_SHAPE = (_IMAGE_SIZE, _IMAGE_SIZE, _N_CHANNELS)


class PosteriorSamplingTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    # Set up config for DPI.
    config = posterior_sampling_config.get_config()
    config.model.n_flow = 1
    config.training.batch_size = 2
    config.training.n_iters = 10
    config.likelihood.likelihood = 'Denoising'
    config.data.image_size = _IMAGE_SIZE
    config.data.num_channels = _N_CHANNELS
    self.config = config

    # Dummy score function.
    self.score_fn = lambda x, t_batch: x / t_batch[0]**2
    self.sde = sde_lib.VPSDE()

  def test_train_dpi(self):
    """This test exercises the workflow for training DPI."""
    config = self.config
    rng = jax.random.PRNGKey(0)

    # Set up `ProbabilityFlow`.
    sde = self.sde
    prob_flow = probability_flow.ProbabilityFlow(
        self.sde,
        self.score_fn,
        solver=diffrax.Heun(),
        stepsize_controller=diffrax.ConstantStepSize(),
        adjoint=diffrax.BacksolveAdjoint(),
        n_trace_estimates=1)

    # Get measurement.
    likelihood = inference_utils.get_likelihood(config)
    x = jax.random.normal(rng, (config.training.batch_size, *_IMAGE_SHAPE))
    y = likelihood.get_measurement(rng, x)

    # Set up RealNVP for training.
    model, model_state, params = model_utils.get_model_and_init_params(
        config, train=True)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(params)

    state = model_utils.State(
        step=0,
        opt_state=opt_state,
        model_state=model_state,
        params=params,
        data_weight=1,
        prior_weight=1,
        entropy_weight=1,
        rng=jax.random.PRNGKey(0))

    # Train step function.
    data_loss_fn = losses.get_data_loss_fn(likelihood, y)
    prior_loss_fn = losses.get_prior_loss_fn(
        config, prob_flow, t0=1e-3, t1=sde.T, dt0=0.1)
    train_step_fn = losses.get_train_step_fn(
        config, model, optimizer, data_loss_fn, prior_loss_fn)
    p_train_step_fn = jax.pmap(
        train_step_fn, axis_name='batch', donate_argnums=1)

    # Training loop.
    pstate = flax.jax_utils.replicate(state)
    for step in range(10):
      rng, *step_rngs = jax.random.split(rng, jax.local_device_count() + 1)
      step_rngs = jnp.asarray(step_rngs)
      pstate, (_, _, _, _), _ = p_train_step_fn(  # pylint:disable=unused-variable, line-too-long
          step_rngs, pstate)

    state = flax.jax_utils.unreplicate(pstate)
    self.assertEqual(state.step, step + 1)


if __name__ == '__main__':
  unittest.main()
