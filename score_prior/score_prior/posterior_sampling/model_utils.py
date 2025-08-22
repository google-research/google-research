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

"""Utility functions for DPI models."""
from typing import Any, Tuple

import flax
import jax
from jaxtyping import PyTree
import optax

from score_prior.posterior_sampling import realnvp_model


@flax.struct.dataclass
class State:
  """Training state."""
  step: int
  opt_state: optax.OptState
  params: PyTree
  model_state: PyTree
  rng: jax.Array
  data_weight: float = 1.
  prior_weight: float = 1.
  entropy_weight: float = 1.


def get_model_and_init_params(config, image_flux=1., train=True):
  """Construct generator and initialize model state and params.

  Args:
    config: `ml_collections.ConfigDict`.
    image_flux: Optional image flux for initializing softplus layer.
    train: Whether model is in training model or not.

  Returns:
    model: The Flax generator.
    init_model_state: A dict containing all mutable states of the model,
      such as `batch_stats`.
    init_params: A dict containing all initial parameters of the model.
  """
  image_dim = config.data.image_size**2 * config.data.num_channels
  if config.model.bijector.lower() == 'glow':
    raise NotImplementedError
  elif config.model.bijector.lower() == 'realnvp':
    orders, reverse_orders = realnvp_model.get_orders(
        image_dim, config.model.n_flow)
    model = realnvp_model.RealNVP(
        out_dim=image_dim,
        n_flow=config.model.n_flow,
        orders=orders,
        reverse_orders=reverse_orders,
        include_softplus=config.model.include_softplus,
        init_softplus_log_scale=image_flux / (0.8 * image_dim),
        batch_norm=config.model.batch_norm,
        init_std=config.model.init_std,
        train=train)

    # Initialize params and model state.
    init_rng = jax.random.PRNGKey(config.seed)
    z = jax.random.normal(init_rng, (config.training.batch_size, image_dim))
    variables = model.init(init_rng, z, reverse=True)
    # `variables` is a `flax.FrozenDict`. It is immutable and respects
    # functional programming.
    init_model_state, init_params = flax.core.pop(variables, 'params')
  else:
    raise ValueError(f'Unrecognized bijector: {config.model.bijector}')
  return model, init_model_state, init_params


def get_sampling_fn(model,
                    params,
                    states,
                    train = False):
  """Create a function to give samples from DPI.

  Args:
    model: A Flax module representing the architecture of the generator.
    params: A dict containing all trainable parameters.
    states: A dict containing all mutable states.
    train: `True` for training and `False` for evaluation.

  Returns:
    A function that returns samples from the generator.
  """

  def sample_fn(rng, shape):
    # Sample latent.
    z_dim = shape[1] * shape[2] * shape[3]
    z = jax.random.normal(rng, (shape[0], z_dim))

    variables = {'params': params, **states}
    if train:
      (x, logdet), new_states = model.apply(
          variables, z, reverse=True, mutable=list(states.keys()))
    else:
      x, logdet = model.apply(variables, z, reverse=True, mutable=False)
      new_states = states
    samples = x.reshape(shape)
    return (samples, logdet), new_states

  return sample_fn
