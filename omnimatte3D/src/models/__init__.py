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

"""File containing function to create a model."""

from typing import Tuple

from clu import parameter_overview
import flax.linen as nn
from flax.training import train_state
import jax
from jax import random
import ml_collections
import optax


from omnimatte3D.src.models import ldi
from omnimatte3D.src.utils import train_utils


MODEL_DICT = {
    'ldi': ldi.create_model,
}


def create_model(config, rng, example_batch):
  """Create and intialize the model.

  Args:
    config: Configuration for model
    rng: JAX PRNG Key
    example_batch: An example batch

  Returns:
    The model and intial parameters
  """
  example_batch = train_utils.prepare_example_batch(example_batch)

  key0, rng = random.split(rng, 2)
  model, variables, metric_collector = MODEL_DICT[config.model.name](
      key0, example_batch, config
  )

  return model, variables, metric_collector


def create_train_state(
    config, rng, learning_rate_fn, example_batch
):
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    learning_rate_fn: learning rate function
    example_batch: for model intialization

  Returns:
    The initialized TrainState with the optimizer.
  """
  model, variables, metric_collector = create_model(config, rng, example_batch)
  params = variables['params']
  parameter_overview.log_parameter_overview(params)
  tx = train_utils.create_optimizer(config, learning_rate_fn)

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=variables['params'],
      tx=tx,
  )
  return model, state, metric_collector
