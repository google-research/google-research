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

"""File containing function to build model"""

from typing import Tuple
from clu import parameter_overview
import flax.linen as nn
from flax.training import train_state
import jax
import ml_collections
import optax

from light_field_neural_rendering.src.models import lfnr
from light_field_neural_rendering.src.models import vanilla_nlf

model_dict = {
    'vanilla_nlf': vanilla_nlf.construct_model,
    'lfnr': lfnr.construct_model,
}


def create_model(config, rng, example_batch):
  """Create and initialize the model

  Args:
    config: Configuration for model
    rng: JAX PRNG Key
    example_batch: An example batch

  Returns:
    The model and intial parameters
  """
  example_batch = prepare_example_batch(example_batch)

  key0, rng = jax.random.split(rng, 2)
  model, variables = model_dict[config.model.name](key0, example_batch, config)

  return model, variables


def prepare_example_batch(example_batch):
  """Function to get rid of extra dimension in batch due to pmap"""
  # Get rid of the pmapping dimension as initialization is done on main process
  example_batch = jax.tree_map(lambda x: x[0], example_batch)
  example_batch.target_view.rays = jax.tree_map(lambda x: x[:4],
                                                example_batch.target_view.rays)

  return example_batch


def create_train_state(
    config, rng, learning_rate_fn,
    example_batch):
  """Create and initialize the model.

  Args:
    config: Configuration for model.
    rng: JAX PRNG Key.
    learning_rate_fn: learning rate function
    example_batch: for model intialization

  Returns:
    The initialized TrainState with the optimizer.
  """
  model, variables = create_model(config, rng, example_batch)
  params = variables['params']
  parameter_overview.log_parameter_overview(params)

  state = train_state.TrainState.create(
      apply_fn=model.apply,
      params=variables['params'],
      tx=optax.adamw(
          learning_rate=learning_rate_fn,
          b1=0.9,
          b2=.98,
          eps=1e-9,
          weight_decay=config.train.weight_decay),
  )
  return model, state
