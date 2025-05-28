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

"""Run DSM on Atari."""

import functools
import operator
from typing import Optional

from absl import flags
from absl import logging
import acme
from acme.agents.jax import dqn
from acme.jax import experiments
from acme.jax import networks as acme_networks
import chex
from etils import epath
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
from orbax import checkpoint
from pvn import networks
from pvn import offline
from pvn.datasets import atari
from pvn.utils import acme_utils
from pvn.utils import config_utils
from pvn.utils import tree_utils

FLAGS = flags.FLAGS
USE_TRAINABLE_ENCODER = flags.DEFINE_bool(
    'use_trainable_encoder', False, 'Whether to use trainable encoder.'
)


class _DistributionalQModule(nn.Module):
  """C51-style head."""

  num_actions: int
  num_atoms: int = 51
  v_min: float = -10.0
  v_max: float = 10.0
  num_hidden_layers: int = 0
  hidden_layer_width: int = 512

  def setup(self):
    self._atoms = jnp.linspace(self.v_min, self.v_max, self.num_atoms)

  @nn.compact
  def __call__(
      self,
      x,
      is_training = False,
      key = None,
  ):
    del key  # Unused.

    kernel_initializer = nn.initializers.xavier_uniform()

    for _ in range(self.num_hidden_layers):
      x = nn.Dense(self.hidden_layer_width, kernel_init=kernel_initializer)(x)
      x = nn.relu(x)

    q_logits = nn.Dense(
        self.num_actions * self.num_atoms, kernel_init=kernel_initializer
    )(x)
    q_logits = jnp.reshape(q_logits, (-1, self.num_actions, self.num_atoms))
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * self._atoms, axis=-1)
    q_values = jax.lax.stop_gradient(q_values)
    if is_training:
      return q_values, q_logits, self._atoms  # pytype: disable=bad-return-type  # jax-ndarray
    return q_values


class _QModule(nn.Module):
  """A simple Q network."""

  num_actions: int
  num_hidden_layers: int = 0
  hidden_layer_width: int = 512

  @nn.compact
  def __call__(
      self,
      x,
      is_training = False,
      key = None,
  ):
    del is_training, key  # Unused.

    kernel_initializer = nn.initializers.xavier_uniform()

    for _ in range(self.num_hidden_layers):
      x = nn.Dense(self.hidden_layer_width, kernel_init=kernel_initializer)(x)
      x = nn.relu(x)

    return nn.Dense(self.num_actions, kernel_init=kernel_initializer)(x)


def create_q_network_with_encoder(
    num_features,
    encoder,
    encoder_params,
    distributional = False,
    num_hidden_layers = 0,
    hidden_layer_width = 512,
):
  """Creates a linear Q-function using a trainable encoder."""

  def network_factory(spec):
    if distributional:
      q_network = _DistributionalQModule(
          spec.actions.num_values,
          num_hidden_layers=num_hidden_layers,
          hidden_layer_width=hidden_layer_width,
      )
    else:
      q_network = _QModule(
          spec.actions.num_values,
          num_hidden_layers=num_hidden_layers,
          hidden_layer_width=hidden_layer_width,
      )
    example_state = jnp.zeros((num_features,), dtype=jnp.float32)

    def init_fn(
        rng_key,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
        unused_example_observation = None,
    ):
      del unused_example_observation
      q_network_params = q_network.init(rng_key, example_state)
      return [encoder_params, q_network_params]

    def apply_fn(
        params,
        observation,
        is_training = True,
        key = None,
    ):
      del is_training, key  # Unused
      if len(observation.shape) == 4:
        observation = jnp.expand_dims(observation, axis=1)
      encoder_params, q_network_params = params

      def q_value(observation):
        # Encoder can only handle observations of shape (1, 84, 84, 4).
        state_representation = encoder.apply(encoder_params, observation)
        return q_network.apply(q_network_params, state_representation)

      return jax.vmap(q_value)(observation)

    return dqn.DQNNetworks(
        policy_network=acme_networks.TypedFeedForwardNetwork(
            init=init_fn, apply=apply_fn
        )
    )

  return network_factory


def create_q_network_without_encoder(
    num_features,
    distributional = False,
    num_hidden_layers = 0,
    hidden_layer_width = 512,
):
  """Creates a linear Q-function on top of fixed representations."""

  def network_factory(spec):
    example_state = jnp.zeros((num_features,), dtype=jnp.float32)
    if distributional:
      q_network = _DistributionalQModule(
          spec.actions.num_values,
          num_hidden_layers=num_hidden_layers,
          hidden_layer_width=hidden_layer_width,
      )
    else:
      q_network = _QModule(
          spec.actions.num_values,
          num_hidden_layers=num_hidden_layers,
          hidden_layer_width=hidden_layer_width,
      )

    # This function is necessary since some pieces pass in
    # an RNG key and an observation, while some pass in just the RNG key.
    def network_init(
        rng_key,  # pytype: disable=annotation-type-mismatch  # jax-ndarray
        unused_example_observation = None,
    ):
      return q_network.init(rng_key, example_state)

    return dqn.DQNNetworks(
        policy_network=acme_networks.TypedFeedForwardNetwork(
            init=network_init, apply=q_network.apply
        )
    )

  return network_factory


def restore_encoder_params(
    checkpoint_dir,
    *,
    config
):  # pyformat: disable
  """Restore encoder parameters from checkpoint."""
  checkpointers = {
      'train': checkpoint.Checkpointer(checkpoint.PyTreeCheckpointHandler())
  }
  ckpt_manager = checkpoint.CheckpointManager(
      checkpoint_dir, checkpointers=checkpointers
  )

  latest_step = ckpt_manager.latest_step()
  if latest_step is None:
    raise RuntimeError(f'Unable to find checkpoint in {checkpoint_dir}')
  logging.info('Restoring checkpoint %d', latest_step)

  element_spec = atari.element_spec(config.game, config.offline.batch_size)

  state_shape = jax.eval_shape(
      functools.partial(
          offline.create_train_state_with_optional_mesh,
          config=config,
          rng=jax.random.PRNGKey(0),
          mesh=None,
      ),
      element_spec,
  )
  state_shape = tree_utils.filter_empty_nodes(state_shape, state_shape)
  state_restore_args = jax.tree_util.tree_map(
      lambda _: checkpoint.ArrayRestoreArgs(), state_shape
  )

  restored_state = ckpt_manager.restore(
      latest_step,
      items={'train': state_shape},
      restore_kwargs={'train': {'restore_args': state_restore_args}},
  )
  restored_state = operator.itemgetter('train')(restored_state)
  encoder_params = restored_state['params']['params']['encoder']
  encoder_params = flax.core.FrozenDict({'params': encoder_params})

  return encoder_params


def train(
    workdir,
    checkpoint_dir,
    *,
    config,
):
  """Trains an agent with a fixed encoder online."""

  game = f'{config.game}NoFrameskip-v4'
  use_distributional_rl = config.online.use_distributional_rl

  # Although the encoder runs on GPU, reverb will automatically convert
  # device arrays to arrays in RAM, so we shouldn't have device memory issues.
  encoder = config_utils.get_configurable(
      networks,
      config.encoder,
      name='encoder',
      dtype=jnp.float32,
      param_dtype=jnp.float32,
  )
  # Load encoder params.
  encoder_params = restore_encoder_params(checkpoint_dir, config=config)
  num_features = encoder.num_features

  if USE_TRAINABLE_ENCODER.value:
    encoder_wrapper = None
    network_factory = create_q_network_with_encoder(  # pytype: disable=wrong-arg-types  # numpy-scalars
        num_features,
        encoder,
        encoder_params,
        distributional=use_distributional_rl,
        num_hidden_layers=config.online.num_hidden_layers,
        hidden_layer_width=config.online.hidden_layer_width,
    )
  else:
    # Create an atari environment that uses the encoder.
    encoder_wrapper = functools.partial(
        acme_utils.EncoderWrapper,
        network_def=encoder,
        params=encoder_params,
        output_dim=num_features,
    )
    network_factory = create_q_network_without_encoder(
        num_features=num_features,
        distributional=use_distributional_rl,
        num_hidden_layers=config.online.num_hidden_layers,
        hidden_layer_width=config.online.hidden_layer_width,
    )
  environment_factory = functools.partial(
      acme_utils.make_environment, level=game, encoder_wrapper=encoder_wrapper
  )

  agent = config.online.agent
  assert agent == 'dqn', 'Only DQN agent is supported at this time'
  agent_config = dqn.DQNConfig(**config.online.dqn)

  if use_distributional_rl:
    # Use C51 loss with double Q-learning.
    agent_loss_fn = dqn.PrioritizedCategoricalDoubleQLearning
  else:
    # Use double Q-learning.
    agent_loss_fn = dqn.PrioritizedDoubleQLearning

  if config.online.use_prioritized_replay:
    # This exponent is used by Dopamine Rainbow.
    agent_config.importance_sampling_exponent = 0.5
    agent_config.priority_exponent = 0.5
  else:
    agent_config.importance_sampling_exponent = 0.0
    agent_config.priority_exponent = 0.0

  loss_fn = agent_loss_fn(
      importance_sampling_exponent=agent_config.importance_sampling_exponent,
      discount=config.discount,
      max_abs_reward=1.0,
  )

  dqn_builder = dqn.DQNBuilder(
      config=agent_config,
      loss_fn=loss_fn,
      # Override actor backend so inference takes place on the default device.
      actor_backend=jax.local_devices()[0].platform,
  )

  if config.seed is None:
    seed = np.random.SeedSequence().generate_state(1).item()
  else:
    seed = config.seed

  checkpointing_config = experiments.CheckpointingConfig(
      max_to_keep=2,  # Just in case we need to fix a checkpoint.
      directory=str(workdir),
      add_uid=True,
      replay_checkpointing_time_delta_minutes=30,  # Might take a long time.
      time_delta_minutes=30,
  )

  experiment_config = experiments.ExperimentConfig(
      builder=dqn_builder,
      environment_factory=environment_factory,
      network_factory=network_factory,
      evaluator_factories=[],
      seed=seed,
      max_num_actor_steps=config.online.num_steps,
      checkpointing=checkpointing_config,
  )
  experiments.run_experiment(
      experiment_config,
      eval_every=config.online.num_steps,
      num_eval_episodes=100,
  )
