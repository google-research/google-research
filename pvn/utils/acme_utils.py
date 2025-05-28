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

"""Utils for running DSM online experiments in Acme."""

import dataclasses
import functools
from typing import Callable, Dict, NamedTuple, Optional, Protocol, Tuple, Sequence, Mapping, Union, Any

from absl import flags
from acme import wrappers
from acme.jax import networks as acme_networks
from acme.utils import loggers
from acme.wrappers import atari_wrapper_dopamine
import dm_env
from dm_env import specs
import flax
from flax import linen as nn
import gym
import jax
from jax import numpy as jnp
import numpy as np
import reverb
import rlax

FLAGS = flags.FLAGS

PrimaryKeyList = Sequence[Tuple[str, Union[int, str]]]


class ReverbUpdate(NamedTuple):
  """Tuple for updating reverb priority information.

  Note: Copied from the Acme codebase since visibility isn't public.
  """

  keys: jnp.ndarray
  priorities: jnp.ndarray


class LossExtra(NamedTuple):
  """Extra information that is returned along with loss value.

  Note: Copied from the Acme codebase since visibility isn't public.
  """

  metrics: Dict[str, jax.Array]
  reverb_update: Optional[ReverbUpdate] = None


class LossFn(Protocol):
  """A LossFn calculates a loss on a single batch of data.

  Note: Copied from the Acme codebase since visibility isn't public.
  """

  def __call__(
      self,
      network: acme_networks.FeedForwardNetwork,
      params: acme_networks.Params,
      target_params: acme_networks.Params,
      batch: reverb.ReplaySample,  # pyformat: disable
      key: acme_networks.PRNGKey,
  ) -> Tuple[jax.Array, LossExtra]:
    """Calculates a loss on a single batch of data."""


@dataclasses.dataclass
class QLearning(LossFn):
  """Deep q learning.

  This matches the original DQN loss: https://arxiv.org/abs/1312.5602.
  It differs by two aspects that improve it on the optimization side
    - it uses Adam intead of RMSProp as an optimizer
    - it uses a square loss instead of the Huber one.

  Note: Copied from the Acme codebase since visibility isn't public.
  """

  discount: float = 0.99
  max_abs_reward: float = 1.0

  def __call__(
      self,
      network: acme_networks.FeedForwardNetwork,
      params: acme_networks.Params,
      target_params: acme_networks.Params,
      batch: reverb.ReplaySample,
      key: acme_networks.PRNGKey,
  ) -> Tuple[jax.Array, LossExtra]:
    """Calculate a loss on a single batch of data."""
    del key
    transitions = batch.data

    # Forward pass.
    q_tm1 = network.apply(params, transitions.observation)
    q_t = network.apply(target_params, transitions.next_observation)

    # Cast and clip rewards.
    d_t = (transitions.discount * self.discount).astype(jnp.float32)
    r_t = jnp.clip(
        transitions.reward, -self.max_abs_reward, self.max_abs_reward
    ).astype(jnp.float32)

    # Compute Q-learning TD-error.
    batch_error = jax.vmap(rlax.q_learning)
    td_error = batch_error(q_tm1, transitions.action, r_t, d_t, q_t)
    batch_loss = jnp.square(td_error)

    loss = jnp.mean(batch_loss)
    extra = LossExtra(metrics={})
    return loss, extra


class EncoderWrapper(wrappers.EnvironmentWrapper):
  """An Atari wrapper that applies an encoder to each observation."""

  def __init__(
      self,
      environment: dm_env.Environment,
      *,
      network_def: nn.Module,
      params: flax.core.FrozenDict,
      output_dim: int,
  ):
    super().__init__(environment)
    self._network_def = network_def
    self._params = params
    self._output_dim = output_dim

    self._apply_encoder = jax.jit(
        functools.partial(self._network_def.apply, params)
    )

  def reset(self) -> dm_env.TimeStep:
    timestep = super().reset()
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward,
        discount=timestep.discount,
        observation=self._apply_encoder(timestep.observation),
    )

  def step(self, action) -> dm_env.TimeStep:
    timestep = super().step(action)
    return dm_env.TimeStep(
        step_type=timestep.step_type,
        reward=timestep.reward,
        discount=timestep.discount,
        observation=self._apply_encoder(timestep.observation),
    )

  def observation_spec(self):
    return specs.Array(shape=(self._output_dim,), dtype=np.float32)


def make_default_logger(
    module: str,
    label: str,
    time_delta: float = 1.0,
    asynchronous: bool = True,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[
        Callable[[Mapping[str, Any]], str]
    ] = loggers.to_numpy,
    steps_key: str = 'steps',
) -> loggers.Logger:
  """Make a default Acme logger."""
  del module  # Unused.
  del steps_key  # Unused.
  print_fn = print if print_fn is None else print_fn

  terminal_logger = loggers.TerminalLogger(label, print_fn=print_fn)

  # We will send logs to each of these targets.
  our_loggers = [terminal_logger]

  # Dispatch to all writers and filter Nones and by time.
  our_logger = loggers.Dispatcher(our_loggers, serialize_fn)
  our_logger = loggers.NoneFilter(our_logger)
  if asynchronous:
    our_logger = loggers.AsyncLogger(our_logger)
  our_logger = loggers.TimeFilter(our_logger, time_delta)

  return our_logger


def make_environment(
    seed: int,
    *,
    level: str,
    encoder_wrapper: Optional[
        Callable[[dm_env.Environment], wrappers.EnvironmentWrapper]
    ] = None,
) -> dm_env.Environment:
  """Loads the Atari environment."""
  env = gym.make(level, full_action_space=False)
  env.seed(seed)

  # Always use episodes of 108k frames as this is standard, matching the paper.
  max_episode_len = 108_000
  wrapper_list = [
      wrappers.GymAtariAdapter,
      functools.partial(
          atari_wrapper_dopamine.AtariWrapperDopamine,
          to_float=True,
          max_episode_len=max_episode_len,
          zero_discount_on_life_loss=False,
      ),
  ]
  if encoder_wrapper:
    wrapper_list.append(encoder_wrapper)
  wrapper_list.append(wrappers.SinglePrecisionWrapper)

  return wrappers.wrap_all(env, wrapper_list)
