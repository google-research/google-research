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

"""Contrastive Value Learning builder."""
from typing import Callable, Iterator, List, Optional

import acme
from acme import adders
from acme import core
from acme import specs
from acme import types
from acme.adders import reverb as adders_reverb
from acme.agents.jax import actor_core as actor_core_lib
from acme.agents.jax import actors
from acme.agents.jax import builders
from acme.jax import networks as networks_lib
from acme.jax import variable_utils
from acme.utils import counting
from acme.utils import loggers
import optax
import reverb
from reverb import rate_limiters
import tensorflow as tf
import tree

from cvl_public import config as contrastive_config
from cvl_public import learning
from cvl_public import networks as contrastive_networks
from cvl_public import utils as contrastive_utils


class ContrastiveBuilder(builders.ActorLearnerBuilder):
  """Contrastive RL builder."""

  def __init__(
      self,
      config,
      logger_fn = lambda: None,
  ):
    """Creates a contrastive RL learner, a behavior policy and an eval actor.

    Args:
      config: a config with contrastive RL hyperparameters
      logger_fn: a logger factory for the learner
    """
    self._config = config
    self._logger_fn = logger_fn

  def make_learner(
      self,
      random_key,
      networks,
      dataset,
      replay_client = None,
      counter = None,
  ):
    # Create optimizers
    policy_optimizer = optax.adam(
        learning_rate=self._config.actor_learning_rate, eps=1e-7)
    q_optimizer = optax.adam(learning_rate=self._config.learning_rate, eps=1e-7)
    return learning.ContrastiveLearner(
        networks=networks,
        rng=random_key,
        policy_optimizer=policy_optimizer,
        q_optimizer=q_optimizer,
        iterator=dataset,
        counter=counter,
        logger=self._logger_fn(),
        config=self._config)

  def make_actor(
      self,
      random_key,
      policy_network,
      adder = None,
      variable_source = None):
    assert variable_source is not None
    actor_core = actor_core_lib.batched_feed_forward_to_actor_core(
        policy_network)
    variable_client = variable_utils.VariableClient(variable_source, 'policy',
                                                    device='cpu')
    if self._config.use_random_actor:
      ACTOR = contrastive_utils.InitiallyRandomActor  # pylint: disable=invalid-name
    else:
      ACTOR = actors.GenericActor  # pylint: disable=invalid-name
    return ACTOR(
        actor_core, random_key, variable_client, adder, backend='cpu')

  def make_replay_tables(
      self,
      environment_spec,
  ):
    """Create tables to insert data into."""
    samples_per_insert_tolerance = (
        self._config.samples_per_insert_tolerance_rate
        * self._config.samples_per_insert)
    min_replay_traj = self._config.min_replay_size  // self._config.max_episode_steps  # pylint: disable=line-too-long
    max_replay_traj = self._config.max_replay_size  // self._config.max_episode_steps  # pylint: disable=line-too-long
    error_buffer = min_replay_traj * samples_per_insert_tolerance
    limiter = rate_limiters.SampleToInsertRatio(
        min_size_to_sample=min_replay_traj,
        samples_per_insert=self._config.samples_per_insert,
        error_buffer=error_buffer)
    return [
        reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_traj,
            rate_limiter=limiter,
            signature=adders_reverb.EpisodeAdder.signature(environment_spec, {}))  # pylint: disable=line-too-long
    ]

  def make_dataset_iterator(
      self, replay_client):
    """Create a dataset iterator to use for learning/updating the agent."""
    @tf.function
    def flatten_fn(sample):
      seq_len = tf.shape(sample.data.observation)[0]
      arange = tf.range(seq_len)
      is_future_mask = tf.cast(arange[:, None] < arange[None], tf.float32)
      discount = self._config.discount ** tf.cast(arange[None] - arange[:, None], tf.float32)  # pylint: disable=line-too-long
      probs = is_future_mask * discount
      # The indexing changes the shape from [seq_len, 1] to [seq_len]
      goal_index = tf.random.categorical(logits=tf.math.log(probs),
                                         num_samples=1)[:, 0]
      state = sample.data.observation[:-1]
      next_state = sample.data.observation[1:]

      # Create the goal observations in three steps.
      # 1. Take all future states (not future goals).
      # 2. Apply obs_to_goal.
      # 3. Sample one of the future states. Note that we don't look for a goal
      # for the final state, because there are no future states.
      goal = tf.gather(sample.data.observation, goal_index[:-1])
      goal_reward = tf.gather(sample.data.reward, goal_index[:-1])
      # new_obs = tf.concat([state, goal], axis=1)
      # new_next_obs = tf.concat([next_state, goal], axis=1)
      transition = types.Transition(
          observation=state,
          action=sample.data.action[:-1],
          reward=sample.data.reward[:-1],
          discount=sample.data.discount[:-1],
          next_observation=next_state,
          extras={
              'next_action': sample.data.action[1:],
              'goal': goal,
              'goal_reward': goal_reward
          })
      # Shift for the transpose_shuffle.
      shift = tf.random.uniform((), 0, seq_len, tf.int32)
      transition = tree.map_structure(lambda t: tf.roll(t, shift, axis=0),
                                      transition)
      return transition

    if self._config.num_parallel_calls:
      num_parallel_calls = self._config.num_parallel_calls
    else:
      num_parallel_calls = tf.data.AUTOTUNE

    def _make_dataset(unused_idx):
      dataset = reverb.TrajectoryDataset.from_table_signature(
          server_address=replay_client.server_address,
          table=self._config.replay_table_name,
          max_in_flight_samples_per_worker=100)
      dataset = dataset.map(flatten_fn)
      # transpose_shuffle
      def _transpose_fn(t):
        dims = tf.range(tf.shape(tf.shape(t))[0])
        perm = tf.concat([[1, 0], dims[2:]], axis=0)
        return tf.transpose(t, perm)
      dataset = dataset.batch(self._config.batch_size, drop_remainder=True)
      dataset = dataset.map(
          lambda transition: tree.map_structure(_transpose_fn, transition))
      dataset = dataset.unbatch()
      # end transpose_shuffle

      dataset = dataset.unbatch()
      return dataset
    dataset = tf.data.Dataset.from_tensors(0).repeat()
    dataset = dataset.interleave(
        map_func=_make_dataset,
        cycle_length=num_parallel_calls,
        num_parallel_calls=num_parallel_calls,
        deterministic=False)

    dataset = dataset.batch(
        self._config.batch_size * self._config.num_sgd_steps_per_step,
        drop_remainder=True)
    @tf.function
    def add_info_fn(data):
      info = reverb.SampleInfo(key=0,
                               probability=0.0,
                               table_size=0,
                               priority=0.0,
                               times_sampled=0)
      return reverb.ReplaySample(info=info, data=data)
    dataset = dataset.map(add_info_fn, num_parallel_calls=tf.data.AUTOTUNE,
                          deterministic=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset.as_numpy_iterator()

  def make_adder(self,
                 replay_client):
    """Create an adder to record data generated by the actor/environment."""
    return adders_reverb.EpisodeAdder(
        client=replay_client,
        priority_fns={self._config.replay_table_name: None},
        max_sequence_length=self._config.max_episode_steps + 1)
