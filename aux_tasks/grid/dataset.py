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

"""Dataset."""
import collections.abc
import functools

from absl import flags
import chex
import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
import jax
import jax.numpy as jnp
from ml_collections import config_dict
import numpy as np
import reverb
import tensorflow as tf

from aux_tasks.minigrid import random_mdp
from minigrid_basics.custom_wrappers import coloring_wrapper
from minigrid_basics.custom_wrappers import mdp_wrapper

flags.DEFINE_string('reverb_address', None,
                    'The address to use to connect to the reverb server.')

FLAGS = flags.FLAGS

# pylint: disable=invalid-name


@jax.jit
def policy_evaluation(P, R, d):
  Pd = jnp.einsum('ast,sa->st', P, d)
  Rd = jnp.einsum('sa,sa->s', R, d)
  return jnp.linalg.solve(Pd, Rd)


def uniform_random_rewards(key, num_tasks, num_states,
                           num_actions):
  return jax.random.uniform(
      key, (num_tasks, num_states, num_actions), minval=-1.0, maxval=1.0)


def random_deterministic_policies(key, num_tasks,
                                  num_states,
                                  num_actions):
  policies = jax.random.randint(
      key, shape=(
          num_tasks,
          num_states,
      ), minval=0, maxval=num_actions)
  return jax.nn.one_hot(policies, num_actions)


def get_dataset(config, key, *,
                num_tasks):
  """"Get dataset."""
  if config.env.name == 'gym' or config.env.name == 'random':
    return MDPDataset(config, key, num_tasks=num_tasks)
  elif config.env.name == 'pw':
    return DistributedSampleDataset(config, key, num_tasks=num_tasks)
  else:
    raise ValueError(f'Invalid value for config.env.name: {config.env.name}')


class MDPDataset(collections.abc.Iterable):
  """Dataset."""

  def __init__(self, config, key, *,
               num_tasks):
    self.config = config
    if config.env.name == 'gym':
      env = gym.make(config.env.gym.id)
      env = RGBImgObsWrapper(env)  # Get pixel observations
      # Get tabular observation and drop the 'mission' field:
      env = mdp_wrapper.MDPWrapper(env, get_rgb=False)
      env = coloring_wrapper.ColoringWrapper(env)
    elif config.env.name == 'random':
      env = random_mdp.RandomMDP(config.env.random.num_states,
                                 config.env.random.num_actions)
    self.env = env

    P = jnp.transpose(env.transition_probs, [1, 0, 2])

    self.key, subkey = jax.random.split(key)
    if config.env.task == 'random_reward':
      R = uniform_random_rewards(subkey, num_tasks, env.num_states,
                                 env.num_actions)
      d = jnp.ones((env.num_states, env.num_actions),
                   dtype=jnp.float32) / env.num_actions
      self.aux_task_matrix = jax.vmap(
          policy_evaluation, in_axes=(None, 0, None), out_axes=(-1))(P, R, d)
    elif config.env.task == 'random_policy':
      R = env.rewards
      d = random_deterministic_policies(subkey, num_tasks, env.num_states,
                                        env.num_actions)
      self.aux_task_matrix = jax.vmap(
          policy_evaluation, in_axes=(None, None, 0), out_axes=(-1))(P, R, d)
    else:
      raise ValueError(f'Invalid value for config.env.task: {config.env.task}')

  def __iter__(self):
    num_states = self.env.num_states
    states = jax.nn.one_hot(jnp.arange(num_states), num_states)
    num_complete_batches, leftover = jnp.divmod(num_states,
                                                self.config.batch_size)
    num_batches = num_complete_batches + bool(leftover)
    assert num_batches > 0

    while True:
      self.key, subkey = jax.random.split(self.key)
      perms = jax.random.permutation(subkey, num_states)

      for i in range(num_batches):
        batch_idx_states = perms[i * self.config.batch_size:(i + 1) *
                                 self.config.batch_size]
        if len(batch_idx_states) != self.config.batch_size:
          break

        inputs = states[batch_idx_states]
        targets = self.aux_task_matrix[batch_idx_states]

        yield inputs, targets


class DistributedSampleDataset(collections.abc.Iterable):
  """DistributedTimestepDataset."""

  def __init__(self,
               config,
               key,
               *,
               num_tasks):
    self.config = config
    self.num_states = config.env.pw.num_bins**2
    assert FLAGS.reverb_address is not None, ('Must specify a reverb address '
                                              'when using Reverb')

    dataset = reverb.TimestepDataset(
        FLAGS.reverb_address,
        'successor_table',
        dtypes=(tf.float32, tf.int64, tf.float32),
        shapes=(tf.TensorShape([2]), tf.TensorShape([]),
                tf.TensorShape([self.num_states])),
        max_in_flight_samples_per_worker=3 * config.batch_size)
    dataset = dataset.batch(
        config.batch_size,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    self.dataset_iter = dataset.as_numpy_iterator()

    self.random_reward_functions = jax.random.uniform(
        key, (num_tasks, self.num_states), minval=-1.0, maxval=1.0)

  @functools.cached_property
  def aux_task_matrix(self):
    """Get empirical aux task matrix from reverb samples."""
    # Get 100k samples from reverb buffer
    samples_to_take = self.config.env.pw.samples_for_ground_truth
    dataset = reverb.TimestepDataset(
        FLAGS.reverb_address,
        'successor_table',
        dtypes=(tf.float32, tf.int64, tf.float32),
        shapes=(tf.TensorShape([2]), tf.TensorShape([]),
                tf.TensorShape([self.num_states])),
        max_in_flight_samples_per_worker=3 * self.config.batch_size)
    dataset = dataset.take(samples_to_take).as_numpy_iterator()

    state_visitations = collections.defaultdict(list)

    def _reverb_transition_to_state_visitation_tuple(
        transition):
      _, state, state_visitation = transition.data
      return state, state_visitation

    # Concat all state-visitations that have the same bin
    for state, state_visitation in map(
        _reverb_transition_to_state_visitation_tuple, dataset):
      state_visitations[state].append(state_visitation)

    # Make sure we sampled every bin
    states_sampled = set(state_visitations.keys())
    assert len(
        states_sampled
    ) == self.num_states, 'aux_task_matrix samples don\'t cover the state space'

    def _average_samples(args):
      return jnp.mean(jnp.stack(args[1], axis=0), axis=0)

    # We must make sure the bins are sorted or else the order will be off
    state_visitations = sorted(state_visitations.items())
    # Average the samples
    state_visitations = list(map(_average_samples, state_visitations))
    # Create a large matrix that is (num_states x num_states)
    state_visitations = jnp.array(state_visitations)

    # Compute the product with the random rewards to obtain expected returns
    expected_returns = jnp.einsum('ts,ns->tn', state_visitations,
                                  self.random_reward_functions)
    return expected_returns

  def __iter__(self):

    @jax.jit
    def calculate_returns(
        reward_functions,
        discounted_state_visitations):
      return jnp.einsum('ts,bs->bt', reward_functions,
                        discounted_state_visitations)

    while self.dataset_iter:
      batch = next(self.dataset_iter)
      states, unused_binned_states, discounted_state_visitations = batch.data
      returns = calculate_returns(self.random_reward_functions,
                                  discounted_state_visitations)

      yield states, returns


class EvalDataset:
  """An eval dataset that retrieves the evaluation batch."""

  def __init__(self, config, key):
    self.num_states = config.env.pw.num_bins**2
    self.eval_dataset = reverb.TimestepDataset(
        FLAGS.reverb_address,
        'eval_table',
        dtypes=(tf.float32, tf.float32),
        shapes=(tf.TensorShape([2]), tf.TensorShape([self.num_states])),
        max_in_flight_samples_per_worker=config.num_eval_points)
    self.eval_dataset = self.eval_dataset.batch(
        config.num_eval_points,
        drop_remainder=True,
        num_parallel_calls=tf.data.AUTOTUNE)

    self.eval_reward_functions = jax.random.uniform(
        key, (config.eval.num_tasks, self.num_states), minval=-1.0, maxval=1.0)

  def get_batch(self):
    """Gets the eval batch.

    Returns:
      A (start_position, returns) tuple, where start_position is an (batch x 2)
        array (2 for (x, y) position) and returns is an (batch x k) array where
        k is the number of auxiliary tasks.
    """
    start_positions, mean_discounted_visitations = (
        next(self.eval_dataset.as_numpy_iterator()).data)

    @jax.jit
    def calculate_returns(visitations, reward_functions):
      return jnp.einsum('bs,ks->bk', visitations, reward_functions)

    returns = calculate_returns(mean_discounted_visitations,
                                self.eval_reward_functions)

    return start_positions, returns


# pylint: enable=invalid-name
