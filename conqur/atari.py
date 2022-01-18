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

# Lint as: python3
"""Implements functionalities associated with Atari."""

from dopamine.discrete_domains import atari_lib
from dopamine.discrete_domains import run_experiment
import gin
import gym
import numpy as np
import tensorflow.compat.v1 as tf

from conqur import conqur_agent
from conqur import env

STATE_DIM = 513  # 512 features at last layer + bias term


def create_agent(session, environment, random_state):
  return conqur_agent.ConqurAgent(session, environment.action_space.n,
                                  random_state)


@gin.configurable
def create_atari_environment(game_name=None, sticky_actions=False):
  assert game_name is not None
  game_version = 'v0' if sticky_actions else 'v4'
  full_game_name = '{}NoFrameskip-{}'.format(game_name, game_version)
  gym_env = gym.make(full_game_name)
  return atari_lib.AtariPreprocessing(gym_env.env)


class Atari(env.Environment):
  """Manages episodic sampling and evaluation of policies."""

  def __init__(self, atari_roms_source, atari_roms_path, gin_files,
               gin_bindings, random_seed, no_op, best_sampling):
    atari_lib.copy_roms(atari_roms_source, destination_dir=atari_roms_path)
    run_experiment.load_gin_configs(gin_files, gin_bindings)
    self.random_state = np.random.RandomState(random_seed)
    self.runner = ConqurRunner(self.random_state, no_op, best_sampling)
    self.num_actions = self.runner.num_actions
    super(Atari, self).__init__()

  def sample_episode(self,
                     eps_explore,
                     random_state,
                     parent_layer,
                     target_layer,
                     max_episode_length=27000):
    """Sample data batches from the Atari environment.

    This function samples batches of experience for training an RL agent.

    Args:
      eps_explore: float, the probability of exploring a random action.
      random_state: np.random.RandomState, for maintaining the random seed.
      parent_layer: keras.layers, A input the agent network to be rollout.
      target_layer: keras weights, input target network used to compute Q-values
        of the target net.
      max_episode_length: int, sample at most this many transitions.

    Returns:
      Batches of experience in the form of a list.
    """
    return self.runner.sample_episode_single_batch(
        eps_explore,
        random_state,
        parent_layer,
        target_layer,
        max_episode_length=max_episode_length)

  def sample_init_states(self, num_samples=1):
    return [(self.runner.init_state_linear_features, 1)] * num_samples

  def evaluate_policy(self,
                      random_state,
                      rollout_layer,
                      max_episode_length=27000,
                      epsilon_eval=0.001):
    return super(Atari, self).evaluate_policy(
        random_state,
        rollout_layer,
        max_episode_length=max_episode_length,
        epsilon_eval=epsilon_eval)

  def evaluate_checkpoint_agent(self,
                                witness_fn,
                                discount_eval=1.0,
                                num_runs=20,
                                max_episode_length=27000):
    return self.runner.evaluate_checkpoint_agent(
        witness_fn,
        discount_eval=discount_eval,
        num_runs=num_runs,
        max_episode_length=max_episode_length)

  def get_last_layer_weights(self):
    return self.runner.get_last_layer_weights()

  def last_layer_weights(self):
    return self.runner.last_layer_weights()

  def last_layer_biases(self):
    return self.runner.last_layer_biases()

  def last_layer_target_weights(self):
    return self.runner.last_layer_target_weights()

  def last_layer_target_biases(self):
    return self.runner.last_layer_target_biases()


@gin.configurable
class ConqurRunner(run_experiment.TrainRunner):
  """Handles policy evaluation on Atari games.

  ConqurRunner will load two agents from different checkpoints. One will serve
  as the policy (providing the actions), while the other will try to learn the
  value function for that policy.
  """

  def __init__(self,
               random_state,
               no_op,
               best_sampling=True,
               representation_checkpoint=None,
               create_environment_fn=create_atari_environment,
               game_name='Pong',
               game_version='v0',
               sticky_actions=False,
               epsilon_eval=0.01,
               checkpoint_file_prefix='ckpt'):
    """Initialize the Runner object in charge of running a full experiment.

      This constructor will take the following actions:
      1. Initialize an environment.
      2. Initialize a `tf.Session`.
      3. Initialize a logger.
      4. Initialize two agents, one as an actor, one as a policy evaluator.
      5. Reload from the latest checkpoint, if available, and initialize the
          Checkpointer object.

    Args:
      random_state: np.random.RandomState, for maintaining the random seed.
      no_op: bool, whether to apply no-ops in beginning of episodes.
      best_sampling: bool, whether to sample episodes from the best node, to
        train all nodes.
      representation_checkpoint: string, full path to the checkpoint to load for
        the network weights, where all but last layer will be frozen.
      create_environment_fn: function, receives a game name and creates an Atari
        2600 Gym environment.
      game_name: str, name of the Atari game to run (required).
      game_version: str, version of the game to run.
      sticky_actions: bool, whether to use sticky actions.
      epsilon_eval: float, the epsilon exploration probability used to evaluate
        policies.
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
    """
    self.random_state = random_state
    self._environment = create_environment_fn(game_name)
    self.num_actions = self._environment.action_space.n
    self.epsilon_eval = epsilon_eval
    self.action_vals = np.zeros((self.num_actions,), dtype='f8')
    self.no_op = no_op
    self.best_sampling = best_sampling

    self._representation_checkpoint = representation_checkpoint
    if self._representation_checkpoint:
      self._control_graph = tf.Graph()
      with self._control_graph.as_default():
        self._control_sess = tf.Session(
            'local', config=tf.ConfigProto(allow_soft_placement=True))
        self._control_agent = create_agent(self._control_sess,
                                           self._environment, self.random_state)
        self._control_sess.run(tf.global_variables_initializer())

    self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    self._save_init_state()

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    self._start_iteration = 0
    if self._control_agent:
      self._agent = self._control_agent
      with self._control_graph.as_default():
        self._agent.reload_checkpoint(self._representation_checkpoint)

  def _save_init_state(self):
    self._agent.eps_action = 0
    _ = self._initialize_episode()
    self.init_state_linear_features = np.append(
        self._agent.observation_to_linear_features(), 1)

  def no_op_initialize_episode(self, random_state, online_layer):
    observation = self._environment.reset()
    self._agent.reset_state()
    self._agent.update_observation(observation, 0.0, False)
    number_no_op = random_state.randint(0, 31)
    for _ in range(number_no_op):
      observation, reward, is_terminal, _ = self._environment.step(0)
      self._agent.update_observation(observation, reward, is_terminal)
      if is_terminal:
        self._agent.reset_state()
        _ = self._environment.reset()
        break
    prev_linear_features = self._agent.observation_to_linear_features()
    action = tf.math.argmax(online_layer(prev_linear_features)[0]).numpy()
    return action

  def sample_episode_single_batch(self,
                                  eps_explore,
                                  random_state,
                                  parent_layer,
                                  target_layer,
                                  max_episode_length=27000):
    """Sample single batch of experience for all agents.

    This function samples one single batch of data experiences for all agents.

    Args:
      eps_explore: float, the probability of exploring a random action.
      random_state: np.random.RandomState, for maintaining the random seed.
      parent_layer: keras.layer, input the agent network to be rolled-out.
      target_layer: kears.layer, input target network used to compute Q-values
        of the target net.
      max_episode_length:  int, sample at most this many transitions.

    Returns:
      Batches of experience in the form of a list.
    """
    max_episode_length = 27000
    online_layer = parent_layer
    no_op = self.no_op
    episode = []
    if no_op:
      action = self.no_op_initialize_episode(random_state, online_layer)
    else:
      action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    step_number = 0
    observation, reward, is_terminal, _ = self._environment.step(action)
    reward = np.clip(reward, -1, 1)

    self._agent.update_observation(observation, reward, is_terminal)
    prev_linear_features = self._agent.observation_to_linear_features()
    prev_action = action
    prev_reward = reward
    while step_number <= max_episode_length:
      action = tf.math.argmax(parent_layer(prev_linear_features)[0]).numpy()
      if random_state.uniform() <= eps_explore:
        # Choose a random action with probability epsilon.
        action = random_state.randint(0, self._agent.num_actions - 1)
      observation, reward, is_terminal, _ = self._environment.step(action)
      reward = np.clip(reward, -1, 1)

      # Update agent's observation
      self._agent.update_observation(observation, reward, is_terminal)

      # Add a bias feature
      prev_linear_features = np.append(prev_linear_features, 1)
      (target_q_max, target_q_for_all_actions,
       no_reward_target_q_for_all_actions
      ) = self._agent.get_target_q_label_multiple_target_layers(
          prev_reward, is_terminal, target_layer, self._agent.num_actions)
      next_linear_features = self._agent.observation_to_linear_features()
      episode.append((prev_linear_features, next_linear_features, prev_action,
                      prev_reward, target_q_max, target_q_for_all_actions,
                      no_reward_target_q_for_all_actions))
      prev_action = action
      prev_reward = reward
      step_number += 1
      if self._environment.game_over or step_number == max_episode_length:
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward)
        self._agent.reset_state()
        if no_op:
          action = self.no_op_initialize_episode(random_state, online_layer)
        else:
          action = self._initialize_episode()
        next_linear_features = self._agent.observation_to_linear_features()
      prev_linear_features = next_linear_features  # the "next_state"

    # This just calls self._agent.end_episode(reward)
    self._end_episode(reward)
    return episode

  def run_one_episode(self,
                      eps_explore,
                      random_state,
                      online_layer,
                      max_episode_length=27000):
    """launch one episode of interaction with the environment.

    Run a single episode for the agent to interact with the Atari env,
    this will be the rollout function for each agent.

    Args:
      eps_explore: float, probability of exploring a random action.
      random_state:  np.random.RandomState, for maintaining the random seed.
      online_layer: Keras.layer , as a copy of the agent
      max_episode_length: int, to determine the length of the episode

    Returns:
      step_number: int, the number of step taken.
      total_reward: float, the total reward earned by the agent.
    """
    step_number = 0
    total_reward = 0.
    no_op = self.no_op
    if no_op:
      action = self.no_op_initialize_episode(random_state, online_layer)
    else:
      action = self._initialize_episode()
    is_terminal = False

    # Keep interacting until we reach a terminal state.
    while True:
      observation, reward, is_terminal, _ = self._environment.step(action)
      self._agent.update_observation(observation, reward, is_terminal)
      total_reward += reward
      step_number += 1

      # Perform reward clipping.
      reward = np.clip(reward, -1, 1)

      if (self._environment.game_over or step_number == max_episode_length):
        # Stop the run loop once we reach the true end of episode.
        break
      elif is_terminal:
        # If we lose a life but the episode is not over, signal an artificial
        # end of episode to the agent.
        self._agent.end_episode(reward)
        self._agent.reset_state()
        if no_op:
          action = self.no_op_initialize_episode(random_state, online_layer)
        else:
          action = self._initialize_episode()
      else:
        prev_linear_features = self._agent.observation_to_linear_features()
        action = tf.math.argmax(online_layer(prev_linear_features)[0]).numpy()
        if random_state.uniform() < eps_explore:
          action = random_state.randint(0, self._agent.num_actions - 1)

    self._end_episode(reward)
    return step_number, total_reward

  def dopamine_monte_carlo_rollout(self,
                                   eps_explore,
                                   random_state,
                                   online_layer,
                                   evaluation_step=125000,
                                   max_episode_length=27000):
    """Sampling one episode, follow Dopamine's implementation.

    Run a single episode for the agent to interact with the Atari env.

    Args:
      eps_explore: float, probability of exploring a random action.
      random_state:  np.random.RandomState, for maintaining the random seed.
      online_layer: keras weights, as a copy of the agent.
      evaluation_step: int, maximum total number of steps across all episodes.
      max_episode_length: int, maximum transitions per sampled episode.

    Returns:
      step_number: int, the number of step taken
      total_reward, float, the total reward earned by the agent.
      num_episodes, int, number of total episode
    """
    self._agent.eps_action = eps_explore

    # Keep interacting until we reach a terminal state.
    step_count = 0
    num_episodes = 0
    sum_returns = 0
    while step_count < evaluation_step:
      episode_length, episode_return = self.run_one_episode(
          eps_explore,
          random_state,
          online_layer,
          max_episode_length=max_episode_length)
      step_count += episode_length
      sum_returns += episode_return
      num_episodes += 1
    return step_count, sum_returns, num_episodes

  def get_last_layer_weights(self):
    return self._agent.last_layer_wts

  def last_layer_weights(self):
    return self._agent.last_layer_weights

  def last_layer_biases(self):
    return self._agent.last_layer_biases

  def last_layer_target_weights(self):
    return self._agent.target_wts

  def last_layer_target_biases(self):
    return self._agent.target_biases
