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

"""Agent-environment training loop based on acme."""

import collections
import csv
import operator
import os
import pickle

from absl import flags
from acme import core
from acme.utils import counting
from acme.utils import loggers
import dm_env
from mime.agent import ScriptAgent
import numpy as np
import tensorflow as tf
from tensorflow.io import gfile
from rrlfd.bc import pickle_dataset

FLAGS = flags.FLAGS


class ActionLogger:
  """Logger to save actions and their base and residual components to pickle."""

  def __init__(self, spec):
    self._spec = spec
    self._reset()

  def _reset(self):
    """Reset episode buffers."""
    if isinstance(self._spec, dict):
      self.actions = collections.defaultdict(list)
      self.base_actions = collections.defaultdict(list)
      self.residual_actions = collections.defaultdict(list)
    else:
      self.actions = []
      self.base_actions = []
      self.residual_actions = []

  def add(self, action, base_action, residual_action):
    """Append an action to the current episode."""
    if isinstance(self._spec, dict):
      for k, v in action.items():
        self.actions[k].append(np.copy(v))
      for k, v in base_action.items():
        self.base_actions[k].append(np.copy(v))
      for k, v in residual_action.items():
        self.residual_actions[k].append(np.copy(v))
    else:
      self.actions.append(np.copy(action))
      self.base_actions.append(np.copy(base_action))
      self.residual_actions.append(np.copy(residual_action))

  def append_to_pickle(self, path):
    if not gfile.exists(os.path.dirname(path)):
      gfile.makedirs(os.path.dirname(path))
    with gfile.GFile(path, 'ab') as f:
      pickle.dump((self.actions, self.base_actions, self.residual_actions), f)
    self._reset()


def equal_observations(obs1, obs2, obs_type):
  # Account for noise (in linear velocity).
  if obs_type in ['visible_state', 'linear_velocity']:
    return np.all(np.isclose(obs1, obs2, rtol=0., atol=1e-09))
  elif obs_type == 'failure_message':
    return np.all(obs1 == obs2)
  else:
    return np.all(np.equal(obs1, obs2))


def equal_dicts(d1, d2):
  """Test whether the dicts d1 and d2 have equal items."""
  equal_keys = sorted(d1.keys()) == sorted(d2.keys())
  equal_values = np.all(
      [equal_observations(d1[k], d2[k], k) for k in d1.keys()])
  return equal_keys and equal_values


def loop_is_stuck(base_obs1, base_obs2, acme_obs1, acme_obs2, a1, a2):
  """Detect if observations and actions are stuck."""
  # If first time step.
  if acme_obs1 is None or acme_obs2 is None or a1 is None or a2 is None:
    return False, False, False, False
  act_stuck = np.all(np.equal(a1, a2))
  base_obs_stuck = True
  # Not used in the case of RL-only agent.
  if base_obs1 is not None and base_obs2 is not None:
    # Base observation is a list of frames (and possibly other features).
    for obs1, obs2 in zip(base_obs1, base_obs2):
      base_obs_stuck = base_obs_stuck and equal_dicts(obs1, obs2)
  acme_obs_stuck = equal_dicts(acme_obs1, acme_obs2)
  obs_stuck = base_obs_stuck and acme_obs_stuck
  stuck = act_stuck and obs_stuck
  return stuck, act_stuck, base_obs_stuck, acme_obs_stuck


class EnvironmentLoop(core.Worker):
  """A custom RL environment loop."""

  def __init__(
      self,
      environment,
      eval_environment,
      cam_environment,
      cam_eval_environment,
      actor,
      counter=None,
      logger=None,
      label='environment_loop',
      summary_writer=None,
  ):
    # Internalize agent and environment.
    self._environment = environment
    self._eval_environment = eval_environment
    self._cam_environment = cam_environment
    self._cam_eval_environment = cam_eval_environment
    self._actor = actor
    self._counter = counter or counting.Counter()
    self._logger = logger or loggers.make_default_logger(label)
    self._summary_writer = summary_writer
    self.min_discounted = np.inf
    self.max_discounted = -np.inf

  @property
  def actor(self):
    return self._actor

  @actor.setter
  def actor(self, actor):
    self._actor = actor

  def _eval_policy(self, num_episodes, out_dir, num_episodes_to_log):
    # Write a single file with num_episodes_to_log episodes.
    start_log_episode = num_episodes - num_episodes_to_log
    self.run(num_episodes, out_dir, ckpt_freq=None,
             log_frames_freq=start_log_episode, eval_freq=None,
             num_episodes_to_log=num_episodes_to_log)

  def _log_action(self, action_logger, action_components):
    action_dicts = []
    for action in action_components:
      action_dicts.append(self._actor.flat_action_to_dict(action))
    action_logger.add(*action_dicts)

  def env_timestep_to_acme_timestep(self, env_timestep, rl_obs):
    return dm_env.TimeStep(
        env_timestep.step_type,
        env_timestep.reward,
        env_timestep.discount,
        rl_obs)

  def keep_shortest_trajectories(
      self, demos_file, num_to_keep, episode_lengths=None):
    # Keep num_to_keep shortest trajectories in the dataset at demos_file.
    if episode_lengths is None:
      episode_lengths = []
      with gfile.GFile(demos_file, 'rb') as f:
        while True:
          try:
            demo = pickle.load(f)
            episode_lengths.append(len(demo['actions']))
          except EOFError:
            break
    sorted_lengths = sorted(
        enumerate(episode_lengths), key=operator.itemgetter(1))
    included_trajectories = set([e[0] for e in sorted_lengths[:num_to_keep]])
    print('Keeping', len(included_trajectories), 'trajectories')
    all_demos_file = (
        demos_file.replace(f'e{num_to_keep}', '').replace('.pkl', 'all.pkl'))
    gfile.Rename(demos_file, all_demos_file)
    new_demo_writer = pickle_dataset.DemoWriter(demos_file)
    i = 0
    with gfile.GFile(all_demos_file, 'rb') as f:
      while True:
        try:
          demo = pickle.load(f)
          if i in included_trajectories:
            new_demo_writer.write_episode(demo['observations'], demo['actions'])
          i += 1
        except EOFError:
          break

  def keep_latest_trajectories(self, demos_file, num_to_keep):
    # Keep num_to_keep shortest trajectories in the dataset at demos_file.
    print(demos_file)
    all_demos_file = (
        demos_file.replace(f'e{num_to_keep}', '').replace('.pkl', 'all.pkl'))
    print(all_demos_file)
    gfile.Rename(demos_file, all_demos_file)
    last_demos = []
    with gfile.GFile(all_demos_file, 'rb') as f:
      while True:
        try:
          demo = pickle.load(f)
          last_demos.append(demo)
          last_demos = last_demos[:num_to_keep]
        except EOFError:
          break
    new_demo_writer = pickle_dataset.DemoWriter(demos_file)
    for demo in last_demos:
      new_demo_writer.write_episode(demo['observations'], demo['actions'])

  def run(self,
          num_episodes=None,
          num_successes=None,
          success_writer=None,
          out_dir=None,
          ckpt_freq=50_000,
          log_frames_freq=1000,
          eval_freq=100_000,
          start_with_eval=False,
          num_eval_episodes=100,
          num_episodes_to_log=0,
          collapse_in_eval=True,
          eval_seed=None,
          increment_eval_seed=False,
          stop_if_stuck=False,
          trajectory_filter='latest',
          summary_writer=None):
    """Run the env for num_episodes episodes or num_successes successes."""
    # If both are set, use num_episodes as the limit but write num_successes
    # trajectories with success_writer.
    # TODO(minttu): OR train until both are satisfied.
    np.set_printoptions(precision=4, suppress=True)
    action_logger = ActionLogger(self._environment.action_spec())
    num_successes_written = 0
    success_lengths = []
    counts = self._counter.get_counts()
    print('Starting env loop with counters', counts)
    total_steps = counts['env_steps'] if 'env_steps' in counts else 0
    prev_total_steps = total_steps
    summary_writer = summary_writer or self._summary_writer
    i = counts['episodes'] if 'episodes' in counts else 0
    # How many episodes have been written in latest log.
    record_count = i % log_frames_freq
    while num_episodes is None or i < num_episodes:
      rewards = []
      episode_steps = 0
      episode_return = 0
      prev_raw_residual = None
      prev_residual_exploration = False
      # For envs with non-Markovian success criteria, track required fields.
      if i % log_frames_freq == 0:
        record_count = 0
        first_to_record = i
        last_to_record = i + num_episodes_to_log - 1
        if out_dir is not None:
          demo_writer = pickle_dataset.DemoWriter(
              os.path.join(out_dir,
                           'episodes',
                           f'episodes_{first_to_record}-{last_to_record}.pkl'))
      if record_count < num_episodes_to_log:  # Log frames for current episode.
        if self._cam_environment is None:
          environment = self._environment
        else:
          environment = self._cam_environment
          self._environment.reset()  # Keep both environments in same state.
          print(f'episode {i}: using cam env')
      else:  # Do not log frames for current episode.
        environment = self._environment
        if self._cam_environment is not None:
          self._cam_environment.reset()  # Keep both environments in same state.
          print(f'episode {i}: using non-cam env')

      timestep = environment.reset()
      if FLAGS.base_controller is not None:
        # Reset script for each episode.
        # TODO(minttu): Recompute at each time step.
        self._actor.base_controller = ScriptAgent(
            environment.env, FLAGS.base_controller)
      if record_count < num_episodes_to_log:
        observations = []
        actions = []
      # Call base agent here (if applicable) in order to pre-shape observation
      # for acme's observe first.
      acme_obs, _, norm_base_act = self._actor.get_acme_observation(
          timestep.observation)
      acme_timestep = self.env_timestep_to_acme_timestep(
          timestep, acme_obs)
      self._actor.observe_first(acme_timestep)

      while not timestep.last():
        (action, base_action, residual_action, raw_residual,
         residual_exploration, policy_mean, policy_std) = (
             self._actor.select_action(
                 acme_obs, norm_base_act, timestep.observation,
                 prev_raw_residual, prev_residual_exploration))
        prev_raw_residual = raw_residual
        prev_residual_exploration = residual_exploration
        self._log_action(action_logger, (action, base_action, residual_action))
        if (i % 100 == 0 and episode_steps == 0
            and summary_writer is not None):
          if policy_std is not None:
            with summary_writer.as_default():
              tf.summary.scalar('grip_policy_std', policy_std[0], step=i)
              tf.summary.scalar(
                  'linear_policy_std', np.mean(policy_std[1:]), step=i)
          if policy_mean is None:
            if residual_exploration:
              # Get deterministic action.
              _, _, policy_mean, _, _, _, _ = (
                  self._actor.select_action(
                      acme_obs, norm_base_act, timestep.observation,
                      add_exploration=False))
            else:
              policy_mean = residual_action
          with summary_writer.as_default():
            tf.summary.scalar(
                'grip_policy_magn', np.abs(policy_mean[0]), step=i)
            tf.summary.scalar(
                'linear_policy_magn', np.mean(np.abs(policy_mean[1:])),
                step=i)

        next_timestep = environment.step(action)
        info = environment.info_from_observation(next_timestep.observation)
        if record_count < num_episodes_to_log:
          observations.append(timestep.observation)
          actions.append(self._actor.flat_action_to_dict(action))

        next_acme_obs, _, next_norm_base_act = (
            self._actor.get_acme_observation(next_timestep.observation))
        acme_timestep = self.env_timestep_to_acme_timestep(
            next_timestep, next_acme_obs)
        self._actor.observe(raw_residual, acme_timestep)
        self._actor.update()

        timestep = next_timestep
        rewards.append(timestep.reward)
        acme_obs = next_acme_obs
        norm_base_act = next_norm_base_act
        episode_return += timestep.reward
        episode_steps += 1
        if ((eval_freq is not None  # and total_steps > 0
             and total_steps % eval_freq == 0)
            or (total_steps == prev_total_steps and start_with_eval)):
          eval_path = None
          eval_task = self._environment.task
          if out_dir is not None:
            increment_str = 'i' if increment_eval_seed else ''
            eval_path = os.path.join(
                out_dir,
                'eval',
                f'eval{eval_task}_s{eval_seed}{increment_str}_'
                f'e{num_eval_episodes}')
          finished_eval = False
          while not finished_eval:
            print(f'Evaluating policy after {total_steps} frames')
            success_rate, finished_eval = self.eval_policy(
                num_episodes=num_eval_episodes,
                trained_steps=total_steps,
                collapse_policy=collapse_in_eval,
                eval_path=eval_path,
                num_videos_to_save=100,
                seed=eval_seed,
                increment_seed=increment_eval_seed,
                stop_if_stuck=stop_if_stuck)
          if summary_writer is not None:
            collapse_str = 'c' if collapse_in_eval else ''
            with summary_writer.as_default():
              tf.summary.scalar(
                  f'{eval_task}_s{eval_seed}_e{num_eval_episodes}{collapse_str}'
                  f'_success_rate',
                  success_rate, step=i)
              tf.summary.scalar(
                  f'{eval_task}_s{eval_seed}_e{num_eval_episodes}{collapse_str}'
                  '_success_rate_env_steps',
                  success_rate, step=total_steps)
        if (ckpt_freq is not None and total_steps % ckpt_freq == 0
            and out_dir is not None):
          # TODO(minttu): Add global step to checkpoint.
          ckpt_path = os.path.join(out_dir, 'policy', f'policy_{total_steps}')
          print('Saving policy weights to', ckpt_path)
          self._actor.save_policy_weights(ckpt_path)

          if self._actor.rl_observation_network_type is not None:
            ckpt_path = os.path.join(
                out_dir, 'observation_net', f'observation_{total_steps}')
            print('Saving observation weights to', ckpt_path)
            checkpoint = tf.train.Checkpoint(
                module=self._actor.rl_agent._learner._observation_network)  # pylint: disable=protected-access
            checkpoint.save(ckpt_path)
        total_steps += 1

      discounted_returns = [rewards[-1]]
      for r in reversed(rewards[:-1]):
        discounted_returns.append(r + FLAGS.discount * discounted_returns[-1])
      self.min_discounted = min(self.min_discounted, np.min(discounted_returns))
      self.max_discounted = max(self.max_discounted, np.max(discounted_returns))
      print('discounted episode return range:'
            f'[{self.min_discounted}, {self.max_discounted}]')

      # Record counts.
      counts = self._counter.increment(episodes=1, env_steps=episode_steps)
      print(self._counter.get_counts(), 'success' if info['success'] else '')
      # Collect the results and combine with counts.
      result = {
          'episode_length': episode_steps,
          'episode_return': episode_return}
      result.update(counts)
      self._logger.write(result)

      if out_dir is not None:
        if record_count < num_episodes_to_log:
          print('Saving episode', i)
          demo_writer.write_episode(observations, actions)
          actions_path = (
              os.path.join(
                  out_dir,
                  'actions',
                  f'actions_{first_to_record}-{last_to_record}.pkl'))
          action_logger.append_to_pickle(actions_path)
          record_count += 1
      if success_writer is not None and info['success']:
        num_successes_written += 1
        print(f'Saving episode {num_successes_written} / {num_successes}')
        success_writer.write_episode(observations, actions)
        success_lengths.append(episode_steps)
        if num_successes_written >= num_successes and num_episodes is None:
          break
      i += 1
    print('Ending env loop with counters', self._counter.get_counts())
    if success_writer is not None and num_successes_written > num_successes:
      if trajectory_filter == 'shortest':
        self.keep_shortest_trajectories(
            success_writer.path, num_successes, success_lengths)
      else:
        self.keep_latest_trajectories(success_writer.path, num_successes)

  def eval_policy(
      self,
      num_episodes,
      trained_steps=None,
      collapse_policy=True,
      eval_path=None,
      num_videos_to_save=0,
      max_num_steps=None,
      seed=None,
      increment_seed=False,
      stop_if_stuck=False):
    """Evaluate policy on env for num_episodes episodes."""
    if FLAGS.domain == 'mime':
      self._eval_environment.create_env()
    if not increment_seed and seed is not None:
      self._eval_environment.env.seed(seed)
      if self._cam_eval_environment is not None:
        self._cam_eval_environment.env.seed(seed)
    num_successes = 0
    action_logger = ActionLogger(self._environment.action_spec())
    if max_num_steps is None:
      max_num_steps = self._eval_environment.default_max_episode_steps
    if eval_path is None:
      log_f = None
      success_f = None
      episode_length_f = None
      eval_writer = None
    else:
      if not gfile.exists(os.path.dirname(eval_path)):
        gfile.makedirs(os.path.dirname(eval_path))
      collapse_str = 'c' if collapse_policy else ''
      stuck_str = 's' if stop_if_stuck else ''
      eval_summary_path = eval_path + f'_all{collapse_str}{stuck_str}'
      eval_path = eval_path + f'_{trained_steps}{collapse_str}{stuck_str}'
      log_f = gfile.GFile(eval_path + '_log.txt', 'w')
      success_f = gfile.GFile(eval_path + '_success.txt', 'w')
      episode_length_f = gfile.GFile(eval_path + '_lengths.txt', 'w')
      eval_writer = pickle_dataset.DemoWriter(eval_path + '.pkl')
      actions_path = eval_path + '_actions.pkl'
      if gfile.exists(actions_path):
        gfile.Remove(actions_path)
    for e in range(num_episodes):
      rewards = []
      if increment_seed and seed is not None:
        self._eval_environment.env.seed(seed + e)
        if self._cam_eval_environment is not None:
          self._cam_eval_environment.env.seed(seed + e)
      if e % 10 == 0 and e > 0:
        success_rate = num_successes / e * 100
        print(f'Episode {e} / {num_episodes}; Success rate {num_successes} / '
              f'{e} ({success_rate:.4f}%)')
      if (e < num_videos_to_save and eval_writer is not None
          and self._cam_eval_environment is not None):
        environment = self._cam_eval_environment
        self._eval_environment.reset()  # Keep both environments in same state.
        print(f'eval episode {e}: using cam env')
      else:
        environment = self._eval_environment
        if self._cam_eval_environment is not None:
          # Keep both environments in same state.
          self._cam_eval_environment.reset()
          print(f'eval episode {e}: using non-cam env')

      timestep = environment.reset()
      observations = []
      actions = []
      step_count = 0
      if FLAGS.base_controller is not None:
        # Reset script for each episode.
        self._actor.base_controller = ScriptAgent(
            environment.env, FLAGS.base_controller)

      while not timestep.last():
        acme_obs, _, norm_base_act = self._actor.get_acme_observation(
            timestep.observation)
        action, base_action, residual_action, _, _, _, _ = (
            self._actor.select_action(
                acme_obs, norm_base_act, timestep.observation,
                add_exploration=False, collapse=collapse_policy))
        observations.append(timestep.observation)
        actions.append(self._actor.flat_action_to_dict(action))

        self._log_action(action_logger, (action, base_action, residual_action))
        next_timestep = environment.step(action)
        info = environment.info_from_observation(next_timestep.observation)

        timestep = next_timestep
        rewards.append(timestep.reward)
        step_count += 1

      discounted_returns = [rewards[-1]]
      for r in reversed(rewards[:-1]):
        discounted_returns.append(r + FLAGS.discount * discounted_returns[-1])
      self.min_discounted = min(self.min_discounted, np.min(discounted_returns))
      self.max_discounted = max(self.max_discounted, np.max(discounted_returns))
      print('discounted episode return range:'
            f'[{self.min_discounted}, {self.max_discounted}]')

      if info['success']:
        print(f'{e}: success')
        if log_f is not None:
          log_f.write(f'{e}: success\n')
          log_f.flush()
        if success_f is not None:
          success_f.write('success\n')
          success_f.flush()
        num_successes += 1
      else:
        if 'failure_message' in info:
          print(f'{e}: failure:', info['failure_message'])
        elif step_count >= max_num_steps or timestep.last():
          print(f'{e}: failure: time limit')
        else:
          print(f'{e}: failure')
        if log_f is not None:
          if 'failure_message' in info:
            log_f.write(f'{e}: failure:' + info['failure_message'] + '\n')
          elif step_count >= max_num_steps or timestep.last():
            log_f.write(f'{e}: failure: time limit \n')
          else:
            log_f.write(f'{e}: failure\n')
          log_f.flush()
        if success_f is not None:
          success_f.write('failure\n')
          success_f.flush()
      if episode_length_f is not None:
        episode_length_f.write(str(step_count) + '\n')
        episode_length_f.flush()
      if e < num_videos_to_save and eval_writer is not None:
        eval_writer.write_episode(observations, actions)
        action_logger.append_to_pickle(actions_path)

    success_rate = num_successes / num_episodes * 100
    print(
        f'Done; Success rate {num_successes} / {num_episodes} '
        f'({success_rate:.4f}%)')
    if log_f is not None:
      log_f.write(
          f'Done; Success rate {num_successes} / {num_episodes} '
          f'({success_rate:.4f}%)\n')
      log_f.close()
    csv_writer = csv.writer(
        gfile.GFile(eval_summary_path + '_success_rates.csv', 'a'))
    csv_writer.writerow([trained_steps, num_successes / num_episodes])
    return num_successes / num_episodes, True
