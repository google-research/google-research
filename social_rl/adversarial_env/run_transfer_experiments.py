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
"""Tests trained models on transfer environments to generate videos and scores.

Note that this code assumes it will be provided with a .csv file indicating
which checkpoints it should load based on finding the best hyperparameters
for a given metric, such as 'SolvedPathLength_last20%'. It assumes this csv will
have columns labeled 'metric', 'exp_id', 'best_seeds', and 'settings'. Such a
csv can be created using the function utils.save_best_work_units_csv()
"""
import ast
import datetime
import os
import pdb
import pickle
import sys

from absl import app
from absl import flags
from absl import logging

import numpy as np
import pandas as pd
import tensorflow as tf  # tf

from tf_agents.environments import tf_py_environment
from tf_agents.google.utils import mp4_video_recorder
from tf_agents.trajectories import time_step as ts_lib
from tf_agents.trajectories import trajectory

# Import needed to trigger env registration, so pylint: disable=unused-import
from social_rl import gym_multigrid

from social_rl.adversarial_env import adversarial_env
from social_rl.adversarial_env import utils
from social_rl.multiagent_tfagents import multiagent_gym_suite


flags.DEFINE_string(
    'root_dir', None,
    'Directory where videos and transfer results will be saved')
flags.mark_flag_as_required('root_dir')
flags.DEFINE_string(
    'hparam_csv', None,
    'Required to determine which checkpoints to load.')
flags.mark_flag_as_required('hparam_csv')
flags.DEFINE_string(
    'transfer_csv', None,
    'If provided, will load this csv and continue saving to it')
flags.DEFINE_boolean(
    'test_on_test', False,
    'If True, will also test on the test environments')
flags.DEFINE_boolean(
    'test_mini', False,
    'If True, will test on the mini environments instead')
flags.DEFINE_boolean(
    'fill_in_missing', False,
    'If True, will test load all existing transfer dfs and try to fill in the '
    'missing data')
flags.DEFINE_boolean(
    'reverse_order', False,
    'If True, will iterate through experiments in reverse order.')
flags.DEFINE_string(
    'metric', 'SolvedPathLength_best_ever',
    'Metric to use for selecting which seeds to test.')
flags.DEFINE_boolean(
    'debug', False,
    'If True, will only load 1 seed for each experiment for speed')
flags.DEFINE_integer(
    'num_trials', 10,
    'Number of trials to do for each seed in each transfer environment')
flags.DEFINE_boolean(
    'save_video_matrices', False,
    'If True, will also save matrix encodings of the environment state used to'
    'make rendered videos')
flags.DEFINE_string(
    'name', 'Test transfer',
    'Informative name to output to explain what process is running.')
FLAGS = flags.FLAGS


VAL_ENVS = [
    'MultiGrid-TwoRooms-Minigrid-v0',
    'MultiGrid-Cluttered40-Minigrid-v0',
    'MultiGrid-Cluttered10-Minigrid-v0',
    'MultiGrid-SixteenRooms-v0',
    'MultiGrid-Maze2-v0',
    'MultiGrid-Maze3-v0',
    'MultiGrid-Labyrinth2-v0',
]
TEST_ENVS = [
    'MultiGrid-FourRooms-Minigrid-v0',
    'MultiGrid-Cluttered50-Minigrid-v0',
    'MultiGrid-Cluttered5-Minigrid-v0',
    'MultiGrid-Empty-Random-15x15-Minigrid-v0',
    'MultiGrid-SixteenRoomsFewerDoors-v0',
    'MultiGrid-Maze-v0',
    'MultiGrid-Labyrinth-v0',
]
MINI_VAL_ENVS = [
    'MultiGrid-MiniTwoRooms-Minigrid-v0',
    'MultiGrid-Empty-Random-6x6-Minigrid-v0',
    'MultiGrid-MiniCluttered6-Minigrid-v0',
    'MultiGrid-MiniCluttered-Lava-Minigrid-v0',
    'MultiGrid-MiniMaze-v0'
]
MINI_TEST_ENVS = [
    'MultiGrid-MiniFourRooms-Minigrid-v0',
    'MultiGrid-MiniCluttered7-Minigrid-v0',
    'MultiGrid-MiniCluttered1-Minigrid-v0'
]


class Experiment:
  """Loads saved checkpoints, tests on transfer envs, generates videos."""

  def __init__(self, name, exp_id, adv_env_name='MultiGrid-Adversarial-v0',
               seeds=None, root_dir=None, checkpoint_nums=None, fps=4,
               num_blank_frames=4, verbose=True, old_env=False, num_agents=3,
               time_last_updated=10, population=True, save_matrices=False,
               benchmark_against=None):
    self.name = name
    self.exp_id = exp_id
    self.adv_env_name = adv_env_name
    self.seeds = seeds
    self.checkpoint_nums = checkpoint_nums
    self.fps = fps
    self.num_blank_frames = num_blank_frames
    self.verbose = verbose
    self.old_env = old_env
    self.time_last_updated = time_last_updated
    self.population = population
    self.population_size = 1
    self.checkpoint_nums = checkpoint_nums
    self.save_matrices = save_matrices
    self.benchmark_against = benchmark_against
    self.num_agents = num_agents

    # Paths
    self.root_dir = root_dir
    if root_dir is None:
      self.root_dir = '/tmp/adversarial_env/'
    self.videos_dir = os.path.join(self.root_dir, 'videos')
    self.model_path = os.path.join(*[self.root_dir, adv_env_name, 'xm',
                                     str(exp_id)])

    self.policies = {}
    self.py_env = None
    self.tf_env = None
    self.blank_frame = None

    # Store the results of testing transfer on a number of environments
    self.transfer_results = {}

  def get_checkpoints_for_seed(self, seed):
    path = os.path.join(self.model_path, seed)
    path += '/policy_saved_model/agent'
    if self.population:
      path = os.path.join(path, '0')
    return tf.io.gfile.listdir(path)

  def get_latest_checkpoint(self, seed):
    """Finds the latest checkpoint number for a model."""
    checkpoints = self.get_checkpoints_for_seed(seed)
    skip_idx = len('policy_')
    if len(checkpoints) < 2:
      return None
    else:
      # Get second last checkpoint to avoid errors where a currently-running XM
      # job is in the process of saving some checkpoint that cannot actually
      # be loaded yet.
      return checkpoints[-2][skip_idx:]

  def load(self, claim=True):
    """Loads policies for all agents and initializes training environment."""
    # Create directory to claim this experiment as currently being computed
    # (Code prioritizes which checkpoint to load next based on when a file in
    # this directory was last modified)
    if claim:
      claim_dir = os.path.join(self.videos_dir, self.name, 'WallsAreLava')
      if not tf.io.gfile.exists(claim_dir):
        tf.io.gfile.makedirs(claim_dir)
      with tf.gfile.GFile(os.path.join(claim_dir, 'claim.txt'), 'wb') as f:
        f.write('Claiming this experiment in the name of some process!\n')
      print('Claiming this experiment by making file in', claim_dir)

    print('Creating experiment', self.name)
    print('Loading from', self.model_path)

    if self.seeds is None:
      files = tf.io.gfile.listdir(self.model_path)
      self.seeds = [f for f in files if tf.io.gfile.isdir(
          os.path.join(self.model_path, f))]
    print('Checking seeds', ', '.join(self.seeds))

    # Get latest checkpoint
    if self.checkpoint_nums is None:
      self.checkpoint_nums = {}
      for s in self.seeds:
        ckpt_num = self.get_latest_checkpoint(s)
        if ckpt_num is None:
          print("Can't find checkpoint for seed", s)
        else:
          self.checkpoint_nums[s] = ckpt_num

    print('Loading policies...')
    for s in self.seeds:
      if s in self.checkpoint_nums.keys():
        self.policies[s] = self.load_checkpoints_for_seed(s)

    if self.py_env is None or self.tf_env is None:
      print('Loading training environment...')
      self.py_env, self.tf_env = self.load_environment(self.adv_env_name)
      self.tf_env.reset()
      self.blank_frame = self.py_env.render()
      self.blank_frame_encoding = self.py_env._gym_env.grid.encode()  # pylint:disable=protected-access

  def load_checkpoint_from_path(self, path, copy_local=True):
    """Load checkpoint from path. Copy locally first to improve speed."""
    if copy_local:
      # First copy file from server locally to avoid deadline exceeded errors
      # and increase performance
      local_path = path.replace(self.root_dir, '/tmp/adversarial_env')
      tf.io.gfile.makedirs(local_path)
      utils.copy_recursively(path, local_path)
    else:
      local_path = path

    return tf.compat.v2.saved_model.load(local_path)

  def load_checkpoints_for_seed(self, seed):
    """Loads most recent checkpoint for each agent for a given work unit."""
    policy_path = os.path.join(*[self.model_path, seed, 'policy_saved_model'])
    checkpoint_path = 'policy_' + self.checkpoint_nums[seed]

    policies = {}

    agents = ['agent', 'adversary_agent', 'adversary_env']
    if 'unconstrained' in self.name or 'minimax' in self.name:
      agents = ['agent', 'adversary_env']
      policies['adversary_agent'] = [None]
    elif 'domain_randomization' in self.name:
      agents = ['agent']

    for name in agents:
      if not self.population:
        path = os.path.join(*[policy_path, name, checkpoint_path])
        print('\tLoading seed', seed, 'policy for', name, '...')
        sys.stdout.flush()
        policies[name] = [self.load_checkpoint_from_path(path)]
      else:
        # Population-based training runs have several agents of each type.
        agent_path = os.path.join(policy_path, name)
        policies[name] = []
        pop_agents = tf.io.gfile.listdir(agent_path)
        for pop in pop_agents:
          path = os.path.join(*[agent_path, pop, checkpoint_path])
          if tf.io.gfile.exists(path):
            print('\tLoading seed', seed, 'policy for', name, pop, '...')
            policies[name].append(self.load_checkpoint_from_path(path))
    return policies

  def load_environment(self, env_name):
    if 'Adversarial' in env_name:
      py_env = adversarial_env.load(env_name)
      tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)
    else:
      py_env = multiagent_gym_suite.load(env_name)
      tf_env = tf_py_environment.TFPyEnvironment(py_env)
    return py_env, tf_env

  def initialize_video_for_seed(self, seed, env_name, agent_id=None,
                                adv_id=None, adv_agent_id=None):
    """Creates a video recorder which records agents playing an environment."""
    env_start = env_name.find('MultiGrid') + len('MultiGrid-')
    env_end = env_name.find('-v0')
    trim_env_name = env_name[env_start:env_end]

    vid_name = '{0}_xm{1}_{2}_s{3}_{4}'.format(
        self.name, self.exp_id, trim_env_name, seed,
        self.checkpoint_nums[seed])

    if agent_id is not None:
      vid_name += '_p{0}'.format(agent_id)
    if adv_agent_id is not None:
      vid_name += '_a{0}'.format(adv_agent_id)
    if adv_id is not None:
      vid_name += '_e{0}'.format(adv_id)
    matrix_filename = vid_name + '.pkl'
    vid_name += '.mp4'

    vid_path = os.path.join(
        *[self.videos_dir, self.name, trim_env_name, vid_name])
    matrix_path = os.path.join(
        *[self.videos_dir, self.name, trim_env_name, matrix_filename])
    if self.verbose:
      print('Saving video to', vid_path)

    vid_recorder = mp4_video_recorder.Mp4VideoRecorder(vid_path, self.fps)
    return vid_recorder, matrix_path

  def run_adversarial_trial(self, adv_policy, agent_policy, recorder,
                            adv_agent_policy=None):
    """Run a trial in which the environment is generated by an adversary."""
    # Run adversary to create environment
    encoded_images = self.create_env_with_adversary(adv_policy, recorder)

    # Run agent in the environment
    reward, encoded_images = self.run_agent(
        agent_policy, recorder, self.adv_env_name, self.py_env, self.tf_env,
        encoded_images=encoded_images)

    if adv_agent_policy is not None:
      _, encoded_images = self.run_agent(
          adv_agent_policy, recorder, self.adv_env_name, self.py_env,
          self.tf_env, encoded_images=encoded_images)

    return reward, encoded_images

  def run_seed_trials(self, seed, env_name, agent_id=0, num_trials=25,
                      video_record_episodes=1, adv_id=0, adv_agent_id=None,
                      py_env=None, tf_env=None):
    """Run a number of trials in an env for agents from a specific seed."""
    rewards = []

    # Initialize video recorder
    recorder = None
    if video_record_episodes > 0:
      recorder, matrix_filename = self.initialize_video_for_seed(
          seed, env_name, agent_id, adv_id, adv_agent_id)
    encoded_images = []

    for t in range(num_trials):
      # Usually record fewer episodes than the number of trials completed
      if recorder and t == video_record_episodes:
        recorder.end_video()
        recorder = None

      if (env_name == self.adv_env_name and
          'domain_randomization' not in self.name):
        # Run adversarial trial
        if adv_agent_id is not None:
          adv_agent_pol = self.policies[seed]['adversary_agent'][adv_agent_id]
        else:
          adv_agent_pol = None
        r, encoded = self.run_adversarial_trial(
            self.policies[seed]['adversary_env'][adv_id],
            self.policies[seed]['agent'][agent_id],
            recorder, adv_agent_policy=adv_agent_pol)
        rewards.append(r)
        if encoded is not None:
          encoded_images.extend(encoded)

      else:
        # Run agent in a transfer environment
        r, encoded = self.run_agent(
            self.policies[seed]['agent'][agent_id], recorder, env_name,
            py_env, tf_env)
        rewards.append(r)
        if encoded is not None:
          encoded_images.extend(encoded)

    if recorder:
      recorder.end_video()
    if self.save_matrices:
      with tf.gfile.GFile(matrix_filename, 'wb') as f:
        pickle.dump(encoded_images, f)
        f.close()

    return rewards

  def check_how_many_trials_in_df(self, df, env, seed, metric, dr_equiv,
                                  agent_id):
    """Check df to see if these trials have already been run."""
    if df.empty or 'exp_id' not in df.columns.values:
      return False, df

    exp_df = df[df['exp_id'] == self.exp_id]
    if exp_df.empty:
      print('Experiment', self.name, self.exp_id,
            'is not yet in the dataframe.')
      return False, df

    seed_df = exp_df[exp_df['seed'] == int(seed)]
    if seed_df.empty:
      return False, df

    env_df = seed_df[seed_df['env'] == env]
    if env_df.empty:
      return False, df

    ckpt_df = env_df[env_df['domain_rand_comparable_checkpoint'] == dr_equiv]
    if ckpt_df.empty:
      return False, df

    ckpt_df = ckpt_df[ckpt_df['checkpoint'] == int(self.checkpoint_nums[seed])]
    if ckpt_df.empty:
      return False, df

    agent_df = ckpt_df[ckpt_df['agent_id'] == agent_id]
    if agent_df.empty:
      return False, df

    # Check if these results exist for a different metric, and if so duplicate
    if metric is not None and metric not in agent_df['metric'].unique():
      row_dict = agent_df[0:1].to_dict(orient='records')[0]
      print('Found existing records for a different metric. Making a copy for '
            'this metric')
      print(row_dict)
      row_dict['metric'] = metric
      df = df.append(row_dict, ignore_index=True)

    print('Found trials already in the df for', self.name, env,
          'seed', seed, 'dr ckpt', dr_equiv, 'agent_id', agent_id,
          '... Skipping ahead.')
    return True, df

  def find_dr_ckpt_equivalent(self, dr_ckpts, seed):
    if dr_ckpts is None:
      print('!! No DR checkpoints passed to find_dr_ckpt_equivalent, not '
            'possible to find equivalent checkpoint')
      return ''
    equiv_ckpt = int(self.checkpoint_nums[seed]) / self.num_agents
    diffs = np.abs(np.array(dr_ckpts) - equiv_ckpt)
    return dr_ckpts[np.argmin(diffs)]

  def run_trials_in_env(self, env_name, df=None, num_trials=100,
                        video_record_episodes=3, test_type='val', metric=None,
                        dr_ckpts=None):
    """Run all trials of all seeds for a particular environment."""
    if env_name == self.adv_env_name and 'domain_randomization' in self.name:
      print(
          'Skipping adversarial episodes for domain randomization environment')
      return None
    else:
      py_env, tf_env = self.load_environment(env_name)

    if df is None:
      df = pd.DataFrame()

    for s in self.seeds:
      dr_equiv = self.find_dr_ckpt_equivalent(dr_ckpts, s)

      if s not in self.policies.keys():
        continue

      for agent_id in range(len(self.policies[s]['agent'])):
        already_done, df = self.check_how_many_trials_in_df(
            df, env_name, s, metric, dr_equiv, agent_id)
        if already_done:
          continue

        if env_name == self.adv_env_name and 'domain_randomization' not in self.name:
          for adv_id in range(len(self.policies[s]['adversary_env'])):
            for adv_agent_id in range(len(self.policies[s]['adversary_agent'])):
              rewards = self.run_seed_trials(
                  s, env_name, agent_id=agent_id, num_trials=num_trials,
                  video_record_episodes=video_record_episodes, adv_id=adv_id,
                  adv_agent_id=adv_agent_id)
              row_dict = self.log_seed(
                  rewards, s, env_name, agent_id, adv_id, adv_agent_id,
                  test_type=test_type, metric=metric, dr_equiv=dr_equiv)
              df = df.append(row_dict, ignore_index=True)
        else:
          rewards = self.run_seed_trials(
              s, env_name, agent_id=agent_id, num_trials=num_trials,
              video_record_episodes=video_record_episodes, py_env=py_env,
              tf_env=tf_env)
          row_dict = self.log_seed(rewards, s, env_name, agent_id,
                                   test_type=test_type, metric=metric,
                                   dr_equiv=dr_equiv)
          df = df.append(row_dict, ignore_index=True)

    return df

  def log_seed(self, rewards, seed, env_name, agent_id=0, adv_id=None,
               adv_agent_id=None, test_type='val', metric=None, dr_equiv=None):
    """Create a dictionary of all score metrics for a particular seed."""
    print('Average return for', self.name, env_name, 'seed', seed, 'agent',
          agent_id, '=', np.mean(rewards))
    if adv_id:
      print('\twith adversary', adv_id, 'and antagonist', adv_agent_id)

    seed_dict = {
        'seed': seed,
        'checkpoint': self.checkpoint_nums[seed],
        'domain_rand_comparable_checkpoint': dr_equiv,
        'num_solved': np.sum(np.greater(rewards, 0)),
        'sum': np.sum(rewards),
        'mean': np.mean(rewards),
        'std': np.std(rewards),
        'min': np.min(rewards),
        'max': np.max(rewards),
        'n': len(rewards),
        'agent_id': agent_id,
        'env': env_name,
        'name': self.name,
        'exp_id': self.exp_id,
        'run': test_type}
    if metric is not None:
      seed_dict['metric'] = metric
    if adv_id is not None:
      seed_dict['adv_id'] = adv_id
    if adv_agent_id is not None:
      seed_dict['adv_agent_id'] = adv_agent_id
    return seed_dict

  def create_env_with_adversary(self, policy, recorder):
    """Run adversary to create environment."""
    encoded_images = None

    # Add blank frames to make it easier to distinguish between runs/agents
    if recorder:
      for _ in range(self.num_blank_frames):
        recorder.add_frame(self.blank_frame)
    if self.save_matrices:
      encoded_images = [self.blank_frame_encoding] * self.num_blank_frames

    policy_state = policy.get_initial_state(1)
    time_step = self.tf_env.reset()
    if self.old_env:
      time_step = backwards_compatible_timestep(time_step)

    if self.verbose: print('\tAdversary is building the environment...')

    for _ in range(self.py_env._gym_env.adversary_max_steps):  # pylint: disable=protected-access
      policy_step = policy.action(time_step, policy_state=policy_state)
      policy_state = policy_step.state
      time_step = self.tf_env.step_adversary(policy_step.action)

      if self.old_env:
        time_step = backwards_compatible_timestep(time_step)

      if recorder:
        recorder.add_frame(self.py_env.render())
      if self.save_matrices:
        encoded_images.append(self.py_env._gym_env.grid.encode())  # pylint:disable=protected-access

    return encoded_images

  def run_agent(self, policy, recorder, env_name, py_env, tf_env,
                encoded_images=None):
    """Run an agent's policy in a particular environment. Possibly record."""
    if self.save_matrices and encoded_images is None:
      encoded_images = []

    # Add blank frames to make it easier to distinguish between runs/agents
    for _ in range(self.num_blank_frames):
      if recorder:
        recorder.add_frame(self.blank_frame)
      if self.save_matrices:
        encoded_images.append(self.blank_frame_encoding)

    rewards = 0
    policy_state = policy.get_initial_state(1)

    if 'domain_randomization' in self.name and env_name == self.adv_env_name:
      time_step = tf_env.reset_random()
    elif 'Adversarial' in env_name:
      time_step = tf_env.reset_agent()
    else:
      time_step = tf_env.reset()

    if recorder:
      recorder.add_frame(py_env.render())
    if self.save_matrices:
      encoded_images.append(self.py_env._gym_env.grid.encode())  # pylint:disable=protected-access

    num_steps = tf.constant(0.0)
    while True:
      policy_step = policy.action(time_step, policy_state=policy_state)

      policy_state = policy_step.state
      next_time_step = tf_env.step(policy_step.action)

      traj = trajectory.from_transition(time_step, policy_step, next_time_step)
      time_step = next_time_step

      num_steps += tf.math.reduce_sum(tf.cast(~traj.is_boundary(), tf.float32))

      rewards += time_step.reward
      if recorder:
        recorder.add_frame(py_env.render())
      if self.save_matrices:
        encoded_images.append(self.py_env._gym_env.grid.encode())  # pylint:disable=protected-access

      if traj.is_last():
        break

    return rewards.numpy().sum(), encoded_images


def backwards_compatible_timestep(new_ts):
  """Remove new observations added in later versions of the environment."""
  old_obs = {
      'image': new_ts.observation['image'],
      'time_step': new_ts.observation['time_step']
  }
  return ts_lib.TimeStep(
      new_ts.step_type,
      new_ts.reward,
      new_ts.discount,
      old_obs)


def prioritize_experiments(experiments, videos_dir):
  """Prioritizes experiments based on recency of generated transfer videos."""
  for exp in experiments:
    exp_dir = os.path.join(*[videos_dir, exp.name, 'WallsAreLava'])
    if tf.io.gfile.exists(exp_dir):
      files = tf.io.gfile.listdir(exp_dir)
      if files:
        # Gets the update time of most recently updated file
        update_times = [tf.io.gfile.stat(
            os.path.join(exp_dir, f)).mtime for f in files]
        update_times.sort()
        exp.time_last_updated = update_times[-1]

  experiments.sort(key=lambda x: x.time_last_updated)
  return experiments


def test_experiment_in_environments(exp, envs, df, transfer_df_path,
                                    unsuccessful_trials, test_type='val',
                                    num_trials=25, debug=True, metric=None,
                                    dr_ckpts=None):
  """Test checkpoints for an experiment in a collection of environments."""
  for env in envs:
    try:
      if debug:
        print('Testing', test_type, 'env', env)
      df = exp.transfer_results[env] = exp.run_trials_in_env(
          env, df=df, num_trials=num_trials, video_record_episodes=1,
          test_type=test_type, metric=metric, dr_ckpts=dr_ckpts)
      save_df_and_log(df, transfer_df_path)
    except Exception as e:  # pylint: disable=broad-except
      logging.error('ERROR! with experiment %s in environment %s',
                    exp.name, env)
      print(e)
      print('\n')
      unsuccessful_trials.append(exp.name + '_' + env)
  return df


def load_existing_transfer_file(transfer_dir, transfer_csv, test_on_test=False,
                                mini_str='', name=''):
  """Load existing transfer file if it exists. Otherwise initialize new."""
  # If filename of transfer csv was provided but does not exist, ignore.
  if transfer_csv and not tf.io.gfile.exists(
      os.path.join(transfer_dir, transfer_csv)):
    print('Error! Could not find transfer CSV file', transfer_csv)
    transfer_csv = None

  if transfer_csv:
    transfer_df_path = os.path.join(transfer_dir, transfer_csv)
    with tf.gfile.GFile(transfer_df_path, 'rb') as f:
      df = pd.read_csv(f)
    print('Loaded existing transfer results file: %s', transfer_df_path)
  else:
    # Name file containing transfer results based on current time.
    test_str = ''
    if test_on_test:
      test_str = 'test_'

    csv_name = 'transfer_' + test_str + mini_str + name + '_results_' + \
      datetime.datetime.now().strftime('%d.%m.%Y.%H:%M:%S') + '.csv'
    transfer_df_path = os.path.join(transfer_dir, csv_name)
    df = pd.DataFrame()
  return df, transfer_df_path


def generate_results(root_dir, experiments, test_on_test=False, test_mini=False,
                     transfer_csv=None, debug=False, num_trials=25,
                     name='', hparam_csv=None, fill_in_missing=False,
                     metric='SolvedPathLength_best_ever', reverse_order=False):
  """Generates transfer results for all experiments, saves videos and a csv.

  Args:
    root_dir: Base directory where experiments are saved and where videos and
      transfer results will be saved.
    experiments: A list of Experiment instances.
    test_on_test: If True, will also test on test environments (not just
      validation)
    test_mini: If True, will test on miniature environments.
    transfer_csv: Path to an alreaady existing csv file of transfer results. If
      provided, will load and append to this file.
    debug: If True, will output additional logging statements.
    num_trials: How many times to test each seed/agent in an environment. Used
      to get error bars on transfer results.
    name: Name of process which will be printed occasionally while it is
      running. Use to disambiguate separate processes for testing e.g. mini
      envs.
    hparam_csv: If provided, will loop through experiments in the csv and use
      hparams specified in the csv.
    fill_in_missing: If True, will load existing transfer dfs until it fills
      in as much data as it can, and then use this to only fill in missing data
      from there.
    metric: The metric used to calculate which hparam settings will be tested.
    reverse_order: If True, will iterate through experiments in reverse order.
  """
  transfer_dir = os.path.join(root_dir, 'transfer_results')

  # Set test environments
  if test_mini:
    mini_str = 'mini_'
    val_envs = MINI_VAL_ENVS
    test_envs = MINI_TEST_ENVS
  else:
    mini_str = ''
    val_envs = VAL_ENVS
    test_envs = TEST_ENVS

  if not test_on_test:
    test_envs = None
    test_str = ''
  else:
    test_str = 'test_'

  if fill_in_missing:
    df = utils.combine_existing_transfer_data(
        transfer_dir, after_date='29.09.2020.00:00:00')
    print('Loaded', len(df), 'existing transfer results')

    csv_name = 'transfer_' + test_str + mini_str + name + '_results_' + \
      datetime.datetime.now().strftime('%d.%m.%Y.%H:%M:%S') + '.csv'
    transfer_df_path = os.path.join(transfer_dir, csv_name)
  else:
    df, transfer_df_path = load_existing_transfer_file(
        transfer_dir, transfer_csv, test_on_test, mini_str, name)

  # Open hparam file and format experiment IDs
  with tf.gfile.GFile(hparam_csv, 'rb') as f:
    hparam_df = pd.read_csv(f)
  hparam_df['exp_id'] = [int(x) for x in hparam_df['exp_id']]

  metric_df = hparam_df[hparam_df['metric'] == metric]
  print('Conducting all transfer experiments for metric', metric)
  checkpoints_to_check = {}
  unsuccessful_trials = []

  # Assign the seeds from the csv for this metric to all experiments
  for exp in experiments:
    exp_df = metric_df[metric_df['exp_id'] == exp.exp_id]

    if ('domain_randomization' in exp.name and
        metric == 'adversary_env_AdversaryReward_last20%'):
      exp_df = hparam_df[hparam_df['metric'] == 'SolvedPathLength_last20%']
      exp_df = exp_df[exp_df['exp_id'] == exp.exp_id]

    # If experiment not in hparams csv, skip
    if exp_df.empty:
      print('No hparams available for experiment', exp.exp_id, exp.name,
            'metric', metric, 'so skipping')
      continue

    checkpoints_to_check[exp.exp_id] = []
    exp.seeds = ast.literal_eval(exp_df['best_seeds'].tolist()[0])
    if isinstance(exp.seeds, int):
      exp.seeds = [str(exp.seeds)]

    assert exp.seeds

    # Adjust number of agents if training with a combined population
    settings_dict = ast.literal_eval(exp_df['settings'].tolist()[0])
    combined_population = (
        'combined_pop' in exp.name or
        ('xm_combined_population' in settings_dict and
         settings_dict['xm_combined_population']))
    exp.num_agents = utils.calculate_num_agents_based_on_population(
        settings_dict, exp.num_agents, combined_population, is_dict=True)
    print('Calculated that experiment', exp.name, 'has', exp.num_agents,
          'agents, based on:')
    print(settings_dict)

  # Find a checkpoint such that all benchmark experiments have the equivalent
  # checkpoint for each experiment
  for exp in experiments:
    if not exp.seeds:
      continue

    if exp.benchmark_against is not None:
      benchmark_exps = [
          e for e in experiments if e.exp_id in exp.benchmark_against]
      equivalent_ckpts = get_checkpoints_to_match_benchmarks(exp,
                                                             benchmark_exps)
      if equivalent_ckpts:
        for exp_id, ckpt in equivalent_ckpts.items():
          if equivalent_ckpts[exp_id] not in checkpoints_to_check[exp_id]:
            checkpoints_to_check[exp_id].append(equivalent_ckpts[exp_id])

  # Find domain randomization checkpoints for easy benchmarking later
  dr_ckpts = None
  for exp in experiments:
    if 'domain_randomization' in exp.name:
      # There should only be one domain randomization experiment per transfer
      # test
      assert dr_ckpts is None
      dr_ckpts = [int(
          c[len('policy_'):]) for c in checkpoints_to_check[exp.exp_id]]
  print('Domain randomization checkpoints are:', dr_ckpts)

  print('Checking the following checkpoints for metric', metric,
        'transfer task', name)
  for exp_id, ckpts in checkpoints_to_check.items():
    print(exp_id, ckpts)
  print('')

  if reverse_order:
    experiments = experiments[::-1]

  for i, exp in enumerate(experiments):  #
    for ckpt in checkpoints_to_check[exp.exp_id]:
      # Assign checkpoint to checkpoint nums before loading
      exp.checkpoint_nums = {}
      for s in exp.seeds:
        exp.checkpoint_nums[s] = ckpt[len('policy_'):]

      # Save seeds to re-assign later
      exp_seeds = exp.seeds

      print('Checking the following for experiment', exp.exp_id,
            exp.name)
      print(exp.checkpoint_nums)

      exp.load(claim=False)

      # Produce video of adversary generating a training environment
      exp.run_trials_in_env(exp.adv_env_name, num_trials=1, dr_ckpts=dr_ckpts)

      df = test_experiment_in_environments(
          exp, val_envs, df, transfer_df_path, unsuccessful_trials,
          test_type='val', num_trials=num_trials, debug=debug,
          metric=metric, dr_ckpts=dr_ckpts)

      # Save the transfer results after every exp since they're slow to obtain
      with tf.gfile.GFile(transfer_df_path, 'wb') as f:
        df.to_csv(f)

      save_df_and_log(df, transfer_df_path)

      if test_envs:
        df = test_experiment_in_environments(
            exp, test_envs, df, transfer_df_path, unsuccessful_trials,
            test_type='test', num_trials=num_trials, debug=debug,
            metric=metric, dr_ckpts=dr_ckpts)

      # Re-assign seeds for next metric
      exp.seeds = exp_seeds

      save_df_and_log(df, transfer_df_path)
      print('Finished checkpoint', ckpt, ' for experiment', exp.exp_id,
            exp.name, '\n')

    print('****** Finished experiment', exp.exp_id, exp.name, '*****\n')
    print('Have finished', i+1, '/', len(experiments), 'experiments, or',
          float(i+1) / float(len(experiments)) * 100.0, '%')

  if unsuccessful_trials:
    print('The following experiments were unsuccessful:')
    print(unsuccessful_trials)

  print('Finished testing all seeds for all experiments!!!')


def save_df_and_log(df, transfer_df_path):
  """Save the transfer results after every exp since they're slow to obtain."""
  with tf.gfile.GFile(transfer_df_path, 'wb') as f:
    df.to_csv(f)
  print('Updated dataframe at', transfer_df_path)


def get_checkpoints_to_match_benchmarks(exp, benchmark_exps):
  """Get checkpoint numbers equivalent to the exps it's benchmarked against."""
  checkpoints = {}
  for s in exp.seeds:
    checkpoints[s] = exp.get_checkpoints_for_seed(s)

  benchmark_checkpoints = {}
  for bexp in benchmark_exps:
    benchmark_checkpoints[bexp.exp_id] = {}
    if not bexp.seeds:
      continue
    for s in bexp.seeds:
      benchmark_checkpoints[bexp.exp_id][s] = bexp.get_checkpoints_for_seed(s)

  equivalent_checkpoints = {}

  for ckpt in checkpoints[exp.seeds[0]][::-1]:
    ckpt_num = int(ckpt[len('policy_'):])

    # Verify that all other seeds have this checkpoint
    if not checkpoint_present_for_all_seeds(ckpt, checkpoints):
      continue

    # Verify that the models being benchmarked against have the equivalent
    # checkpoint
    equivalent_checkpoints = {}
    all_benchmarks = True
    for bexp in benchmark_exps:
      if not bexp.seeds:
        continue
      equiv_num = int(ckpt_num / exp.num_agents * bexp.num_agents)

      # Find checkpoint within 20000
      bstr = None
      for bckpt in benchmark_checkpoints[bexp.exp_id][bexp.seeds[0]][::-1]:
        bckpt_num = int(bckpt[len('policy_'):])
        if np.abs(bckpt_num - equiv_num) <= 20000:
          bstr = bckpt
          break
      if bstr is None:
        all_benchmarks = False
        break

      if not checkpoint_present_for_all_seeds(
          bstr, benchmark_checkpoints[bexp.exp_id]):
        all_benchmarks = False
        break
      equivalent_checkpoints[bexp.exp_id] = bstr

    if not all_benchmarks:
      continue

    equivalent_checkpoints[exp.exp_id] = ckpt
    return equivalent_checkpoints

  print('Uh oh! Could not find a set of equivalent checkpoints for experiment',
        exp.exp_id, exp.name)
  pdb.set_trace()


def checkpoint_present_for_all_seeds(ckpt, seed_checkpoints):
  for s in seed_checkpoints.keys():
    if ckpt not in seed_checkpoints[s]:
      return False
  return True


def main(_):
  logging.set_verbosity(logging.INFO)

  experiments = [
      Experiment('paired', 17485886, num_agents=3,
                 benchmark_against=[17486166, 17486367]),
      Experiment('minimax', 17486367, num_agents=2),
      Experiment('domain_randomization', 17486166, num_agents=1),
  ]

  print('Name is', FLAGS.name)

  if FLAGS.save_video_matrices:
    for e in experiments:
      e.save_matrices = True

  generate_results(FLAGS.root_dir, experiments,
                   transfer_csv=FLAGS.transfer_csv,
                   hparam_csv=FLAGS.hparam_csv,
                   test_on_test=FLAGS.test_on_test, test_mini=FLAGS.test_mini,
                   debug=FLAGS.debug, num_trials=FLAGS.num_trials,
                   name=FLAGS.name, fill_in_missing=FLAGS.fill_in_missing,
                   metric=FLAGS.metric, reverse_order=FLAGS.reverse_order)


if __name__ == '__main__':
  app.run(main)
