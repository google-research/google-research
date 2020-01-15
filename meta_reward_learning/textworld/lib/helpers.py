# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Helper functions for environment and agent class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import numpy as np
from six.moves import map
from six.moves import zip
from tensorflow.compat.v1 import gfile

from meta_reward_learning.textworld import common_flags
from meta_reward_learning.textworld.lib import environment
from meta_reward_learning.textworld.lib.graph_search import dfs_paths
from meta_reward_learning.textworld.lib.replay_buffer import AllGoodReplayBuffer
from meta_reward_learning.textworld.lib.replay_buffer import Traj

# pylint: disable=g-import-not-at-top
try:
  import six.moves.cPickle as pickle
except ImportError:
  import pickle
# pylint: enable=g-import-not-at-top


def pad_sequence(l, value, length):
  """Pad a list with value upto the given length."""
  to_append = [value] * (length - len(l))
  if isinstance(l, np.ndarray):
    to_append = np.array(to_append, dtype=l.dtype)
    return np.concatenate((l, to_append))
  elif isinstance(l, list):
    return l + to_append


def pad_sequences(sequence_list, pad_value, maxlen=None):
  """Pads a list of sequences with pad_value."""
  sequence_lengths = [len(x) for x in sequence_list]
  if maxlen is None:
    maxlen = max(sequence_lengths)
  sequence_list = [pad_sequence(x, pad_value, maxlen) for x in sequence_list]
  return sequence_list, sequence_lengths, maxlen


def all_paths(env, maxlen, maxpaths=10):
  """Calculate all the possible paths for reaching the goal in env."""
  env.reset()
  agent = environment.GridAgent(pos=env.agent.pos)
  graph, goal = env.grid.grid, env.grid.goal_pos
  path_generator = dfs_paths(graph, agent, goal, env.num_actions, maxlen)
  paths = sorted(list(path_generator), key=len)[:maxpaths]
  return paths


def create_env_and_paths(seed, n_plants, grid_size, maxlen, mode='train'):
  name = '{}-{}'.format(mode, seed)
  env = environment.Environment(
      seed=seed, n_plants=n_plants, grid_size=grid_size, name=name)
  paths = all_paths(env, maxlen)
  return (env, paths)


def create_traj(env, path):
  """Creates a trajectory given an evironment and a trajectory."""
  rewards = []
  index = 0
  done = False
  env.reset()
  while (not done) and index < len(path):
    ac = path[index]
    index += 1
    reward, done = env.step(ac)  # Down
    rewards.append(reward)
  if not done:
    raise RuntimeError('Done must be true since we reach the goal')
  traj = dict(env_name=env.name, rewards=rewards, actions=path)
  return traj


def create_trajs(env, paths):
  trajs = [create_traj(env, path) for path in paths]
  env_state = dict(
      grid=env.grid.grid, goal_id=env.grid.goal_id, seed=env.rand_seed)
  return (env.name, dict(env_state=env_state, trajs=trajs))


def create_features(inp_sequence):
  """Create a feature vector composed of unary and pairwise counts."""
  # count[1:4] represents the single token counts
  # Example: count[I] = #I in `inp_sequence`
  # count[5:20] represents the pairwise token counts
  # Example: count[4*I + J] = #IJ  in `inp_sequence`
  counts = collections.defaultdict(int)
  prev = str(inp_sequence[0])
  counts[prev] = 1
  for val in map(str, inp_sequence[1:]):
    counts[val] += 1
    counts[prev + val] += 1
    prev = val
  return counts


def create_pairwise_features(x1, x2, keys):
  """Creates features for pairwise count-based similarity."""
  # pylint: disable=g-complex-comprehension
  return [(x1[i] == x2[j]) * ((x1[i] > 0) or (x2[j] > 0))
          for i in keys
          for j in keys]
  # pylint: enable=g-complex-comprehension


def create_joint_features(seq1, seq2):
  f1 = create_features(seq1)
  f2 = create_features(seq2)
  single_matches = create_pairwise_features(f1, f2, common_flags.FEATURE_KEYS)
  pair_matches = create_pairwise_features(f1, f2,
                                          common_flags.PAIR_FEATURE_KEYS)
  feature_arr = np.array(single_matches + pair_matches, dtype=np.float32)
  feature_arr /= len(f2)  # Divide by the length of the language command
  return feature_arr


def create_replay_traj(env, actions):
  traj_dict = create_traj(env, actions)
  features = create_joint_features(traj_dict['actions'], env.context)
  return Traj(features=features, **traj_dict)


def get_top_trajs(buffer_scorer, trajs):
  features = [t.features for t in trajs]
  scores = buffer_scorer.get_scores(features)
  max_score = max(scores)
  top_indices = [i for i, s in enumerate(scores) if s == max_score]
  return [trajs[i] for i in top_indices]


def load_pickle(pickle_file):
  """Load a pickle file (py2/3 compatible)."""
  try:
    with gfile.Open(pickle_file, 'rb') as f:
      pickle_data = pickle.load(f)
  except UnicodeDecodeError as e:
    with gfile.Open(pickle_file, 'rb') as f:
      pickle_data = pickle.load(f, encoding='latin1')
  except Exception as e:
    print('Unable to load {}: {}'.format(pickle_file, e))
    raise
  return pickle_data


def create_dataset(data_file,
                   grid_size,
                   n_plants,
                   return_trajs=False,
                   seed=None,
                   num_envs=None,
                   use_gold_trajs=False,
                   buffer_scorer=None):
  """Create the environment dataset to be used for training/evaluation."""
  data_dict = load_pickle(data_file)
  env_names = list(data_dict.keys())
  if num_envs is not None:
    env_names = sorted(env_names)[:num_envs]
  env_dict = {}
  all_trajs = []
  np.random.seed(seed)
  for name in env_names:
    data = data_dict[name]
    env_state, trajs = data['env_state'], data['trajs']
    goal_id, seed = env_state['goal_id'], env_state['seed']
    env = environment.TextEnvironment(
        name=name,
        goal_id=goal_id,
        grid_size=grid_size,
        n_plants=n_plants,
        seed=seed)
    env.reset()
    env.grid.grid = env_state['grid']
    index = np.random.choice(len(trajs))
    # Reverse the trajectory to create the context
    env.context = list(reversed(trajs[index]['actions']))
    env_dict[name] = env
    if use_gold_trajs:
      trajs = [trajs[index]]
    if return_trajs:
      features = [
          create_joint_features(traj['actions'], env.context) for traj in trajs
      ]
      new_trajs = [
          Traj(features=f, env_name=name, **x) for f, x in zip(features, trajs)
      ]
      if buffer_scorer is not None:
        new_trajs = get_top_trajs(buffer_scorer, new_trajs)
      all_trajs += new_trajs

  if return_trajs:
    return env_dict, all_trajs
  else:
    return env_dict


def create_replay_buffer(data_file,
                         grid_size,
                         n_plants,
                         seed=None,
                         num_envs=None,
                         use_gold_trajs=False,
                         save_trajs=True,
                         buffer_scorer=None):
  """Create a replay buffer with high reward programs."""
  env_dict, all_trajs = create_dataset(
      data_file,
      grid_size=grid_size,
      n_plants=n_plants,
      return_trajs=True,
      seed=seed,
      num_envs=num_envs,
      use_gold_trajs=use_gold_trajs,
      buffer_scorer=buffer_scorer)
  # Save the generated trajectories into the replay buffer
  replay_buffer = AllGoodReplayBuffer(env_dict)
  if save_trajs:
    replay_buffer.save_trajs(all_trajs)
  return replay_buffer


def eval_agent(agent, env_dict):
  trajs = agent.sample_trajs(list(env_dict.values()), greedy=True)
  rews = np.array([sum(t.rewards) for t in trajs])
  accuracy = (np.sum(rews > 0) / len(rews)) * 100
  return accuracy


def cross_product(l1, l2):
  return (x + y for x, y in itertools.product(l1, l2))
