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

"""Image dataset."""

import os
import pickle

import numpy as np


class Dataset:
  """A simple image dataset class."""

  def __init__(self, path):
    """A simple RGB-D image dataset."""
    self.path = path
    self.sample_set = []
    self.max_seed = -1
    self.n_episodes = 0

    # Track existing dataset if it exists.
    color_path = os.path.join(self.path, 'action')
    if os.path.exists(color_path):
      for fname in sorted(os.listdir(color_path)):
        if '.pkl' in fname:
          seed = int(fname[(fname.find('-') + 1):-4])
          self.n_episodes += 1
          self.max_seed = max(self.max_seed, seed)

    self._cache = {}

  def add(self, seed, episode):
    """Add an episode to the dataset.

    Args:
      seed: random seed used to initialize the episode.
      episode: list of (obs, act, reward, info) tuples.
    """
    color, depth, action, reward, info = [], [], [], [], []
    for obs, act, r, i in episode:
      color.append(obs['color'])
      depth.append(obs['depth'])
      action.append(act)
      reward.append(r)
      info.append(i)

    color = np.uint8(color)
    depth = np.float32(depth)

    def dump(data, field):
      field_path = os.path.join(self.path, field)
      if not os.path.exists(field_path):
        os.makedirs(field_path)
      fname = f'{self.n_episodes:06d}-{seed}.pkl'  # -{len(episode):06d}
      pickle.dump(data, open(os.path.join(field_path, fname), 'wb'))

    dump(color, 'color')
    dump(depth, 'depth')
    dump(action, 'action')
    dump(reward, 'reward')
    dump(info, 'info')

    self.n_episodes += 1
    self.max_seed = max(self.max_seed, seed)

  def set(self, episodes):
    """Limit random samples to specific fixed set."""
    self.sample_set = episodes

  def load(self, episode_id, images=True, cache=False):
    """Load data from a saved episode.

    Args:
      episode_id: the ID of the episode to be loaded.
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      episode: list of (obs, act, reward, info) tuples.
      seed: random seed used to initialize the episode.
    """

    def load_field(episode_id, field, fname):

      # Check if sample is in cache.
      if cache:
        if episode_id in self._cache:
          if field in self._cache[episode_id]:
            return self._cache[episode_id][field]
        else:
          self._cache[episode_id] = {}

      # Load sample from files.
      path = os.path.join(self.path, field)
      data = pickle.load(open(os.path.join(path, fname), 'rb'))
      if cache:
        self._cache[episode_id][field] = data
      return data

    # Get filename and random seed used to initialize episode.
    seed = None
    path = os.path.join(self.path, 'action')
    for fname in sorted(os.listdir(path)):
      if f'{episode_id:06d}' in fname:
        seed = int(fname[(fname.find('-') + 1):-4])

        # Load data.
        color = load_field(episode_id, 'color', fname)
        depth = load_field(episode_id, 'depth', fname)
        action = load_field(episode_id, 'action', fname)
        reward = load_field(episode_id, 'reward', fname)
        info = load_field(episode_id, 'info', fname)

        # Reconstruct episode.
        episode = []
        for i in range(len(action)):
          obs = {'color': color[i], 'depth': depth[i]} if images else {}
          episode.append((obs, action[i], reward[i], info[i]))
        return episode, seed

  def sample(self, images=True, cache=False):
    """Uniformly sample from the dataset.

    Args:
      images: load image data if True.
      cache: load data from memory if True.

    Returns:
      sample: randomly sampled (obs, act, reward, info) tuple.
      goal: the last (obs, act, reward, info) tuple in the episode.
    """

    # Choose random episode.
    if len(self.sample_set) > 0:  # pylint: disable=g-explicit-length-test
      episode_id = np.random.choice(self.sample_set)
    else:
      episode_id = np.random.choice(range(self.n_episodes))
    episode, _ = self.load(episode_id, images, cache)

    # Return random observation action pair (and goal) from episode.
    i = np.random.choice(range(len(episode) - 1))
    sample, goal = episode[i], episode[-1]
    return sample, goal
