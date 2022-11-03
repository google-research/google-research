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

"""Utilities to read and write episodes in pickle format."""

import copy
import math
import os
import pickle
import re
from absl import flags
import numpy as np
from PIL import Image
from tensorflow.io import gfile

from rrlfd.data_utils import compress

flags.DEFINE_boolean('include_val_in_stats', False,
                     'If True, compute statistics over train and validation '
                     'data. Else use train only.')
flags.DEFINE_boolean('stats_from_large_dataset', False,
                     'If True, compute mean and std over whole dataset.')
FLAGS = flags.FLAGS


class DemoWriter:
  """Write a demonstration dataset."""

  def __init__(self, path, compress=True):  # pylint: disable=redefined-outer-name
    self.path = path
    self.compress = compress
    if gfile.exists(self.path):
      gfile.Remove(self.path)
    elif not gfile.exists(os.path.dirname(self.path)):
      gfile.makedirs(os.path.dirname(self.path))

  def compress_images(self, observations):
    """Compress image fields in observations."""
    compressed = []
    for t, obs in enumerate(observations):
      image_keys = [
          k for k in obs.keys() if 'depth' in k or 'mask' in k or 'rgb' in k]
      compressed.append(copy.deepcopy(observations[t]))
      for k in image_keys:
        if (isinstance(obs[k], bytes)
            or (len(obs[k].shape) > 2 and obs[k].shape[2] > 3)):
          compressed[t][k] = obs[k]
        else:
          compressed[t][k] = compress.compress_image(obs[k])
    return compressed

  def write_episode(self, observations, actions, rewards=None):
    if self.compress:
      observations = self.compress_images(observations)
    demo_dict = {'observations': observations, 'actions': actions}
    if rewards is not None:
      demo_dict['rewards'] = rewards
    with gfile.GFile(self.path, 'ab') as f:
      pickle.dump(demo_dict, f)


class DemoReader:
  """Load and preprocess a demonstration dataset."""

  def __init__(self, path, input_type='depth', max_to_load=None,
               max_demo_length=None, augment_frames=False, load_in_memory=True,
               decompress_once=True, agent=None, skip=0):
    self.path = path
    self.input_type = input_type
    self.max_to_load = max_to_load
    self.max_demo_length = max_demo_length
    self.augment_frames = augment_frames
    self.in_memory = load_in_memory
    self.episode_lengths = None
    self.num_timesteps = None
    self.action_mean = None
    self.action_std = None
    self.agent = agent
    self._stats_fixed = False
    self._decompress_once = decompress_once
    self._skipped_demos = skip

    if self.in_memory:
      self._load_demos()
      if self._decompress_once:
        # Keep decompressed observations in RAM.
        self.observations = self.decompress_all_images(self.observations)
      if self.max_to_load is None:
        self.max_to_load = len(self.observations)
    else:
      # TODO(minttu): 'skip' functionality for not-in-memory datasets.
      self.num_demos = None

  def _update_incremental_stats(
      self, timestep, running_mean, running_var, m2, num_timesteps,
      feats_to_skip=()):
    """Update running mean and variance with an incoming data point."""
    if isinstance(timestep, dict):
      if running_mean is None:
        running_mean, running_var, m2 = {}, {}, {}
      for k, v in timestep.items():
        if np.any([feat in k for feat in feats_to_skip]):
          continue
        if k not in running_mean:
          running_mean[k] = np.array(v)
          running_var[k] = 0
          m2[k] = 0
        else:
          new_mean = (running_mean[k] * num_timesteps + v) / (num_timesteps + 1)
          m2[k] += (v - running_mean[k]) * (v - new_mean)
          running_var[k] = m2[k] / (num_timesteps + 1)
          running_mean[k] = new_mean
    else:
      v = timestep
      if running_mean is None:
        running_mean = np.array(v)
        running_var = 0
        m2 = 0
      else:
        new_mean = (running_mean * num_timesteps + v) / (num_timesteps + 1)
        m2 += (v - running_mean) * (v - new_mean)
        running_var = m2 / (num_timesteps + 1)
        running_mean = new_mean
    return running_mean, running_var, m2

  def compute_dataset_stats(self):
    """Compute statistics that require a full pass over the data."""
    if self._stats_fixed:
      print('Stats unchanged')
      print('action mean', self.action_mean)
      print('action std', self.action_std)
      return
    if self.episode_lengths is None or self.action_mean is None:
      action_mean, action_var, action_m2 = None, None, None
      signal_mean, signal_var, signal_m2 = None, None, None
      episode_lengths = []
      num_timesteps = 0
      demo_idx = 0
      with gfile.GFile(self.path, 'rb') as f:
        while True:
          try:
            if self.in_memory:
              observations = self.observations[demo_idx]
              actions = self.actions[demo_idx]
            else:
              demo = pickle.load(f)
              observations = demo['observations']
              actions = demo['actions']
              if self.max_demo_length is not None:
                observations = observations[:self.max_demo_length]
                actions = actions[:self.max_demo_length]
            episode_lengths.append(len(observations))
            # For stats after preprocessing:
            # if self.agent is not None:
            #   actions = self.agent.preprocess_demo_actions(
            #       actions, observations)
            for t in range(len(observations)):
              obs_timestep = observations[t]
              act_timestep = actions[t]
              if FLAGS.include_val_in_stats or (
                  self.split_by_demo and self.episode_train_split[demo_idx]
                  or not self.split_by_demo
                  and self.episode_train_split[demo_idx][t]):
                signal_mean, signal_var, signal_m2 = (
                    self._update_incremental_stats(
                        obs_timestep, signal_mean, signal_var, signal_m2,
                        num_timesteps, feats_to_skip=['depth', 'mask', 'rgb']))
                action_mean, action_var, action_m2 = (
                    self._update_incremental_stats(
                        act_timestep, action_mean, action_var, action_m2,
                        num_timesteps))
                num_timesteps += 1

            demo_idx += 1
            if (self.max_to_load is not None
                and len(episode_lengths) >= self.max_to_load):
              break
            if self.num_demos is not None and demo_idx >= self.num_demos:
              break
          except EOFError:
            break
      if self.episode_lengths is None:
        self.episode_lengths = episode_lengths
        self.num_episodes = len(episode_lengths)
        self.num_timesteps = num_timesteps
      self.action_mean = action_mean
      self.signal_mean = signal_mean
      if isinstance(action_var, dict):
        action_std = {k: np.sqrt(v) for k, v in action_var.items()}
      else:
        action_std = np.sqrt(action_var)
      if isinstance(signal_var, dict):
        signal_std = {k: np.sqrt(v) for k, v in signal_var.items()}
      else:
        signal_std = np.sqrt(signal_var)
      self.action_std = action_std
      self.signal_std = signal_std
      print('action mean', self.action_mean)
      print('action std', self.action_std)
      print('signal mean', self.signal_mean)
      print('signal std', self.signal_std)

  def create_split(self, val_size=None, split_by_demo=None, seed=None,
                   episode_train_split=None):
    """Create fixed split or reuse an existing split for a dataset on disk."""
    self.split_by_demo = split_by_demo
    if self.in_memory:
      self.episode_lengths = [len(e) for e in self.observations]
    else:
      episode_lengths = []
      print('Loading episode lengths')
      with gfile.GFile(self.path, 'rb') as f:
        while True:
          try:
            demo = pickle.load(f)
            episode_lengths.append(len(demo['observations']))
            if (self.max_to_load is not None
                and len(episode_lengths) >= self.max_to_load):
              break
          except EOFError:
            break
      print('Loaded episode lengths for', len(episode_lengths), 'episodes')
      self.episode_lengths = episode_lengths
    self.num_episodes = len(self.episode_lengths)
    self.num_timesteps = np.sum(self.episode_lengths)
    # if self.episode_lengths is None:
    #   # Compute stats to calculate mean and std even if split is already set.
    #   self.compute_dataset_stats()
    if episode_train_split is None:
      if self.split_by_demo:
        data_size = self.num_episodes
      else:
        data_size = self.num_timesteps
      if val_size < 1:  # Treat as fraction of data size if < 1.
        val_size = (
            max(1, int(np.round(val_size * data_size)))
            if val_size > 0 else 0)
      else:
        val_size = int(np.round(val_size))
      train_split = (
          np.concatenate(
              [np.ones([data_size - val_size]), np.zeros([val_size])])
          .astype(np.bool))
      if seed is not None:
        np_random = np.random.RandomState(seed)
        np_random.shuffle(train_split)

      if self.split_by_demo:
        self.episode_train_split = train_split
        num_train_episodes = np.sum(train_split)
        num_val_episodes = len(train_split) - num_train_episodes
        print(f'Training on {num_train_episodes}, '
              f'validating on {num_val_episodes} episodes')
        print(f'Val episodes (seed {seed})')
        print([t for t in range(len(train_split)) if not train_split[t]])
      else:
        cum_len = 0
        self.episode_train_split = []
        for e_len in self.episode_lengths:
          self.episode_train_split.append(
              train_split[cum_len:cum_len + e_len])
          cum_len += e_len
        print('Training on', np.mean(train_split), 'of timesteps')
    else:
      self.episode_train_split = episode_train_split
    self.compute_dataset_stats()  # To compute on train split only

  def _keep_input_type(self, observation):
    if isinstance(observation, list):
      return np.stack([self._keep_input_type(obs) for obs in observation])
    elif self.input_type == 'depth':
      return np.expand_dims(observation['depth0'], axis=2)
    elif self.input_type == 'rgb':
      return observation['rgb0']
    else:
      return np.concatenate(
          [observation['rgb0'], np.expand_dims(observation['depth0'], axis=2)],
          axis=2)

  def decompress_all_images(self, observations):
    """Decompress frames of self.input_type, including multiple cameras.

    Drop image observations of other types from observation dictionaries but
    keep other non-image observations.

    Args:
      observations: List of observation dictionaries for a single episode or a
        list of lists of observation dictionaries for multiple episodes.

    Returns:
      uncompressed: observations with uncompressed image fields.
    """
    uncompressed = []
    for t, obs in enumerate(observations):
      if isinstance(obs, list):
        uncompressed.append(self.decompress_all_images(obs))
      else:
        uncompressed.append({})
        for k in obs.keys():
          if self.input_type in k:
            if isinstance(obs[k], np.ndarray):  # If image is not compressed.
              v = obs[k]
            else:
              v = compress.decompress_image(obs[k])
            uncompressed[t][k] = v
          elif 'depth' not in k and 'mask' not in k and 'rgb' not in k:
            uncompressed[t][k] = copy.deepcopy(obs[k])
    return uncompressed

  def _strip_digit_suffix(self, string):
    while string and string[-1].isdigit():
      string = string[:-1]
    return string

  def decompress_images(self, observations, randomize_camera=True):
    """Decompress frames of self.input_type and keep non-image observations."""
    uncompressed = []
    for t, obs in enumerate(observations):
      uncompressed.append({})
      image_key = None
      if self.input_type is not None:
        # TODO(minttu): Handle rgbd input.
        regex = re.compile(self.input_type + '.*')
        input_type_keys = filter(regex.match, obs.keys())
        if randomize_camera:
          image_key = np.random.choice(list(input_type_keys))
        elif self.input_type in input_type_keys:
          image_key = self.input_type
        else:
          image_key = self.input_type + '0'
      for k in obs.keys():
        if image_key is not None and k == image_key:
          if isinstance(obs[k], np.ndarray):  # If image is not compressed.
            v = obs[k]
          else:
            v = compress.decompress_image(obs[k])
          # Return one camera only. If keys have a digit suffix, change it to 0.
          if k[-1].isdigit():
            uncompressed[t][self._strip_digit_suffix(k) + '0'] = v
          else:
            uncompressed[t][k] = v
        elif 'depth' not in k and 'mask' not in k and 'rgb' not in k:
          uncompressed[t][k] = copy.deepcopy(obs[k])
    return uncompressed

  def _load_demos(self):
    """Load demonstrations as episodes.

    Assumes dataset fits in memory.
    """
    observations = []
    actions = []
    num_demos = 0
    # Hack to accommodate hand_vil data format.
    # rrlfd format:
    # observations: list of list of dict
    # actions: list of list of np array
    try:
      with gfile.GFile(self.path, 'rb') as f:
        skipped_demos = 0
        while True:
          try:
            demo = pickle.load(f)
            if skipped_demos < self._skipped_demos:
              skipped_demos += 1
              continue
            num_demos += 1
            obs_demo = demo['observations']
            act_demo = demo['actions']
            if self.max_demo_length is not None:
              obs_demo = obs_demo[:self.max_demo_length]
              act_demo = act_demo[:self.max_demo_length]
            observations.append(obs_demo)
            actions.append(np.stack(act_demo))
            if not FLAGS.stats_from_large_dataset:
              if self.max_to_load is not None and num_demos >= self.max_to_load:
                break
          except EOFError:
            break
    except IndexError:
      with gfile.GFile(self.path, 'rb') as f:
        dataset = pickle.load(f)
      dataset = dataset[:-self._skipped_demos]
      num_demos = len(dataset)
      if self.max_to_load is not None:
        num_demos = min(num_demos, self.max_to_load)
      for d in range(num_demos):
        obs_demo = []
        act_demo = []
        for t in range(len(dataset[d]['robot_info'])):
          obs_t = {self.input_type: dataset[d]['image_pixels'][t]}
          qvel = None
          # Splicing is specific to each env.
          if '_hand_door_' in self.path:
            qpos, qvel, palm, tactile = np.split(dataset[d]['robot_info'][t],
                                                 [28, 56, 59])
          elif '_hand_hammer_' in self.path:
            qpos, qvel, palm, tactile = np.split(dataset[d]['robot_info'][t],
                                                 [27, 54, 57])
          elif '_hand_pen_' in self.path:
            qpos, qvel, palm, tactile = np.split(dataset[d]['robot_info'][t],
                                                 [24, 48, 51])
          else:
            qpos, palm, tactile = np.split(dataset[d]['robot_info'][t],
                                           [30, 33])
          obs_t['qpos'] = qpos
          if qvel is not None:
            obs_t['qvel'] = qvel
          obs_t['palm_pos'] = palm
          obs_t['tactile'] = tactile
          act_t = dataset[d]['actions'][t]
          obs_demo.append(obs_t)
          act_demo.append(act_t)
        observations.append(obs_demo)
        actions.append(act_demo)
    self.observations = observations
    self.actions = actions
    self.num_demos = num_demos

  def add_demos(self, path, max_to_load=None):
    """Add additional demos to training set without affecting dataset stats."""
    # TODO(minttu): not-in-memory option
    self._stats_fixed = True
    if self.max_to_load is not None and FLAGS.stats_from_large_dataset:
      self.observations = self.observations[:self.max_to_load]
      self.actions = self.actions[:self.max_to_load]
    length_before = len(self.observations)
    observations = self.observations
    actions = self.actions
    lengths = self.episode_lengths
    num_new_demos = 0
    print('Before adding demos')
    print(len(self.observations), len(self.actions),
          len(self.episode_train_split))
    with gfile.GFile(path, 'rb') as f:
      while True:
        try:
          demo = pickle.load(f)
          num_new_demos += 1
          obs_demo = demo['observations']
          act_demo = demo['actions']
          if self.max_demo_length is not None:
            obs_demo = obs_demo[:self.max_demo_length]
            act_demo = act_demo[:self.max_demo_length]
          observations.append(obs_demo)
          actions.append(np.stack(act_demo))
          lengths.append(len(obs_demo))
          if self.split_by_demo:
            self.episode_train_split = np.concatenate(
                [self.episode_train_split, np.ones(1)])
          else:
            self.episode_train_split = np.concatenate(
                [self.episode_train_split, np.ones(len(obs_demo))])
          if max_to_load is not None and num_new_demos >= max_to_load:
            break
        except EOFError:
          break
    assert len(self.observations) == length_before + num_new_demos
    print('Addeed', num_new_demos, 'demos')
    print(len(self.observations), len(self.actions),
          len(self.episode_train_split))

  def transform_image(self, image, angle=(0, 0), translation=(0, 0),
                      scale=(1.0, 1.0)):
    """Apply affine transformation to PIL Image."""
    center = np.array(image.size) / 2
    angle = -np.array(angle) / 180. * math.pi
    x, y = center
    nx, ny = center + translation
    scale_x, scale_y = scale
    cos = math.cos(angle)
    sin = math.sin(angle)
    a = cos / scale_x
    b = sin / scale_x
    c = x - nx * a - ny * b
    d = -sin / scale_y
    e = cos / scale_y
    f = y - nx * d - ny * e
    image = image.transform(
        image.size, Image.AFFINE, (a, b, c, d, e, f), fillcolor=255)
    return image

  def augment_images(self, observations):
    """Apply image augmentation to images in observations."""
    augmented = []
    for t, obs in enumerate(observations):
      image_keys = [
          k for k in obs.keys() if 'depth' in k or 'mask' in k or 'rgb' in k]
      augmented.append(copy.deepcopy(observations[t]))
      for k in image_keys:
        im = obs[k]
        angle = 5
        translate = 0.04 * im.shape[0]
        rotation = np.random.uniform(low=-angle, high=angle)
        translation = np.random.uniform(
            low=-translate, high=translate, size=(2))
        translation = np.rint(translation).astype(np.int)
        im = Image.fromarray(im)
        im = self.transform_image(im, rotation, translation)
        im = np.asarray(im)
        augmented[t][k] = im
    return augmented

  def demo_in_train(self, demo_idx):
    return (self.split_by_demo and self.episode_train_split[demo_idx]
            or (not self.split_by_demo
                and np.any(self.episode_train_split[demo_idx])))

  def demo_in_val(self, demo_idx):
    return (self.split_by_demo and not self.episode_train_split[demo_idx]
            or (not self.split_by_demo
                and not np.all(self.episode_train_split[demo_idx])))

  def generate_train_timestep(self):
    """Generator to iterate over the train split."""
    demo_idx = 0
    train_pointer = gfile.GFile(self.path, 'rb')
    while True:
      try:
        if not self.in_memory:
          demo = pickle.load(train_pointer)
        if self.demo_in_train(demo_idx):
          # Preprocess only if (at least some time step) in train split.
          if self.in_memory:
            observations = self.observations[demo_idx]
            actions = self.actions[demo_idx]
          else:
            observations = demo['observations']
            actions = demo['actions']
            if self.max_demo_length is not None:
              observations = observations[:self.max_demo_length]
              actions = actions[:self.max_demo_length]
          if not self.in_memory or not self._decompress_once:
            # Decompress images of all viewpoints (if applicable) to keep
            # viewpoint constant across stacked frames.
            observations = self.decompress_all_images(observations)
          signals = [None for _ in observations]
          if FLAGS.clip_actions:
            actions = list(np.clip(actions, -1, 1))
          if self.agent is not None:
            observations, signals, actions = self.agent.normalize_demo(
                observations, actions, augment_frames=self.augment_frames,
                randomize_camera=True)
          for t in range(len(actions)):
            if self.split_by_demo:
              if self.episode_train_split[demo_idx]:
                yield observations[t], signals[t], actions[t], demo_idx, t
            elif self.episode_train_split[demo_idx][t]:
              yield observations[t], signals[t], actions[t], demo_idx, t
        demo_idx += 1
        if self.max_to_load is not None and demo_idx >= self.max_to_load:
          return
      except EOFError:
        return

  def generate_val_timestep(self):
    """Generator to iterate over the validation split."""
    demo_idx = 0
    val_pointer = gfile.GFile(self.path, 'rb')
    while True:
      try:
        if not self.in_memory:
          demo = pickle.load(val_pointer)
        if self.demo_in_val(demo_idx):
          # Preprocess only if (at least some time step) in val split.
          if self.in_memory:
            observations = self.observations[demo_idx]
            actions = self.actions[demo_idx]
          else:
            observations = demo['observations']
            actions = demo['actions']
            if self.max_demo_length is not None:
              observations = observations[:self.max_demo_length]
              actions = actions[:self.max_demo_length]
          if not self.in_memory or not self._decompress_once:
            # Decompress images of all viewpoints (if applicable) to keep
            # viewpoint constant across stacked frames.
            observations = self.decompress_all_images(observations)
          signals = [None for _ in observations]
          if FLAGS.clip_actions:
            actions = list(np.clip(actions, -1, 1))
          if self.agent is not None:
            observations, signals, actions = self.agent.normalize_demo(
                observations, actions, augment_frames=self.augment_frames,
                randomize_camera=True)
          for t in range(len(actions)):
            if self.split_by_demo:
              if not self.episode_train_split[demo_idx]:
                yield observations[t], signals[t], actions[t], demo_idx, t
            elif not self.episode_train_split[demo_idx][t]:
              yield observations[t], signals[t], actions[t], demo_idx, t
        demo_idx += 1
        if self.max_to_load is not None and demo_idx >= self.max_to_load:
          return
      except EOFError:
        return
