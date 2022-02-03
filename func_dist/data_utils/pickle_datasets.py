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

"""Load offline datasets stored in pickle format."""

import abc
import gzip
import io
import pickle
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.io import gfile
from func_dist.data_utils import image_utils

TensorflowSignature = Tuple[tf.dtypes.DType, Ellipsis]


class Dataset(abc.ABC):
  """Base class for pickle datasets."""

  def __init__(self, path, val_fraction = None):
    self._path = path
    self._val_fraction = val_fraction
    self._train_split = None
    self._split_by_episodes = None

  def _load_gzipped_pickle(self):
    with gfile.GFile(self._path, 'rb') as f:
      f = gzip.GzipFile(fileobj=f)
      data = io.BytesIO(f.read())
      data = pickle.load(data)
    return data

  def _create_train_val_split(
      self, data_size, shuffle = False, seed = None
      ):
    """Create a binary array of length data_size, equal to 1 for training data.

    Args:
      data_size: the number of data points to split into two sets.
      shuffle: If True, the split is randomized. Else the contiguous data points
        from the beginning of the dataset will form the training split.
      seed: random seed used for shuffling, if applicable.

    Returns:
      train_split: a binary numpy array of size [data_size], with 1 for each
          data point in the training split and 0 for each data point in the
          validation split.
    """
    val_size = int(np.round(data_size * self._val_fraction))
    val_size = max(1, val_size) if self._val_fraction > 0 else 0
    train_size = data_size - val_size
    train_split = np.concatenate(
        [np.ones([train_size], dtype=np.int32),
         np.zeros([val_size], dtype=np.int32)])
    if shuffle:
      np.random.RandomState(seed).shuffle(train_split)
    return train_split

  @abc.abstractmethod
  def _get_iterator(self, dataset_type, eval_mode, **kwargs):
    """Creates an iterator over the dataset."""

  @abc.abstractmethod
  def get_iterator_signature(self, dataset_type):
    """Get tensorflow signature of an iterator over the dataset."""

  def get_train_split(self, _):
    return self._train_split

  def generate_train_data(self, dataset_type, eval_mode, **kwargs):
    iterator = self._get_iterator(dataset_type, eval_mode, **kwargs)
    train_split = self.get_train_split(dataset_type)
    for e, example in enumerate(iterator):
      if self._split_by_episodes:  # Use episode index to determine split.
        example, e = example
      if train_split[e]:
        yield example

  def generate_val_data(self, dataset_type, eval_mode, **kwargs):
    iterator = self._get_iterator(dataset_type, eval_mode, **kwargs)
    train_split = self.get_train_split(dataset_type)
    for e, example in enumerate(iterator):
      if self._split_by_episodes:  # Use episode index to determine split.
        example, e = example
      if not train_split[e]:
        yield example


class PairedDataset(Dataset):
  """Dataset consisting of paired observations from two different domains."""

  def __init__(self, path, val_fraction = None,
               augment_frames = True):
    super().__init__(path, val_fraction)
    data = self._load_gzipped_pickle()
    self._pairs = (image_utils.shape_img(data['observations']),
                   image_utils.shape_img(data['next_observations']))
    self._train_split = self._create_train_val_split(self.__len__())
    self._split_by_episodes = False
    self._augment_frames = augment_frames

  def __len__(self):
    return len(self._pairs[0])

  def _get_iterator(self, unused_dataset_type, eval_mode,
                    augment_frames=None):
    augment_frames = (
        augment_frames if augment_frames is not None
        else self._augment_frames and not eval_mode)
    for obs1, obs2 in zip(*self._pairs):
      if augment_frames:
        obs1 = image_utils.random_crop_image(obs1)
        obs2 = image_utils.random_crop_image(obs2)
      yield obs1, obs2

  def get_iterator_signature(self, _):
    return (tf.float32, tf.float32)


class EpisodicDataset(Dataset):
  """Dataset consisting of episodes, i.e. continuous series of observations.

  Supports datasets stored as either flat time steps (with episode breaks
  denoted by terminals field) or as a list of episodes.
  """

  def __init__(self,
               path,
               val_fraction = 0.,
               end_on_success = True,
               include_zero_distance_pairs = False,
               split_by_episodes = True,
               shuffle_splits = True,
               shuffle_seed = None,
               augment_goals = True,
               augment_frames = True,
               subsampling_threshold = 0.0):
    super().__init__(path, val_fraction)
    data = self._load_gzipped_pickle()
    if isinstance(data, dict):
      data = self._flat_data_to_episodes(data)

    for e in range(len(data)):
      if end_on_success:
        success_t = np.argmax(data[e]['rewards'])
        for k in data[e]:
          data[e][k] = data[e][k][:success_t + 1]
        data[e]['terminals'][-1] = 1
      for image_k in ['observations', 'next_observations']:
        data[e][image_k] = image_utils.shape_img(data[e][image_k])

    # Enforce a minimum difference between consecutive timesteps.
    subsampled_data = []
    num_total_frames = 0
    num_total_summary_frames = 0
    for e in range(len(data)):
      subsampled_data.append({k: [v[0]] for k, v in data[e].items()})
      num_total_frames += 1
      num_total_summary_frames += 1
      for t in range(1, len(data[e]['observations'])):
        num_total_frames += 1
        if np.linalg.norm(
            data[e]['observations'][t]
            - subsampled_data[e]['observations'][-1]) >= subsampling_threshold:
          for k in data[e]:
            subsampled_data[e][k].append(data[e][k][t])
          num_total_summary_frames += 1
      # Add one of each for the final next_observation.
      num_total_frames += 1
      num_total_summary_frames += 1
      for k in subsampled_data[e]:
        subsampled_data[e][k] = np.array(subsampled_data[e][k])

    print('total frames', num_total_frames, 'summary frames',
          num_total_summary_frames, 'ratio:',
          num_total_summary_frames / num_total_frames)
    data = subsampled_data

    self._flat_observations = []
    self._is_terminal = []
    for episode in data:
      for t, observation in enumerate(episode['observations']):
        self._flat_observations.append(observation)
        self._is_terminal.append(False)
        if episode['terminals'][t]:
          self._flat_observations.append(episode['next_observations'][t])
          self._is_terminal.append(True)

    self._data = data

    self._train_split: Dict[str, np.ndarray] = {}
    split_length = self.num_episodes() if split_by_episodes else self.__len__()
    self._train_split['obs'] = self._create_train_val_split(
        split_length, shuffle=shuffle_splits, seed=shuffle_seed)
    # Goal augmentation increases distance dataset size:
    if augment_goals and not split_by_episodes:
      raise NotImplementedError(
          'Train split by time steps not implemented with goal augmentation.')
    self._train_split['dist'] = np.copy(self._train_split['obs'])

    self._augment_goals = augment_goals
    self._augment_frames = augment_frames
    self._split_by_episodes = split_by_episodes

    self._include_zero_distance_pairs = include_zero_distance_pairs

  def _flat_data_to_episodes(self, flat_data):
    """Separate a flat dataset of time steps to a list of episodes."""
    data = []
    episode_start = 0
    for t in range(len(flat_data['observations'])):
      if flat_data['terminals'][t]:
        episode = {k: v[episode_start:t + 1] for k, v in flat_data.items()}
        data.append(episode)
        episode_start = t + 1
    return data

  def __len__(self):
    return len(self._flat_observations)

  def episode_lengths(self):
    return np.array(
        [len(episode['observations']) + 1 for episode in self._data])

  def num_nonterminal_observations(self):
    return np.sum([len(episode['observations']) for episode in self._data])

  def num_episodes(self):
    return len(self._data)

  def _get_iterator(self, dataset_type, eval_mode, **kwargs):
    if dataset_type == 'obs':
      iterator = self.generate_observations(eval_mode, **kwargs)
    elif dataset_type == 'dist':
      iterator = self.generate_observation_pairs(eval_mode, **kwargs)
    return iterator

  def get_iterator_signature(self, dataset_type='obs'):
    if dataset_type == 'obs':
      signature = tf.float32
    elif dataset_type == 'dist':
      signature = (tf.float32, tf.float32, tf.int64, tf.int64, tf.int64)
    return signature

  def get_train_split(self, dataset_type):
    return self._train_split[dataset_type]

  def get_goal_image(self, episode_idx=0):
    return self._data[episode_idx]['next_observations'][-1]

  def generate_observations(self, eval_mode, augment_frames=None):
    """Generate all observations s sequentially from the dataset."""
    episode_idx = 0
    augment_frames = (
        augment_frames if augment_frames is not None
        else self._augment_frames and not eval_mode)
    for t, obs in enumerate(self._flat_observations):
      if augment_frames:
        obs = image_utils.random_crop_image(obs)
      if self._split_by_episodes:
        yield obs, episode_idx
      else:
        yield obs
      if self._is_terminal[t]:
        episode_idx += 1

  def generate_observation_pairs(
      self, eval_mode, augment_frames=None, augment_goals=None):
    """Generate (s, g, d) tuples, where d is the duration from state s to g."""
    def _get_observation_pair(
        observations, start, end, episode_idx, augment_frames):
      obs = observations[start]
      goal_obs = observations[end]
      if augment_frames:
        obs, goal_obs = image_utils.random_crop_image_pair(obs, goal_obs)
      t_offset = end - start
      if self._split_by_episodes:
        # Return episode idx so we can check against train split.
        return (obs, goal_obs, t_offset, episode_idx, start), episode_idx
      else:
        return obs, goal_obs, t_offset, episode_idx, start

    augment_frames = (
        augment_frames if augment_frames is not None
        else self._augment_frames and not eval_mode)
    augment_goals = (
        augment_goals if augment_goals is not None
        else self._augment_goals and not eval_mode)

    for e, episode in enumerate(self._data):
      all_obs = np.concatenate(
          [episode['observations'], episode['next_observations'][-1:]], axis=0)
      for t1 in range(len(all_obs)):
        if augment_goals:
          t2s = range(t1, len(all_obs))
          if eval_mode:
            # Choose just one t2 in order to keep evaluation time manageable.
            t2s = [np.random.choice(t2s)]
        else:
          t2s = [len(all_obs) - 1]
        for t2 in t2s:
          t_offset = t2 - t1
          if t_offset > 0 or self._include_zero_distance_pairs:
            yield _get_observation_pair(all_obs, t1, t2, e, augment_frames)


class InteractionDataset(EpisodicDataset):
  """Dataset consisting of (s, a, [r,] s') transitions.

  Stored data should contain at least 'observations', 'actions', 'rewards',
  'next_observations' and 'terminals'.
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    is_nonterminal = np.logical_not(self._is_terminal)
    self._train_split['interaction'] = np.copy(self._train_split['obs'])
    if not self._split_by_episodes:
      # All time steps are included in 'obs' split: not all are nonterminal.
      self._train_split['interaction'] = (
          self._train_split['interaction'][is_nonterminal])

  def _get_iterator(self, dataset_type, eval_mode, **kwargs):
    if dataset_type == 'interaction':
      iterator = self.generate_transitions(eval_mode, **kwargs)
    else:
      iterator = super()._get_iterator(dataset_type, eval_mode, **kwargs)
    return iterator

  def generate_transitions(self, eval_mode, augment_frames=None):
    """Generate (state, action, reward, next state) tuples from the dataset."""
    augment_frames = (
        augment_frames if augment_frames is not None
        else self._augment_frames and not eval_mode)
    for e, episode in enumerate(self._data):
      for t in range(len(episode['observations'])):
        s = episode['observations'][t]
        a = episode['actions'][t]
        r = episode['rewards'][t]
        next_s = episode['next_observations'][t]
        if augment_frames:
          s, next_s = image_utils.random_crop_image_pair(s, next_s)
        if self._split_by_episodes:
          yield (s, a, r, next_s), e
        else:
          yield s, a, r, next_s


class ObservationDataset(EpisodicDataset):
  """Dataset consisting of (observation)->(next observation) transitions.

  Stored data should contain at least 'observations', 'next_observations' and
  'terminals'.
  """

  def __init__(self, path, val_fraction, *args, **kwargs):
    end_on_success = False
    super().__init__(path, val_fraction, end_on_success, *args, **kwargs)
