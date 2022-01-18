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
"""Helper classes for sample generation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import collections
import heapq
import random
import numpy as np
import six
import tensorflow.compat.v1 as tf

Sample = collections.namedtuple('Sample', 'traj prob')
Traj = collections.namedtuple('Traj', 'env_name features actions rewards')
ACTION_MAP = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}


def chunker(seq, size):
  # https://stackoverflow.com/questions/434287
  return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def path_to_str(path):
  return ''.join([ACTION_MAP[a] for a in path])


def traj_to_program(traj):
  return path_to_str(traj.actions)


def normalize_probs(samples, smoothing=1e-8):
  """Normalize the probability of the samples (in each env) to sum to 1.0."""
  sum_prob_dict = {}
  for s in samples:
    name = s.traj.env_name
    if name in sum_prob_dict:
      sum_prob_dict[name] += s.prob + smoothing
    else:
      sum_prob_dict[name] = s.prob + smoothing
  new_samples = []
  for s in samples:
    new_prob = (s.prob + smoothing) / sum_prob_dict[s.traj.env_name]
    new_samples.append(Sample(traj=s.traj, prob=new_prob))
  return new_samples


@six.add_metaclass(abc.ABCMeta)
class ReplayBuffer():

  @abc.abstractmethod
  def save(self, samples):
    raise NotImplementedError

  @abc.abstractmethod
  def replay(self, envs):
    raise NotImplementedError


class AllGoodReplayBuffer(ReplayBuffer):
  """Replay Buffer containing only the high rewarding programs."""

  def __init__(self, env_dict=None):
    self._buffer = dict()
    self.program_dict = dict()
    self.prob_sum_dict = dict()
    self._env_dict = env_dict

  def has_found_solution(self, env_name):
    return env_name in self._buffer and self._buffer[env_name]

  def contain(self, traj):
    name = traj.env_name
    if name not in self.program_dict:
      return False
    program = traj_to_program(traj)
    if program in self.program_dict[name]:
      return True
    else:
      return False

  @property
  def env_dict(self):
    return self._env_dict

  @property
  def size(self):
    n = 0
    for _, v in self._buffer.iteritems():
      n += len(v)
    return n

  def save(self, samples):
    trajs = [s.traj for s in samples]
    self.save_trajs(trajs)

  def save_trajs(self, trajs):
    """Save the trajs to the replay buffer."""
    total_returns = [sum(t.rewards) for t in trajs]
    for t, return_ in zip(trajs, total_returns):
      name = t.env_name
      program = traj_to_program(t)
      if (return_ == 1.0 and not (name in self.program_dict and
                                  (program in self.program_dict[name]))):
        if name in self.program_dict:
          self.program_dict[name].add(program)
        else:
          self.program_dict[name] = set([program])
        if name in self._buffer:
          self._buffer[name].append(t)
        else:
          self._buffer[name] = [t]

  def all_samples(self, env_names, agent=None):
    """Returns all the samples in the replay buffer."""
    trajs = []
    # Collect all the trajs for the selected envs.
    for name in env_names:
      if name in self._buffer:
        trajs += self._buffer[name]
    if agent is None:
      # All traj has the same probability, since it will be
      # normalized later, we just assign them all as 1.0.
      probs = [1.0] * len(trajs)
    else:
      probs = agent.compute_probs(trajs, self._env_dict)
    samples = [Sample(traj=t, prob=p) for t, p in zip(trajs, probs)]
    return samples

  def get_score_sums(self, samples):
    return [self._score_sum_dict[s.traj.env_name] for s in samples]

  def replay(self, env_names, n_samples=1, agent=None, use_top_k_samples=False):
    all_samples = self.all_samples(env_names, agent)
    env_sample_dict = {name: [] for name in env_names if name in self._buffer}
    for s in all_samples:
      env_sample_dict[s.traj.env_name].append(s)

    replay_samples = []
    for name, samples in env_sample_dict.iteritems():
      # Compute the sum of prob of replays in the buffer.
      self.prob_sum_dict[name] = sum([sample.prob for sample in samples])
      # Randomly samples according to their probs.
      if use_top_k_samples:
        len_samples = len(samples)
        if len_samples <= n_samples:
          # Repeat the samples to get n_samples
          to_repeat = n_samples // len_samples + (n_samples % len_samples > 0)
          if to_repeat > 1:
            samples = sorted(samples, key=lambda s: s.prob, reverse=True)
          selected_samples = (samples * to_repeat)[:n_samples]
        else:
          # Select the top k samples weighted by their probs.
          selected_samples = heapq.nlargest(
              n_samples, samples, key=lambda s: s.prob)
        replay_samples += normalize_probs(selected_samples)
      else:
        samples = normalize_probs(samples)
        # pylint: disable=g-complex-comprehension
        selected_samples = [
            samples[i] for i in np.random.choice(
                len(samples), n_samples, p=[sample.prob for sample in samples])
        ]
        # pylint: enable=g-complex-comprehension
        replay_samples += [
            Sample(traj=s.traj, prob=1.0 / n_samples) for s in selected_samples
        ]

    return replay_samples


class SampleGenerator(object):
  """Simple class for batch generation of samples."""

  def __init__(self,
               replay_buffer,
               agent=None,
               min_replay_weight=0.1,
               n_samples=1,
               explore=False,
               use_top_k_samples=False,
               objective='mapo'):
    self.replay_buffer = replay_buffer
    self.env_dict = replay_buffer.env_dict
    self.agent = agent
    self.objective = objective
    self._min_replay_weight = min_replay_weight
    self._n_samples = n_samples
    self._explore = explore
    self._use_top_k_samples = use_top_k_samples
    self._counter = 0

  def reweight_samples(self, samples, use_clipping=False, fn='replay'):
    """Reweights the prob. of each sample according to the MAPO objective."""
    new_samples = []
    for sample in samples:
      name = sample.traj.env_name
      if name in self.replay_buffer.prob_sum_dict:
        replay_prob = self.replay_buffer.prob_sum_dict[name]
        if use_clipping:
          replay_prob = max(self._min_replay_weight, replay_prob)
      else:
        replay_prob = 0.0
      if fn == 'policy':
        replay_prob = 1.0 - replay_prob
      new_samples.append(
          Sample(traj=sample.traj, prob=sample.prob * replay_prob))
    return new_samples

  def get_score_sums(self, samples):
    return self.replay_buffer.get_score_sums(samples)

  def generate_samples(self, batch_size, debug=False):
    """Generate training samples."""
    all_env_names = list(self.env_dict.keys())
    drop_last = len(all_env_names) >= batch_size
    if self.objective != 'iml':
      agent = self.agent
    else:
      agent = None
    while True:
      # Randomly shuffle the env_names
      random.shuffle(all_env_names)
      for env_names in chunker(all_env_names, batch_size):
        if drop_last and (len(env_names) < batch_size):
          continue
        envs = [self.env_dict[name] for name in env_names]
        # Sample trajectories using the agent's policy for exploration
        if self._explore:
          explore_samples = self.agent.generate_samples(envs, greedy=True)
          explore_samples = [
              s for s in explore_samples if sum(s.traj.rewards) > 0
          ]
          # Update the replay buffer
          self.replay_buffer.save(explore_samples)

        # Sample trajectories from the replay buffer
        replay_samples = self.replay_buffer.replay(
            env_names,
            agent=agent,
            n_samples=self._n_samples,
            use_top_k_samples=self._use_top_k_samples)
        if debug and self._counter % 100 == 0:
          replay_names = list(self.replay_buffer.prob_sum_dict.keys())
          tf.logging.info('Replay: {}'.format([
              self.replay_buffer.prob_sum_dict[name]
              for name in replay_names[:10]
          ]))
        replay_samples = self.reweight_samples(
            replay_samples, use_clipping=True)
        train_samples = replay_samples

        contexts = [
            self.env_dict[sample.traj.env_name].context + 1
            for sample in train_samples
        ]
        self._counter += 1
        yield (train_samples, contexts)


class BufferScorer(object):
  """Replay buffer ranker."""

  def __init__(self, score_init):
    tf.logging.info('Score init {}'.format(score_init))
    self.score_weights = score_init

  def get_scores(self, feature_list):
    scores = [np.dot(self.score_weights, f) for f in feature_list]
    return scores
