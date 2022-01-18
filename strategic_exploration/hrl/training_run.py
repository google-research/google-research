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

import gym
import random
import numpy as np
import os
import copy
import torch
import torch.optim as optim
import logging
from collections import deque
from codecs import open
from gtd.ml.torch.training_run import TorchTrainingRun
from gtd.ml.torch.utils import try_gpu
from gtd.ml.training_run import TrainingRuns
from strategic_exploration.hrl import data
from strategic_exploration.hrl.abstract_state import configure_abstract_state
from strategic_exploration.hrl.master import Master
from strategic_exploration.hrl.env_wrapper import get_env, OriginalPixelsWrapper, GifRecorder
from strategic_exploration.hrl.dqn import DQNPolicy
from strategic_exploration.hrl.graph_update import Traverse
from strategic_exploration.hrl.replay import ReplayBuffer
from strategic_exploration.hrl.rl import Experience, Episode
from strategic_exploration.hrl.state import State
from strategic_exploration.hrl.systematic_exploration import SystematicExploration
from strategic_exploration.hrl.utils import mean_with_default
from PIL import Image
from torch.nn.utils import clip_grad_norm
from tqdm import tqdm
from strategic_exploration.hrl.vecenv import SubprocVecEnv


class HRLTrainingRuns(TrainingRuns):

  def __init__(self, check_commit=True):
    data_dir = data.workspace.experiments
    # root of the Git repo
    src_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    super(HRLTrainingRuns, self).__init__(
        data_dir,
        src_dir,
        HRLTrainingRun.from_config,
        check_commit=check_commit)


class HRLTrainingRun(TorchTrainingRun):

  def __init__(self, config, save_dir):
    super(HRLTrainingRun, self).__init__(config, save_dir)

    self._env = get_env(config.env)

  @staticmethod
  def from_config(config, save_dir):
    """Returns the appropriate HRLTrainingRun subclass given the config.

        Necessary because different model types require different train
        methods.

        Args: config (Config)

        Returns:
            HRLTrainingRun
        """
    if config.policy.type == "dqn":
      return DQNTrainingRun(config, save_dir)
    elif config.policy.type == "systematic_exploration":
      return SystematicExplorationTrainingRun(config, save_dir)
    else:
      raise ValueError("policy type: {} not supported".format(
          config.policy.type))

  def close(self):
    pass


class DQNTrainingRun(HRLTrainingRun):
  """The TrainingRun corresponding to DQN policies."""

  def __init__(self, config, save_dir):
    super(DQNTrainingRun, self).__init__(config, save_dir)

    self._dqn = try_gpu(
        DQNPolicy.from_config(config.policy, self._env.action_space.n))
    optimizer = optim.Adam(self._dqn.parameters(), lr=config.learning_rate)
    self._train_state = self.checkpoints.load_latest(self._dqn, optimizer)

    self._replay_buffer = ReplayBuffer.from_config(config.buffer)

    # See configs/default-base.txt for documentation about these
    self._max_episode_len = config.max_episode_len
    self._buffer_size_start = config.buffer_size_start
    self._batch_size = config.batch_size
    self._sync_target_freq = config.sync_target_freq
    self._evaluate_freq = config.evaluate_freq
    self._episodes_to_evaluate = config.episodes_to_evaluate
    self._max_frames = config.max_frames
    self._update_freq = config.update_freq
    self._max_grad_norm = config.max_grad_norm

    self.workspace.add_dir("video", "video")

  def _evaluate(self, record=True):
    """Rolls out a single episode from the environment and returns the

        reward received (with test flag=True everywhere)

        Returns:
            float: reward for that episode
            list[Image]: all the frames, empty if record=False
        """
    if record:
      env = GifRecorder(self._env)
    state = env.reset()
    episode_reward = 0.
    for step in range(self._max_episode_len):
      action = self._dqn.act(state, True)
      next_state, reward, done, info = env.step(action)
      next_state = next_state
      episode_reward += reward
      state = next_state
      if done:
        break

    images = []
    if record:
      images = env.images
    return episode_reward, images

  def train(self):
    rewards = deque(maxlen=100)
    take_grad_step = lambda loss: self._take_grad_step(self._train_state, loss,
                                                       self._max_grad_norm)
    frames = 0  # number of training frames seen
    episodes = 0  # number of training episodes that have been played
    with tqdm(total=self._max_frames) as progress:
      # Each loop completes a single episode
      while frames < self._max_frames:
        state = self._env.reset()
        episode_reward = 0.
        episode_frames = 0
        # Each loop completes a single step, duplicates _evaluate() to
        # update at the appropriate frame #s
        for _ in range(self._max_episode_len):
          frames += 1
          episode_frames += 1
          action = self._dqn.act(state)
          next_state, reward, done, info = self._env.step(action)
          next_state = next_state
          episode_reward += reward
          # NOTE: state and next_state are LazyFrames and must be
          # converted to np.arrays
          self._replay_buffer.add(
              Experience(state, action, reward, next_state, done))
          state = next_state

          if len(self._replay_buffer) > self._buffer_size_start and \
                  frames % self._update_freq == 0:
            experiences, weights, indices = \
                    self._replay_buffer.sample(self._batch_size)
            td_error = self._dqn.update_from_experiences(
                experiences, weights, take_grad_step)
            new_priorities = \
                    np.abs(td_error.cpu().data.numpy()) + 1e-6
            self._replay_buffer.update_priorities(indices, new_priorities)

          if frames % self._sync_target_freq == 0:
            self._dqn.sync_target()

          if done:
            break

        episodes += 1
        rewards.append(episode_reward)
        stats = self._dqn.stats()
        stats["Episode Reward"] = episode_reward
        stats["Avg Episode Reward"] = mean_with_default(rewards, None)
        stats["Num Episodes"] = episodes
        progress.set_postfix(stats, refresh=False)
        progress.update(episode_frames)
        episode_frames = 0

        for k, v in stats.items():
          if v is not None:
            self.tb_logger.log_value(k, v, step=frames)

        if episodes % self._evaluate_freq == 0:
          test_rewards = []
          gif_images = []
          for _ in tqdm(range(self._episodes_to_evaluate), desc="Evaluating"):
            test_reward, images = self._evaluate()
            gif_images += images
            test_rewards.append(test_reward)
          save_path = os.path.join(self.workspace.video,
                                   "{}.gif".format(episodes))
          durations = [20] * len(gif_images)
          durations[-1] = 1000
          gif_images[0].save(
              save_path,
              append_images=gif_images[1:],
              save_all=True,
              duration=durations,
              loop=0)
          avg_test_reward = \
              sum(test_rewards) / float(len(test_rewards))
          print("Evaluation Reward: {}".format(avg_test_reward))
          self.tb_logger.log_value(
              "Evaluation Reward", avg_test_reward, step=frames)


class SystematicExplorationTrainingRun(HRLTrainingRun):
  """The TrainingRun corresponding to SystematicExploration policies."""

  def __init__(self, config, save_dir):
    super(SystematicExplorationTrainingRun, self).__init__(config, save_dir)

    def make_env(i):

      def _thunk():
        # Only ever try to visualize index 0
        if i == 0:
          return OriginalPixelsWrapper(get_env(config.env))
        return get_env(config.env)

      return _thunk

    configure_abstract_state(config.env.domain)
    self._nproc = config.num_processes
    self._env = SubprocVecEnv([make_env(i) for i in range(self._nproc)])

    Traverse.configure(config.edge_expansion_coeff)
    self._policy = try_gpu(
        Master.from_config(config.policy, self._env.action_space.n,
                           self._env.reset()[0], self._nproc,
                           config.env.domain))
    self._max_episodes = config.max_episodes
    self._video_freq = config.video_freq
    self._text_freq = config.text_freq
    self._stats_freq = config.stats_freq
    self._eval_freq = config.eval_freq
    self._eval_video_freq = config.eval_video_freq
    self._checkpoint_freq = config.checkpoint_freq
    self._max_checkpoints = config.max_checkpoints
    self._permanent_checkpoint = config.get("permanent_checkpoint", None)
    self.workspace.add_dir("visualizations", "visualizations")
    self.workspace.add_dir("traces", "traces")
    self.workspace.add_dir("video", "video")
    self.workspace.add_file("log", "log")
    logging.basicConfig(filename=self.workspace.log)
    self._checkpoint_number = 0

    self._teleport_frames = 0
    self._true_frames = 0  # Number of actually frames run, excludes teleport
    self._dead_frames = 0  # Number of frames simulated on dead edges
    self._dead_episodes = 0  # Number of episodes simulated on dead edges
    self._episode_nums = [0] * self._nproc  # Episode number of each proc
    self._best_reward = 0.  # Highest reward episode seen so far for proc 0

    self._load_latest_checkpoint()

  def train(self):
    with tqdm(total=self._max_episodes) as progress:
      # Set progress to right spot if reloading
      progress.update(sum(self._episode_nums))

      states = self._env.reset()
      test = False

      # Index 0 specific data
      episode_visualizer = EpisodeVisualizer()
      episode_length = 0
      episode_reward = 0.

      while sum(self._episode_nums) < self._max_episodes:
        actions_and_justifications, dead_count, episodes = \
                self._policy.act(states, test=test)
        self._dead_frames += dead_count
        self._dead_episodes += episodes
        actions, justifications = zip(*actions_and_justifications)
        next_states, rewards, dones, infos = self._env.step(actions)
        self._policy.observe(states, actions, rewards, next_states, dones)
        episode_visualizer.step(states[0], justifications[0])
        states = next_states

        for i, done in enumerate(dones):
          if dones[i]:
            self._episode_nums[i] += 1
            progress.update(1)

        self._true_frames += self._nproc
        self._teleport_frames += sum(info["steps"] - 1 for info in infos)
        episode_reward += rewards[0]
        episode_length += 1

        if dones[0]:
          total_frames = self._teleport_frames + \
                  self._true_frames + self._dead_frames
          if self._episode_nums[0] % self._checkpoint_freq == 0:
            self._save_checkpoint(total_frames)

          if self._episode_nums[0] % self._stats_freq == 0:
            stats = self._policy.stats()
            stats["METRICS/Episode Reward"] = episode_reward
            stats["METRICS/Episode Length"] = episode_length
            stats["METRICS/Total Frames"] = total_frames
            stats["METRICS/Dead Frames"] = self._dead_frames
            stats["METRICS/Teleport Frames"] = self._teleport_frames
            stats["METRICS/True Frames"] = self._true_frames
            stats["METRICS/Episode"] = \
                    sum(self._episode_nums) + self._dead_episodes

            for k, v in stats.items():
              if v is not None:
                self.tb_logger.log_value(k, v, step=total_frames)

            self.tb_logger.log_value(
                "METRICS/Reward Vs Frames", episode_reward, step=total_frames)

          if self._episode_nums[0] % self._video_freq == 0:
            episode_visualizer.save_video(self.workspace.video,
                                          self._episode_nums[0], False)
            self._policy.visualize(self.workspace.visualizations,
                                   self._episode_nums[0])

          if self._episode_nums[0] % self._text_freq == 0:
            with open(
                os.path.join(self.workspace.traces,
                             "{}.txt".format(self._episode_nums[0])),
                "w+") as f:
              f.write(str(self._policy))

          # Periodically evaluate
          if self._episode_nums[0] % self._eval_freq == 0:
            test = True
          else:
            # Logs the test episode
            if test:
              self.tb_logger.log_value(
                  "METRICS/Test Reward", episode_reward, step=total_frames)
              self.tb_logger.log_value(
                  "METRICS/Test Reward Vs Frames",
                  episode_reward,
                  step=total_frames)
              if (self._episode_nums[0] - 1) % self._eval_video_freq == 0:
                episode_visualizer.save_video(self.workspace.video,
                                              self._episode_nums[0], True)
            test = False

          if episode_reward > self._best_reward:
            self._best_reward = episode_reward
            episode_visualizer.save_video(self.workspace.video,
                                          self._episode_nums[0], False)
            with open(
                os.path.join(self.workspace.traces,
                             "{}.txt".format(self._episode_nums[0])),
                "w+") as f:
              f.write(str(self._policy))

          episode_visualizer.clear()
          episode_reward = 0.
          episode_length = 0

  def state_dict(self):
    # Saves master, checkpoint number, does not need to save env
    state = {
        "policy": self._policy.state_dict(),
        "checkpoint_num": self._checkpoint_number,
        "teleport_frames": self._teleport_frames,
        "true_frames": self._true_frames,
        "dead_frames": self._dead_frames,
        "dead_episodes": self._dead_episodes,
        "episode_nums": self._episode_nums,
        "best_reward": self._best_reward,
    }
    return state

  def load_state_dict(self, state_dict):
    self._policy.load_state_dict(state_dict["policy"])
    self._checkpoint_number = state_dict["checkpoint_num"]
    self._true_frames = state_dict["true_frames"]
    self._dead_frames = state_dict["dead_frames"]
    self._dead_episodes = state_dict["dead_episodes"]
    self._episode_nums = state_dict["episode_nums"]
    self._best_reward = state_dict["best_reward"]

  def _save_checkpoint(self, frames):
    metadata = os.path.join(self.workspace.checkpoints, "metadata")
    with open(metadata, "a") as f:
      f.write("{}: {}\n".format(self._checkpoint_number, frames))

    print("Saving checkpoint: {} at frames: {}".format(self._checkpoint_number,
                                                       frames))

    # Remove checkpoint
    if self._checkpoint_number >= self._max_checkpoints:
      stale_checkpoint = os.path.join(
          self.workspace.checkpoints,
          "{}.ckpt".format(self._checkpoint_number - self._max_checkpoints))
      try:
        os.remove(stale_checkpoint)
      except OSError as e:
        logging.warning(e)

    extension = "ckpt"
    if self._permanent_checkpoint is not None and \
            frames >= self._permanent_checkpoint:
      self._permanent_checkpoint = None
      extension = "perm"

    # self._checkpoint_number needs to get saved with the NEXT checkpoint
    # number
    self._checkpoint_number += 1
    state_dict = self.state_dict()
    save_path = os.path.join(
        self.workspace.checkpoints, "{}.{}".format(self._checkpoint_number - 1,
                                                   extension))
    torch.save(state_dict, save_path)

  def _load_latest_checkpoint(self):
    checkpoints = [
        c for c in os.listdir(self.workspace.checkpoints) if c[-5:] == ".ckpt"
    ]
    checkpoints.sort(key=lambda path: int(path[:-5]))

    if len(checkpoints) == 0:
      print("No checkpoints found, starting new job")
    else:
      checkpoint_file = os.path.join(self.workspace.checkpoints,
                                     checkpoints[-1])
      print("Resuming from checkpoint: {}".format(checkpoint_file))
      state_dict = torch.load(checkpoint_file)
      self.load_state_dict(state_dict)


class EpisodeVisualizer(object):
  """Takes data from an episode and logs it."""

  def __init__(self):
    self._states = []
    self._justifications = []

  def step(self, state, justification):
    """Appends data to the current episode.

        Args:
            state (State): current state in the episode
            justification (Justification): justification for the action taken in
              the state.
    """
    self._justifications.append(justification)
    self._states.append(state)

  def clear(self):
    """Marks the start of a new episode, clears out data from previous

        episode.
        """
    for state in self._states:
      state.drop_unmodified_pixels()

    self._justifications = []
    self._states = []

  def save_video(self, save_dir, episode_num, test):
    """Saves a video in save_dir, associated with episode_num.

        Args:
            save_dir (str): directory to save in
            episode_num (int): current episode number
            test (bool): whether episode was generated with test flag or not
    """

    def save_gif(frames, path):
      durations = [20] * len(frames)
      durations[-1] = 1000
      if len(durations) > 1:
        frames[0].save(
            path,
            append_images=frames[1:],
            save_all=True,
            duration=durations,
            loop=0)
      else:
        frames[0].save(path)

    test = "test-" if test else ""

    # Clean visualizations
    clean_frames = [
        Image.fromarray(state.unmodified_pixels, "RGB")
        for state in self._states
    ]
    clean_path = os.path.join(save_dir,
                              "{}clean-{}.gif".format(test, episode_num))
    save_gif(clean_frames, clean_path)

    # Abstract state visualizations
    visualizations = [
        j.visualize(state)
        for j, state in zip(self._justifications, self._states)
    ]
    vis_path = os.path.join(save_dir, "{}{}.gif".format(test, episode_num))
    save_gif(visualizations, vis_path)
