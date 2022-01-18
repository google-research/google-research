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
"""Atari env factory."""

import collections
import functools as fct
import math
import os
import random
import tempfile

from absl import flags
from absl import logging
import atari_py  # pylint: disable=unused-import
import gym
from gym.spaces import box
import gym.wrappers
import inflection
import numpy as np
import tensorflow as tf
from tf_agents.environments import suite_atari

from muzero import core as mzcore
import cv2

flags.DEFINE_string('game_name', 'Pong', 'Name of the Atari game.')
flags.DEFINE_string(
    'game_mode', 'Deterministic',
    'Mode of the Atari game ["", "NoFrameskip", "Deterministic"].')
flags.DEFINE_string('game_version', 'v4',
                    'Version of the Atari game ["v0", "v4"].')
flags.DEFINE_integer('screen_size', 84, 'Screen size of the Atari env.')
flags.DEFINE_integer(
    'full_action_space', 0,
    'If yes (1) then use the full action space of 18 actions, otherwise (0) use the minimal action set for the given game.'
)
flags.DEFINE_integer(
    'n_history_frames', 16,
    'Number of historic frames to encode for the observation.')
flags.DEFINE_integer(
    'encode_actions', 1,
    'Whether the actions should be encoded as part of the observation (1) or not (0).'
)
flags.DEFINE_integer(
    'quit_on_invalid_action', 0,
    'If yes (1) then game is terminated after invalid action with score of -value_range - 1'
)
flags.DEFINE_integer(
    'terminal_on_life_lost', 0,
    'If yes (1) then game is terminated after first lost life. Otherwise (0), the game is played unitl the environment returns game over.'
)
flags.DEFINE_integer('grayscale', 1, 'Use grayscale observation.')
flags.DEFINE_integer('pretrain_buffer_size', 100,
                     'Buffer size for pretraining collection.')

FLAGS = flags.FLAGS

RANGES = ((-300., 300.), (-300., 300.))


def get_game_id():
  copy_rom()
  return suite_atari.game(
      name=FLAGS.game_name, mode=FLAGS.game_mode, version=FLAGS.game_version)


@fct.lru_cache(maxsize=None)
def copy_rom():
  """Copies the ROM file to a local directory."""
  local_roms = tempfile.mkdtemp(prefix='tmp_atari_roms')
  rom_name = inflection.underscore(FLAGS.game_name) + '.bin'
  logging.info('Copy ROM %s from %s to %s', rom_name, FLAGS.atari_roms_path,
               local_roms)
  tf.io.gfile.copy(
      os.path.join(FLAGS.atari_roms_path, rom_name),
      os.path.join(local_roms, rom_name))
  FLAGS.atari_roms_path = local_roms


def get_descriptor():
  """Creates an EnvironmentDescriptor."""
  plane_size = FLAGS.n_history_frames if (FLAGS.encode_actions == 0) else int(
      2 * FLAGS.n_history_frames)
  if not FLAGS.grayscale:
    plane_size *= 3
  reward_range, value_range = RANGES
  reward_range = tuple(map(mzcore.inverse_contractive_mapping, reward_range))
  value_range = tuple(map(mzcore.inverse_contractive_mapping, value_range))
  observation_space = box.Box(
      low=0.,
      high=1.,
      shape=(FLAGS.screen_size, FLAGS.screen_size, plane_size),
      dtype=np.float32)
  return mzcore.EnvironmentDescriptor(
      observation_space=observation_space,
      action_space=gym.make(
          get_game_id(),
          full_action_space=(FLAGS.full_action_space == 1)).action_space,
      reward_range=reward_range,
      value_range=value_range,
      pretraining_space=gym.spaces.Tuple([
          observation_space,
          observation_space,
      ]),
  )


def create_environment(task, training=True):  # pylint: disable=missing-docstring
  logging.info('Creating environment: {}'.format(get_game_id()))  # pylint: disable=logging-format-interpolation
  env = AtariEnv.from_game_name(
      name=get_game_id(),
      max_episode_steps=int(FLAGS.num_action_repeats * 27000),
      frame_skip=FLAGS.num_action_repeats,
      quit_on_invalid_action=bool(FLAGS.quit_on_invalid_action == 1),
      penalty_for_invalid_action=RANGES[0][0],
      screen_size=FLAGS.screen_size,
      encode_actions=bool(FLAGS.encode_actions == 1),
      n_history_frames=FLAGS.n_history_frames)
  env.seed(task)
  if not training and FLAGS.debug:
    env = gym.wrappers.Monitor(
        env,
        '/tmp/atari_video',
        force=True,
        video_callable=lambda x: int(math.pow(x, 1 / 3) + .5)**3 == x)
  return env


class AtariEnv(gym.core.Wrapper):
  """Environment for Atari games."""

  def __init__(self,
               env,
               n_history_frames=8,
               encode_actions=True,
               frame_skip=4,
               quit_on_invalid_action=True,
               penalty_for_invalid_action=0,
               terminal_on_life_loss=True,
               screen_size=96,
               n_noop=30,
               normalize_observation=True,
               dtype=np.float32):
    """Constructor for an Atari 2600 preprocessor.

    Args:
      env: Gym environment whose observations are preprocessed.
      n_history_frames: int, number of historic frames to return as observation.
      encode_actions: bool, if True actions are part of the observation.
      frame_skip: int, the frequency at which the agent experiences the game.
      quit_on_invalid_action: bool, if True, the game is lost when issuing an
        invalid action.
      penalty_for_invalid_action: int, Penalty for issuing invalid action.
      terminal_on_life_loss: bool, If True, the step() method returns
        is_terminal=True whenever a life is lost. See Mnih et al. 2015.
      screen_size: int, Size of a resized Atari 2600 frame.
      n_noop: int, Number of NOOP actions at the beginning of an episode.
      normalize_observation: bool, If True the observation is between 0 and 1,
        otherwise it is between 0 and 255.
      dtype: dtype of the observation.

    Raises:
      ValueError: if frame_skip or screen_size are not strictly positive.
    """
    super().__init__(env)
    self.observation_space = get_descriptor().observation_space

    if frame_skip <= 0:
      raise ValueError(
          'Frame skip should be strictly positive, got {}'.format(frame_skip))
    if screen_size <= 0:
      raise ValueError(
          'Target screen size should be strictly positive, got {}'.format(
              screen_size))

    self.terminal_on_life_loss = terminal_on_life_loss
    self.frame_skip = frame_skip
    self.screen_size = screen_size
    self.normalize_observation = normalize_observation
    self.encode_actions = encode_actions
    self.n_noop = n_noop

    self.screen_buffer = ScreenBuffer(
        size=n_history_frames,
        shape=(screen_size, screen_size) if FLAGS.grayscale else
        (screen_size, screen_size, 3),
        action_space=self.action_space.n,
        normalize=self.normalize_observation,
        dtype=dtype)

    self.game_over = False
    self.lives = 0  # Will need to be set by reset().

    self.quit_on_invalid_action = quit_on_invalid_action
    self.penalty_for_invalid_action = penalty_for_invalid_action
    self.minimal_action_set = self.ale.getMinimalActionSet()
    logging.info('Minimal action set: %s', str(self.minimal_action_set))
    logging.info('Action meanings: %s',
                 str(env.unwrapped.get_action_meanings()))

    if len(self.minimal_action_set) == self.action_space.n:
      # environment in minimal action set setting
      self.minimal_action_set = list(range(self.action_space.n))

  def reset(self):
    """Resets the environment.

    Returns:
      observation: numpy array, the initial observation emitted by the
        environment.
    """

    def noop_action(n):
      for _ in range(n):
        self.env.step(0)
        self.screen_buffer.add(self._get_observation(), action=0)

    self.env.reset()
    self.screen_buffer.reset()
    self.screen_buffer.add(self._get_observation(), action=0)
    noop_action(random.randrange(self.n_noop))

    self.lives = self.env.ale.lives()
    self.game_over = False

    observation = self.screen_buffer.as_numpy(
        include_actions=self.encode_actions)

    return observation, {}

  def step(self, action, training_steps=0):
    """Applies the given action in the environment.

    Args:
      action: The action to be executed.
      training_steps: The number of training steps done on the learner.

    Returns:
      observation: numpy array, the observation following the action.
      reward: float, the reward following the action.
      game_over: bool, whether the environment has reached a terminal state.
        This is true when a life is lost and terminal_on_life_loss, or when the
        episode is over.
      info: Gym API's info data structure.
    """
    accumulated_reward = 0.

    for _ in range(self.frame_skip):
      _, reward, game_over, info = self.env.step(action)
      accumulated_reward += reward
      self.screen_buffer.add(self._get_observation(), action=action)

      if self.quit_on_invalid_action and action not in self.minimal_action_set:
        # disqualify agent
        accumulated_reward = self.penalty_for_invalid_action
        game_over = True
        break

      if not game_over and self.terminal_on_life_loss:
        new_lives = self.env.ale.lives()
        game_over = new_lives < self.lives  # did we just lose one?
        self.lives = new_lives

      if game_over:
        break

    observation = self.screen_buffer.as_numpy(
        include_actions=self.encode_actions)
    self.game_over = game_over
    return observation, accumulated_reward, game_over, info

  def _get_grayscale_observation(self):
    """Returns the current observation in grayscale.

    Does a re-size on the image (and additional pre-processing for Pong).

    Args:

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    raw_observation = self.env.ale.getScreenGrayscale()
    observation = raw_observation.squeeze()

    # special processing for PONG
    if 'Pong' in self.spec.id:
      # crop away the upper part (scoring board)
      observation = observation[35:195, :]
      observation[observation == 87] = 0  # remove the background
      observation[observation != 0] = 255

    # resize the image
    observation = cv2.resize(
        observation, (self.screen_size, self.screen_size),
        interpolation=cv2.INTER_LINEAR)
    return observation

  def _get_color_observation(self):
    """Returns the current observation in grayscale.

    Does a re-size on the image (and additional pre-processing for Pong).

    Args:

    Returns:
      observation: numpy array, the current observation in grayscale.
    """
    observation = self.env.ale.getScreenRGB2()

    # special processing for PONG
    if 'Pong' in self.spec.id:
      # crop away the upper part (scoring board)
      observation = observation[35:195]

    # resize the image
    observation = cv2.resize(
        observation, (self.screen_size, self.screen_size),
        interpolation=cv2.INTER_LINEAR)
    return observation

  def _get_observation(self):
    if FLAGS.grayscale:
      return self._get_grayscale_observation()
    else:
      return self._get_color_observation()

  @classmethod
  def from_game_name(cls, name, max_episode_steps, *args, **kwargs):
    env = gym.make(name, full_action_space=FLAGS.full_action_space)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return cls(env, *args, **kwargs)

  def _augment_image(self, x):
    w, h, _ = x.shape
    minw = int(w * .25)
    minh = int(h * .25)
    cw = np.random.randint(minw, w + 1)
    ch = np.random.randint(minh, h + 1)
    dw = np.random.randint(0, w - cw + 1)
    dh = np.random.randint(0, h - ch + 1)
    x_crop = x[dw:dw + cw, dh:dh + ch]
    x = cv2.resize(x_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    brightness_factor = np.random.uniform(.85, 1.15, x.shape)
    x = x * brightness_factor
    x = x.clip(0., (1. if self.normalize_observation else 255.))
    x = x.astype(np.float32)
    return x

  def get_pretraining_sample(self):
    episode_buffer = []
    obs, _ = self.reset()
    episode_buffer.append(obs)
    done = False
    step = 1
    while not done:
      action = np.random.choice(self.action_space.n)
      obs, _, done, _ = self.step(action)
      if step < FLAGS.pretrain_buffer_size:
        episode_buffer.append(obs)
      else:
        r = np.random.randint(0, step)
        if r < FLAGS.pretrain_buffer_size:
          episode_buffer[r] = obs
      step += 1
    random.shuffle(episode_buffer)
    return [
        (self._augment_image(s), self._augment_image(s)) for s in episode_buffer
    ]


class ScreenBuffer(object):
  """Queue for the historic screen and action observations of the Atari game."""

  def __init__(self,
               size,
               shape,
               action_space,
               dtype=np.float32,
               normalize=True):
    """Constructor for the ScreenBuffer.

    Args:
      size: int, The size of the buffer, i.e. how many past frames should it
        store.
      shape: tuple, The shape of the screen observations.
      action_space: int, Size of the action space.
      dtype: Required type for the output observation.
      normalize: bool, If True the observation will be returned between 0 and 1.
    """
    self.size = size
    self.shape = shape
    self.action_space = action_space
    self.normalize = normalize
    self.dtype = dtype

    self.reset()

  def reset(self):
    self.buffer = collections.deque([], maxlen=self.size)
    self.action_buffer = collections.deque([], maxlen=self.size)

  def add(self, screen, action):
    assert screen.shape == self.shape, ('Trying to add screen of shape {}. '
                                        'Expected shape is: {}').format(
                                            screen.shape, self.shape)
    self.buffer.append(screen)
    self.action_buffer.append(action)

  def as_numpy(self, include_actions=True):
    """Returns the observation buffer as numpy array.

    Args:
      include_actions: bool, If True include the past actions as 'action frames'
        of the same size as the observation frames.

    Returns:
      observation: np.array of self.dtype, Observation of shape (self.shape,
      self.size). If include action is true, the actions are concatenated.
    """
    total_size = self.size if not include_actions else 2 * self.size
    out = np.zeros((total_size, *self.shape))
    out[total_size - len(self.buffer):] = np.array(self.buffer)
    if self.normalize:
      out = out / 255.

    if include_actions:
      action_planes = [
          np.ones(self.shape) * action / float(self.action_space)
          for action in self.action_buffer
      ]
      out[self.size -
          len(self.action_buffer):self.size] = np.array(action_planes)

    out = np.moveaxis(out, 0, -1)  # move the planes axis to the right
    out = out.astype(self.dtype)
    out = out.reshape((out.shape[0], out.shape[1], -1))
    return out
