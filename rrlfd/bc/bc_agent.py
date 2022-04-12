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

"""Defines agent class for learning and executing behavioural cloning policies.
"""
import collections
import copy
import enum
import math
import os
import pickle
import re

from absl import flags
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
import gym
from mime.envs import utils as mime_env_utils
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.io import gfile
from rrlfd.bc import network

FLAGS = flags.FLAGS


class Normalization(enum.Enum):
  UNIT = 1  # [-1, 1]
  UNIT_NONNEG = 2  # [0, 1]
  ZEROMEAN_UNITVAR = 3  #  mean = 0, var = 1
  CENTERED = 4  #  mean unchanged, var = 1
  NONE = 5


class NormalizationSpace:
  """Base normalization space with unit and zero-mean unit-variance norms."""

  def __init__(self, norm_type, scale=1.0):
    if isinstance(norm_type, str):
      norm_type = Normalization[norm_type.upper()]
    assert (
        norm_type in
        [Normalization.UNIT, Normalization.ZEROMEAN_UNITVAR, Normalization.NONE,
         Normalization.CENTERED]
    )
    self.norm_type = norm_type
    self.scale = scale

  def normalize(self, action):
    """Scale action from original action space to range expected by a policy."""
    if isinstance(action, list):
      return [self.normalize(a) for a in action]
    if isinstance(action, dict):
      norm_action = collections.OrderedDict()
      for k, v in action.items():
        if self.norm_type == Normalization.UNIT:
          norm_action[k] = v / self.maxima[k]
        elif self.norm_type == Normalization.ZEROMEAN_UNITVAR:
          if k == 'grip_velocity':
            norm_action[k] = v
          else:
            norm_action[k] = (v - self.mean[k]) / (self.std[k] + 1e-8)
        elif self.norm_type == Normalization.CENTERED:
          if k == 'grip_velocity':
            norm_action[k] = v
          else:
            norm_action[k] = v / (self.std[k] + 1e-8)
        else:
          norm_action[k] = v
        if k != 'grip_velocity':
          norm_action[k] = norm_action[k] / self.scale
    else:
      norm_action = self.normalize_flat(action)
    return norm_action

  def denormalize(self, action):
    """Scale action returned by a policy to the original action space."""
    if isinstance(action, dict):
      denorm_action = collections.OrderedDict()
      for k, v in action.items():
        if self.norm_type == Normalization.UNIT:
          denorm_action[k] = v * self.maxima[k] * self.scale
        elif self.norm_type == Normalization.ZEROMEAN_UNITVAR:
          if k == 'grip_velocity':
            denorm_action[k] = v
          else:
            denorm_action[k] = v * self.std[k] * self.scale + self.mean[k]
        elif self.norm_type == Normalization.CENTERED:
          if k == 'grip_velocity':
            denorm_action[k] = v
          else:
            denorm_action[k] = v * self.std[k] * self.scale
        else:
          denorm_action[k] = v * self.scale
    else:
      denorm_action = np.copy(action)
      if self.norm_type == Normalization.UNIT:
        denorm_action = action * self.maxima * self.scale
      elif self.norm_type == Normalization.ZEROMEAN_UNITVAR:
        denorm_action = action * self.std * self.scale + self.mean
      elif self.norm_type == Normalization.CENTERED:
        denorm_action = action * self.std * self.scale
    return denorm_action


class ActionSpace(NormalizationSpace):
  """Defines normalization for actions."""

  def __init__(self, norm_type, env=None, scale=1.0):
    super().__init__(norm_type, scale)
    if self.norm_type == Normalization.UNIT:
      if isinstance(env.action_space, gym.spaces.dict.Dict):
        self.maxima = {k: space.high
                       for k, space in env.action_space.spaces.items()}
      else:
        self.maxima = env.action_space.high
        print('action minima', env.action_space.low)
        print('action maxima', self.maxima)

  def normalize_flat(self, action):
    """Normalize an action as a flat array, assuming sorted keys."""
    norm_action = np.copy(action)
    if self.norm_type == Normalization.UNIT:
      if isinstance(self.maxima, dict):
        i = 0
        for k in sorted(self.maxima.keys()):
          size = len(self.maxima[k])
          norm_action[i:(i + size)] = (
              action[i:(i + size)] / (self.maxima[k] + 1e-8))
          i += size
      else:
        norm_action = action / self.maxima
    elif self.norm_type == Normalization.ZEROMEAN_UNITVAR:
      if isinstance(self.mean, dict):
        i = 0
        for k in sorted(self.mean.keys()):
          size = len(self.mean[k])
          if k != 'grip_velocity':
            norm_action[i:(i + size)] = (
                (action[i:(i + size)] - self.mean[k]) / (self.std[k] + 1e-8))
          i += size
      else:
        norm_action = (action - self.mean) / (self.std + 1e-8)
    elif self.norm_type == Normalization.CENTERED:
      if isinstance(self.std, dict):
        i = 0
        for k in sorted(self.std.keys()):
          size = len(self.std[k])
          if k != 'grip_velocity':
            norm_action[i:(i + size)] = (
                (action[i:(i + size)]) / (self.std[k] + 1e-8))
          i += size
      else:
        norm_action = action / (self.std + 1e-8)
    norm_action /= self.scale
    return norm_action


class ObservationSpace(NormalizationSpace):
  """Defines normalization for custom scalar observations."""

  def __init__(self, norm_type, env=None):
    super().__init__(norm_type)
    self._env = env

  def normalize_flat(self, features, key_order=None):
    """Apply normalization to concatenated features."""
    if key_order is None:
      key_order = sorted(self.mean.keys())
    norm_features = np.copy(features)
    if self.norm_type == Normalization.ZEROMEAN_UNITVAR:
      if isinstance(self.mean, dict):
        i = 0
        for k in sorted(self.mean.keys()):
          size = len(self.mean[k])
          norm_features[Ellipsis, i:(i + size)] = (
              (features[Ellipsis, i:(i + size)] - self.mean[k])
              / (self.std[k] + 1e-8))
          i += size
      else:
        norm_features = (features - self.mean) / (self.std + 1e-8)
    elif self.norm_type == Normalization.UNIT:
      raise NotImplementedError
    return norm_features

  def normalize_feature(self, key, feature):
    """Scale observation from the environment to range expected by a policy."""
    if isinstance(feature, list):
      feature = np.array(feature)
    if self.norm_type == Normalization.UNIT:
      low, high = self._env.unwrapped.scene.workspace
      low = np.array(low)
      high = np.array(high)
      if key == 'linear_velocity':
        # Action space limits not the same for all tasks.
        norm_feature = feature / 0.075  # [-1,1]
      elif key == 'grip_velocity':
        norm_feature = feature / 2.0  # [-1,1]
      elif key == 'gripper_opened':
        norm_feature = feature.astype(np.int)
      elif 'position' in key:
        # mostly [-1,1]
        norm_feature = (feature - low) / (high - low) * 2.0 - 1.0
      elif key == 'grip_state':
        norm_feature = feature * 2.0 - 1.0  # mostly [-1,1]
      else:
        print('No normalization defined for', key, feature)
        norm_feature = np.copy(feature)
      norm_feature = norm_feature.astype(feature.dtype)
    elif self.norm_type == Normalization.ZEROMEAN_UNITVAR:
      norm_feature = super().normalize({key: feature})[key]
      norm_feature = norm_feature.astype(feature.dtype)
    elif self.norm_type == Normalization.NONE:
      norm_feature = np.copy(feature)
    return norm_feature

  def normalize(self, feature):
    norm_feature = {}
    for k, v in feature.items():
      norm_feature[k] = self.normalize_feature(k, v)
    return norm_feature

  def denormalize(self, key, feature):
    raise NotImplementedError('Denormalization not implemented for', key)


class ImageObservationSpace:
  """Defines normalization for uint8 based inputs."""

  def __init__(self, norm_type):
    if isinstance(norm_type, str):
      norm_type = Normalization[norm_type.upper()]
    assert (
        norm_type in
        [Normalization.UNIT, Normalization.UNIT_NONNEG, Normalization.NONE])
    self.norm_type = norm_type

  def normalize(self, observation):
    """Scale observation from the environment to range expected by a policy."""
    if isinstance(observation, list):
      observation = np.array(observation)
    if self.norm_type == Normalization.UNIT:
      observation = (observation - 127.5) / 127.5
    elif self.norm_type == Normalization.UNIT_NONNEG:
      observation = observation / 255
    return observation


class BCAgent:
  """BC agent consisting of a network and preprocessing config."""

  def __init__(
      self, network_type, input_type='depth', binary_grip_action=True,
      grip_action_from_state=True, zero_action_keeps_state=False,
      early_closing=True, num_input_frames=1, crop_frames=True,
      full_image_size=240, crop_size=224, target_offsets=(1,),
      visible_state_features=(), allow_missing_features=False,
      action_norm=Normalization.UNIT, obs_norm=Normalization.UNIT,
      signals_norm=Normalization.UNIT, last_activation=None, fc_layer_sizes=(),
      weight_decay=5e-4, action_space='tool_lin', env=None, late_fusion=False,
      init_scheme='v1'):
    self.input_type = input_type
    self.binary_grip_action = binary_grip_action
    self.grip_action_from_state = grip_action_from_state
    self.zero_action_keeps_state = zero_action_keeps_state
    self.early_closing = early_closing
    self.num_input_frames = num_input_frames
    self.crop_frames = crop_frames
    num_input_channels = {
        None: 0,
        'depth': 1,
        'rgb': 3,
        'rgbd': 4
    }[self.input_type]
    self.num_input_channels = num_input_channels
    self.late_fusion = late_fusion
    if self.crop_frames:
      self.full_image_size = full_image_size
      self.crop_size = crop_size
      im_size = self.crop_size
    else:
      im_size = full_image_size
    if self.late_fusion:
      self.input_shape = (
          num_input_frames, im_size, im_size, num_input_channels)
    else:
      self.input_shape = (
          im_size, im_size, num_input_frames * num_input_channels)

    self.action_space_type = action_space
    self.action_pred_dim = {
        'tool_lin': 4,
        'tool_lin_ori': 7,
        'door': 28,
        'hammer': 26,
        'pen': 24,
        'relocate': 30
    }[action_space]
    self.action_target_dim = self.action_pred_dim
    # Classify gripper state with first two outputs.
    if self.binary_grip_action:
      self.action_pred_dim += 1
    self.target_offsets = target_offsets
    self.visible_state_features = visible_state_features
    self.skip_missing_features = allow_missing_features
    self.num_outputs = self.action_pred_dim * len(self.target_offsets)
    self.network = None
    if network_type is not None:
      network_class = {
          # 'resnet18': network.Resnet18,
          'resnet18_narrow32': network.Resnet18Narrow32,
          'resnet50': network.resnet50,
          'simple_cnn': network.SimpleCNN,
          'hand_vil': network.HandVilNet,
      }[network_type]
      network_kwargs = {}
      # TODO(minttu): mime robot info dimensions.
      # TODO(minttu): Sizes of partially visible states.
      if (self.visible_state_features
          and action_space in ['door', 'hammer', 'pen', 'relocate']):
        self.robot_info_dim = {
            'door': 100,
            'hammer': 98,
            'pen': 72,
            'relocate': 54,
        }[action_space]
      if network_class == network.HandVilNet and not fc_layer_sizes:
        if self.visible_state_features:
          fc_layer_sizes = (200, 128 + self.robot_info_dim)
        else:
          fc_layer_sizes = (200,)
        network_kwargs['late_fusion'] = self.late_fusion
      if input_type is None:
        network_class = network.FeedForwardNet
      if network_class == network.resnet50:
        network_kwargs['input_shape'] = self.input_shape
      print('Creating network with fc layers', fc_layer_sizes)
      self.network = network_class(
          n_classes=self.num_outputs,
          last_activation=last_activation,
          fc_layer_sizes=fc_layer_sizes,
          weight_decay=weight_decay,
          init_scheme=init_scheme,
          **network_kwargs)

      # For initializing network weights manually (for debugging).
      if FLAGS.policy_init_path is not None:
        with gfile.GFile(FLAGS.policy_init_path, 'rb') as f:
          layers = pickle.load(f)
        np.random.seed(0)
        dummy_input = np.random.rand(128, 3, 128, 128, 3)
        dummy_scalars = np.random.rand(128, self.robot_info_dim)
        _ = self.network.call(dummy_input, dummy_scalars, interrupt=False)
        conv_transpose = [2, 3, 1, 0]
        self.network.conv1.weights[0].assign(np.transpose(layers[0],
                                                          conv_transpose))
        self.network.conv1.weights[1].assign(np.transpose(layers[1]))

        self.network.bn1.weights[0].assign(np.transpose(layers[2]))
        self.network.bn1.weights[1].assign(np.transpose(layers[3]))

        self.network.conv2.weights[0].assign(np.transpose(layers[4],
                                                          conv_transpose))
        self.network.conv2.weights[1].assign(np.transpose(layers[5]))

        self.network.bn2.weights[0].assign(np.transpose(layers[6]))
        self.network.bn2.weights[1].assign(np.transpose(layers[7]))

        self.network.conv3.weights[0].assign(np.transpose(layers[8],
                                                          conv_transpose))
        self.network.conv3.weights[1].assign(np.transpose(layers[9]))

        self.network.bn3.weights[0].assign(np.transpose(layers[10]))
        self.network.bn3.weights[1].assign(np.transpose(layers[11]))

        self.network.conv4.weights[0].assign(np.transpose(layers[12],
                                                          conv_transpose))
        self.network.conv4.weights[1].assign(np.transpose(layers[13]))

        self.network.bn4.weights[0].assign(np.transpose(layers[14]))
        self.network.bn4.weights[1].assign(np.transpose(layers[15]))

        self.network.fcs[0].weights[0].assign(np.transpose(layers[16]))
        self.network.fcs[0].weights[1].assign(np.transpose(layers[17]))

        self.network.fcs[1].weights[0].assign(np.transpose(layers[18]))
        self.network.fcs[1].weights[1].assign(np.transpose(layers[19]))

        self.network.fc_out.weights[0].assign(np.transpose(layers[20]))
        self.network.fc_out.weights[1].assign(np.transpose(layers[21]))

        self.network.bn1.weights[2].assign(np.transpose(layers[22]))
        self.network.bn1.weights[3].assign(np.transpose(layers[23]))
        self.network.bn2.weights[2].assign(np.transpose(layers[24]))
        self.network.bn2.weights[3].assign(np.transpose(layers[25]))
        self.network.bn3.weights[2].assign(np.transpose(layers[26]))
        self.network.bn3.weights[3].assign(np.transpose(layers[27]))
        self.network.bn4.weights[2].assign(np.transpose(layers[28]))
        self.network.bn4.weights[3].assign(np.transpose(layers[29]))
        raw_pred = self.network.call(dummy_input, dummy_scalars)
        print('raw pred\n', raw_pred)

    self.env = env
    self.action_space = ActionSpace(action_norm, env)
    self.img_obs_space = ImageObservationSpace(obs_norm)
    self.signals_obs_space = ObservationSpace(signals_norm, env)

  def restore_from_ckpt(self, ckpt, compile_model=False):
    """Restore network weights from a checkpoint."""
    if compile_model:
      try:
        obs_space = mime_env_utils.make_dict_space(
            self.env.scene, *self.visible_state_features)
      except AttributeError:
        obs_space = self.env.observation_space
      scalar_sizes = [
          obs_space[feat].shape for feat in self.visible_state_features]
      scalars_size = int(np.sum([np.prod(a) for a in scalar_sizes]))
      dummy_input = np.random.rand(1, *self.input_shape)
      dummy_scalars = np.random.rand(1, scalars_size)
      _ = self.network(dummy_input, dummy_scalars)
    self.network.load_weights(ckpt)

  @property
  def log_std(self):
    return self.network.log_std

  def __call__(self, *args, **kwargs):
    return self.network.__call__(*args, **kwargs)

  def save(self, path):
    # TODO(minttu): Save full model definition.
    self.network.save_weights(path)

  def load(self, path):
    self.network.load_weights(path)

  def reset_action_stats(self, action_mean, action_std):
    self.action_space.mean = action_mean
    self.action_space.std = action_std

  def reset_observation_stats(self, signal_mean, signal_std):
    self.signals_obs_space.mean = signal_mean
    self.signals_obs_space.std = signal_std

  def preprocess_image(self,
                       state_hist,
                       augment_frames=False,
                       randomize_camera=False):
    """Stack images, apply image augmentations and reshape."""
    stacked_state = state_hist[-(self.num_input_frames):]
    img = self.stack_frames(stacked_state, randomize_camera)
    # Use the same augmentation parameters for each frame in history.
    if augment_frames:
      rotation, translation = self.draw_augment_params(img.shape[0])
      for i in range(img.shape[2]):
        img[:, :, i] = self.transform_image(img[:, :, i], rotation, translation)
    img = self.crop_image(img, augment_frames)
    img = self.fuse_dimensions(img)
    return img

  def state_to_observation(self, state_hist):
    """Prepare observation from environment as input for policy.

    Args:
      state_hist: history of states from environment [1, ..., t].

    Returns:
      obs: observation for time t.
    """
    signals = self.keep_visible_features(state_hist[-1])
    signals = np.expand_dims(signals, axis=0)
    if self.input_type is None:
      return None, signals
    img = self.preprocess_image(state_hist)
    img = np.expand_dims(img, axis=0)  # Add batch dimension.
    img = self.img_obs_space.normalize(img)
    return img, signals

  def flat_action_to_dict(self, action):
    if self.action_space_type == 'tool_lin':
      action = {'grip_velocity': action[:1],
                'linear_velocity': action[1:]}
    elif self.action_space_type == 'tool_lin_ori':
      action = {'angular_velocity': action[:3],
                'grip_velocity': action[3:4],
                'linear_velocity': action[4:]}
    return action

  def get_raw_action(self, states, return_feats=False,
                     return_stacked_obs=False):
    """Get action before denormalization or clipping."""
    # TODO(minttu): Return logits.
    # TODO(minttu): other action space types
    img, signals = self.state_to_observation(states)
    full_action, feats = self.network.call(img, signals, return_feats=True)
    full_action = full_action[0]
    # First action_pred_dim predictions: action at time t + 1.
    action = np.array(full_action[:self.action_pred_dim])
    if self.binary_grip_action:
      grip_actions = [{'grip_velocity': [2.0]}, {'grip_velocity': [-2.0]}]
      grip_actions = self.action_space.normalize(grip_actions)
      grip_actions = [act['grip_velocity'] for act in grip_actions]
      # First logit: open, second: close.
      grip_pred = grip_actions[np.argmax(action[:2])]
      action = np.concatenate([grip_pred, action[2:]], axis=0)
    action = self.flat_action_to_dict(action)
    returns = [action]
    if return_feats:
      returns.append(feats)
    if return_stacked_obs:
      returns.append((img, signals))
    if len(returns) == 1:
      returns = returns[0]
    return returns

  def get_action(self, state, state_hist, eval_env=None, return_feats=False,
                 return_stacked_obs=False):
    """Shape action returned by policy on state and clip to limits."""
    state_hist = state_hist + [state]
    # TODO(minttu): Separate image preprocessing.
    raw_action, feats, stacked_obs = self.get_raw_action(
        state_hist, return_feats=True, return_stacked_obs=True)
    action = self.action_space.denormalize(raw_action)
    env = eval_env or self.env
    # TODO(minttu): Verify whether clipping is useful with mime.
    # (For Adroit, it is redundant as the environment re-applies clipping).
    if isinstance(action, dict):
      for k, v in action.items():
        action[k] = np.clip(
            v, env.action_space.spaces[k].low, env.action_space.spaces[k].high)
    else:
      action = np.clip(
          action, env.action_space.low, env.action_space.high)
    returns = [action]
    if return_feats:
      returns.append(feats)
    if return_stacked_obs:
      returns.append(stacked_obs)
    if len(returns) == 1:
      returns = returns[0]
    return returns

  def keep_visible_features(self, observation, visible_state_features=None):
    """Concatenate visible feature fields in observation."""
    if isinstance(observation, list):
      return np.stack([self.keep_visible_features(obs) for obs in observation])
    visible_features = []
    visible_state_features = (
        visible_state_features or self.visible_state_features)
    for key in visible_state_features:
      if key not in observation and self.skip_missing_features:
        continue
      feature = observation[key]
      if isinstance(feature, int) or isinstance(feature, float):
        feature = [feature]
      feature = np.array(feature)
      feature = self.signals_obs_space.normalize_feature(key, feature)
      visible_features.append(feature.flatten())
    if visible_features:
      visible_features = np.concatenate(visible_features, axis=0)
    return visible_features

  def transform_image(self, image, angle=0., translation=(0, 0),
                      scale=(1.0, 1.0)):
    """Apply affine transformation to a 2d numpy array."""
    image = Image.fromarray(image)
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
    return np.asarray(image)

  def draw_augment_params(self, im_size):
    angle = 5
    translate = 0.04 * im_size
    rotation = np.random.uniform(low=-angle, high=angle)
    translation = np.random.uniform(
        low=-translate, high=translate, size=(2))
    translation = np.rint(translation).astype(np.int)
    return rotation, translation

  def stack_frames(self, frame_history, randomize_camera=False):
    """Stack frames in frame_history, repeating oldest frames if needed.

    Args:
      frame_history: List of past dictionary observations, where newest is last.
      randomize_camera: Whether to sample randomly among available camera views.

    Returns:
      obs: numpy array with stacked input_type frames of shape
        [h, w, history, c].
    """
    input_type_keys = [
        k for k in frame_history[0] if re.search(self.input_type + '*', k)]
    # Only apply randomize camera flag if dataset contains multiple cameras.
    if len(input_type_keys) == 1:
      input_key = input_type_keys[0]
    else:
      # Use the same viewpoint for each frame in history.
      input_key = (
          np.random.choice(input_type_keys) if randomize_camera
          else self.input_type + '0')

    num_missing = self.num_input_frames - len(frame_history)
    repeat_frames = [frame_history[0][input_key]] * num_missing
    history_frames = [frame[input_key] for frame in frame_history]
    full_obs = np.stack(repeat_frames + history_frames, axis=2)
    return full_obs

  def crop_image(self, obs, augment_frames=False):
    """Crop stacked images if applicable."""
    if self.crop_frames:
      crop_size = self.crop_size
      if augment_frames:
        y_margin = np.random.randint(obs.shape[0] - crop_size)
        x_margin = np.random.randint(obs.shape[1] - crop_size)
      else:  # central crop
        y_margin = int((obs.shape[0] - crop_size) / 2)
        x_margin = int((obs.shape[1] - crop_size) / 2)
    else:
      y_margin = 0
      x_margin = 0
      crop_size = obs.shape[0]
    obs = obs[y_margin:y_margin + crop_size, x_margin:x_margin + crop_size]
    return obs

  def fuse_dimensions(self, obs):
    """If using late fusion, set history first. Else merge history and channels.

    Args:
      obs: Input image observation of shape [height, width, history, channels].

    Returns:
      obs: Reshaped image observation.
    """
    if self.late_fusion:
      # Set history as the first dimension.
      obs = np.moveaxis(obs, 2, 0)
    else:
      # Merge history and image channels (if any).
      obs = np.reshape(obs, [*obs.shape[:2], np.prod(obs.shape[2:])])
    return obs

  def normalize_demo_observations(self, ep_obs, augment_frames=False,
                                  randomize_camera=True):
    """Normalize observations of an episode.

    Args:
      ep_obs: list of an episode's observations
      augment_frames: whether to add image augmentation.
      randomize_camera: whether to sample a camera position.

    Returns:
      norm_ep: list of reshaped and normalized observations, of shape
      [t, crop_size, crop_size, num_dim * num_input_frames].
      Oldest frame is first.
    """
    signals = self.keep_visible_features(ep_obs)
    if self.input_type is None:
      return [None] * len(signals), signals
    ep_frames = copy.deepcopy(ep_obs)
    norm_frames = []
    for t in range(len(ep_frames)):
      obs = self.preprocess_image(
          ep_frames[:t + 1], augment_frames, randomize_camera)

      obs = self.img_obs_space.normalize(obs)
      norm_frames.append(obs)
    return norm_frames, signals

  def preprocess_demo_actions(self, episode_actions, episode_observations=None):
    """Preprocess gripper actions."""
    if self.grip_action_from_state or self.zero_action_keeps_state:
      episode_actions = copy.deepcopy(episode_actions)
      if self.grip_action_from_state:
        for t in range(len(episode_actions)):
          episode_actions[t]['grip_velocity'] = (
              np.array([episode_observations[t]['grip_velocity']]))
      elif self.zero_action_keeps_state:
        prev_action = 2.0  # First state should be open if fist action is zero.
        for t in range(len(episode_actions)):
          if episode_actions[t]['grip_velocity'][0] == 0:
            episode_actions[t]['grip_velocity'] = np.array([prev_action])
          else:
            prev_action = episode_actions[t]['grip_velocity'][0]
    return episode_actions

  def normalize_demo_actions(self, episode_actions, episode_observations=None):
    """Normalize actions of an episode."""
    episode_actions = self.preprocess_demo_actions(
        episode_actions, episode_observations)
    norm_ep = []
    for t in range(len(episode_actions)):
      full_action = []
      for offset in self.target_offsets:
        idx_at_offset = min(t + offset, len(episode_actions) - 1)
        act = copy.deepcopy(episode_actions[idx_at_offset])
        if self.early_closing:
          idx_future = min(idx_at_offset + 5, len(episode_actions) - 1)
          # To match rlbc implementation.
          # idx_future = min(idx_future, t + max(self.target_offsets))
          if (episode_actions[idx_at_offset]['grip_velocity']
              * episode_actions[idx_future]['grip_velocity'] < 0):
            act['grip_velocity'] = episode_actions[idx_future]['grip_velocity']
        act = self.action_space.normalize(act)
        if isinstance(act, dict):
          act = np.concatenate([act[k] for k in sorted(act)])
        full_action.append(act)
      full_action = np.concatenate(full_action)
      norm_ep.append(full_action)
    return norm_ep

  def normalize_demo(self, observations, actions, augment_frames=False,
                     randomize_camera=False):
    actions = self.normalize_demo_actions(actions, observations)
    observations, signals = self.normalize_demo_observations(
        observations, augment_frames, randomize_camera)
    return observations, signals, actions


class EpisodeSnapshotter(tf2_savers.Snapshotter):
  """Snapshotter subclass with a custom output dir (without process_path)."""

  def __init__(
      self, objects_to_save, *, directory='~/acme/', time_delta_minutes=30.0):
    """Builds the saver object.

    Args:
      objects_to_save: Mapping specifying what to snapshot.
      directory: Which directory to put the snapshot in.
      time_delta_minutes: How often to save the snapshot, in minutes.
    """
    objects_to_save = objects_to_save or {}

    self._time_delta_minutes = time_delta_minutes
    self._last_saved = 0.
    self._snapshots = {}

    # Save the base directory path so we can refer to it if needed.
    self.directory = directory

    # Save a dictionary mapping paths to snapshot capable models.
    for name, module in objects_to_save.items():
      path = os.path.join(self.directory, name)
      self._snapshots[path] = tf2_savers.make_snapshot(module)


class ResidualBCAgent:
  """Agent which predicts residual actions on top of a given base agent.

  The class matches the BCAgent interface, i.e., defines its own preprocessing
  and forward passes in eval and train modes.
  TODO(minttu): Extract shared interface subclass.
  """

  def __init__(self, base_agent, residual_spec, policy_network, action_norm,
               action_norm_scale=1.0, env=None, visible_state_features=()):
    self.base_agent = base_agent
    obs_spec = residual_spec.observations

    tf2_utils.create_variables(policy_network, [obs_spec])
    self.network = policy_network
    self.visible_state_features = visible_state_features

    self.env = env
    self.action_space = ActionSpace(
        action_norm, env=env, scale=action_norm_scale)
    # Reuse stats; normalization scheme may still be different.
    self.action_space.mean = self.base_agent.action_space.mean
    self.action_space.std = self.base_agent.action_space.std

    # Options that may want to be shared between base and residual agent.
    self.binary_grip_action = self.base_agent.binary_grip_action
    # self.grip_action_from_state = self.base_agent.grip_action_from_state
    # self.zero_action_keeps_state = self.base_agent.zero_action_keeps_state
    # self.early_closing = self.base_agent.early_closing

    # For convenience, might want to revisit later.
    self.action_pred_dim = self.base_agent.action_pred_dim
    self.action_target_dim = self.base_agent.action_target_dim
    # TODO(minttu): log_std?
    # Should target offsets be enabled?

  def __call__(self, *args, **kwargs):
    return self.network.__call__(*args, **kwargs)

  def save(self, path):
    """Save policy network as a checkpoint at path."""
    # TODO(minttu): Save in a way that allows to continue training.
    # Create new Snapshotter to write to a different path each time.
    snapshotter = EpisodeSnapshotter(
        objects_to_save={os.path.basename(path): self.network},
        directory=os.path.dirname(path))
    snapshotter.save(force=True)

  def load(self, path):
    self.network = tf.saved_model.load(path)

  def preprocess_dataset(self, dataset, batch_size):
    """Normalize demonstrations and add base agent actions and features."""
    # Assuming dataset in RAM.
    for d in range(len(dataset.observations)):
      base_observations, signals, _ = self.base_agent.normalize_demo(
          dataset.observations[d], dataset.actions[d])
      residual_signals = self.base_agent.keep_visible_features(
          dataset.observations[d], self.visible_state_features)
      # Downcast to float32 as is later done in generator (in train_utils).
      base_observations = np.array(base_observations, dtype=np.float32)
      signals = np.array(signals, dtype=np.float32)
      base_dataset = tf.data.Dataset.from_tensor_slices(
          (base_observations, signals, residual_signals))
      base_dataset = base_dataset.batch(batch_size)
      base_actions = []
      residual_observations = []
      for obs_batch, signals_batch, residual_signals_batch in base_dataset:
        raw_base_actions_batch, feats_batch = self.base_agent.network.call(
            obs_batch, signals_batch, return_feats=True)
        base_actions_batch = self.base_agent.action_space.denormalize(
            raw_base_actions_batch)
        residual_observations_batch = np.concatenate(
            [residual_signals_batch, raw_base_actions_batch, feats_batch],
            axis=1)
        residual_observations.extend(residual_observations_batch)
        base_actions.extend(base_actions_batch)
      dataset.observations[d] = residual_observations
      if isinstance(dataset.actions[d][0], dict):
        residual_actions = []
        for t in dataset.actions[d]:
          residual_actions.append({
              k: v - base_actions[t][k]
              for k, v in dataset.actions[d][t].items()})
      else:
        residual_actions = np.array(dataset.actions[d]) - np.array(base_actions)
      residual_actions = self.action_space.normalize(residual_actions)
      dataset.actions[d] = list(residual_actions)

  def normalize_demo(self, observations, actions, augment_frames=False,
                     randomize_camera=False):
    del augment_frames, randomize_camera
    residual_signals = [None for _ in observations]
    return observations, residual_signals, actions

  def preprocess_demo(self, observations, actions, augment_frames=False,
                      randomize_camera=False):
    """Preprocess (apply transformations and normalizations) a single demo."""
    # If preprocessing cannot be done once (e.g. dataset does not fit in RAM),
    # preprocess each demo individually.
    observations = copy.deepcopy(observations)
    actions = copy.deepcopy(actions)
    observations, signals, _ = self.base_agent.normalize_demo(
        observations, actions, augment_frames, randomize_camera)
    # TODO(minttu): get base action for a batch at a time
    residual_observations = []
    residual_signals = [None for _ in observations]
    for t in range(len(observations)):
      obs_batch = np.expand_dims(observations[t], 0)
      signals_batch = np.expand_dims(signals[t], 0)
      raw_base_action, feats = self.base_agent.network.call(
          obs_batch, signals_batch, return_feats=True)
      raw_base_action = raw_base_action[0]
      feats = feats[0]
      base_action = self.base_agent.action_space.denormalize(raw_base_action)
      actions[t] -= base_action
      actions[t] = self.action_space.normalize(actions[t])
      concat_obs = np.concatenate([signals[t], raw_base_action, feats])
      residual_observations.append(concat_obs)
    return residual_observations, residual_signals, actions

  # def state_to_observation(self, states):
  #   img, signals = self.base_agent.state_to_observation(states)
  #   base_action, feats = self.base_agent.network.call(
  #         img, signals, return_feats=True)
  #   base_action = base_action[0]
  #   feats = feats[0]
  #   signals = signals[0]
  #   obs = np.concatenate([signals, base_action, feats])
  #   return obs

  def get_raw_action(self, obs):
    """Get action before denormalization or clipping."""
    obs_dict = {'residual_obs': obs.astype(np.float32)}
    full_action = self.network(obs_dict).mean()
    full_action = full_action[0]
    # First action_pred_dim predictions: action at time t + 1.
    action = np.array(full_action[:self.action_pred_dim])
    if self.binary_grip_action:
      grip_actions = [{'grip_velocity': [2.0]}, {'grip_velocity': [-2.0]}]
      grip_actions = self.action_space.normalize(grip_actions)
      grip_actions = [act['grip_velocity'] for act in grip_actions]
      # First logit: open, second: close.
      grip_pred = grip_actions[np.argmax(action[:2])]
      action = np.concatenate([grip_pred, action[2:]], axis=0)
    # Simply break up array action into mime action dicitonary, if applicable.
    action = self.base_agent.flat_action_to_dict(action)
    return action

  def clip_action(self, action, env):
    """Clip an action to env action limits."""
    if env is None:
      return action
    if isinstance(action, dict):
      clipped_action = {}
      for k, v in action.items():
        clipped_action[k] = np.clip(
            v, env.action_space.spaces[k].low, env.action_space.spaces[k].high)
    else:
      clipped_action = np.clip(
          action, env.action_space.low, env.action_space.high)
    return clipped_action

  def get_action(self, state, state_hist, eval_env=None, return_feats=False,
                 return_stacked_obs=False):
    """Shape action returned by policy on state and clip to limits."""
    state_hist = state_hist + [state]
    unused_img, signals = self.base_agent.state_to_observation(state_hist)
    raw_base_action, feats, stacked_obs = self.base_agent.get_raw_action(
        state_hist, return_feats=True, return_stacked_obs=True)
    raw_base_action = np.expand_dims(raw_base_action, axis=0)
    residual_obs = np.concatenate([signals, raw_base_action, feats], axis=1)
    raw_action = self.get_raw_action(residual_obs)
    # raw_action = np.zeros_like(raw_base_action)
    residual_action = self.action_space.denormalize(raw_action)
    # assert np.array_equal(residual_action, raw_action)
    base_action = self.base_agent.action_space.denormalize(raw_base_action)

    clipped_base_action = self.clip_action(base_action, eval_env)
    assert np.array_equal(
        clipped_base_action[0],
        self.base_agent.get_action(state, state_hist[:-1]))
    if isinstance(residual_action, dict):
      action = {}
      for k, v in residual_action.items():
        action[k] = v + base_action[k]
    else:
      action = residual_action + base_action
    action = self.clip_action(action, eval_env)
    returns = [action]
    if return_feats:
      returns.append(feats)
    if return_stacked_obs:
      returns.append(stacked_obs)
    if len(returns) == 1:
      returns = returns[0]
    return returns
