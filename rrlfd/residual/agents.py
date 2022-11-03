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

"""Defines a residual RL agent."""

import collections

from acme import core
from acme.agents.tf import d4pg
from acme.tf import utils as tf_utils
import numpy as np
import tensorflow as tf

from rrlfd.bc import bc_agent
from rrlfd.bc import network


class ObservationNet(tf.keras.Model):
  """Observation network which handles image input preprocessing."""

  def __init__(self, network_type, input_type, add_linear_layer,
               crop_frames, full_image_size, crop_margin_size, late_fusion,
               obs_norm=bc_agent.Normalization.UNIT):
    super(ObservationNet, self).__init__(name='observation_net')
    network_class = {
        # 'resnet18': network.Resnet18,
        'resnet18_narrow32': network.Resnet18Narrow32,
        'resnet50': network.resnet50,
        'simple_cnn': network.SimpleCNN,
        'feedforward': network.FeedForwardNet,
        'hand_vil': network.HandVilNet,
    }[network_type]
    # We might want to add a linear layer if initializing from a BC checkpoint.
    if add_linear_layer:
      # TODO(minttu): Replace each magic number.
      self.network = network_class(n_classes=20 * 4)
    else:
      self.network = network_class()
    self.input_type = input_type
    self.crop_frames = crop_frames
    self.full_image_size = full_image_size
    self.crop_margin_size = crop_margin_size
    self.crop_size = full_image_size - crop_margin_size
    self.late_fusion = late_fusion
    self.obs_space = bc_agent.ImageObservationSpace(obs_norm)

  def __call__(self, observation):
    # Only normalize depth here in order to save as uint8 in replay.
    if self.input_type in ['depth', 'rgb']:
      img = observation[self.input_type]
      if 'visible_state' in observation:
        scalar_feats = observation['visible_state']
        if len(scalar_feats.shape) < 2:
          scalar_feats = tf.expand_dims(scalar_feats, 0)
    if self.input_type in ['depth', 'rgb']:
      img = tf.cast(img, tf.float32)
      if self.input_type == 'depth':
        # TODO(minttu): Is this correct / needed?
        img = tf.reshape(img,
                         [-1, self.full_image_size, self.full_image_size, 3])
      if self.input_type == 'rgb':
        if self.late_fusion:
          raise NotImplementedError(
              'Late fusion not implemented for ObservationNet.')
      if self.crop_frames:
        crop_size = self.crop_size
        y_margin = int((img.shape[1] - crop_size) / 2)
        x_margin = int((img.shape[2] - crop_size) / 2)
        img = img[
            :,
            y_margin:y_margin + crop_size,
            x_margin:x_margin + crop_size]
      img = self.obs_space.normalize(img)
    else:
      raise NotImplementedError(
          'ObservationNet behavior not implemented for input type '
          f'{self.input_type}.')
    # TODO(minttu): check
    training = img.shape[0] is not None and img.shape[0] > 1
    # if training:
    #   print(img.shape, 'training =', training)

    # If n_classes is not set, features are always returned as the only output.
    # Otherwise, (output, features) are returned as a tuple.
    features = self.network(img, training=training, return_feats=True)
    if isinstance(features, tuple):
      features = features[1]
    if 'visible_state' in observation:
      full_observation = tf.concat([features, scalar_feats], axis=1)
    else:
      full_observation = features
    return full_observation


class ResidualAgent(core.Actor):
  """Residual agent which additively corrects a base agent's actions."""

  def __init__(
      self,
      base_agent,
      rl_agent,
      action_space,
      action_norm,
      action_norm_scale,
      signals_norm,
      rl_eval_policy=None,
      feats_spec=(),
      state_keys=(),
      bernoulli_rate=0.,
      sticky_rate=0.,
      rl_observation_network_type=None,
      rl_input_type='depth',
      rl_num_input_frames=3,
      base_controller=None,
      env=None):
    self.base_agent = base_agent
    # If set, a black-box controller used for base actions. base_agent may
    # still be used for features.
    self.base_controller = base_controller
    self.rl_agent = rl_agent
    self.rl_input_type = rl_input_type
    if self.rl_input_type != 'full_state':
      self.rl_num_input_channels = {
          None: 0,
          'depth': 1,
          'rgb': 3,
          'rgbd': 4
      }[self.rl_input_type]
    self.rl_num_input_frames = rl_num_input_frames
    self.rl_eval_policy = rl_eval_policy
    self._gaussian_residual = not isinstance(self.rl_agent, d4pg.D4PG)
    self.feats_spec = feats_spec
    self.state_keys = sorted(state_keys)
    self.bernoulli_rate = bernoulli_rate
    self.sticky_rate = sticky_rate
    self.rl_observation_network_type = rl_observation_network_type
    self.action_space_type = action_space
    self.action_space = bc_agent.ActionSpace(
        action_norm, env=env, scale=action_norm_scale)
    self.signals_obs_space = bc_agent.ObservationSpace(signals_norm, env)
    if action_norm in ['centered', 'zeromean_unitvar']:
      # Reuse stats; normalization scheme may still be different.
      self.action_space.mean = self.base_agent.action_space.mean
      self.action_space.std = self.base_agent.action_space.std
    if signals_norm == 'zeromean_unitvar':
      self.signals_obs_space.mean = self.base_agent.signals_obs_space.mean
      self.signals_obs_space.std = self.base_agent.signals_obs_space.std

  def denormalize_flat(self, norm_action):
    # flat_action_to_dict converts an action into a dict without normalization.
    norm_action = self.flat_action_to_dict(norm_action)
    action = self.action_space.denormalize(norm_action)
    if isinstance(norm_action, dict):
      action = np.concatenate([v for k, v in sorted(action.items())])
    return action

  def denormalize_flat_base_action(self, norm_action):
    # flat_action_to_dict converts an action into a dict without normalization.
    norm_action = self.flat_action_to_dict(norm_action)
    action = self.base_agent.action_space.denormalize(norm_action)
    if isinstance(norm_action, dict):
      action = np.concatenate([v for k, v in sorted(action.items())])
    return action

  def flat_action_to_dict(self, action):
    if self.action_space_type == 'tool_lin':
      action = {'grip_velocity': action[:1],
                'linear_velocity': action[1:]}
    elif self.action_space_type == 'tool_lin_ori':
      action = {'angular_velocity': action[:3],
                'grip_velocity': action[3:4],
                'linear_velocity': action[4:]}
    return action

  def rl_policy_params(self, observation):
    if self.rl_observation_network_type is None:
      batched_observation = tf_utils.add_batch_dim(observation)
    else:
      batched_observation = (
          self.rl_agent._learner._observation_network(observation))  # pylint: disable=protected-access
    policy_distr = self.rl_agent._learner._policy_network(batched_observation)  # pylint: disable=protected-access
    mean = policy_distr.loc.numpy().squeeze(axis=0)
    std = policy_distr.scale.diag.numpy().squeeze(axis=0)
    return mean, std

  def shape_image_input(self, observation, input_type, num_input_channels):
    """Separate stacked observation to a list."""
    image_obs = observation[input_type]
    if input_type == 'depth':
      obs_list = []
      for c in range(0, image_obs.shape[2], num_input_channels):
        # History is stacked along 3rd dimension.
        obs_list.append({})
        obs_list[-1][input_type] = image_obs[:, :, c:(c + num_input_channels)]
    else:
      # History is stacked along 1st dimension.
      obs_list = [{input_type: image} for image in image_obs]
    return obs_list

  def shape_base_agent_input(self, observation):
    obs_list = self.shape_image_input(
        observation, self.base_agent.input_type,
        self.base_agent.num_input_channels)
    # Keep other keys.
    for key in observation:
      if key != self.base_agent.input_type:
        obs_list[-1][key] = observation[key]
    return obs_list

  def shape_rl_agent_input(self, norm_base_act, feats, observation):
    state_feats = []
    for key in self.state_keys:
      feature = observation[key]
      if isinstance(feature, int) or isinstance(feature, float):
        feature = np.array([feature])
      feature = self.signals_obs_space.normalize_feature(key, feature)
      state_feats.append(feature.flatten())
      if not state_feats[-1].shape:
        state_feats[-1] = np.expand_dims(state_feats[-1], 0)
    feats_dict = collections.OrderedDict()
    if 'base_action' in self.feats_spec:
      feats_dict['base_action'] = norm_base_act
    if 'feats' in self.feats_spec:
      feats_dict['feats'] = feats
    if self.rl_input_type in self.feats_spec:
      # Convert stacked array into a list of past time steps.
      obs_history = self.shape_image_input(
          observation, self.rl_input_type, self.rl_num_input_channels)
      # Convert list back to stacked history, with possible repetitions.
      stacked_img = self.base_agent.shape_image_observation(obs_history)
      feats_dict[self.rl_input_type] = stacked_img
    if state_feats and 'visible_state' in self.feats_spec:
      feats_dict['visible_state'] = np.concatenate(state_feats, axis=0)
    return feats_dict

  def get_acme_observation(self, observation):
    feats = None
    base_obs = None
    if self.base_controller is not None:
      base_dict = self.base_controller.get_action()
      if base_dict is None:
        base_dict = self.base_controller._env.action_space.sample()  # pylint: disable=protected-access
        for k, v in base_dict.items():
          base_dict[k] = np.zeros_like(v)
      norm_base_act = self.base_agent.action_space.normalize(base_dict)
    elif self.base_agent.network is not None:
      base_obs = self.shape_base_agent_input(observation)
      # Signals are normalized if needed.
      norm_base_act, feats = self.base_agent.get_raw_action(
          base_obs, return_feats=True)
      feats = np.squeeze(feats)
    if isinstance(norm_base_act, dict):
      # TODO(minttu): Clip norm_base_act to action space limits before adding
      # residual?
      norm_base_act = np.concatenate(
          [v for k, v in sorted(norm_base_act.items())]).astype(np.float32)
    rl_obs = self.shape_rl_agent_input(norm_base_act, feats, observation)
    return rl_obs, base_obs, norm_base_act

  def select_action(
      self, rl_obs, norm_base_act, full_obs=None, prev_residual=None,
      prev_exploration=False, add_exploration=True, collapse=False,
      verbose=False):
    mean, std = None, None
    chose_exploration = False
    if collapse and self.rl_eval_policy is not None:
      # Call custom deterministic policy; overrides gripper exploration.
      batched_rl_obs = tf_utils.add_batch_dim(rl_obs)
      batched_residual = self.rl_eval_policy(batched_rl_obs)
      residual = batched_residual.numpy().squeeze(axis=0)
    else:
      residual = self.rl_agent.select_action(rl_obs)
      if add_exploration:
        if prev_exploration and np.random.rand() < self.sticky_rate:
          residual[0] = prev_residual[0]
          chose_exploration = True
        elif np.random.rand() < self.bernoulli_rate:
          # Only explore open if gripper not currently opened (& vice versa).
          grip_state = full_obs['grip_state']
          if grip_state < 0.4:
            residual[0] = -2  # Close
          else:
            residual[0] = 2  # Open
          # residual[0] = (np.random.rand() < 0.5) * 4 - 2
          chose_exploration = True
      if self._gaussian_residual:
        mean, std = self.rl_policy_params(rl_obs)
        if collapse:  # Collapse overrides gripper exploration.
          if verbose:
            print(f'Collapsing {residual} to mean {mean} (std {std})')
          residual = mean
        elif chose_exploration and verbose:
          print(f'Exploring {residual} from mean {mean}, std {std})')
        elif verbose:
          print(f'Drew {residual} from mean {mean}, std {std})')

    base_action = self.denormalize_flat_base_action(norm_base_act)
    residual_action = self.denormalize_flat(residual)
    if self.action_space.norm_type != self.base_agent.action_space.norm_type:
      # Normalize each action separately. denorm(r) + denorm(b).
      # Makes sense with residual_norm == centered
      if isinstance(base_action, dict):
        action = {k: v + residual_action[k] for k, v in base_action.items()}
      else:
        action = base_action + residual_action
    else:
      # Normalize once only. denorm(r + b).
      # Makes sense with base_norm == residual_norm == zeromean_unitvar.
      norm_act = norm_base_act + residual
      action = self.denormalize_flat(norm_act)
    return (action, base_action, residual_action, residual, chose_exploration,
            mean, std)

  def observe(self, action, timestep):
    self.rl_agent.observe(action, timestep)

  def observe_first(self, timestep):
    self.rl_agent.observe_first(timestep)

  def update(self, *args, **kwargs):
    self.rl_agent.update(*args, **kwargs)

  def save_policy_weights(self, out_path):
    checkpoint = tf.train.Checkpoint(
        module=self.rl_agent._learner._policy_network)  # pylint: disable=protected-access
    checkpoint.save(out_path)


class FixedObservationAgent(ResidualAgent):
  """Non-residual RL agent with fixed observation network.

  Reuses acme observation preprocessing (and other functionality) from
  ResidualAgent but only uses base agent for input features, not base action.
  """

  def select_action(
      self, rl_obs, norm_base_act, full_obs=None, prev_residual=None,
      prev_exploration=False, add_exploration=True, collapse=False,
      verbose=False):
    mean, std = None, None
    chose_exploration = False
    if collapse and self.rl_eval_policy is not None:
      # Call custom deterministic policy; overrides gripper exploration.
      batched_rl_obs = tf_utils.add_batch_dim(rl_obs)
      batched_residual = self.rl_eval_policy(batched_rl_obs)
      residual = batched_residual.numpy().squeeze(axis=0)
    else:
      residual = self.rl_agent.select_action(rl_obs)
      if add_exploration:
        if prev_exploration and np.random.rand() < self.sticky_rate:
          residual[0] = prev_residual[0]
          chose_exploration = True
        elif np.random.rand() < self.bernoulli_rate:
          # Only explore open if gripper not currently opened (& vice versa).
          grip_state = full_obs['grip_state']
          if grip_state < 0.4:
            residual[0] = -2  # Close
          else:
            residual[0] = 2  # Open
          # residual[0] = (np.random.rand() < 0.5) * 4 - 2
          chose_exploration = True
      if self._gaussian_residual:
        mean, std = self.rl_policy_params(rl_obs)
        if collapse:  # Collapse overrides gripper exploration.
          if verbose:
            print(f'Collapsing {residual} to mean {mean} (std {std})')
          residual = mean
        elif chose_exploration and verbose:
          print(f'Exploring {residual} from mean {mean}, std {std})')
        elif verbose:
          print(f'Drew {residual} from mean {mean}, std {std})')

    base_action = np.zeros_like(residual)
    residual_action = self.denormalize_flat(residual)
    action = residual_action
    return (action, base_action, residual_action, residual, chose_exploration,
            mean, std)


class RLAgent(ResidualAgent):
  """An RL agent with a no-op base agent defining action normalization."""
  # TODO(minttu): Which action normalization to use? Unit?
  # I guess it would be good to verify how much zeromean_unitvar helps BC
  # compared to unit action norm. It would be cleanest to have RL-from-scratch
  # experiments with unit var only, without using demonstrations. But if the
  # difference is significant for BC, might not be justified to compare
  # RL-from-scratch with uninformative normalization with residual RL with an
  # informative scheme.

  def shape_rl_agent_input(self, observation):
    """Concatenate included keys from observation into visible state."""
    state_feats = []
    for key in self.state_keys:
      feature = observation[key]
      feature = self.signals_obs_space.normalize_feature(key, feature)
      state_feats.append(feature)
      if not state_feats[-1].shape:
        state_feats[-1] = np.expand_dims(state_feats[-1], 0)
    feats_dict = collections.OrderedDict()
    if state_feats and 'visible_state' in self.feats_spec:
      visible_state = np.concatenate(state_feats, axis=0)
      feats_dict['visible_state'] = visible_state
    return feats_dict

  def get_acme_observation(self, observation):
    rl_obs = self.shape_rl_agent_input(observation)
    if self.rl_input_type in observation:
      rl_obs[self.rl_input_type] = observation[self.rl_input_type]
    # if self.rl_input_type != 'full_state':
    # base_obs = observation[self.rl_input_type]
    # Frame stacking wrapper stacks frames on final dimension.
    # history_length = int(base_obs.shape[-1])
    # If history is shorter than num input frames, repeat oldest state.
    # history_length = int(base_obs.shape[2] / self.rl_num_input_channels)
    # num_missing = self.rl_num_input_frames - history_length
    # repeat_frames = np.repeat(base_obs[:, :, :self.rl_num_input_channels],
    # num_missing, axis=2)
    # Oldest frame is first.
    # img = np.concatenate([repeat_frames, base_obs], axis=2)
    # rl_obs[self.rl_input_type] = img
    return rl_obs, None, None

  def select_action(
      self, rl_obs, unused_norm_base_act, prev_residual=None,
      prev_exploration=False, add_exploration=True, collapse=False,
      verbose=False):
    mean, std = None, None
    chose_exploration = False
    if collapse and self.rl_eval_policy is not None:
      # Call custom deterministic policy; overrides gripper exploration.
      batched_rl_obs = tf_utils.add_batch_dim(rl_obs)
      batched_residual = self.rl_eval_policy(batched_rl_obs)
      residual = batched_residual.numpy().squeeze(axis=0)
    else:
      residual = self.rl_agent.select_action(rl_obs)
      if add_exploration:
        if prev_exploration and np.random.rand() < self.sticky_rate:
          residual[0] = prev_residual[0]
          chose_exploration = True
        elif np.random.rand() < self.bernoulli_rate:
          # TODO(minttu): Only explore open if gripper not fully opened (& vice
          # versa).
          residual[0] = (np.random.rand() < 0.5) * 4 - 2
          chose_exploration = True
      if self._gaussian_residual:
        batched_rl_obs = tf_utils.add_batch_dim(rl_obs)
        mean, std = self.rl_policy_params(batched_rl_obs)
        if collapse:  # Collapse overrides gripper exploration.
          if verbose:
            print(f'Collapsing {residual} to mean {mean} (std {std})')
          residual = mean
        elif chose_exploration and verbose:
          print(f'Exploring {residual} from mean {mean}, std {std})')
        elif verbose:
          print(f'Drew {residual} from mean {mean}, std {std})')

    action = self.denormalize_flat(residual)
    residual_action = self.denormalize_flat(residual)
    base_action = np.zeros_like(residual)
    return (action, base_action, residual_action, residual, chose_exploration,
            mean, std)
