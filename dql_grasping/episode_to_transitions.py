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

"""Functions that map episode data to transitions for RL.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import numpy as np
from PIL import Image
from PIL import ImageFile

import tensorflow as tf
import gin.tf

_bytes_feature = (
    lambda v: tf.train.Feature(bytes_list=tf.train.BytesList(value=v)))
_int64_feature = (
    lambda v: tf.train.Feature(int64_list=tf.train.Int64List(value=v)))
_float_feature = (
    lambda v: tf.train.Feature(float_list=tf.train.FloatList(value=v)))


def jpeg_string(image, jpeg_quality=90):
  """Returns given PIL.Image instance as jpeg string.

  Args:
    image: A PIL image.
    jpeg_quality: The image quality, on a scale from 1 (worst) to 95 (best).

  Returns:
    a jpeg_string.
  """
  # This fix to PIL makes sure that we don't get an error when saving large
  # jpeg files. This is a workaround for a bug in PIL. The value should be
  # substantially larger than the size of the image being saved.
  ImageFile.MAXBLOCK = 640 * 512 * 64

  output_jpeg = io.BytesIO()
  image.save(output_jpeg, 'jpeg', quality=jpeg_quality, optimize=True)
  return output_jpeg.getvalue()


@gin.configurable
def episode_to_transitions_v0(episode_data, continuous=True):
  """Converts episode data to a series of TFExample transitions.

  Writes continuous actions as vector-encoded. If action is discrete, can
  represent as one-hot encoding.

  Args:
    episode_data: List of episode transition tuples (obs_t, action, reward,
      obs_tp1, done, debug).
    continuous: If True, encode vector-encoded action. Otherwise, encode int64
      action (discrete action space).

  Returns:
    List of TFExample transitions.

  """
  transitions = []
  for t, transition in enumerate(episode_data):
    (obs_t, action, reward, obs_tp1, done, debug) = transition
    del debug
    features = {}
    obs_t = Image.fromarray(obs_t)
    obs_tp1 = Image.fromarray(obs_tp1)
    features['S/img'] = _bytes_feature([jpeg_string(obs_t)])
    features['S_p1/img'] = _bytes_feature([jpeg_string(obs_tp1)])
    if continuous:
      if isinstance(action, np.ndarray):
        action = action.flatten().tolist()
      features['A'] = _float_feature(action)
    else:
      features['A'] = _int64_feature([action])
    features['R'] = _float_feature([reward])
    features['done'] = _int64_feature([int(done)])
    features['t'] = _int64_feature([t])
    transitions.append(
        tf.train.Example(features=tf.train.Features(feature=features)))
  return transitions


@gin.configurable
def episode_to_sequence_v0(episode_data, continuous=True, episode_length=16):
  """Converts episode data to a single TFExample.

  Writes continuous actions as vector-encoded. If action is discrete, can
  represent as one-hot encoding.

  Args:
    episode_data: List of episode transition tuples (obs_t, action, reward,
      obs_tp1, done, debug).
    continuous: If True, encode vector-encoded action. Otherwise, encode int64
      action (discrete action space).
    episode_length: Length to pad each episode to.

  Returns:
    List of length 1 with a single TFExample.

  Raises:
    ValueError: If input episode has length greater than episode_length.
  """
  input_length = len(episode_data)
  if input_length > episode_length:
    raise ValueError('Received episode length %d; expected %d' %
                     (input_length, episode_length))

  padding = episode_length - input_length
  episode_data += episode_data[-1:] * padding

  (all_obs_t, all_action, all_reward, all_obs_tp1,
   all_done, all_debug) = zip(*episode_data)
  del all_debug  # unused

  feature = {}
  feature['S/img'] = _bytes_feature(
      [jpeg_string(Image.fromarray(obs)) for obs in all_obs_t])
  feature['S_p1/img'] = _bytes_feature(
      [jpeg_string(Image.fromarray(obs)) for obs in all_obs_tp1])

  if continuous:
    all_action = np.array(all_action)
    all_action = all_action.flatten().tolist()
    feature['A'] = _float_feature(all_action)
  else:
    feature['A'] = _int64_feature(all_action)

  feature['R'] = _float_feature(all_reward)
  feature['done'] = _int64_feature([int(done) for done in all_done])
  feature['t'] = _int64_feature(range(episode_length))

  return [tf.train.Example(features=tf.train.Features(feature=feature))]


@gin.configurable
def episode_to_transitions_etrace(episode_data,
                                  lmbda=0.1,
                                  base_fn=episode_to_transitions_v0):
  """Forward-view elegibility trace returns.

  See https://arxiv.org/pdf/1704.05495.pdf for background. Reward at each step
  is distributed backwards through time so that total reward remains unchanged,
  but reward signal is no longer sparse.

  Args:
    episode_data: See episode_to_transitions_v0.
    lmbda: Fraction of reward mass to "shift" over from time t+1 to t.
    base_fn: Which fn to call to convert to TFExample.
  Returns:
    See episode_to_transitions_v0.
  """
  assert lmbda > 0 and lmbda < 1
  episode_data = [list(transition) for transition in episode_data]
  for t in reversed(range(len(episode_data)-1)):
    next_reward = episode_data[t+1][2]
    episode_data[t][2] += lmbda * next_reward
    episode_data[t+1][2] -= lmbda * next_reward
  return base_fn(episode_data)


@gin.configurable
def episode_to_transitions_mc(episode_data,
                              base_fn=episode_to_transitions_v0):
  """Monte-Carlo returns.

  Entire trajectory's return is used as target value of current state.

  Args:
    episode_data: See episode_to_transitions_v0.
    base_fn: Which function to call to convert episode_data to TFExample.
  Returns:
    See episode_to_transitions_v0.
  """
  episode_data = [list(transition) for transition in episode_data]
  total_reward = np.sum([transition[2] for transition in episode_data])
  for t in range(len(episode_data)):
    episode_data[t][2] = total_reward
  return base_fn(episode_data)


@gin.configurable
def episode_to_transitions_supervised(episode_data,
                                      base_fn=episode_to_transitions_v0):
  """N-step returns with synthetic actions to convert to 1-step problem.

  TODO(ejang) - refactor episode_data to deal with SARSTransitions.

  Args:
    episode_data: See episode_to_transitions_v0.
    base_fn: Which function to call to convert episode_data to TFExample.
  Returns:
    See episode_to_transitions_v0.
  """
  episode_data = [list(transition) for transition in episode_data]
  # Action of final transition.
  cumulative_action = np.array(episode_data[-1][1])
  cumulative_reward = np.sum([transition[2] for transition in episode_data])
  # Construct synthetic actions for all transitions leading up to the final
  # transition.
  for t in reversed(range(len(episode_data)-1)):
    episode_data[t][1] += cumulative_action
    episode_data[t][2] = cumulative_reward
    # Mark all transitions as terminal.
    episode_data[t][4] = 1
    cumulative_action = episode_data[t][1]
  return base_fn(episode_data)
