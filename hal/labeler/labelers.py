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

"""Object for relabeling."""
# pylint: disable=unused-variable
# pylint: disable=undefined-variable
# pylint: disable=wildcard-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v2 as tf

from hal.labeler.labeler_utils import *


class Labeler:
  """A object that approximates the oracle.

  Attributes:
    generated_label_num: number of lables generated per transition
    max_sequence_length: maximum length of sequence generated
    temperature: temperature of sampling
    encoder: encoder of the captioning model
    decoder: decoder of the captioning model
    answering_encoder: encoder of the answering model
    answering_decoder: decoder of the answering model
    answering_projection_layer: final projection layer of the answering model
  """

  def __init__(self, labeler_config, name='labeler'):
    """Initializes the labeler.

    Args:
      labeler_config: configuration of the labeler
      name: optional name
    """
    self._name = name
    self.generated_label_num = labeler_config['generated_label_num']
    self.max_sequence_length = labeler_config['max_sequence_length']
    self.temperature = labeler_config['sampling_temperature']

  def set_captioning_model(self, labeler_config, saved_weight_path=None):
    """Set up the captinong model and maybe load weights.

    Args:
      labeler_config: configuration of the labeler
      saved_weight_path: optional path where weights are loaded from
    """
    self.encoder = get_captioning_encoder(labeler_config['captioning_encoder'])
    self.decoder = get_captioning_decoder(labeler_config['captioning_decoder'])
    if saved_weight_path:
      ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder)
      latest = tf.train.latest_checkpoint(saved_weight_path)
      assert latest, 'Captioning model ckpt not found in {}.'.format(
          saved_weight_path)
      print('Loading captioning model from: {}'.format(latest))
      ckpt.restore(latest)

  def set_answering_model(self, labeler_config, saved_weight_path=None):
    """Set up and load models of the answering model and maybe load weights.

    Args:
      labeler_config: configuration of the labeler
      saved_weight_path: optional path where weights are loaded from
    """
    self.answering_encoder = get_answering_encoder(
        labeler_config['answering_encoder'])
    self.answering_decoder = get_answering_decoder(
        labeler_config['answering_decoder'])
    self.answering_projection_layer = tf.keras.layers.Dense(
        1, activation='sigmoid', name='answering_projection')
    if saved_weight_path:
      ckpt = tf.train.Checkpoint(
          encoder=self.answering_encoder,
          decoder=self.answering_decoder,
          projection_layer=self.answering_projection_layer)
      latest = tf.train.latest_checkpoint(saved_weight_path)
      assert latest, 'Answering model ckpt not found in {}.'.format(
          saved_weight_path)
      print('Loading answering model from: {}'.format(latest))
      ckpt.restore(latest)

  def label_trajectory(self, trajectory, null_token=None):
    """Generate valid instructions for a trajectory of transitions.

    Args:
      trajectory: configuration of the labeler
      null_token: optional token that indicates the transition has no label

    Returns:
      labels for each transition in trajecotry, if any
    """
    instructions = self.label_batch_transition(trajectory)
    post_achieved_indicator = self.verify_batch_observation_batch_instruction(
        trajectory[:, 1], instructions)
    pre_achieved_indicator = self.verify_batch_observation_batch_instruction(
        trajectory[:, 0], instructions)
    filtered_inst = []
    # prune the false answers
    for i in range(len(trajectory)):
      if null_token:
        all_token = instructions[i].flatten()
        num_null = np.float32(all_token == null_token).sum()
        if num_null > (self.generated_label_num / 3.):
          filtered_inst.append([])
          continue
      filtered_transition_inst = []
      for pre, achieved, inst in zip(pre_achieved_indicator[i],
                                     post_achieved_indicator[i],
                                     instructions[i]):
        if achieved > 0.5 and pre < 0.5 and null_token not in inst:
          filtered_transition_inst.append(list(inst)[1:])  # cut sos symbol
      filtered_inst.append(filtered_transition_inst)
    return filtered_inst

  def label_transition(self, obs, obs_next):
    """Generate an instruction for two neighboring frames.

    Args:
      obs: one frame
      obs_next: the subsequent frame

    Returns:
      possible labels for that transition
    """
    instructions = self.label_batch_transition([(obs, obs_next)])
    return instructions[0]

  def label_batch_transition(self, transition_pairs):
    """Generate a batch of instructions for a batch of transitions.

    Args:
      transition_pairs: a batch of (obs, obs_next)

    Returns:
      possible labels for each transition
    """
    transition_pairs = tf.convert_to_tensor(transition_pairs)
    result = self._label_batch_transition(transition_pairs).numpy()
    return result

  def _label_batch_transition(self, transition_pairs_tensor):
    """Generate instructions from a batch of transitions."""
    num_pairs = len(transition_pairs_tensor)
    transition_pairs = self.encoder.preprocess(transition_pairs_tensor)
    features = self.encoder(transition_pairs)
    features = tf.expand_dims(features, axis=1)
    features_shape = tf.shape(features)
    features_rank = len(features_shape)
    tile_spec = tf.Variable(tf.ones([features_rank], dtype=tf.int32))
    tile_spec[1].assign(self.generated_label_num)
    features = tf.tile(features, tile_spec)
    # tile each pair for generated label num times
    features = tf.reshape(features, tf.concat([[-1], features_shape[2:]],
                                              axis=0))
    hidden = self.decoder.reset_state(batch_size=self.generated_label_num *
                                      num_pairs)
    result = [np.array([[1]] * self.generated_label_num * num_pairs)]
    dec_input = tf.Variable(result[0], dtype=tf.int32)
    for _ in tf.range(1, self.max_sequence_length):
      # passing the features through the decoder
      predictions, hidden, _ = self.decoder(dec_input, features, hidden)
      predicted_id = tf.random.categorical(
          predictions / self.temperature, 1, dtype=tf.int32).numpy()
      result.append(predicted_id)
      dec_input = predicted_id
    result = tf.transpose(tf.squeeze(tf.stack(result), axis=-1), [1, 0])
    result = tf.reshape(
        result, (num_pairs, self.generated_label_num, self.max_sequence_length))
    return result

  def verify_batch_observation_batch_instruction(self, batch_obs, batch_inst):
    """Verify whether each instruction fits each observation.

    Args:
      batch_obs: a batch of observation
      batch_inst: a batch of single label for each transition

    Returns:
      an array of boolean indicating if each instruction is valid for each
        transitions
    """
    bo_t = tf.convert_to_tensor(batch_obs)
    bi_t = tf.convert_to_tensor(batch_inst)
    return self._verify_batch_observation_batch_instruction(bo_t, bi_t)

  def verify_instruction(self, obs, instruction):
    """Verify a single instruction fits a single observation.

    Args:
      obs: a single observation
      instruction: a tokenized instruction

    Returns:
      whether the instruction is valid for the observation
    """
    obs, inst = tf.convert_to_tensor(obs), tf.convert_to_tensor(instruction)
    inst = tf.expand_dims(inst, axis=0)
    return self._verify_observation_batch_instruction(obs, inst)

  @tf.function
  def _verify_observation_batch_instruction(self, obs, batch_inst):
    """Verify if a single observation satisfy a batch of instruction."""
    batch_size = len(batch_inst)
    batch_inst = tf.expand_dims(batch_inst, axis=-1)
    features = self.answering_encoder(tf.expand_dims(obs, axis=0))
    features = tf.concat([features] * batch_size, axis=0)
    hidden = self.answering_decoder.reset_state(batch_size=batch_size)
    for i in tf.range(self.max_sequence_length):
      _, hidden, _ = self.answering_decoder(batch_inst[:, i], features, hidden)
    batch_answer = self.answering_projection_layer(hidden)
    return batch_answer

  @tf.function
  def _verify_batch_observation_batch_instruction(self, batch_obs, batch_inst):
    """Verify if an batch of observation satisfy a batch of instruction."""
    batch_size = len(batch_inst)
    batch_inst = tf.expand_dims(batch_inst, axis=-1)
    features = self.answering_encoder(batch_obs)
    hidden = self.answering_decoder.reset_state(batch_size=batch_size)
    for i in tf.range(self.max_sequence_length):
      _, hidden, _ = self.answering_decoder(batch_inst[:, i], features, hidden)
    batch_answer = self.answering_projection_layer(hidden)
    return batch_answer
