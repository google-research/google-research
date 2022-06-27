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

"""An Environment class storing global configuration data for Contrack."""

import json
import os
import pprint
from typing import Any, Dict, Text

import tensorflow as tf

from contrack import encoding


class Config(object):
  """Base class for the config objects defined below."""

  def as_dict(self):
    res = {}
    for attr in filter(lambda a: not a.startswith('__'), dir(self)):
      val = getattr(self, attr)
      if isinstance(val, (str, bool, int, float)):
        res[attr] = val
      elif isinstance(val, Config):
        res[attr] = val.as_dict()
    return res

  def __repr__(self):
    return pprint.pformat(self.as_dict())


class AttentionConfig(Config):
  """Configuration for an attention layer."""

  def __init__(self):
    self.filter_layer_size = 800
    self.num_heads = 9
    self.dropout_rate = 0.0

  @classmethod
  def load_from_dict(cls, att_dict):
    config = AttentionConfig()
    config.filter_layer_size = att_dict['filter_layer_size']
    config.num_heads = att_dict['num_heads']
    config.dropout_rate = att_dict['dropout_rate']
    return config


class ContrackConfig(Config):
  """Configuration data for Contrack model."""

  def __init__(self):
    self.new_id_attention = AttentionConfig()
    self.tracking_attention = AttentionConfig()
    self.timing_signal_size = 10
    self.new_id_false_negative_cost = 6.0
    self.max_seq_len = 100
    self.batch_size = 20
    self.shuffle_buffer_size = 50
    self.rotate_enref_ids = True
    self.max_steps = 50000
    self.steps_per_epoch = 1000
    self.learning_rate = 0.001

  @classmethod
  def load_from_json(cls, config_str):
    """Parses the config from a json string."""
    config = ContrackConfig()
    config_dict = json.loads(config_str)
    config.new_id_attention = AttentionConfig.load_from_dict(
        config_dict['new_id_attention'])
    config.tracking_attention = AttentionConfig.load_from_dict(
        config_dict['tracking_attention'])
    config.timing_signal_size = config_dict['timing_signal_size']
    config.new_id_false_negative_cost = config_dict[
        'new_id_false_negative_cost']
    config.max_seq_len = config_dict['max_seq_len']
    config.batch_size = config_dict['batch_size']
    config.shuffle_buffer_size = config_dict['shuffle_buffer_size']
    config.rotate_enref_ids = config_dict['rotate_enref_ids']
    config.max_steps = config_dict['max_steps']
    config.steps_per_epoch = config_dict['steps_per_epoch']
    config.learning_rate = config_dict['learning_rate']

    return config

  @classmethod
  def load_from_path(cls, path):
    """Reads the config from a json file."""
    with tf.io.gfile.GFile(path, 'r') as config_file:
      config_str = config_file.read()

    return cls.load_from_json(config_str)

  def save(self, path):
    filepath = os.path.join(path, 'contrack_config.json')
    with tf.io.gfile.GFile(filepath, 'w') as file:
      json.dump(self.as_dict(), file, indent=2)


class Env(object):
  """Environment singleton storing global configuration data."""
  instance = None

  @classmethod
  def get(cls):
    return cls.instance

  @classmethod
  def init(cls, config, encodings):
    cls.instance = Env(config, encodings)

  @classmethod
  def init_from_saved_model(cls, path):
    config = ContrackConfig.load_from_path(
        os.path.join(path, 'contrack_config.json'))
    encodings = encoding.Encodings.load_from_json(
        os.path.join(path, 'encodings.json'))
    cls.instance = Env(config, encodings)

  def __init__(self, config, encodings):
    self.config = config
    self.encodings = encodings
