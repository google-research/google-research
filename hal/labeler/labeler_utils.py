# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Utilities for labelers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from hal.labeler.decoders import FilmDecoder
from hal.labeler.decoders import RNNAttentionDecoder
from hal.labeler.decoders import RNNDecoder
from hal.labeler.encoders import CNNEncoder
from hal.labeler.encoders import CNNEncoderSingleFrame
from hal.labeler.encoders import MLPEncoder


def get_captioning_encoder(config):
  """Get captinoning model's encoder based on config."""
  if config['name'] == 'state':
    return MLPEncoder(
        embedding_dim=config['embedding_dim'],
        name='captioning_encoder',
        subtract_neighboring_observation=True)
  elif config['name'] == 'image':
    return CNNEncoder(
        embedding_dim=config['embedding_dim'],
        name='captioning_encoder')
  else:
    raise ValueError('Unrecognized model type: {}'.format(config['name']))


def get_captioning_decoder(config):
  """Returns captinoning model's decoder based on config."""
  if config['name'] == 'state':
    return RNNDecoder(
        embedding_dim=config['word_embedding_dim'],
        hidden_units=config['hidden_units'],
        vocab_size=config['vocab_size'],
        name='captioning_encoder')
  elif config['name'] == 'attention':
    return RNNAttentionDecoder(
        embedding_dim=config['word_embedding_dim'],
        units=config['hidden_units'],
        vocab_size=config['vocab_size'],
        name='captioning_encoder')
  else:
    raise ValueError('Unrecognized model type: {}'.format(config['name']))


def get_answering_encoder(config):
  """Get answering model's encoder based on config."""
  if config['name'] == 'state':
    return MLPEncoder(
        embedding_dim=config['embedding_dim'], name='answering_encoder')
  elif config['name'] == 'state-film':
    return tf.keras.layers.Lambda(lambda x: x)
  elif config['name'] == 'image':
    return CNNEncoderSingleFrame(
        embedding_dim=config['embedding_dim'],
        name='answering_encoder')
  else:
    raise ValueError('Unrecognized model type: {}'.format(config['name']))


def get_answering_decoder(config):
  """Get answering model's decoder based on config."""
  if config['name'] == 'state':
    return RNNDecoder(
        embedding_dim=config['word_embedding_dim'],
        hidden_units=config['hidden_units'],
        vocab_size=config['vocab_size'],
        name='answering_decoder')
  elif config['name'] == 'state-film':
    return FilmDecoder(
        vocab_size=config['vocab_size'], name='answering_decoder')
  elif config['name'] == 'attention':
    return RNNAttentionDecoder(
        embedding_dim=config['word_embedding_dim'],
        units=config['hidden_units'],
        vocab_size=config['vocab_size'],
        name='answering_decoder')
  else:
    raise ValueError('Unrecognized model type: {}'.format(config['name']))
