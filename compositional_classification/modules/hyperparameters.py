# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

# coding=utf-8
"""Hyperparameter configurations for all the classification models."""

import argparse

from absl import flags

FLAGS = flags.FLAGS

# All the flag arguments below should have None for their default values to
# recognize the values entered by command line arguments. Please specify the
# actual default values in the basic_hparams() method.

# Basic hyper-parameters
flags.DEFINE_string('initializer', None, 'kernel initialization method')
flags.DEFINE_float('initializer_range', None, 'Range for initializer')
flags.DEFINE_string('optimizer', None, 'Optimizer')
flags.DEFINE_float('optimizer_adam_epsilon', None, 'Epsilon of Adam optimizer')
flags.DEFINE_float('optimizer_adam_beta1', None, 'Beta_1 of Adam optimizer')
flags.DEFINE_float('optimizer_adam_beta2', None, 'Beta_2 of Adam optimizer')
flags.DEFINE_float('weight_decay', None, 'L2 regularization strength')
flags.DEFINE_float('learning_rate', None, 'Learning rate')
flags.DEFINE_enum('learning_rate_schedule', None, ['constant', 'exp', 'rsqrt'],
                  'Learning rate scheduler (constant, exp, rsqrt)')
flags.DEFINE_integer('learning_rate_decay_steps', None,
                     'Steps per decay period')
flags.DEFINE_boolean('learning_rate_decay_staircase', None, 'Staircase decay')
flags.DEFINE_float('learning_rate_decay_rate', None, 'Decay rate per period')
flags.DEFINE_float('learning_rate_minimum', None, 'The minimum learing rate')
flags.DEFINE_integer('learning_rate_warmup_steps', None, 'Learning rate warmup')

# Model hyper-parameters
flags.DEFINE_integer('num_hidden_layers', None, 'Number of LSTM layers')
flags.DEFINE_float('dropout', None, 'Dropout rate (probability to drop)')
flags.DEFINE_float('attn_dropout', None, 'Dropout rate in multi-head attention')
flags.DEFINE_integer('hidden_size', None, 'Dimensionality of hidden vectors')
flags.DEFINE_integer('num_dense_layers', None, 'Number of dense layers')
flags.DEFINE_integer('num_encoder_layers', None, 'Number of encoder layers')
flags.DEFINE_integer('num_heads', None,
                     'Number of heads in multi-head attention')
flags.DEFINE_string('hidden_act', None,
                    'Activation function in multi-head attention')
flags.DEFINE_integer('filter_size', None,
                     'hidden_size in feed forward layer of encoder')
flags.DEFINE_boolean('learned_position_encoding', None,
                     'Whether to train positional encoding')
flags.DEFINE_integer('maximum_position_encoding', None,
                     'The maximum input length of positional encoding')

# Dataset options
flags.DEFINE_boolean('parse_tree_input', None,
                     'Whether to use parse tree input')
flags.DEFINE_boolean('use_attention_mask', None,
                     'Whether to use hard attention using attention mask')
flags.DEFINE_boolean('use_relative_attention', None,
                     'Whether to use relative attention')
flags.DEFINE_boolean('parse_tree_attention', None,
                     'Whether to use parse tree attentions')
flags.DEFINE_boolean('block_attention', None, 'Whether to use block attentions')
flags.DEFINE_boolean('block_attention_sep', None,
                     'Whether to include SEP token to the second block')
flags.DEFINE_boolean('cls_global_token', None,
                     'Whether to make CLS token global that can attend others')
flags.DEFINE_boolean(
    'entity_cross_link', None,
    'Whether to use cross link attention between same entities')
flags.DEFINE_boolean('cross_link_exact', None,
                     'Whether to make cross link only for exact token matches')
flags.DEFINE_boolean('restart_query_pos', None,
                     'Whether to restart positional ids of query')
flags.DEFINE_boolean('share_pos_embed', None,
                     'Whether to share pos embeddings of quesiton and query')
flags.DEFINE_boolean(
    'unique_structure_token_pos', None,
    'Whether to use a unique position id for structure tokens')
flags.DEFINE_integer('batch_size', None, 'Size of mini-batch')
flags.DEFINE_boolean('add_cls_token', None, 'Whether to insert CLS at begining')
flags.DEFINE_boolean('add_eos_token', None, 'Whether to insert EOS at end')

# Loss options
flags.DEFINE_float('class_imbalance', None, 'The ratio of examples (neg/pos)')
flags.DEFINE_enum('loss_weight_method', None, ['linear', 'sqrt'],
                  'Weight method on BCE loss (linear, sqrt)')


def get_hparams_from_flags():
  """Collects hyper-parameters from command-line arguments."""
  hparams = argparse.Namespace()
  for key in FLAGS.__flags:  # pylint:disable=protected-access
    value = getattr(FLAGS, key)
    if value is not None:
      setattr(hparams, key, value)
  return hparams


def basic_hparams(hparams):
  """Basic hyper-parameters."""
  hparams.initializer = getattr(hparams, 'initializer', 'orthogonal')
  hparams.optimizer = getattr(hparams, 'optimizer', 'adam')
  hparams.optimizer_adam_epsilon = getattr(hparams, 'optimizer_adam_epsilon',
                                           1e-6)
  hparams.optimizer_adam_beta1 = getattr(hparams, 'optimizer_adam_beta1', 0.85)
  hparams.optimizer_adam_beta2 = getattr(hparams, 'optimizer_adam_beta2', 0.997)
  hparams.weight_decay = getattr(hparams, 'weight_decay', 1e-6)
  hparams.learning_rate = getattr(hparams, 'learning_rate', 0.1)
  hparams.learning_rate_schedule = getattr(hparams, 'learning_rate_schedule',
                                           'constant')  # constant, exp, rsqrt
  hparams.learning_rate_decay_steps = getattr(hparams,
                                              'learning_rate_decay_steps', 5000)
  hparams.learning_rate_decay_staircase = getattr(
      hparams, 'learning_rate_decay_staircase', False)
  hparams.learning_rate_decay_rate = getattr(hparams,
                                             'learning_rate_decay_rate', 1.0)
  hparams.learning_rate_minimum = getattr(hparams, 'learning_rate_minimum',
                                          None)
  hparams.learning_rate_warmup_steps = getattr(hparams,
                                               'learning_rate_warmup_steps',
                                               100)
  hparams.batch_size = getattr(hparams, 'batch_size', 4096)
  hparams.parse_tree_input = getattr(hparams, 'parse_tree_input', False)
  hparams.use_attention_mask = getattr(hparams, 'use_attention_mask', False)
  hparams.use_relative_attention = getattr(hparams, 'use_relative_attention',
                                           False)
  hparams.add_cls_token = getattr(hparams, 'add_cls_token', False)
  hparams.add_eos_token = getattr(hparams, 'add_eos_token', False)
  hparams.class_imbalance = getattr(hparams, 'class_imbalance', 1.0)
  hparams.loss_weight_method = getattr(hparams, 'loss_weight_method',
                                       'linear')  # linear, sqrt


def lstm_model_hparams():
  """LSTM hyper-parameters for CFQ classification."""
  hparams = get_hparams_from_flags()
  hparams.batch_size = getattr(hparams, 'batch_size', 1024)
  hparams.dropout = getattr(hparams, 'dropout', 0.4)
  hparams.hidden_size = getattr(hparams, 'hidden_size', 512)
  hparams.num_hidden_layers = getattr(hparams, 'num_hidden_layers', 2)
  hparams.num_dense_layers = getattr(hparams, 'num_dense_layers', 2)
  hparams.initializer = getattr(hparams, 'initializer', 'VarianceScaling')
  hparams.learning_rate = getattr(hparams, 'learning_rate', 0.001)
  hparams.weight_decay = getattr(hparams, 'weight_decay', 0.0)
  hparams.add_eos_token = getattr(hparams, 'add_eos_token', True)
  basic_hparams(hparams)
  return hparams


def transformer_model_hparams():
  """Transformer hyper-paramters for CFQ classification."""
  hparams = get_hparams_from_flags()
  hparams.batch_size = getattr(hparams, 'batch_size', 512)
  hparams.dropout = getattr(hparams, 'dropout', 0.1)
  hparams.hidden_size = getattr(hparams, 'hidden_size', 128)
  hparams.num_encoder_layers = getattr(hparams, 'num_encoder_layers', 2)
  hparams.filter_size = getattr(hparams, 'filter_size', 2048)
  hparams.maximum_position_encoding = getattr(hparams,
                                              'maximum_position_encoding', 256)
  hparams.num_heads = getattr(hparams, 'num_heads', 16)
  hparams.optimizer_adam_epsilon = getattr(hparams, 'optimizer_adam_epsilon',
                                           1e-9)
  hparams.optimizer_adam_beta1 = getattr(hparams, 'optimizer_adam_beta1', 0.9)
  hparams.optimizer_adam_beta2 = getattr(hparams, 'optimizer_adam_beta2', 0.997)
  hparams.learning_rate = getattr(hparams, 'learning_rate', 0.001)
  hparams.learning_rate_schedule = getattr(hparams, 'learning_rate_schedule',
                                           'rsqrt')
  hparams.learning_rate_warmup_steps = getattr(hparams,
                                               'learning_rate_warmup_steps',
                                               1000)
  hparams.weight_decay = getattr(hparams, 'weight_decay', 0.0)
  hparams.add_cls_token = getattr(hparams, 'add_cls_token', True)
  basic_hparams(hparams)
  return hparams


def relative_transformer_model_hparams():
  """Transformer hyper-paramters for CFQ classification."""
  hparams = get_hparams_from_flags()
  hparams.batch_size = getattr(hparams, 'batch_size', 336)
  hparams.dropout = getattr(hparams, 'dropout', 0.1)
  hparams.attn_dropout = getattr(hparams, 'attn_dropout', 0.0)
  hparams.hidden_size = getattr(hparams, 'hidden_size', 128)
  hparams.num_encoder_layers = getattr(hparams, 'num_encoder_layers', 2)
  hparams.filter_size = getattr(hparams, 'filter_size', 512)
  hparams.learned_position_encoding = getattr(hparams,
                                              'learned_position_encoding', True)
  hparams.maximum_position_encoding = getattr(hparams,
                                              'maximum_position_encoding', 256)
  hparams.num_heads = getattr(hparams, 'num_heads', 16)
  hparams.hidden_act = getattr(hparams, 'hidden_act', 'gelu')
  hparams.initializer_range = getattr(hparams, 'initializer_range', 0.1)
  hparams.optimizer_adam_epsilon = getattr(hparams, 'optimizer_adam_epsilon',
                                           1e-9)
  hparams.optimizer_adam_beta1 = getattr(hparams, 'optimizer_adam_beta1', 0.9)
  hparams.optimizer_adam_beta2 = getattr(hparams, 'optimizer_adam_beta2', 0.997)
  hparams.learning_rate = getattr(hparams, 'learning_rate', 0.001)
  hparams.learning_rate_schedule = getattr(hparams, 'learning_rate_schedule',
                                           'rsqrt')
  hparams.learning_rate_warmup_steps = getattr(hparams,
                                               'learning_rate_warmup_steps',
                                               1000)
  hparams.weight_decay = getattr(hparams, 'weight_decay', 0.0)
  hparams.parse_tree_input = getattr(hparams, 'parse_tree_input', True)
  hparams.use_attention_mask = getattr(hparams, 'use_attention_mask', False)
  hparams.use_relative_attention = getattr(hparams, 'use_relative_attention',
                                           False)
  hparams.parse_tree_attention = getattr(hparams, 'parse_tree_attention', False)
  hparams.block_attention = getattr(hparams, 'block_attention', False)
  hparams.block_attention_sep = getattr(hparams, 'block_attention_sep', False)
  hparams.cls_global_token = getattr(hparams, 'cls_global_token', True)
  hparams.entity_cross_link = getattr(hparams, 'entity_cross_link', False)
  hparams.cross_link_exact = getattr(hparams, 'cross_link_exact', True)
  hparams.restart_query_pos = getattr(hparams, 'restart_query_pos', True)
  hparams.share_pos_embed = getattr(hparams, 'share_pos_embed', False)
  hparams.unique_structure_token_pos = getattr(hparams,
                                               'unique_structure_token_pos',
                                               True)
  hparams.add_cls_token = getattr(hparams, 'add_cls_token', True)
  basic_hparams(hparams)
  return hparams
