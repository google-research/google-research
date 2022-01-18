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

"""Experiment configurations for screen2words models."""

_BASE_CONFIG = {
    'debug':
        False,
    'train_steps':
        100 * 1000,
    'batch_size':
        8,
    'beam_size':
        5,
    'dropout':
        0.2,
    'layer_prepostprocess_dropout':
        0.2,
    'attention_dropout':
        0.2,
    'relu_dropout':
        0.2,
    'glove_trainable':
        False,
    'train_pixel_encoder':
        True,
    'add_pixel_skip_link':
        False,
    'learning_rate_warmup_steps':
        2000,
    'learning_rate_constant':
        1,  # Learning rate constant used for schedule of t2t optimizer.
    'learning_rate':
        0.001,  # Learning rate used by Adam optimizer.
    'caption_optimizer':
        'adam',
    'clip_norm':
        1,
    'screen_encoder':
        'pixel_transformer',
    'num_decoder_layers':
        6,  # Decoder layer.
    'num_hidden_layers':
        6,  # View hierarchy encoder layer.
    'hidden_size':
        128,
    'num_heads':
        8,
    'filter_size':
        512,
    'obj_text_aggregation':
        'sum',
    'vocab_size':
        10000,
    'phrase_vocab_size':
        10000,
    'train_buffer_size':
        2000,
    'eval_buffer_size':
        100,
    'embedding_file': 'TO_BE_ADDED',
    'word_vocab_path': 'TO_BE_ADDED',
    'eval_files': 'TO_BE_ADDED',
    'train_files': 'TO_BE_ADDED',
    'screen_embedding_feature': ['text', 'type', 'pos', 'click', 'dom'],
    'use_app_description':
        True,
}

_DEBUG_CONFIG = {
    'debug':
        True,
    'train_steps':
        8,
    'eval_steps':
        1,
    'batch_size':
        2,
    'eval_batch_size':
        4,
    'dropout':
        0,
    'beam_size':
        5,
    'hidden_size':
        8,
    'filter_size':
        4,
    'num_heads':
        2,
    'layer_prepostprocess_dropout':
        0.2,
    'attention_dropout':
        0.2,
    'relu_dropout':
        0.2,
    'glove_trainable':
        False,
    'learning_rate_warmup_steps':
        2000,
    'learning_rate_constant':
        1,
    'learning_rate':
        0.001,
    'caption_optimizer':
        'adam',
    'clip_norm':
        1,
    'screen_encoder':
        'pixel_transformer',
    'num_decoder_layers':
        2,  # Decoder layer.
    'num_hidden_layers':
        4,  # View hierarchy encoder layer.
    'obj_text_aggregation':
        'sum',
    'train_buffer_size':
        2,
    'eval_buffer_size':
        2,
    'vocab_size':
        1000,
    'phrase_vocab_size':
        1000,
    'embedding_file': 'TO_BE_ADDED',
    'word_vocab_path': 'TO_BE_ADDED',
    'eval_files': 'TO_BE_ADDED',
    'train_files': 'TO_BE_ADDED',
    'use_app_description':
        True,
}


def _create_config(hparams):
  config = {}
  config.update(_BASE_CONFIG)
  config.update(hparams)
  return config


experiments = {
    'base':
        _BASE_CONFIG,
    'debug':
        _DEBUG_CONFIG,
    # Uses layout properties only
    'layout_only':
        _create_config({
            'screen_encoder': 'vh_only',
            'batch_size': 128,
            'learning_rate_constant': 2,
            'screen_embedding_feature': ['type', 'pos', 'click', 'dom'],
            'use_app_description': False,
        }),
    # Uses pixel only
    'caption_pixel_baseline':
        _create_config({
            'screen_encoder': 'pixel_only',
            'batch_size': 128,
            'learning_rate_constant': 2,
            'use_app_description': False,
        }),
    # Uses pixel + layout properties
    'caption_pixel_and_layout_baseline':
        _create_config({
            'screen_encoder': 'pixel_transformer',
            'batch_size': 128,
            'learning_rate_constant': 2,
            'screen_embedding_feature': ['type', 'pos', 'click', 'dom'],
            'use_app_description': False,
        }),
    # Uses view hierarchy + pixel, no app description
    'caption_no_appdesc':
        _create_config({
            'screen_encoder': 'pixel_transformer',
            'batch_size': 128,
            'learning_rate_constant': 2,
            'use_app_description': False,
        }),
    # This is the screen2words full model, uses vh+px+appdesc
    'caption_no_obj_no_bbx':
        _create_config({
            'screen_encoder': 'pixel_transformer',
            'batch_size': 128,
            'learning_rate_constant': 2,
            'use_app_description': True,
        }),
}
