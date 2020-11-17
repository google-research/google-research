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

"""Hyperparameters for widget captioning model."""

_BASE_CONFIG = {
    'encode_screen_with_context': False,
    'debug': False,
    'train_with_one_node': False,
    'train_steps': 100 * 1000 * 1000,
    'batch_size': 64,
    'decoding_task': True,
    'classification_task': False,
    'classification_loss_weight': 1,
    'dropout': 0.2,
    'layer_prepostprocess_dropout': 0.2,
    'attention_dropout': 0.2,
    'relu_dropout': 0.2,
    'glove_trainable': False,
    'train_pixel_encoder': True,
    'add_pixel_skip_link': False,
    'learning_rate_warmup_steps': 2000,
    'learning_rate_constant':
        1,  # Learning rate constant used for schedule of t2t optimizer.
    'learning_rate': 0.001,  # Learning rate used by Adam optimizer.
    'caption_optimizer': 'adam',
    'clip_norm': 1,
    'screen_encoder': 'pixel_transformer',
    'num_decoder_layers': 6,  # Decoder layer.
    'num_hidden_layers': 6,  # View hierarchy encoder layer.
    'hidden_size': 128,
    'num_heads': 8,
    'filter_size': 512,
    'obj_text_aggregation': 'max',
    'vocab_size': 10000,
    'phrase_vocab_size': 10000,
    'train_buffer_size': 2000,
    'eval_buffer_size': 100,
    'embedding_file': None,
    'word_vocab_path': None,
    'phrase_vocab_path': None,
    'eval_files': None,
    'train_files': None,
    'widget_encoder_checkpoint': None,
    'use_developer_node_as_label': False,
    'use_worker_node_as_label': True,
}


def _create_config(hparams):
  config = {}
  config.update(_BASE_CONFIG)
  config.update(hparams)
  return config


experiments = {
    'pixel_transformer':
        _create_config({
            'screen_encoder': 'pixel_transformer',
            'caption_optimizer': 't2t',
        }),
}
