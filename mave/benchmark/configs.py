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

"""Configs for sequence tagging models."""
import datetime
import os

import ml_collections


def get_model_common_config():
  """Returns model common config."""
  config = ml_collections.ConfigDict()
  # For static input sequence length.
  config.seq_length: int = 512
  config.category_attribute_type_id: int = 0
  config.paragraph_type_id: int = 1
  config.cls_id: int = 101
  config.sep_id: int = 102

  return config


def get_bilstm_crf_config():
  """Returns BiLSTM-CRF sequence tagger config."""
  config = ml_collections.ConfigDict()
  config.vocab_size: int = 30522
  config.word_embeddding_size: int = 768
  config.lstm_units: int = 512
  config.use_attention_layer: bool = True
  config.use_attention_scale: bool = True
  config.recurrent_dropout: float = 0.4
  config.attention_dropout: float = 0.0
  config.num_tags: int = 2

  config.init_word_embedding_from_bert: bool = True

  return config


def get_bert_config():
  """Returns BERT sequence tagger config."""
  config = ml_collections.ConfigDict()
  config.bert_base_dir: str = ''
  config.bert_config_file: str = (
      config.get_ref('bert_base_dir') + '/bert_config.json')
  config.vocab_file: str = config.get_ref('bert_base_dir') + '/vocab.txt'

  ## Initialization
  # The model parameters will be initialized in the following order.
  # (1) If `bert_hub_module_url != ''`, BERT backbone model will be first loaded
  # from TFHub.
  config.bert_hub_module_url: str = ''
  config.bert_trainable: bool = True
  # (2) If `initial_checkpoint != ''` , BERT tagger model will be initialized
  # from `initial_checkpoint`.
  config.initial_checkpoint: str = ''
  # (3) If `initial_checkpoint == ''` and `pretrain_checkpoint != ''`, BERT
  # backbone will be initialized from `pretrain_checkpoint`.
  config.pretrain_checkpoint: str = ''
  # (4) If latest checkpoint present in the model_dir, the BERT tagger will be
  # restored from the latest checkpoint

  return config


def get_etc_config():
  """Returns ETC sequence tagger config."""
  config = ml_collections.ConfigDict()
  config.global_seq_length: int = 64
  config.long_seq_length: int = 1024
  config.etc_base_dir: str = ''
  config.etc_config_file: str = (
      config.get_ref('etc_base_dir') + '/etc_config.json')
  config.vocab_file: str = (
      config.get_ref('etc_base_dir') + '/vocab_bert_uncased_english.txt')

  ## Initialization
  # The model parameters will be initialized in the following order.
  # (1) If `initial_checkpoint != ''` , ETC tagger model will be initialized
  # from `initial_checkpoint`.
  config.initial_checkpoint: str = ''
  # (2) If `initial_checkpoint == ''` and `pretrain_checkpoint != ''`, ETC
  # backbone will be initialized from `pretrain_checkpoint`.
  config.pretrain_checkpoint: str = (
      config.get_ref('etc_base_dir') + '/model.ckpt')
  # (3) If latest checkpoint present in the model_dir, the ETC tagger will be
  # restored from the latest checkpoint

  config.use_one_hot_embeddings: bool = True
  config.use_one_hot_relative_embeddings: bool = True

  # Global token ids. Global and long tokens share a same vocab.
  # Not recommend to change, in-consistent with pretrained models。
  config.category_global_token_id: int = 3
  config.attribute_global_token_id: int = 4
  config.paragraph_global_token_id: int = 5

  # Token type ids. Global and long token types share a same vocab of size 16.
  # Global token type id. Using 0 for transfer learning.
  config.global_token_type_id: int = 0
  config.category_token_type_id: int = 1
  config.attribute_token_type_id: int = 2
  config.title_token_type_id: int = 3
  config.description_token_type_id: int = 4
  config.other_token_type_id: int = 5

  # Inference options.
  # Whether to filter long span predictions by global span predictions.
  config.filter_by_global: bool = True

  return config


def get_train_config():
  """Returns the config for training."""
  config = ml_collections.ConfigDict()
  config.experiment_base_dir: str = ''
  datatime_str: str = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
  config.model_dir: str = os.path.join(
      config.get_ref('experiment_base_dir').get(), f'{datatime_str}_test')

  config.tf_examples_filespec: str = ''
  # Random seed for data processing, e.g. shuffling.
  config.data_random_seed: int = 111

  config.train_batch_size: int = 128
  config.num_train_steps: int = 200000
  config.initial_learning_rate: float = 3e-5
  config.end_learning_rate: float = 0.0
  config.num_warmup_steps: int = 10000

  config.steps_per_loop: int = 1000
  config.save_summary_steps: int = 1000
  # Currently only a single checkpoint is saved at training end. For more
  # frequent checkpointing, we suggest to use TF2 customized training loop.
  config.save_checkpoints_steps: int = config.get_ref('num_train_steps')
  config.checkpoints_to_keep: int = 1

  # This is for keras `model.fit` function arguments. Metrcs will be evaluated
  # after each epoch. For all models, we only use the first 5000 batches of data
  # for training, which corresponds to a random sampling of
  # `steps_per_epoch * train_batch_size` examples.
  config.steps_per_epoch: int = 5000
  config.epochs: int = (
      config.get_ref('num_train_steps') // config.get_ref('steps_per_epoch'))

  return config


def get_eval_config():
  """Returns the config for evaluation."""
  config = ml_collections.ConfigDict()
  config.tf_examples_filespec: str = ''
  config.eval_batch_size: int = 128
  # Max number of eval steps. The actual eval steps will be calculated based on
  # eval dataset size.
  config.num_eval_steps: int = (50000 // config.get_ref('eval_batch_size'))

  return config


def get_data_config():
  """Returns the config for dataset."""
  config = ml_collections.ConfigDict()
  config.version: str = '00_All'
  config.use_category: bool = True
  config.use_attribute_key: bool = True
  config.use_cls: bool = True
  config.use_sep: bool = True
  config.debug: bool = True

  return config


def get_config():
  """Returns the config."""
  config = ml_collections.ConfigDict()
  config.model = get_model_common_config()
  # One of 'bert', 'bilstm_crf', 'etc'
  config.model_type: str = 'bert'
  config.bilstm_crf = get_bilstm_crf_config()
  config.bert = get_bert_config()
  config.etc = get_etc_config()
  config.train = get_train_config()
  config.eval = get_eval_config()
  config.data = get_data_config()

  return config
