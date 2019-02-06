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

"""Embedding Models for SQUAD.
"""

import tensorflow as tf
import tensorflow_hub as hub

from qanet import model_base
from qanet import squad_helper
from qanet.util import misc_util

MAX_BATCH_SIZE = 128


class SubwordEmbedder(model_base.Module):
  """Subword embedding layer for SQuADDataset model."""

  @staticmethod
  def _config():
    return {'trainable': False, 'module': ''}

  def _build(self, features):
    cfg = self.config
    if cfg.module == 'learned':
      # Handle test cases gracefully by falling back to an emb lookup
      emb_mat = tf.get_variable('emb_mat', [9000, 128])
    else:
      tf.logging.info('Using hub module: %s', self.config.module)
      module = hub.Module(cfg.module, trainable=cfg.trainable)

    def encode(x):
      if cfg.module == 'learned':
        return tf.nn.embedding_lookup(emb_mat, x)
      else:
        return module(x, as_dict=True)['emb']

    question = encode(features['question_ids'])
    context = encode(features['context_ids'])
    # Word encoding for both context and question
    return dict(xw=context, qw=question)


class QANetEmbeddingLayer(model_base.Module):
  """Embedding layer using word vectors, ELMO, and MT models."""

  @staticmethod
  def _config():
    cfg = {
        'char_emb_size': 200,  # character emb size, not word emb size
        'char_vocab_size': 262,
        'word_embedding_dropout': 0.1,
        'char_embedding_dropout': 0.1,
        'mt_ckpt_path': '',
        'elmo_path': '',
        'include_mt_embeddings': True,
        'num_gpus': 4,
        'base_gpu_elmo': 2,
        'mt_elmo': False,
        'use_glove': True,
        'use_char': True,
        'elmo': True,
        'elmo_option': 'elmo',  # 'elmo' or other options
    }
    return cfg

  def _build(self, features):
    (xc, qc, xw, qw, x_mt, q_mt, x_elmo, q_elmo) = (
        build_embedding_layer(
            features, mode=self.mode, params=self.config))
    # Word encoding for both context and question
    return dict(
        xc=xc,
        qc=qc,
        xw=xw,
        qw=qw,
        x_mt=x_mt,
        q_mt=q_mt,
        x_elmo=x_elmo,
        q_elmo=q_elmo)


def build_embedding_layer(features, mode, params, reuse=False):
  """Common embedding layer for feature and kernel functions.

  Args:
    features: A dictionary containing features, directly copied from `model_fn`.
    mode: Mode.
    params: Contains parameters, directly copied from `model_fn`.
    reuse: Reuse variables.

  Returns:
    `(x, q)` where `x` is embedded representation of context, and `q` is the
    embedded representation of the question.
  """
  with tf.variable_scope('embedding_layer', reuse=reuse):
    training = mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('embedding'):
      if params.get('use_char', True):
        tf.logging.info('# Char embeddings')
        # self-trained character embedding
        char_emb_mat = tf.get_variable(
            'char_emb_mat',
            [params['char_vocab_size'], params['char_emb_size']])
        if training:
          char_emb_mat = tf.nn.dropout(
              char_emb_mat,
              keep_prob=1.0 - params['char_embedding_dropout'],
              noise_shape=[params['char_vocab_size'], 1])
        xc = tf.nn.embedding_lookup(
            char_emb_mat, features['indexed_context_chars'][:, 1:-1, :])
        qc = tf.nn.embedding_lookup(
            char_emb_mat, features['indexed_question_chars'][:, 1:-1, :])
        xc = tf.reduce_max(xc, 2)
        qc = tf.reduce_max(qc, 2)
      else:
        xc, qc = None, None

      # glove embedding
      if params['use_glove']:
        _, xw, qw = squad_helper.glove_layer(features, mode, params)
      else:
        xw, qw = None, None

      # MT ELMO
      x_mt, q_mt = None, None
      gpu_id = 1
      if params['mt_elmo']:
        tf.logging.info('# MT ELMO gpu_id %d/%d', gpu_id, params['num_gpus'])
        with tf.device(misc_util.get_device_str(gpu_id, params['num_gpus'])):
          # Translation vectors
          x_mt = squad_helper.embed_translation(
              features['context_words'], features['context_num_words'],
              params['mt_ckpt_path'], params['include_mt_embeddings'])
          q_mt = squad_helper.embed_translation(
              features['question_words'], features['question_num_words'],
              params['mt_ckpt_path'], params['include_mt_embeddings'])

      # ELMO
      x_elmo, q_elmo = None, None
      if params['elmo']:
        gpu_id += 1
        tf.logging.info('# ELMO gpu_id %d/%d', gpu_id, params['num_gpus'])
        with tf.device(misc_util.get_device_str(gpu_id, params['num_gpus'])):
          # elmo vectors
          if params['elmo_option'] == 'elmo':
            x_elmo = squad_helper.embed_elmo_chars(
                features['indexed_context_chars'], 128,
                params['elmo_path'], training,
                params['num_gpus'],
                params['base_gpu_elmo'])
            q_elmo = squad_helper.embed_elmo_chars(
                features['indexed_question_chars'], 128,
                params['elmo_path'], training,
                params['num_gpus'],
                params['base_gpu_elmo'])
          else:
            x_elmo = squad_helper.embed_elmo_sentences(
                features['tokenized_context'], MAX_BATCH_SIZE,
                params['elmo_path'], training, params['elmo_option'])
            q_elmo = squad_helper.embed_elmo_sentences(
                features['tokenized_question'], MAX_BATCH_SIZE,
                params['elmo_path'], training, params['elmo_option'])

  tf.logging.info('# Done build_embedding_layer')

  return xc, qc, xw, qw, x_mt, q_mt, x_elmo, q_elmo
