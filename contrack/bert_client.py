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

"""API to compute BERT tokenization and embeddings."""

import os
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text  # pylint: disable=unused-import

DEFAULT_TOKENIZER_TFHUB_HANDLE = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
DEFAULT_ENCODER_TFHUB_HANDLE = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3'


class Tokenizer(object):
  """Tokenizes words to token_ids."""

  def __init__(self, tfhub_handle = DEFAULT_TOKENIZER_TFHUB_HANDLE):
    preprocessor = hub.load(tfhub_handle)
    model_path = hub.resolve(tfhub_handle)
    vocab_file_path = os.path.join(model_path, 'assets/vocab.txt')
    with tf.io.gfile.GFile(vocab_file_path, 'r') as vocab_file:
      self.vocab = [token.strip() for token in vocab_file.readlines()]

    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string)
    tokenize_layer = hub.KerasLayer(preprocessor.tokenize)
    outputs = tokenize_layer(text_input)

    self.model = tf.keras.Model(
        inputs=text_input, outputs=outputs, name='tokenizer')

  def tokenize(self, utterance):
    """Returns tokens and token_ids in utterance (used for BERT embeddings)."""
    model_input = tf.constant([utterance], dtype=tf.string)
    token_ids = self.model(model_input).numpy()
    token_ids = list(np.concatenate(token_ids[0]).flat)
    tokens = [self.vocab[tid] for tid in token_ids]
    return tokens, token_ids


class BertClient(object):
  """Computes BERT embeddings for input tokens."""

  def __init__(self, tf_hub_source = DEFAULT_ENCODER_TFHUB_HANDLE):
    self.batch_size = 1
    self.max_seq_length = 128
    input_shape = (self.max_seq_length)
    encoder_inputs = {
        'input_type_ids': tf.keras.layers.Input(input_shape, dtype=tf.int32),
        'input_word_ids': tf.keras.layers.Input(input_shape, dtype=tf.int32),
        'input_mask': tf.keras.layers.Input(input_shape, dtype=tf.int32)
    }
    encoder = hub.KerasLayer(tf_hub_source, trainable=True)
    encoder_outputs = encoder(encoder_inputs)
    sequence_output = encoder_outputs['sequence_output']

    self.model = tf.keras.Model(
        inputs=encoder_inputs, outputs=sequence_output, name='bert_model')

  def predict_batch(self, msg_ids, token_batch,
                    mask_batch):
    """Run BERT on one batch of input data."""
    res = {}
    model_input = {
        'input_type_ids':
            tf.zeros([self.batch_size, self.max_seq_length], dtype=tf.int32),
        'input_word_ids':
            tf.constant(token_batch, dtype=tf.int32),
        'input_mask':
            tf.constant(mask_batch, dtype=tf.int32)
    }
    output = self.model(model_input).numpy()
    for i, msg_id in enumerate(msg_ids):
      res[msg_id] = output[i, :, :].tolist()
    return res

  def lookup(self, messages):
    """Look up BERT embeddings for the tokens in messages."""
    result = {}
    token_batch = []
    mask_batch = []
    msg_ids = []
    for msg_id, msg in messages.items():
      token_batch.append(msg + [0] * (self.max_seq_length - len(msg)))
      mask_batch.append([1] * len(msg) + [0] * (self.max_seq_length - len(msg)))
      msg_ids.append(msg_id)

      if len(msg_ids) == self.batch_size:
        res = self.predict_batch(msg_ids, token_batch, mask_batch)
        result.update(res)
        token_batch = []
        mask_batch = []
        msg_ids = []

    if msg_ids and self.batch_size > len(msg_ids):
      to_add = self.batch_size - len(msg_ids)
      msg_ids.extend([-1] * to_add)
      token_batch.extend([0] * to_add)
      mask_batch.extend([0] * to_add)

      res = self.predict_batch(msg_ids, token_batch, mask_batch)
      result.update(res)

    return result
