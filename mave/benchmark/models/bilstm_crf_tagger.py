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

"""Library for BiLSTM-CRF sequence tagger."""
import collections
from typing import Any, Optional, Union

import ml_collections
import tensorflow as tf
import tensorflow_hub as hub

from mave.benchmark.models import crf_layer
from mave.benchmark.models import crf_utils
from official.nlp import modeling as tfnlp

_Initializer = Union[str, Any]


class BiLSTMCRFSequenceTagger(tf.keras.Model):
  """BiLSTM-CRF Sequence tagger model."""

  def __init__(self,
               seq_length,
               vocab_size,
               word_embeddding_size,
               lstm_units,
               recurrent_dropout,
               use_attention_layer,
               use_attention_scale,
               attention_dropout,
               num_tags = 2,
               word_embeddings_initializer = None,
               **kwargs):
    word_embeddings_initializer = (
        word_embeddings_initializer or
        tf.keras.initializers.TruncatedNormal(stddev=0.02))

    input_word_ids = tf.keras.layers.Input(
        shape=(seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(
        shape=(seq_length,), dtype=tf.int32, name='input_mask')
    boolean_mask = tf.cast(input_mask, tf.bool)
    word_embeddings = tfnlp.layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=word_embeddding_size,
        initializer=word_embeddings_initializer,
        name='word_embeddings')(
            input_word_ids)

    bilstm_outputs = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            units=lstm_units,
            recurrent_dropout=recurrent_dropout,
            return_sequences=True),
        name='BiLSTM')(
            word_embeddings, mask=boolean_mask)
    if use_attention_layer:
      attention_outputs = tf.keras.layers.Attention(
          use_scale=use_attention_scale,
          dropout=attention_dropout,
          name='Attention')(
              inputs=[bilstm_outputs, bilstm_outputs],
              mask=[boolean_mask, boolean_mask])
    else:
      attention_outputs = bilstm_outputs

    crf = crf_layer.CRF(units=num_tags, name='CRF')
    outputs = crf(attention_outputs, mask=input_mask)

    inputs = {
        'input_word_ids': input_word_ids,
        'input_mask': input_mask,
    }

    bisltm_crf_encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='core_model')

    super(BiLSTMCRFSequenceTagger, self).__init__(
        inputs=inputs, outputs=outputs[0], **kwargs)
    self._bisltm_crf_encoder = bisltm_crf_encoder
    config_dict = {
        'seq_length': seq_length,
        'vocab_size': vocab_size,
        'word_embeddding_size': word_embeddding_size,
        'lstm_units': lstm_units,
        'recurrent_dropout': recurrent_dropout,
        'use_attention_layer': use_attention_layer,
        'use_attention_scale': use_attention_scale,
        'attention_dropout': attention_dropout,
        'num_tags': num_tags,
        'word_embeddings_initializer': word_embeddings_initializer,
    }
    # We are storing the config dict as a namedtuple here to ensure checkpoint
    # compatibility with an earlier version of this model which did not track
    # the config dict attribute. TF does not track immutable attrs which
    # do not contain Trackables, so by creating a config namedtuple instead of
    # a dict we avoid tracking it.
    config_cls = collections.namedtuple('Config', config_dict.keys())
    self._config = config_cls(**config_dict)

  def train_step(self, data):
    """The logic for one training step."""
    x, y, sample_weight = [*data, None][:3]
    # Run forward pass.
    with tf.GradientTape() as tape:
      y_pred, potentials, sequence_length, chain_kernel = (
          self._bisltm_crf_encoder(x, training=True))
      log_likelihood, _ = crf_utils.crf_log_likelihood(potentials, y,
                                                       sequence_length,
                                                       chain_kernel)
      if sample_weight is not None:
        log_likelihood *= sample_weight
      loss = tf.reduce_sum(-log_likelihood) + sum(self.losses)
    # Run backwards pass.
    self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
    self.compiled_metrics.update_state(y, y_pred)
    # Collect metrics to return
    return_metrics = {'loss': loss}
    for metric in self.metrics:
      result = metric.result()
      if isinstance(result, dict):
        return_metrics.update(result)
      else:
        return_metrics[metric.name] = result
    return return_metrics

  def get_config(self):
    return dict(self._config._asdict())

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)


def build_model(config):
  """Builds the BiLSTM-CRF sequence tagger model."""
  if config.bilstm_crf.init_word_embedding_from_bert:
    #  Initializes word embedding layer from BERT word embedding layer.
    bert_model = hub.KerasLayer(
        config.bert.bert_hub_module_url, trainable=False)
    for varlable in bert_model.variables:
      if varlable.name == 'word_embeddings/embeddings:0':
        bert_word_embedding = tf.constant(varlable.value())
    else:
      bert_word_embedding = None

    if bert_word_embedding is not None:

      def word_embeddings_initializer(shape, dtype=None, **kwargs):
        del shape, kwargs  # Unused.
        return tf.cast(bert_word_embedding, dtype=dtype)
    else:
      word_embeddings_initializer = None
    del bert_model
  else:
    word_embeddings_initializer = None

  return BiLSTMCRFSequenceTagger(
      seq_length=config.model.seq_length,
      vocab_size=config.bilstm_crf.vocab_size,
      word_embeddding_size=config.bilstm_crf.word_embeddding_size,
      lstm_units=config.bilstm_crf.lstm_units,
      recurrent_dropout=config.bilstm_crf.recurrent_dropout,
      use_attention_layer=config.bilstm_crf.use_attention_layer,
      use_attention_scale=config.bilstm_crf.use_attention_scale,
      attention_dropout=config.bilstm_crf.attention_dropout,
      num_tags=config.bilstm_crf.num_tags,
      word_embeddings_initializer=word_embeddings_initializer)
