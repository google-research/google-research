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

# Lint as: python3
"""Factory to build language models."""

import logging
from typing import Any, Dict, Optional, Text

import tensorflow as tf

from vatt.configs import factory as configs_factory
from vatt.configs import text as text_config
from vatt.modeling.backbones.text import bert
from vatt.modeling.backbones.text import linear
from vatt.modeling.backbones.text import t5


LANGUAGE_MODEL_HEADS = {
    "linear": linear.LinearLM,
    "t5": t5.T5Encoder,
    "bert": bert.BertEncoder,
}


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class LanguageModel(tf.keras.layers.Layer):
  """Constructs a language model given the configs."""

  def __init__(self,
               base_lm_head,
               params):
    """Main language model.

    Args:
      base_lm_head: the language model, either linear or Transformer-based.
      params: Hyperparameters of the model.

    """
    super(LanguageModel, self).__init__(name="text_module")

    self._vocab_size = params.vocab_size
    self._d_embedding = params.d_embedding
    self._use_agg_token = params.use_agg_token
    self._trainable_embeddings = params.trainable_embeddings
    self._is_transformer = params.is_transformer
    self.embedding = tf.keras.layers.Embedding(
        self._vocab_size,
        self._d_embedding,
        trainable=self._trainable_embeddings,
        )

    # if specified, there would be a dense projection before feeding to the LM
    if params.d_pre_proj is not None:
      self.pre_proj = tf.keras.layers.Dense(
          params.d_pre_proj,
          use_bias=False,
          activation=params.activation,
          name="pre_proj"
          )
    else:
      self.pre_proj = tf.identity

    self.base_lm = base_lm_head(**params.as_dict())

    # if specified, there would be a dense projection after the LM outputs
    if params.d_post_proj is not None:
      self.post_proj = tf.keras.layers.Dense(
          params.d_post_proj,
          use_bias=False,
          activation=params.activation,
          name="post_proj"
          )
    else:
      self.post_proj = tf.identity

  def build(self, input_shape):
    if self._use_agg_token:
      self.agg_token_embd = self.add_weight(
          name="agg_embedding",
          shape=(self._d_embedding,),
          initializer=tf.keras.initializers.get("glorot_normal"),
          trainable=True,
          dtype=tf.float32,
          )

  def _append_special_token(self, word_embeddings, attention_mask):
    batch_size = get_shape(word_embeddings)[0]
    agg_embeddings = tf.tile(self.agg_token_embd[None, None, :],
                             [batch_size, 1, 1])
    word_embeddings = tf.concat([agg_embeddings, word_embeddings],
                                axis=1)
    attention_mask = tf.concat([tf.ones((batch_size, 1),
                                        dtype=attention_mask.dtype),
                                attention_mask],
                               axis=1)
    return word_embeddings, attention_mask

  def call(self,
           inputs_ids,
           training=True,
           attention_mask=None):
    """Connects graph to sentence representation."""

    # word_id to embeddings
    word_embeddings = self.embedding(inputs_ids)

    # generate padding mask
    if attention_mask is None:
      attention_mask = tf.where(inputs_ids == 0, 0, 1)

    # append special aggregation token if T5
    # the BERT implementation already supports AGG append
    if self._use_agg_token:
      word_embeddings, attention_mask = self._append_special_token(
          word_embeddings,
          attention_mask,
          )

    word_embeddings = self.pre_proj(word_embeddings)
    base_outputs = self.base_lm(
        inputs=None,
        inputs_embeddings=word_embeddings,
        attention_mask=attention_mask,
        training=training,
        )

    last_hidden_states = self.post_proj(base_outputs["hidden_states"][-1])
    if self._use_agg_token or self._is_transformer:
      word_embeddings = last_hidden_states[:, 1:, :]
      sentence_embeddings = last_hidden_states[:, 0, :]
    else:
      word_embeddings = last_hidden_states
      sentence_embeddings = tf.reduce_max(last_hidden_states, axis=1)

    outputs = {
        "word_embeddings": word_embeddings,
        "sentence_embeddings": sentence_embeddings,
    }

    return outputs


def build_model(
    params = None,
    override_params = None,
    backbone = None,
    ):
  """Build language model by name."""

  if params is None:
    assert backbone is not None, (
        "either params or backbone should be specified")
    params = configs_factory.build_model_configs(backbone)

  if override_params is not None:
    params.override(override_params)

  model_name = params.name.lower()
  if model_name.startswith("linear"):
    base_lm_head = LANGUAGE_MODEL_HEADS["linear"]
  elif model_name.startswith("t5"):
    base_lm_head = LANGUAGE_MODEL_HEADS["t5"]
  elif model_name.startswith("bert"):
    base_lm_head = LANGUAGE_MODEL_HEADS["bert"]
  else:
    raise ValueError("Unknown model name {!r}".format(params.name))

  model = LanguageModel(
      base_lm_head=base_lm_head,
      params=params,
      )

  logging.info("Text model %s created successfully.", params.name)

  return model
