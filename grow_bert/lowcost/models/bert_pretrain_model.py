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

"""Trainer network for BERT-style models."""

from typing import List, Optional

import tensorflow as tf

from grow_bert.lowcost.layers import masked_lm_layer


class BertPretrainModel(tf.keras.Model):
  """BERT pretraining model.

  Modified from BertPretrainerV2 in
  third_party/tensorflow_models/official/nlp/modeling/models/bert_pretrainer.py.
  Major difference lies in the masked_lm_layer:
  here we separate masked and unmasked tokens in order to apply unmasked-only
  pooling to reduce input sequence length. Hence, instead of applying
  gather_index on the encoder output to fetch masked token representation,
  we just use slicing to get masked tokens from encoder output as "target
  embedding" to pass to the masked_lm_layer for classification.
  """

  def __init__(
      self,
      encoder_network,
      mlm_initializer='glorot_uniform',
      mlm_activation=None,
      classification_heads = None,
      name = 'bert',
      **kwargs):
    """Initialize.

    Arguments:
      encoder_network: A transformer network. This network should output a
        sequence output and a classification output.
      mlm_initializer: The initializer (if any) to use in the masked LM. Default
        to a Glorot uniform initializer.
      mlm_activation: mlm_activation,
      classification_heads: A list of optional head layers to transform on
        encoder sequence outputs.
      name: The name of the model.
      **kwargs: **kwargs.
    """
    super(BertPretrainModel, self).__init__(name=name)
    self._config = {
        'encoder_network': encoder_network,
        'mlm_activation': mlm_activation,
        'mlm_initializer': mlm_initializer,
        'classification_heads': classification_heads,
        'name': name,
    }

    self.mlm_activation = mlm_activation
    self.mlm_initializer = mlm_initializer
    self.encoder_network = encoder_network
    self.classification_heads = classification_heads or []
    if len(set([cls.name for cls in self.classification_heads])) != len(
        self.classification_heads):
      raise ValueError('Classification heads should have unique names.')

    hidden_size = encoder_network.get_config()['hidden_size']
    self.masked_lm = masked_lm_layer.MaskedLMLayer(
        embedding_table=self.encoder_network.get_embedding_table(),
        activation=self.mlm_activation,
        initializer=self.mlm_initializer,
        name='cls/predictions')

  @property
  def checkpoint_items(self):
    """Returns a dictionary of items to be additionally checkpointed."""
    items = dict(encoder=self.encoder_network, masked_lm=self.masked_lm)
    for head in self.classification_heads:
      for key, item in head.checkpoint_items.items():
        items['.'.join([head.name, key])] = item
    return items

  def get_config(self):
    return self._config

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)

  def call(self, inputs):
    sequence_output, _, target_outputs = self.encoder_network(inputs)

    outputs = dict()
    outputs['mlm_logits'] = self.masked_lm(target_outputs)
    for cls_head in self.classification_heads:
      outputs[cls_head.name] = cls_head(sequence_output)
    return outputs
