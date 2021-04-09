# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Transformer-based text encoder network."""

from absl import logging
import tensorflow as tf

from grow_bert.lowcost.layers import position_embedding
from grow_bert.lowcost.layers import resolution_layer
from grow_bert.lowcost.layers import transformer_layer
from official.modeling import activations
from official.modeling import tf_utils
from official.nlp.modeling import layers


class TransformerEncoder(tf.keras.Model):
  """Bi-directional Transformer-based encoder network.

  This network implements a bi-directional Transformer-based encoder as
  described in "BERT: Pre-training of Deep Bidirectional Transformers for
  Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
  embedding lookups and transformer layers, but not the masked language model
  or classification task networks.

  The default values for this object are taken from the BERT-Base implementation
  in "BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding".
  """

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_layers=12,
               num_attention_heads=12,
               sequence_length=512,
               max_sequence_length=None,
               type_vocab_size=16,
               intermediate_size=3072,
               activation=activations.gelu,
               dropout_rate=0.1,
               attention_dropout_rate=0.1,
               initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
               return_all_encoder_outputs=False,
               output_range=None,
               embedding_width=None,
               net2net_ratio=None,
               net2net_layers=None,
               lightatt_layers=None,
               input_pool_name=None,
               input_pool_size=None,
               **kwargs):
    """Bi-directional Transformer-based encoder network.

    This network implements a bi-directional Transformer-based encoder as
    described in "BERT: Pre-training of Deep Bidirectional Transformers for
    Language Understanding" (https://arxiv.org/abs/1810.04805). It includes the
    embedding lookups and transformer layers, but not the masked language model
    or classification task networks.

    The default values for this object are taken from the BERT-Base
    implementation
    in "BERT: Pre-training of Deep Bidirectional Transformers for Language
    Understanding".

    Arguments:
      vocab_size: The size of the token vocabulary.
      hidden_size: The size of the transformer hidden layers.
      num_layers: The number of transformer layers.
      num_attention_heads: The number of attention heads for each transformer.
        The hidden size must be divisible by the number of attention heads.
      sequence_length: The sequence length that this encoder expects. If None,
        the sequence length is dynamic; if an integer, the encoder will require
        sequences padded to this length.
      max_sequence_length: The maximum sequence length that this encoder can
        consume. If None, max_sequence_length uses the value from sequence
        length. This determines the variable shape for positional embeddings.
      type_vocab_size: The number of types that the 'type_ids' input can take.
      intermediate_size: The intermediate size for the transformer layers.
      activation: The activation to use for the transformer layers.
      dropout_rate: The dropout rate to use for the transformer layers.
      attention_dropout_rate: The dropout rate to use for the attention layers
        within the transformer layers.
      initializer: The initialzer to use for all weights in this encoder.
      return_all_encoder_outputs: Whether to output sequence embedding outputs
        of all encoder transformer layers.
      output_range: the sequence output range, [0, output_range), by slicing the
        target sequence of the last transformer layer. `None` means the entire
        target sequence will attend to the source sequence, which yeilds the
        full output.
      embedding_width: The width of the word embeddings. If the embedding width
        is not equal to hidden size, embedding parameters will be factorized
        into two matrices in the shape of ['vocab_size', 'embedding_width'] and
        ['embedding_width', 'hidden_size'] ('embedding_width' is usually much
        smaller than 'hidden_size').
       net2net_ratio: net2net ratio for the small fully connected matrices.
       net2net_layers: number of layers with net2net treatment.
       lightatt_layers: number of layers with light attention,
       input_pool_name: input_pool_name,
       input_pool_size: input_pool_size,
       **kwargs: **kwargs
    """
    super(TransformerEncoder, self).__init__()

    activation = tf.keras.activations.get(activation)
    initializer = tf.keras.initializers.get(initializer)

    if not max_sequence_length:
      max_sequence_length = sequence_length
    self.net2net_ratio = net2net_ratio
    self.net2net_layers = net2net_layers
    self.lightatt_layers = lightatt_layers
    self.input_pool_name = input_pool_name
    self.input_pool_size = input_pool_size

    if embedding_width is None:
      embedding_width = hidden_size
    self._config_dict = {
        'vocab_size': vocab_size,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'num_attention_heads': num_attention_heads,
        'sequence_length': sequence_length,
        'max_sequence_length': max_sequence_length,
        'type_vocab_size': type_vocab_size,
        'intermediate_size': intermediate_size,
        'activation': tf.keras.activations.serialize(activation),
        'dropout_rate': dropout_rate,
        'attention_dropout_rate': attention_dropout_rate,
        'initializer': tf.keras.initializers.serialize(initializer),
        'return_all_encoder_outputs': return_all_encoder_outputs,
        'output_range': output_range,
        'embedding_width': embedding_width,
        'net2net_ratio': net2net_ratio,
        'net2net_layers': net2net_layers,
        'lightatt_layers': lightatt_layers,
        'input_pool_name': input_pool_name,
        'input_pool_size': input_pool_size,
    }

    self._embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        name='word_embeddings')
    # Always uses dynamic slicing for simplicity.
    self._position_embedding_layer = position_embedding.PositionEmbedding(
        embed_dim=hidden_size,
        initializer=initializer,
        use_dynamic_slicing=True,
        max_sequence_length=max_sequence_length,
        name='position_embedding')
    self._type_embedding_layer = layers.OnDeviceEmbedding(
        vocab_size=type_vocab_size,
        embedding_width=embedding_width,
        initializer=initializer,
        use_one_hot=True,
        name='type_embeddings')

    self._embedding_norm_layer = tf.keras.layers.LayerNormalization(
        name='embeddings/layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    self._dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)

    self._embedding_projection_layer = tf.keras.layers.experimental.EinsumDense(
        '...x,xy->...y',
        output_shape=hidden_size,
        bias_axes='y',
        kernel_initializer=initializer,
        name='embedding_projection')

    self._self_attention_mask_layer = layers.SelfAttentionMask()

    self._transformer_layers = []
    print('!!!! building transformer layers !!!')
    logging.info('!!!! building transformer layers !!!')
    for i in range(num_layers):
      if i == num_layers - 1 and output_range is not None:
        transformer_output_range = output_range
      else:
        transformer_output_range = None

      group_size = num_layers // net2net_layers if net2net_layers is not None else None
      layer_net2net_ratio = None if (net2net_layers is None or
                                     i % group_size != 0) else net2net_ratio

      group_size = num_layers // lightatt_layers if lightatt_layers is not None else None
      use_lightatt = False if (lightatt_layers is None or i % group_size !=
                               (group_size - 1)) else True

      logging.info(i)
      logging.info(layer_net2net_ratio)
      logging.info(use_lightatt)
      layer = transformer_layer.TransformerLayer(
          num_attention_heads=num_attention_heads,
          intermediate_size=intermediate_size,
          intermediate_activation=activation,
          dropout_rate=dropout_rate,
          attention_dropout_rate=attention_dropout_rate,
          output_range=transformer_output_range,
          kernel_initializer=initializer,
          name='transformer/layer_%d' % i,
          use_lightatt=use_lightatt,
          net2net_ratio=layer_net2net_ratio)
      self._transformer_layers.append(layer)
    print('!!!! finish building transformer layers !!!')
    logging.info('!!!! finish building transformer layers !!!')

    self._squeeze_layer = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x[:, 0:1, :], axis=1))

    self._pooler_layer = tf.keras.layers.Dense(
        units=hidden_size,
        activation='tanh',
        kernel_initializer=initializer,
        name='pooler_transform')

    nocls = input_pool_name != 'concat'
    input_pool_size = 1 if input_pool_name is None else input_pool_size
    self._mask_resolution_layer = resolution_layer.MaskPoolLayer(
        input_pool_size, nocls=nocls, name='mask_resolution')
    self._embed_resolution_layer = resolution_layer.EmbedPoolLayer(
        hidden_size, input_pool_size, input_pool_name, name='embed_resolution')

  def call(self, inputs):
    mask = inputs['input_mask']
    word_ids = inputs['input_word_ids']
    type_ids = inputs['input_type_ids']
    input_positions = inputs['input_positions']
    batch_size, input_seqlen = tf_utils.get_shape_list(word_ids)

    word_embeddings = self._embedding_layer(word_ids)
    type_embeddings = self._type_embedding_layer(type_ids)
    position_embeddings = self._position_embedding_layer(input_positions)
    embeddings = word_embeddings + type_embeddings + position_embeddings
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = self._dropout_layer(embeddings)

    target_positions = inputs['masked_lm_positions']
    target_type_ids = inputs['masked_segment_ids']
    target_input_ids = inputs['masked_input_ids']
    batch_size, target_len = target_positions.shape
    target_mask = tf.cast(inputs['masked_lm_weights'], dtype=mask.dtype)
    target_position_embeddings = self._position_embedding_layer(
        target_positions)
    target_input_embeddings = self._embedding_layer(target_input_ids)
    target_type_embeddings = self._type_embedding_layer(target_type_ids)
    target_embeddings = target_input_embeddings + target_position_embeddings + target_type_embeddings
    target_embeddings = self._embedding_norm_layer(target_embeddings)
    target_embeddings = self._dropout_layer(target_embeddings)

    # We project the 'embedding' output to 'hidden_size' if it is not already
    # 'hidden_size'.
    if self._config_dict['embedding_width'] != self._config_dict['hidden_size']:
      embeddings = self._embedding_projection_layer(embeddings)
      target_embeddings = self._embedding_projection_layer(target_embeddings)

    encoder_outputs = []
    attention_scores_list = []  # score previous attention score mappings

    merged_mask = tf.concat([mask, target_mask], axis=1)
    merged_embeddings = tf.concat([embeddings, target_embeddings], axis=1)
    pooled_query = tf.concat(
        [self._embed_resolution_layer(embeddings), target_embeddings], axis=1)
    first_layer_attention_mask = self._self_attention_mask_layer(
        [pooled_query, merged_mask])
    data, attention_scores = self.transformer_layers[0](
        [merged_embeddings, first_layer_attention_mask],
        _target_tensor=pooled_query)
    encoder_outputs.append(data)
    attention_scores_list.append(None)  # layer 1 will apply identical attention

    pooled_merged_mask = tf.concat(
        [self._mask_resolution_layer(mask), target_mask], axis=1)
    attention_mask = self._self_attention_mask_layer([data, pooled_merged_mask])
    for i in range(1, self._config_dict['num_layers']):
      current_layer = self.transformer_layers[i]
      shared_attention_scores = attention_scores_list[
          i - 1] if current_layer.use_lightatt else None
      data, attention_scores = current_layer(
          [data, attention_mask],
          shared_attention_scores=shared_attention_scores)
      encoder_outputs.append(data)
      attention_scores_list.append(attention_scores)

    first_token_tensor = self._squeeze_layer(encoder_outputs[-1])
    cls_output = self._pooler_layer(first_token_tensor)
    target_outputs = encoder_outputs[-1][:, -target_len:, :]

    if self._config_dict['return_all_encoder_outputs']:
      outputs = [encoder_outputs, cls_output, target_outputs]
    else:
      outputs = [encoder_outputs[-1], cls_output, target_outputs]

    return outputs

  def get_embedding_table(self):
    return self._embedding_layer.embeddings

  def get_position_embedding(self):
    return self._position_embedding_layer

  def get_config(self):
    return self._config_dict

  @property
  def transformer_layers(self):
    """List of Transformer layers in the encoder."""
    return self._transformer_layers

  @property
  def pooler_layer(self):
    """The pooler dense layer after the transformer layers."""
    return self._pooler_layer

  @classmethod
  def from_config(cls, config, custom_objects=None):
    return cls(**config)
