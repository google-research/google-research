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

"""Fast clipping model generators."""
import tensorflow as tf
import tensorflow_models as tfm
from tensorflow_privacy.privacy.fast_gradient_clipping import layer_registry
from tensorflow_privacy.privacy.keras_models import dp_keras_model
from fast_gradient_clipping.src import bert_encoder_utils
from fast_gradient_clipping.src import custom_registry_functions


def reshape_and_sum(tensor):
  """Reshapes and sums along non-batch dims to get the shape [None, 1]."""
  reshaped_2d = tf.reshape(tensor, [tf.shape(tensor)[0], -1])
  return tf.reduce_sum(reshaped_2d, axis=-1, keepdims=True)


def make_fully_connected_model(n, m, p, q):
  """Creates a fully connected Keras model.

  Args:
    n: channel dimension 1
    m: channel dimension 2
    p: input dimension
    q: output dimension

  Returns:
    A one-layer (tf.keras.layers.EinsumDense) fully connected Keras model.
  """
  inputs = tf.keras.Input(shape=(n, m, p))
  esd_layer = tf.keras.layers.EinsumDense(
      equation='bnmp,pq->bnmq',
      output_shape=(n, m, q),
      activation='relu',
      bias_axes='m',
  )
  transformed = esd_layer(inputs)
  outputs = reshape_and_sum(transformed)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(loss=tf.keras.losses.MeanSquaredError())
  return model


def make_direct_bias_model(p, q, r, m):
  """Creates a fully connected Keras model with (direct) bias.

  Args:
    p: input dimension
    q: output dimension
    r: channel dimension
    m: bias dimension

  Returns:
    A one-layer (tf.keras.layers.EinsumDense) fully connected Keras model.
  """
  if r % m != 0:
    raise ValueError(f'Bias dimension {m} must divide channel dimension {r}.')
  return make_fully_connected_model(r // m, m, p, q)


def make_indirect_bias_model(p, q, r, m):
  """Creates a fully connected Keras model with (indirect) bias.

  Args:
    p: input dimension
    q: output dimension
    r: channel dimension
    m: bias dimension

  Returns:
    A one-layer (tf.keras.layers.EinsumDense) fully connected Keras model.
  """
  inputs1 = tf.keras.Input(shape=(r, p))
  esd_layer1 = tf.keras.layers.EinsumDense(
      equation='brp,pq->brq',
      output_shape=(r, q),
      activation='relu',
  )
  transformed1 = esd_layer1(inputs1)
  inputs2 = tf.keras.Input(shape=(r, q, m))
  esd_layer2 = tf.keras.layers.EinsumDense(
      equation='brqm,md->brqd',  # d==1
      output_shape=(r, q, 1),
      activation='relu',
  )
  transformed2 = tf.squeeze(esd_layer2(inputs2), axis=-1)
  transformed = transformed1 + transformed2
  outputs = reshape_and_sum(transformed)
  model = tf.keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
  model.compile(loss=tf.keras.losses.MeanSquaredError())
  return model


def make_embedding_model(vocab_size, num_queries, output_dim):
  """Makes a simple embedding model."""
  inputs = tf.keras.Input(shape=(num_queries))
  emb_layer = tf.keras.layers.Embedding(vocab_size, output_dim)
  transformed = emb_layer(inputs)
  outputs = reshape_and_sum(transformed)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  model.compile(loss=tf.keras.losses.MeanSquaredError())
  return model


def make_bert_layer_registry():
  """Simple layer registry for BERT."""
  registry = layer_registry.LayerRegistry()
  registry.insert(
      tf.keras.layers.Dense, custom_registry_functions.dense_layer_computation
  )
  registry.insert(
      tf.keras.layers.EinsumDense,
      custom_registry_functions.einsum_layer_computation,
  )
  registry.insert(
      tf.keras.layers.LayerNormalization,
      custom_registry_functions.layer_normalization_computation,
  )
  registry.insert(
      tfm.nlp.layers.OnDeviceEmbedding,
      custom_registry_functions.nlp_on_device_embedding_computation,
  )
  registry.insert(
      tfm.nlp.layers.PositionEmbedding,
      custom_registry_functions.nlp_position_embedding_computation,
  )
  registry.insert(
      tfm.nlp.layers.MultiHeadAttention,
      custom_registry_functions.multi_head_attention_layer_computation,
  )
  return registry


def make_dp_bert_model(vocab_size, use_fast_clipping):
  """Creates a simple BERT model.

  Args:
    vocab_size: Vocabulary size.
    use_fast_clipping: Whether to use fast gradient clipping.

  Returns:
    A DP Keras BERT model with `l2_norm_clip=10.0` and `noise_multiplier=1.0`.
  """
  bert_model = tfm.nlp.networks.BertEncoder(
      vocab_size=vocab_size,
      num_layers=1,
      hidden_size=128,
      inner_dim=2048,
  )
  # Unwrap the Transformer Encoder Block.
  unwrapped_bert_encoder = bert_encoder_utils.get_unwrapped_bert_encoder(
      bert_model
  )
  pooled_output = unwrapped_bert_encoder.outputs[1]
  final_head = tf.reduce_sum(pooled_output, axis=-1, keepdims=True)
  # Apply a reduce-sum.
  registry = None
  if use_fast_clipping:
    registry = make_bert_layer_registry()
  # Build the model, add an SGD optimizer, and set MSE loss.
  dp_model = dp_keras_model.DPModel(
      l2_norm_clip=10.0,
      noise_multiplier=1.0,
      inputs=unwrapped_bert_encoder.inputs,
      outputs=final_head,
      layer_registry=registry,
      use_xla=False,
  )
  dp_model.compile(
      optimizer='sgd',
      loss=tf.keras.losses.MeanSquaredError(),
      run_eagerly=False,
  )
  return dp_model
