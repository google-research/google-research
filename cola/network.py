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

"""Network architecture."""
import tensorflow as tf

from cola import constants


class DotProduct(tf.keras.layers.Layer):
  """Normalized dot product."""

  def call(self, anchor, positive):
    anchor = tf.nn.l2_normalize(anchor, axis=-1)
    positive = tf.nn.l2_normalize(positive, axis=-1)
    return tf.linalg.matmul(anchor, positive, transpose_b=True)


class BilinearProduct(tf.keras.layers.Layer):
  """Bilinear product."""

  def __init__(self, dim):
    super().__init__()
    self._dim = dim

  def build(self, _):
    self._w = self.add_weight(
        shape=(self._dim, self._dim),
        initializer="random_normal",
        trainable=True,
        name="bilinear_product_weight",
    )

  def call(self, anchor, positive):
    projection_positive = tf.linalg.matmul(self._w, positive, transpose_b=True)
    return tf.linalg.matmul(anchor, projection_positive)


class ContrastiveModel(tf.keras.Model):
  """Wrapper class for custom contrastive model."""

  def __init__(self, embedding_model, temperature, similarity_layer,
               similarity_type):
    super().__init__()
    self.embedding_model = embedding_model
    self._temperature = temperature
    self._similarity_layer = similarity_layer
    self._similarity_type = similarity_type

  def train_step(self, data):
    anchors, positives = data

    with tf.GradientTape() as tape:
      inputs = tf.concat([anchors, positives], axis=0)
      embeddings = self.embedding_model(inputs, training=True)
      anchor_embeddings, positive_embeddings = tf.split(embeddings, 2, axis=0)

      # logits
      similarities = self._similarity_layer(anchor_embeddings,
                                            positive_embeddings)

      if self._similarity_type == constants.SimilarityMeasure.DOT:
        similarities /= self._temperature
      sparse_labels = tf.range(tf.shape(anchors)[0])

      loss = self.compiled_loss(sparse_labels, similarities)
      loss += sum(self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(sparse_labels, similarities)
    return {m.name: m.result() for m in self.metrics}


def get_efficient_net_encoder(input_shape, pooling):
  """Wrapper function for efficient net B0."""
  efficient_net = tf.keras.applications.EfficientNetB0(
      include_top=False, weights=None, input_shape=input_shape, pooling=pooling)
  # To set the name `encoder` as it is used by supervised module for
  # to trainable value.
  return tf.keras.Model(
      efficient_net.inputs, efficient_net.outputs, name="encoder")


def get_contrastive_network(embedding_dim,
                            temperature,
                            pooling_type="max",
                            similarity_type=constants.SimilarityMeasure.DOT,
                            input_shape=(None, 64, 1)):
  """Creates a model for contrastive learning task."""
  inputs = tf.keras.layers.Input(input_shape)
  encoder = get_efficient_net_encoder(input_shape, pooling_type)
  x = encoder(inputs)
  outputs = tf.keras.layers.Dense(embedding_dim, activation="linear")(x)
  if similarity_type == constants.SimilarityMeasure.BILINEAR:
    outputs = tf.keras.layers.LayerNormalization()(outputs)
    outputs = tf.keras.layers.Activation("tanh")(outputs)
  embedding_model = tf.keras.Model(inputs, outputs)
  if similarity_type == constants.SimilarityMeasure.BILINEAR:
    embedding_dim = embedding_model.output.shape[-1]
    similarity_layer = BilinearProduct(embedding_dim)
  else:
    similarity_layer = DotProduct()
  return ContrastiveModel(embedding_model, temperature, similarity_layer,
                          similarity_type)
