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
"""Models for next-sentence prediction task on ROCStories.
"""

import collections

from absl import logging
import gin
import gin.tf
import tensorflow.compat.v2 as tf

gfile = tf.io.gfile


@gin.configurable
class LinearModel(tf.keras.Model):
  """Multi-layer perceptron with embedding matrix at end."""

  def __init__(
      self,
      num_input_sentences=None,
      embedding_matrix=None,
      embedding_dim=None):
    """Creates a small MLP, then multiplies outputs by embedding matrix.

    Either an embedding matrix or an embedding dimension should be specified.
    If the former, predictions are made by multiplying the NN outputs by this
    embedding matrix. If only an embedding dimension is provided, call()
    outputs an embedding, but no predictions.

    Args:
      num_input_sentences: Integer number of input sentences.
      embedding_matrix: Matrix of size [embedding_dim * num_last_ouputs]
      embedding_dim: Matrix of size [embedding_dim * num_last_ouputs]
    """
    super(LinearModel, self).__init__()
    assert (embedding_matrix is None) ^ (embedding_dim is None)

    self._loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    self._num_input_sentences = num_input_sentences
    self.embedding_matrix = embedding_matrix

    if self.embedding_matrix is not None:
      self._embedding_dim = self.embedding_matrix.shape[1]
    else:
      self._embedding_dim = embedding_dim

    x_input, x_output = self._build_network()
    super(LinearModel, self).__init__(
        inputs=x_input, outputs=x_output, name='model')

  @gin.configurable('LinearModel.hparams')
  def _build_network(self,
                     relu_layers=(2048, 1024),
                     dropout_amount=0.5,
                     normalize_embeddings=False,
                     final_dropout=True,
                     small_context_loss_weight=0.0,
                     max_num_distractors=-1):
    """Builds the network.

    Args:
      relu_layers: Dimensions of linear+RELU layers to add to MLP. These do not
        need to include the final projection down to embedding_dim.
      dropout_amount: If training, how much dropout to use in each layer.
      normalize_embeddings: If True, normalize sentence embeddings (both
        input and predicted) to mean 0, unit variance.
      final_dropout: If True, adds dropout to the final embedding layer.
      small_context_loss_weight: If >0, in addition to the loss with many
        distractors, add another loss where the only distractors are the
        sentences of the context.
      max_num_distractors: If non-negative, randomly pick a window of this many
        distractors around the true 5th sentence.

    Returns:
        A Keras model.
    """
    self.small_context_loss_weight = small_context_loss_weight
    self._max_num_distractors = max_num_distractors

    # x starts off with dimension [batch_size x num_sentences x emb_size].
    # Convert it to [batch_size x (num_sentences*emb_size)].
    x_input = tf.keras.Input(
        shape=[self._num_input_sentences, self._embedding_dim])
    flattened_shape = [-1, self._num_input_sentences * self._embedding_dim]
    x = tf.reshape(x_input, flattened_shape)

    mlp = tf.keras.Sequential()
    if normalize_embeddings:
      mlp.add(tf.keras.layers.LayerNormalization(axis=1))
    for layer_output_dim in relu_layers:
      mlp.add(
          tf.keras.layers.Dense(layer_output_dim, activation='relu'))
      mlp.add(tf.keras.layers.Dropout(dropout_amount))

    # Final layer bring us back to embedding dimension.
    mlp.add(tf.keras.layers.Dense(self._embedding_dim, activation='linear'))
    if final_dropout:
      mlp.add(tf.keras.layers.Dropout(dropout_amount))
    if normalize_embeddings:
      mlp.add(tf.keras.layers.LayerNormalization(axis=1))
    return x_input, mlp(x)

  def call(self, x, training=True):
    embedding = super(LinearModel, self).call(x, training)
    if self.embedding_matrix is not None:
      scores = tf.matmul(
          embedding, self.embedding_matrix, transpose_b=True)
      return scores, embedding
    else:
      return None, embedding

  def compute_loss(self, labels, scores):
    if (self._max_num_distractors != -1 and
        self._max_num_distractors <= scores.shape[1]):
      # Truncates the number of distractors and redefines labels and scores.

      # TODO(dei): Add gin config arg for choosing random num distractor.s
      # max_num_dist = tf.random.uniform(
      #     [], 1, self.embedding_matrix.shape[0], dtype=tf.int32)
      max_num_dist = self._max_num_distractors

      def slice_to_max_num_distractors_fn(inputs):
        """Reduces the number of distractors to the max number."""
        label_for_ex, scores_for_ex = inputs

        scores_nocorrect = tf.concat(
            [scores_for_ex[0:label_for_ex],
             scores_for_ex[(label_for_ex+1):]],
            axis=0)
        random_start_index = tf.random.uniform(
            shape=[],
            minval=0,
            maxval=scores_for_ex.shape[0]-max_num_dist,
            dtype=tf.int32)

        new_scores = scores_nocorrect[
            random_start_index:random_start_index+max_num_dist]

        # Put the groundtruth embedding in position 0 to make labels easy.
        new_scores = tf.concat(
            [tf.expand_dims(scores_for_ex[label_for_ex], 0), new_scores],
            axis=0)

        return new_scores

      # Truncates the number of distractors being scores to the max number.
      scores = tf.map_fn(slice_to_max_num_distractors_fn,
                         [labels, scores], dtype=tf.float32)

      logging.warning('HERE: scores=%s, labels%s',
                      str(scores.shape), str(labels.shape))
      # Since we moved the correct embedding to position 0.
      labels = tf.zeros_like(labels)

    main_loss = self._loss_object(labels, scores)
    return main_loss

  def create_metrics(self):
    """Outputs a dictionary containing all the metrics we want to log."""

    metrics = [
        tf.keras.metrics.Mean(name='train_loss'),
        tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc'),
        tf.keras.metrics.Accuracy(name='valid_nolabel_acc'),
        tf.keras.metrics.Accuracy(name='train_subset_acc'),
        tf.keras.metrics.Accuracy(name='valid_spring2016_acc'),
        tf.keras.metrics.Accuracy(name='valid_winter2018_acc')]

    if self.small_context_loss_weight > 0.0:
      metrics.append(tf.keras.metrics.Mean(name='main_loss'))
      metrics.append(tf.keras.metrics.Mean(name='small_context_loss'))

    metrics = collections.OrderedDict((m.name, m) for m in metrics)
    return metrics


@gin.configurable
class ResidualModel(LinearModel):
  """Residual multi-layer perceptron with embedding matrix at end."""

  @gin.configurable('ResidualModel.hparams')
  def _build_network(self,
                     residual_layer_size=1024,
                     num_residual_layers=2,
                     dropout_amount=0.5,
                     small_context_loss_weight=0.0,
                     max_num_distractors=-1):
    """Builds an MLP with residual connections.

    Args:
      residual_layer_size: Dimension for linear layer to add to MLP.
      num_residual_layers: Number of residual layer.
      dropout_amount: If training, how much dropout to use in each layer.
      small_context_loss_weight: If >0, in addition to the loss with many
        distractors, add another loss where the only distractors are the
        sentences of the context.
      max_num_distractors: The maximum number of distractors provided at each
        train step.

    Returns:
      The input and output tensors for the network, with the input being a
      placeholder variable.
    """
    self.small_context_loss_weight = small_context_loss_weight
    self._max_num_distractors = max_num_distractors

    # x starts off with dimension [batch_size x num_sentences x emb_size].
    # Convert it to [batch_size x (num_sentences*emb_size)].
    x_input = tf.keras.Input(
        shape=[self._num_input_sentences, self._embedding_dim])
    flattened_shape = [-1, self._num_input_sentences * self._embedding_dim]
    x = tf.reshape(x_input, flattened_shape)

    def block(start_x, embedding_size):
      x = tf.keras.layers.Dense(embedding_size, activation='relu')(start_x)
      x = tf.keras.layers.Dropout(dropout_amount)(x)
      x = tf.keras.layers.Dense(embedding_size, activation='relu')(x)
      return x + start_x

    x = tf.keras.layers.LayerNormalization(axis=1)(x)

    # First bring dimension down to desired.
    x = tf.keras.layers.Dense(residual_layer_size)(x)

    # Add specified number of residual layers.
    for _ in range(num_residual_layers):
      x = block(x, residual_layer_size)

    # Go back up to desired dimension.
    x = tf.keras.layers.Dense(self._embedding_dim, activation='linear')(x)
    x = tf.keras.layers.LayerNormalization(axis=1)(x)
    return x_input, x


@gin.configurable(allowlist=['network_class'])
def build_model(num_input_sentences,
                embedding_matrix=None,
                embedding_dim=None,
                network_class=None):
  """Creates the model object and returns it."""
  if network_class is None:
    # Default to the fully connected model.
    model = LinearModel(num_input_sentences, embedding_matrix, embedding_dim)
  else:
    model = network_class(num_input_sentences, embedding_matrix, embedding_dim)
  return model
