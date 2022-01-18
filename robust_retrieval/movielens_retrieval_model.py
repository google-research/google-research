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

"""Distributionally-robust retrieval model for MovieLens dataset."""

import tensorflow as tf
import tensorflow_recommenders as tfrs
from robust_retrieval.tasks import RobustRetrieval


class EmbeddingModel(tf.keras.Model):
  """Defines the user/item embedding model."""

  def __init__(self,
               hidden_dims,
               feature_columns=None,
               final_layer_l2norm=False,
               kernel_l2=1e-4,
               bias_l2=None,
               activation_fn="relu",
               name="embed_model"):
    """Embedding model for user/item input features.

    Args:
      hidden_dims: A list of integers where the i-th entry represents the number
        of units in the i-th layer.
      feature_columns: None or A list of feature columns.
      final_layer_l2norm: A bool, if True will L2-normalize the last layer
        output.
      kernel_l2: L2-regularization for kernel.
      bias_l2: L2-regularization for bias.
      activation_fn: A string, activation function, default to use "relu".
      name: [Optional] A string, name of the model.
    """
    super().__init__()
    self.embed_model = tf.keras.Sequential(name=name)
    if feature_columns:
      self.embed_model.add(tf.keras.layers.DenseFeatures(feature_columns))

    self._kernel_l2 = tf.keras.regularizers.l2(kernel_l2)
    self._bias_l2 = bias_l2

    for layer_dim in hidden_dims[:-1]:
      self.embed_model.add(
          tf.keras.layers.Dense(
              layer_dim,
              activation=activation_fn,
              kernel_regularizer=self._kernel_l2,
              bias_regularizer=self._bias_l2))

    # No activation for the last layer.
    self.embed_model.add(
        tf.keras.layers.Dense(
            hidden_dims[-1],
            kernel_regularizer=self._kernel_l2,
            bias_regularizer=self._bias_l2))

    # Add L2-normalization on the output of the final layer.
    if final_layer_l2norm:
      self.embed_model.add(
          tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))

  def call(self, inputs):
    return self.embed_model(inputs)


class MovieLensRetrievalModel(tfrs.models.Model):
  """A two-tower model with ERM or Robust Learning for ML-1M."""

  def __init__(self, user_feature_columns, item_feature_columns, candidates,
               model_configs):
    """Initialize the two-tower model.

    Args:
      user_feature_columns: A list of user-related feature columns to be passed
        to the user embedding model.
      item_feature_columns: A list of item-related feature columns to be passed
        to the item embedding model.
      candidates: tf.data.Dataset. A set of candidate items with the same format
        as item_feature_columns. It will be used by tfrs.tasks to calculate
        retrieval performance.
      model_configs: An instance of BaseModelConfig, defines Model
        hyper-parameters and configurations. Refer to
        research/sir/rep_learning/robust_retrieval/configs/model_params.py for
        details.
    """
    super().__init__()
    self._model_configs = model_configs
    self.global_step = tf.compat.v1.train.create_global_step()
    self.query_model = EmbeddingModel(
        feature_columns=user_feature_columns,
        hidden_dims=self._model_configs.hidden_dims,
        final_layer_l2norm=self._model_configs.final_layer_l2norm,
        activation_fn=self._model_configs.activation_fn,
        name="user_model")
    self.candidate_model = EmbeddingModel(
        feature_columns=item_feature_columns,
        hidden_dims=self._model_configs.hidden_dims,
        final_layer_l2norm=self._model_configs.final_layer_l2norm,
        activation_fn=self._model_configs.activation_fn,
        name="item_model")
    mapped_candidates = candidates.map(self.candidate_model)
    # Initialize retrieval task type.
    if self._model_configs.task_type == "erm":
      self.task = tfrs.tasks.Retrieval(
          metrics=tfrs.metrics.FactorizedTopK(candidates=mapped_candidates),
          temperature=self._model_configs.softmax_temperature,
      )
    elif self._model_configs.task_type == "robust":
      # Use robust retrieval task for computing robust loss, e.g. DRO.
      self.task = RobustRetrieval(
          group_labels=self._model_configs.group_labels,
          group_loss_init=self._model_configs.group_loss_init,
          group_metric_init=self._model_configs.group_metric_init,
          group_weight_init=self._model_configs.group_weight_init,
          group_reweight_strategy=self._model_configs.group_reweight_strategy,
          dro_temperature=self._model_configs.dro_temperature,
          streaming_group_loss=self._model_configs.streaming_group_loss,
          streaming_group_loss_lr=self._model_configs.streaming_group_loss_lr,
          streaming_group_metric_lr=self._model_configs
          .streaming_group_metric_lr,
          metric_update_freq=self._model_configs.metric_update_freq,
          metrics=tfrs.metrics.FactorizedTopK(candidates=mapped_candidates),
          candidates=mapped_candidates,
          temperature=self._model_configs.softmax_temperature,
      )

  def compute_loss(self, inputs, training=False):
    """Defines the loss function.

    Args:
      inputs: Dict[Text, tf.Tensor], a data structure of tensors: raw inputs to
        the model. These will usually contain labels and weights as well as
        features.
      training: Whether the model is in training mode.

    Returns:
      Loss tensor.
    """
    self.global_step.assign_add(1)

    # Shape of user_embeddings/item_embeddings: [batch_size, embedding_dim].
    user_embeddings = self.query_model(inputs)
    item_embeddings = self.candidate_model(inputs)

    if self._model_configs.task_type == "erm":
      # Perform ERM training.
      return self.task(
          user_embeddings, item_embeddings, compute_metrics=not training)

    else:
      # Perform DRO training.
      if self._model_configs.group not in inputs:
        raise ValueError("f{self._model_configs.group} not found in inputs.")

      # Shape of group_identity: [batch_size].
      group_identity = inputs[self._model_configs.group]

      if group_identity is None:
        raise ValueError("group_identity is None.")

      if isinstance(group_identity, tf.SparseTensor):
        group_identity = group_identity.values
      group_identity = tf.squeeze(group_identity)

      if group_identity.shape != tf.TensorShape([user_embeddings.shape[0]]):
        raise ValueError("Expected a tensor of shape [batch_size].")

      return self.task(
          user_embeddings,
          item_embeddings,
          group_identity=group_identity,
          step_count=self.global_step,
          compute_metrics=not training)
