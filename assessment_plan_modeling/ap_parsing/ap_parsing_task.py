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

"""AP parsing TF-NLP tagging task."""

import dataclasses
from typing import Any, Dict, List, Optional, Sequence, Tuple

from absl import logging
import tensorflow as tf
import tensorflow_addons.text as tfa_text

from assessment_plan_modeling.ap_parsing import ap_parsing_dataloader
from assessment_plan_modeling.ap_parsing import constants
from official.core import base_task
from official.core import config_definitions as cfg
from official.core import exp_factory
from official.modeling.hyperparams import base_config

InputsDict = Dict[str, tf.keras.layers.Layer]


@dataclasses.dataclass
class LSTMEncoderConfig(base_config.Config):
  """LSTM encoder config."""
  num_layers: int = 2
  hidden_size: int = 32
  dropout_rate: float = 0.1


@dataclasses.dataclass
class EmbeddingShape(base_config.Config):
  vocab_size: int = 0
  embedding_size: int = 0


@dataclasses.dataclass
class EmbeddingShapeConfig(base_config.Config):
  """The embedding shape config."""
  token_ids: EmbeddingShape = dataclasses.field(
      default_factory=lambda: EmbeddingShape(  # pylint: disable=g-long-lambda
          vocab_size=25000, embedding_size=250
      )
  )
  token_type: EmbeddingShape = dataclasses.field(
      default_factory=lambda: EmbeddingShape(vocab_size=6, embedding_size=16)
  )
  is_upper: EmbeddingShape = dataclasses.field(
      default_factory=lambda: EmbeddingShape(vocab_size=2, embedding_size=8)
  )
  is_title: EmbeddingShape = dataclasses.field(
      default_factory=lambda: EmbeddingShape(vocab_size=2, embedding_size=8)
  )


@dataclasses.dataclass
class EmbeddingConfig(base_config.Config):
  """Embedding block config for AP parsing."""
  shape_configs: EmbeddingShapeConfig = dataclasses.field(
      default_factory=EmbeddingShapeConfig
  )
  dropout_rate: float = 0.1
  pretrained_embedding_path: Optional[str] = None


@dataclasses.dataclass
class ModelConfig(base_config.Config):
  """A base span labeler configuration."""
  encoder: LSTMEncoderConfig = dataclasses.field(
      default_factory=LSTMEncoderConfig
  )
  input_embedding: EmbeddingConfig = dataclasses.field(
      default_factory=EmbeddingConfig
  )


@dataclasses.dataclass
class APParsingConfig(cfg.TaskConfig):
  """The task config."""
  model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
  use_crf: bool = False
  train_data: ap_parsing_dataloader.APParsingDataConfig = dataclasses.field(
      default_factory=ap_parsing_dataloader.APParsingDataConfig
  )
  validation_data: ap_parsing_dataloader.APParsingDataConfig = (
      dataclasses.field(
          default_factory=ap_parsing_dataloader.APParsingDataConfig
      )
  )


class CrfLayer(tf.keras.layers.Layer):
  """A Keras layer that performs CRF over labels and computes CRF loss."""

  def __init__(self, num_classes, **kwargs):
    super().__init__(**kwargs)
    self._num_classes = int(num_classes)

  def build(self, input_shapes):
    del input_shapes
    self._transition_matrix = self.add_weight(
        name="crf_transition_matrix",
        shape=(self._num_classes, self._num_classes),
        initializer="orthogonal",
        trainable=True,
        dtype=tf.float32)

  def call(self,
           inputs,
           training = False):
    """Decodes the highest scoring sequence of tags.

    If training, calculates and records the CRF log-likelihood loss (length
    normalized).
    Args:
      inputs: A list with three tensors. The first tensor is [batch_size,
        max_seq_len, num_tags] tensor of logits. The second tensor is a
        [batch_size] vector of true sequence lengths. The third tensor is
        [batch_size, max_seq_len] tensor of expected ids (only used in training
        mode).
      training: Whether it runs in training mode.

    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
      Contains the highest scoring tag indices.
    """
    logits, sequence_length, labels = inputs

    decode_tags, _ = tfa_text.crf_decode(logits, self._transition_matrix,
                                         sequence_length)
    decode_tags = tf.cast(decode_tags, tf.int32)

    if training:
      # Clip right-padding which equals -1 and messes with the loss calculation.
      labels = tf.maximum(labels, 0)
      log_likelihood, _ = tfa_text.crf_log_likelihood(logits, labels,
                                                      sequence_length,
                                                      self._transition_matrix)
      self.add_loss(tf.reduce_mean(-log_likelihood))

    return decode_tags


def load_embedding_matrix(pretrained_embedding_path):
  """Loads embedding matrix from checkpoint.

  Expects checkpoint to contain a tf.Variable called "embedding_matrix"
  and thus an endpoint called "embedding_matrix/.ATTRIBUTES/VARIABLE_VALUE".
  Args:
    pretrained_embedding_path: Path to checkpoint.

  Returns:
    tf.Tensor of the embedding matrix.
  """
  ckpt = tf.train.load_checkpoint(pretrained_embedding_path)
  logging.info("Loaded pretrained embeddings from checkpoint: %s",
               pretrained_embedding_path)
  return ckpt.get_tensor("embedding_matrix/.ATTRIBUTES/VARIABLE_VALUE")


def get_input_block(names):
  return {
      k: tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=k)
      for k in names
  }


def build_embedding_block(
    inputs,
    embedding_cfg):
  """Builds embedding block.

  Args:
    inputs: A dict of keras input layers.
    embedding_cfg: Config for embedding block.

  Returns:
    Embeddings and boolean mask as tensors.
  """
  initializer = tf.keras.initializers.TruncatedNormal()
  if embedding_cfg.pretrained_embedding_path:
    embedding_matrix = load_embedding_matrix(
        embedding_cfg.pretrained_embedding_path)
    token_ids_initializer = tf.keras.initializers.Constant(embedding_matrix)
  else:
    token_ids_initializer = initializer

  all_embeddings = []
  shape_config_dict = embedding_cfg.shape_configs.as_dict()
  for k, v in inputs.items():
    layer = tf.keras.layers.Embedding(
        input_dim=shape_config_dict[k]["vocab_size"],
        output_dim=shape_config_dict[k]["embedding_size"],
        embeddings_initializer=token_ids_initializer
        if k == constants.TOKEN_IDS else initializer,
        name=f"{k}_embedding")

    all_embeddings.append(layer(v))

  embeddings = tf.keras.layers.Concatenate(axis=-1)(all_embeddings)

  norm = tf.keras.layers.LayerNormalization(
      name="embeddings/layer_norm", axis=-1)
  dropout = tf.keras.layers.Dropout(
      rate=embedding_cfg.dropout_rate, name="embeddings/dropout")

  embeddings = dropout(norm(embeddings))

  # Mask zeroes.
  mask = tf.cast(inputs[constants.TOKEN_IDS] > 0, tf.bool)

  return embeddings, mask


def build_lstm_encoder(emb_data, mask,
                       encoder_cfg):
  """Builds Bi-LSTM encoder.

  Args:
    emb_data: Tensor output of the embedding block [batch_size, seq_len,
      emb_dim].
    mask: Boolean mask [batch_size, seq_length].
    encoder_cfg: Encoder config dataclass.

  Returns:
    The encoder output as a tf.Tensor
  """
  # Project to hidden size:
  proj = tf.keras.layers.Dense(
      encoder_cfg.hidden_size * 2, name="embedding/projection")
  data = proj(emb_data)

  for i in range(encoder_cfg.num_layers):
    # Block logic is:
    # LayerNorm(Dropout(BiLSTM(previous_layer)) + previous_layer)

    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(encoder_cfg.hidden_size, return_sequences=True),
        name=f"BiLSTM{i+1}/BiLSTM")
    dropout = tf.keras.layers.Dropout(
        name=f"BiLSTM{i+1}/dropout", rate=encoder_cfg.dropout_rate)
    add = tf.keras.layers.Add(name=f"BiLSTM{i+1}/Add")
    norm = tf.keras.layers.LayerNormalization(
        name=f"BiLSTM{i+1}/layer_norm", axis=-1)

    orig_data = data
    data = bilstm(data, mask=mask)
    data = dropout(data)
    data = add([data, orig_data])
    data = norm(data)

  return data


def build_heads(encoder_output,
                crf_inputs,
                use_crf = False,
                dropout_rate = 0.1):
  """Builds classification heads (logit outputs) based on the heads config.

  Args:
    encoder_output: Tensor of encoder outputs [batch_size, seq_len,
      hidden_size*2].
    crf_inputs: Optional, inputs containing label ids, required to calculate
      training loss for the crf (if use_crf). Can be zeroes at eval.
    use_crf: Whether to use a CRF on top of the head.
    dropout_rate: Dropout rate to be applied to the encoder outputs before each
      head.

  Returns:
    Dict of {label_name: head_outputs}.
  """

  outputs = {}
  heads_shape = [(k, len(v)) for k, v in constants.CLASS_NAMES.items()]
  for head_name, n_classes in heads_shape:
    cur_output = tf.keras.layers.Dropout(
        rate=dropout_rate, name=f"{head_name}/dropout")(
            encoder_output)
    head_classifier = tf.keras.layers.Dense(
        n_classes, name=f"{head_name}/logits", activation=None)
    cur_output = head_classifier(cur_output)
    outputs[f"{head_name}/logits"] = cur_output

    if use_crf:
      crf = CrfLayer(n_classes, name=f"{head_name}/CRF")
      seq_lengths = tf.reduce_sum(
          tf.cast(tf.greater_equal(crf_inputs[head_name], 0), tf.int32), -1)
      predict_ids = crf((cur_output, seq_lengths, crf_inputs[head_name]))
    else:
      predict_ids = tf.argmax(cur_output, axis=-1, output_type=tf.int32)

    # Use linear activation (pass-through) to have an output name.
    outputs[f"{head_name}/predict_ids"] = tf.keras.layers.Activation(
        None, name=f"{head_name}/predict_ids")(
            predict_ids)
  return outputs


class APParsingTaskBase(base_task.Task):
  """Task object for AP Parsing."""

  def build_model(self):
    """Builds a tf.keras.Model for ap parsing.

    Returns:
      A tf.keras.Model constructed via the functional API containing embeddings,
      BiLSTM based encoder and heads with or without a CRF.
    """
    embedding_cfg = self.task_config.model.input_embedding
    encoder_cfg = self.task_config.model.encoder

    emb_inputs = get_input_block(constants.FEATURE_NAMES)
    emb_data, mask = build_embedding_block(emb_inputs, embedding_cfg)
    encoder_output = build_lstm_encoder(emb_data, mask, encoder_cfg)

    outputs = build_heads(
        encoder_output,
        use_crf=False,
        crf_inputs=None,
        dropout_rate=encoder_cfg.dropout_rate)

    inputs = emb_inputs
    return tf.keras.Model(inputs=inputs, outputs=outputs)

  def build_inputs(self,
                   params,
                   input_context = None):
    """Returns tf.data.Dataset for AP parsing task."""
    loader = ap_parsing_dataloader.APParsingDataLoader(params)
    return loader.load(input_context)

  def build_losses(
      self,
      labels,
      model_outputs,
      aux_losses = None):
    total_loss = tf.constant(0.0, dtype=tf.float32)
    if aux_losses is not None:
      total_loss += tf.reduce_sum(aux_losses)

    for k in constants.LABEL_NAMES:
      cur_labels = tf.cast(labels[k], tf.int32)
      logits = tf.cast(model_outputs[f"{k}/logits"], tf.float32)
      masked_weights = tf.greater_equal(cur_labels, 0)
      total_loss += masked_sparse_categorical_crossentropy(
          cur_labels, logits, masked_weights)
    return total_loss

  def build_metrics(self,
                    training = None):
    del training
    metrics = [
        tf.keras.metrics.Mean(name="all/accuracy"),
    ]

    # Classification metrics.
    for k, class_names in constants.CLASS_NAMES.items():
      metrics.extend([
          tf.keras.metrics.Accuracy(name=f"{k}/accuracy"),
          tf.keras.metrics.Mean(name=f"{k}/macro-f1"),
          tf.keras.metrics.Mean(name=f"{k}/CE"),
      ])
      for class_name in class_names:
        if class_name != "O":
          metrics.extend([
              tf.keras.metrics.Mean(name=f"{k}/{class_name}/precision"),
              tf.keras.metrics.Mean(name=f"{k}/{class_name}/recall"),
              tf.keras.metrics.Mean(name=f"{k}/{class_name}/f1"),
          ])
    return metrics

  def process_metrics(self, metrics,
                      labels,
                      model_outputs):
    metrics_dict = {metric.name: metric for metric in metrics}

    for k, class_names in constants.CLASS_NAMES.items():
      cur_labels = labels[k]
      predicted = model_outputs[f"{k}/predict_ids"]
      logits = model_outputs[f"{k}/logits"]

      # Calculate mask.
      mask = tf.greater_equal(cur_labels, 0)
      tp = (cur_labels == predicted) & mask

      metrics_dict[f"{k}/accuracy"].update_state(cur_labels, predicted,
                                                 tf.cast(mask, tf.int32))
      metrics_dict[f"{k}/CE"].update_state(
          masked_sparse_categorical_crossentropy(cur_labels, logits, mask))

      macro_f1 = 0
      class_names: List[str] = [x for x in class_names if x != "O"]
      for i, class_name in enumerate(class_names):
        tp_i = tf.reduce_sum(tf.cast(tp & (cur_labels == i), tf.float32))
        real_pos_i = tf.reduce_sum(tf.cast(cur_labels == i, tf.float32))
        pred_pos_i = tf.reduce_sum(tf.cast(predicted == i, tf.float32))

        recall = tf.math.divide_no_nan(tp_i, real_pos_i)
        precision = tf.math.divide_no_nan(tp_i, pred_pos_i)
        f1 = tf.math.divide_no_nan(2 * recall * precision, recall + precision)
        macro_f1 += f1
        metrics_dict[f"{k}/{class_name}/recall"].update_state(recall)
        metrics_dict[f"{k}/{class_name}/precision"].update_state(precision)
        metrics_dict[f"{k}/{class_name}/f1"].update_state(f1)
      metrics_dict[f"{k}/macro-f1"].update_state(macro_f1 / len(class_names))

    metrics_dict["all/accuracy"].update_state(
        tf.reduce_mean([
            metrics_dict[f"{k}/accuracy"].result()
            for k in constants.LABEL_NAMES
        ]))


def masked_sparse_categorical_crossentropy(y_true, logits,
                                           mask):
  masked_labels = tf.where(mask, y_true, 0)
  mask = tf.cast(mask, tf.float32)
  loss = tf.keras.losses.sparse_categorical_crossentropy(
      masked_labels, logits, from_logits=True)
  return tf.math.divide_no_nan(tf.reduce_sum(loss * mask), tf.reduce_sum(mask))


class APParsingTaskCRF(APParsingTaskBase):
  """Task object for AP Parsing with CRF."""

  def build_model(self):
    """Builds a tf.keras.Model for ap parsing.

    Returns:
      A tf.keras.Model constructed via the functional API containing embeddings,
      BiLSTM based encoder and heads with or without a CRF.
    """
    embedding_cfg = self.task_config.model.input_embedding
    encoder_cfg = self.task_config.model.encoder
    emb_inputs = get_input_block(constants.FEATURE_NAMES)
    emb_data, mask = build_embedding_block(emb_inputs, embedding_cfg)
    encoder_output = build_lstm_encoder(emb_data, mask, encoder_cfg)

    crf_inputs = get_input_block(constants.LABEL_NAMES)

    outputs = build_heads(
        encoder_output,
        crf_inputs=crf_inputs,
        use_crf=True,
        dropout_rate=encoder_cfg.dropout_rate)

    inputs = {**emb_inputs, **crf_inputs}
    return tf.keras.Model(inputs=inputs, outputs=outputs)

  def build_losses(
      self,
      labels,
      model_outputs,
      aux_losses = None):
    # Guarding logic, should always have the internal CRF nll loss.
    if aux_losses is None:
      losses = [tf.constant(0.0, dtype=tf.float32)]
    else:
      losses = aux_losses
    return tf.reduce_sum(losses)


@exp_factory.register_config_factory("ap_parsing")
def ap_parsing_base_config():
  """Experiment config for AP Parsing."""
  return cfg.ExperimentConfig(
      task=APParsingConfig(),
      restrictions=[
          "task.train_data.is_training != None",
          "task.validation_data.is_training != None"
      ])
