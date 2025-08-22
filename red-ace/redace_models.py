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

"""Defines loss layers and RED-ACE models."""

from official.modeling import activations
from official.nlp.modeling import layers
from official.nlp.modeling import losses
import redace_bert_encoder
import tensorflow as tf


class BertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_example_loss):
    """Adds metrics.

    Args:
      lm_output: [batch_size, max_predictions_per_seq, vocab_size] tensor with
        language model logits.
      lm_labels: [batch_size, max_predictions_per_seq] tensor with gold outputs.
      lm_label_weights: [batch_size, max_predictions_per_seq] tensor with
        per-token weights.
      lm_example_loss: scalar loss.
    """
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    numerator = tf.reduce_sum(accuracy * lm_label_weights)
    denominator = tf.reduce_sum(lm_label_weights)
    masked_lm_accuracy = tf.math.divide_no_nan(numerator, denominator)

    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    masked_lm_predictions = tf.argmax(lm_output, axis=-1, output_type=tf.int32)
    per_token_accuracy = tf.equal(lm_labels, masked_lm_predictions)
    sent_level_accuracy = tf.cast(
        tf.reduce_all(
            tf.logical_or(per_token_accuracy, (lm_label_weights <= 0)), axis=1),
        tf.float32)
    self.add_metric(sent_level_accuracy, name='exact_match', aggregation='mean')

  def call(self, lm_output, lm_label_ids, lm_label_weights):
    """Implements call() for the layer.

    Args:
      lm_output: [batch_size, max_predictions_per_seq, vocab_size] tensor with
        language model logits.
      lm_label_ids: [batch_size, max_predictions_per_seq] tensor with gold
        outputs.
      lm_label_weights: [batch_size, max_predictions_per_seq] tensor with
        per-token weights.

    Returns:
      final_loss: scalar loss.
    """
    lm_label_weights = tf.cast(lm_label_weights, tf.float32)
    lm_output = tf.cast(lm_output, tf.float32)

    mask_label_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        labels=lm_label_ids,
        predictions=lm_output,
        weights=lm_label_weights,
        from_logits=True)

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      mask_label_loss)
    return mask_label_loss

  def get_config(self):
    return self._config


class RedAceLoss(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for RED-ACE.

  Attributes:
    tagging_weight: Tensor (scalar) used to control the importance of the
      tagging loss.
  """

  def __init__(self):
    self.tagging_weight = tf.Variable(1.0, trainable=False, dtype=tf.float32)
    super(RedAceLoss, self).__init__()

  def _add_metrics(
      self,
      tag_logits,
      tag_labels,
      tag_loss,
      input_mask,
      labels_mask,
      total_loss,
  ):
    """Adds metrics.

    Args:
      tag_logits: [batch_size, seq_length, vocab_size] tensor with tag logits.
      tag_labels: [batch_size, seq_length] tensor with gold outputs.
      tag_loss: [batch_size, seq_length] scalar loss for tagging model.
      input_mask: [batch_size, seq_length] tensor with mask (1s or 0s).
      labels_mask: [batch_size, seq_length] mask for labels, may be a binary
        mask or a weighted float mask.
      total_loss:   scalar loss for whole model.
    """
    self.add_metric(tag_loss, name='tag_loss', aggregation='mean')

    tag_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        tag_labels, tag_logits)
    numerator = tf.reduce_sum(tag_accuracy * tf.cast(labels_mask, tf.float32))
    denominator = tf.reduce_sum(tf.cast(labels_mask, tf.float32))
    tag_accuracy = tf.math.divide_no_nan(numerator, denominator)
    self.add_metric(tag_accuracy, name='tag_accuracy', aggregation='mean')

    tag_predictions = tf.argmax(tag_logits, axis=-1, output_type=tf.int32)
    tag_acc = tf.equal(tag_labels, tag_predictions)
    sent_level_tag_accuracy = tf.cast(
        tf.reduce_all(tf.logical_or(tag_acc, (labels_mask <= 0)), axis=1),
        tf.float32)
    self.add_metric(
        sent_level_tag_accuracy, name='tag_exact_match', aggregation='mean')
    sent_level_accuracy = sent_level_tag_accuracy

    self.add_metric(
        sent_level_accuracy, name='estimated_exact_match', aggregation='mean')

    self.add_metric(total_loss, name='total_loss', aggregation='mean')

  def call(
      self,
      tag_logits,
      tag_labels,
      input_mask,
      labels_mask,
  ):
    """Implements call() for the layer.

    Args:
      tag_logits: [batch_size, seq_length, vocab_size] tensor with tag logits.
      tag_labels: [batch_size, seq_length] tensor with gold outputs.
      input_mask: [batch_size, seq_length]tensor with mask (1s or 0s).
      labels_mask: [batch_size, seq_length] mask for labels, may be a binary
        mask or a weighted float mask.

    Returns:
      Scalar loss of the model.
    """

    # Apply abstain mask. A tag_label of 0 is a pad token.
    input_mask = tf.cast(input_mask, tf.float32) * tf.cast(
        tf.clip_by_value(tag_labels, 0, 1), tf.float32)
    labels_mask = tf.cast(labels_mask, tf.float32) * tf.cast(
        tf.clip_by_value(tag_labels, 0, 1), tf.float32)
    # Add to prevent NaNs.
    tag_logits = tf.cast(tag_logits, tf.float32) + 0.00001
    # labels_mask is length normalized, this undoes this normalization.
    labels_mask = labels_mask * tf.math.reduce_sum(
        tf.cast(input_mask, tf.float32), axis=-1, keepdims=True)

    tag_logits_loss = losses.weighted_sparse_categorical_crossentropy_loss(
        labels=tag_labels,
        predictions=tag_logits,
        weights=tf.cast(labels_mask, tf.float32),
        from_logits=True,
    )
    total_loss = tag_logits_loss * self.tagging_weight

    self._add_metrics(
        tag_logits,
        tag_labels,
        tag_logits_loss,
        input_mask,
        labels_mask,
        total_loss,
    )

    return total_loss

  def get_config(self):
    return self._config


class BalanceLossCallback(tf.keras.callbacks.Callback):
  """Creates a Callback that automatically balances the 2/3 RED-ACE losses."""

  def __init__(
      self,
      loss_weight_tensor_1,
      loss_weight_tensor_2,
      loss_weight_tensor_3=None,
      score_name_1='tag_exact_match',
      score_name_2='point_exact_match',
      score_name_3=None,
  ):
    """Creates a callback which balances multiple losses.

    Args:
      loss_weight_tensor_1: Tensor that determines the importances of the first
        loss.
      loss_weight_tensor_2: Tensor that determines the importances of the second
        loss.
      loss_weight_tensor_3: Tensor that determines the importances of the third
        loss. If there is no third loss this can be None.
      score_name_1: The name of the first score in the logs.
      score_name_2: The name of the second score in the logs.
      score_name_3: The name of the third score in the logs. If there is no
        third score this can be None.
    """
    super(BalanceLossCallback, self).__init__()

    loss_1_features = {
        'loss_weight': 1.0,
        'loss_weight_tensor': loss_weight_tensor_1,
        'score_name': score_name_1,
        'score_weighted_average': 0
    }

    loss_2_features = {
        'loss_weight': 1.0,
        'loss_weight_tensor': loss_weight_tensor_2,
        'score_name': score_name_2,
        'score_weighted_average': 0,
    }
    self._loss_features = [loss_1_features, loss_2_features]
    if loss_weight_tensor_3 is not None:
      loss_3_features = {
          'loss_weight': 1.0,
          'loss_weight_tensor': loss_weight_tensor_3,
          'score_name': score_name_3,
          'score_weighted_average': 0,
      }
      self._loss_features.append(loss_3_features)

    self._adjustment_rate = 1.25
    # The following values are arbitrary. They try to balance multiple losses,
    # however a very high/low value could restrict one of the losses from
    # reaching 100%, if it is impossible for the other loss to reach 100%.
    self._min_lambda_weight = 0.01
    self._max_lambda_weight = 10
    # A decayed weighted average is used to score the different losses.
    self._decay_rate = 0.9

  def on_batch_end(self, epoch, logs=None):
    for loss_feature in self._loss_features:
      loss_feature['score_weighted_average'] = loss_feature[
          'score_weighted_average'] * self._decay_rate + logs[
              loss_feature['score_name']] * (1 - self._decay_rate)

    sorted(self._loss_features, key=lambda x: x['score_weighted_average'])
    # Increase the weight of the worst performing loss and decrease the
    # weight of the best performing loss.
    smallest_loss_feature = self._loss_features[0]
    smallest_loss_feature['loss_weight'] = min(
        self._max_lambda_weight,
        smallest_loss_feature['loss_weight'] * self._adjustment_rate)
    tf.keras.backend.set_value(smallest_loss_feature['loss_weight_tensor'],
                               smallest_loss_feature['loss_weight'])

    biggest_loss_feature = self._loss_features[0]
    biggest_loss_feature['loss_weight'] = max(
        self._min_lambda_weight,
        biggest_loss_feature['loss_weight'] / self._adjustment_rate)
    tf.keras.backend.set_value(biggest_loss_feature['loss_weight_tensor'],
                               biggest_loss_feature['loss_weight'])


def get_model(
    redace_config,
    seq_length,
    is_training=True,
    force_model_batch_size=False,
    batch_size=None,
):
  """Returns model to be used for pre-training.

  Args:
      redace_config: Configuration that defines the core BERT model.
      seq_length: Maximum sequence length of the training data.
      is_training: Will the model be trained or is it inferance time.
      force_model_batch_size: If True model tensors are explicitly set to have
        the given batch_size as the first dimension, otherwise this is infered
        with tf.shape. This is used for exporting, but could also be used for
        prediction. Note this works for all cases apart from when a TPU is
        bigger than 1x1.
      batch_size: Only needed when use_joint_model is True and the model is
        being exported or performing inference.

  Returns:
      The RED-ACE model, the core BERT submodel and callbacks used for training.
  """
  if force_model_batch_size and batch_size is not None:
    model_batch_size = batch_size
  else:
    model_batch_size = None
  input_word_ids = tf.keras.layers.Input(
      shape=(seq_length,),
      name='input_word_ids',
      dtype=tf.int32,
      batch_size=batch_size)
  input_mask = tf.keras.layers.Input(
      shape=(seq_length,),
      name='input_mask',
      dtype=tf.int32,
      batch_size=batch_size)
  input_type_ids = tf.keras.layers.Input(
      shape=(seq_length,),
      name='input_type_ids',
      dtype=tf.int32,
      batch_size=batch_size)
  input_confidence_scores = tf.keras.layers.Input(
      shape=(seq_length,),
      name='input_confidence_scores',
      dtype=tf.int32,
      batch_size=batch_size,
  )
  pre_trained_model = redace_bert_encoder.RedAceBertEncoder(
      vocab_size=redace_config.vocab_size,
      hidden_size=redace_config.hidden_size,
      num_layers=redace_config.num_hidden_layers,
      num_attention_heads=redace_config.num_attention_heads,
      intermediate_size=redace_config.intermediate_size,
      activation=activations.gelu,
      dropout_rate=redace_config.hidden_dropout_prob,
      attention_dropout_rate=redace_config.attention_probs_dropout_prob,
      sequence_length=seq_length,
      max_sequence_length=redace_config.max_position_embeddings,
      type_vocab_size=redace_config.type_vocab_size,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=redace_config.initializer_range),
  )

  redace_model = RedAceModel(
      pre_trained_model,
      seq_length=seq_length,
      redace_config=redace_config,
      is_training=is_training,
      batch_size=model_batch_size,
  )
  redace_inputs = [
      input_word_ids,
      input_mask,
      input_type_ids,
      input_confidence_scores,
  ]
  if is_training:
    edit_tags = tf.keras.layers.Input(
        shape=(seq_length,),
        name='edit_tags',
        dtype=tf.int32,
        batch_size=batch_size)

    redace_inputs.append(edit_tags)

    redace_outputs = redace_model(redace_inputs)
    labels_mask = tf.keras.layers.Input(
        shape=(seq_length,),
        name='labels_mask',
        dtype=tf.float32,
        batch_size=batch_size)
    redace_inputs.append(labels_mask)

    tag_logits, _ = redace_outputs

    loss_function = RedAceLoss()
    redace_tag_loss = loss_function(tag_logits, edit_tags, input_mask,
                                    labels_mask)

    keras_model = tf.keras.Model(inputs=redace_inputs, outputs=redace_tag_loss)
    return keras_model, pre_trained_model
  else:
    redace_inputs = [
        input_word_ids,
        input_mask,
        input_type_ids,
        input_confidence_scores,
    ]
    redace_outputs = redace_model(redace_inputs)
    tag_logits, *remaining_outputs, _ = redace_outputs
    assert not remaining_outputs  # Should be empty now.
    keras_model = tf.keras.Model(
        inputs=redace_inputs,
        outputs={
            'redace_tagger': tag_logits,
        },
    )
    return keras_model, pre_trained_model


class RedAceModel(tf.keras.Model):
  """RED-ACE tagger model based on a BERT-style transformer-based encoder."""

  def __init__(
      self,
      network,
      redace_config,
      initializer='glorot_uniform',
      seq_length=128,
      is_training=True,
      batch_size=None,
  ):
    """Creates RED-ACE Tagger.

    Setting up all of the layers needed for call.

    Args:
      network: An encoder network, which should output a sequence of hidden
        states.
      redace_config: A config file which in addition to the  RedAceConfig values
        also includes: num_classes, hidden_dropout_prob, and query_transformer.
      initializer: The initializer (if any) to use in the classification
        networks. Defaults to a Glorot uniform initializer.
      seq_length:  Maximum sequence length.
      is_training: The model is being trained.
      batch_size: Size of the batch, only needed when exporting the model.
    """

    super(RedAceModel, self).__init__()
    self._network = network
    self._seq_length = seq_length
    self._redace_config = redace_config
    self._is_training = is_training
    self._batch_size = batch_size

    self._tag_logits_layer = tf.keras.layers.Dense(
        self._redace_config.num_classes)

  def _create_sets_of_layers(self, number_of_layers, shared):
    transformer_layers = [
        layers.TransformerEncoderBlock(
            num_attention_heads=self._redace_config.num_attention_heads,
            inner_dim=self._redace_config.intermediate_size,
            inner_activation=activations.gelu,
            output_dropout=self._redace_config.hidden_dropout_prob,
            attention_dropout=self._redace_config.hidden_dropout_prob,
            output_range=self._seq_length,
        )
    ]
    for _ in range(1, number_of_layers):
      if shared:
        transformer_layers.append(transformer_layers[0])
      else:
        transformer_layers.append(
            layers.TransformerEncoderBlock(
                num_attention_heads=self._redace_config.num_attention_heads,
                inner_dim=self._redace_config.intermediate_size,
                inner_activation=activations.gelu,
                output_dropout=self._redace_config.hidden_dropout_prob,
                attention_dropout=self._redace_config.hidden_dropout_prob,
                output_range=self._seq_length,
            ))
    return transformer_layers

  def call(
      self,
      inputs,
      is_training=None,
  ):
    """Forward pass of the model.

    Args:
      inputs: A list of tensors. In training the following 4 tensors are
        required, [input_word_ids, input_mask, input_type_ids,edit_tags]. Only
        the first 3 are required in test. input_word_ids[batch_size,
        seq_length], input_mask[batch_size, seq_length],
        input_type_ids[batch_size, seq_length], edit_tags[batch_size,
        seq_length]. If using output variants, these should also be provided.
        output_variant_ids[batch_size, 1].
      is_training: Indict whether forward pass in training or test (false) mode.

    Returns:
      The logits of the edit tags and optionally the logits of the pointer
        network.
    """

    (
        input_word_ids,
        input_mask,
        input_type_ids,
        input_confidence_scores,
        *inputs,
    ) = inputs
    if self._is_training:
      _, *inputs = inputs

    assert not inputs  # Should be empty now.

    bert_output = self._network(
        [input_word_ids, input_mask, input_type_ids,
         input_confidence_scores])[0]

    tag_logits = self._tag_logits_layer(bert_output)

    return [tag_logits, bert_output]
