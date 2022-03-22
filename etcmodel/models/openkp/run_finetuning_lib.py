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

"""Library for OpenKP finetuning."""

import functools
import math
from typing import Dict, Mapping, Text

import attr
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator

from etcmodel import layers as etc_layers
from etcmodel import tensor_utils
from etcmodel.models import input_utils
from etcmodel.models import modeling
from etcmodel.models import optimization
from etcmodel.models.openkp import generate_examples_lib


_DENSE_FEATURES = [
    'global_x_coords',
    'global_y_coords',
    'global_widths',
    'global_heights',
    'global_parent_x_coords',
    'global_parent_y_coords',
    'global_parent_widths',
    'global_parent_heights',
]

_INDICATOR_FEATURES = [
    'global_block_indicator',
    'global_inline_indicator',
    'global_heading_indicator',
    'global_leaf_indicator',
    'global_bold_indicator',
    'global_parent_heading_indicator',
    'global_parent_leaf_indicator',
    'global_parent_bold_indicator',
]


@attr.s(auto_attribs=True)
class DenseFeatureScaler:
  """Clips and scales dense Tensors to a certain range."""

  # Minimum input value. Inputs below this will be clipped to this value.
  min_value: float = -1.0

  # Maximum input value. Inputs above this will be clipped to this value.
  max_value: float = 1.0

  def __attrs_post_init__(self):
    if self.min_value + 1e-6 >= self.max_value:
      raise ValueError(
          f'`min_value` ({self.min_value}) must be less than `max_value` '
          f'({self.max_value}).')

  def transform(self, tensor: tf.Tensor):
    """Clips and scales all input values to the range [-1.0, 1.0].

    First all input values are clipped to `min_value` and `max_value`. Then
    the result is linearly scaled such that `min_value` maps to -1.0 and
    `max_value` maps to 1.0.

    Args:
      tensor: A float32 Tensor of any shape.

    Returns:
      A float32 Tensor of the same shape as `tensor`, with all values scaled
      to the inclusive range [-1.0, 1.0].
    """
    min_float = tf.constant(self.min_value, dtype=tf.float32)
    max_float = tf.constant(self.max_value, dtype=tf.float32)
    clipped_values = tf.clip_by_value(tensor, min_float, max_float)

    midpoint = (max_float + min_float) / 2
    scale = (max_float - min_float) / 2

    return (clipped_values - midpoint) / scale


def _get_scalers_from_flags(flags) -> Mapping[Text, DenseFeatureScaler]:
  result = {}
  for name in ('x_coords', 'y_coords', 'widths', 'heights'):
    result[name] = DenseFeatureScaler(
        min_value=getattr(flags, f'{name}_min'),
        max_value=getattr(flags, f'{name}_max'))
  return result


def indicators_to_id(*indicators: tf.Tensor) -> tf.Tensor:
  """Returns the integer id resulting from crossing the (binary) indicators.

  Args:
    *indicators: int32 Tensors containing only `1` and `0` values. All tensors
      must have the same shape.

  Returns:
    An int32 Tensor of the same shape as the inputs, with values in the range
    from 0, inclusive, to 2**len(indicators), exclusive.

  Raises:
    ValueError: If `indicators` is empty.
  """
  if not indicators:
    raise ValueError('`indicators` must not be empty.')

  result = tf.zeros_like(indicators[0])
  for i, tensor in enumerate(reversed(indicators)):
    result += tensor * 2**i
  return result


def input_fn_builder(input_file,
                     flags,
                     model_config,
                     is_training,
                     drop_remainder,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""
  name_to_features = {
      'url_code_points':
          tf.FixedLenFeature([flags.url_num_code_points], tf.int64),
      'label_start_idx':
          tf.FixedLenFeature([flags.num_labels], tf.int64),
      'label_phrase_len':
          tf.FixedLenFeature([flags.num_labels], tf.int64),
      'long_token_ids':
          tf.FixedLenFeature([flags.long_max_length], tf.int64),
      'long_word_idx':
          tf.FixedLenFeature([flags.long_max_length], tf.int64),
      'long_vdom_idx':
          tf.FixedLenFeature([flags.long_max_length], tf.int64),
      'long_input_mask':
          tf.FixedLenFeature([flags.long_max_length], tf.int64),
      'long_word_input_mask':
          tf.FixedLenFeature([flags.long_max_length], tf.int64),
      'global_token_ids':
          tf.FixedLenFeature([flags.global_max_length], tf.int64),
      'global_input_mask':
          tf.FixedLenFeature([flags.global_max_length], tf.int64),
  }

  if flags.use_visual_features_in_global or flags.use_visual_features_in_long:
    name_to_features.update({
        'global_font_ids':
            tf.FixedLenFeature([flags.global_max_length], tf.int64),
        'global_parent_font_ids':
            tf.FixedLenFeature([flags.global_max_length], tf.int64),
    })
    for name in _DENSE_FEATURES:
      name_to_features[name] = tf.FixedLenFeature([flags.global_max_length],
                                                  tf.float32)
    for name in _INDICATOR_FEATURES:
      name_to_features[name] = tf.FixedLenFeature([flags.global_max_length],
                                                  tf.int64)

  def decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
      tensor = example[name]
      if tensor.dtype == tf.int64:
        tensor = tf.cast(tensor, tf.int32)
      example[name] = tensor

    if (not flags.use_visual_features_in_global and
        not flags.use_visual_features_in_long):
      return example

    # Transform visual features.
    dense_scalers = _get_scalers_from_flags(flags)
    dense_features = []
    for name in _DENSE_FEATURES:
      feature = example.pop(name)
      # We use the same scaler for a feature in a VDOM element and its parent.
      short_name = name.replace('global_', '', 1).replace('parent_', '', 1)
      dense_features.append(dense_scalers[short_name].transform(feature))

    for name in _INDICATOR_FEATURES:
      if name in flags.indicators_to_cross:
        continue
      dense_features.append(tf.cast(example.pop(name), tf.float32))

    example['global_dense_features'] = tf.stack(dense_features, axis=-1)

    indicators_to_cross = [
        example.pop(name) for name in flags.indicators_to_cross
    ]
    example['global_indicator_cross'] = indicators_to_id(*indicators_to_cross)

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params['batch_size']
    d = tf.data.Dataset.list_files(input_file, shuffle=is_training)
    d = d.apply(
        tf.data.experimental.parallel_interleave(
            functools.partial(tf.data.TFRecordDataset),
            cycle_length=num_cpu_threads,
            sloppy=is_training))
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))
    d = d.prefetch(tf.data.experimental.AUTOTUNE)
    d = d.map(
        functools.partial(_add_side_input_features, model_config),
        tf.data.experimental.AUTOTUNE)
    return d.prefetch(tf.data.experimental.AUTOTUNE)

  return input_fn


def _add_side_input_features(
    model_config: modeling.EtcConfig,
    features: Mapping[Text, tf.Tensor]) -> Dict[Text, tf.Tensor]:
  """Replaces raw input features with derived ETC side inputs.

  Args:
    model_config: An `EtcConfig`.
    features: A dictionary of Tensor features, crucially including
      `long_breakpoints`, `global_breakpoints`, `sentence_ids`.

  Returns:
    A new `features` dictionary with global-local transformer side inputs.
  """
  features = dict(features)
  side_inputs = (
      input_utils.make_global_local_transformer_side_inputs_from_example_ids(
          long_example_ids=features['long_input_mask'],
          global_example_ids=features['global_input_mask'],
          sentence_ids=features['long_vdom_idx'],
          local_radius=model_config.local_radius,
          relative_pos_max_distance=model_config.relative_pos_max_distance,
          use_hard_g2l_mask=model_config.use_hard_g2l_mask,
          use_hard_l2g_mask=model_config.use_hard_l2g_mask))
  features.update(side_inputs.to_dict())
  return features


def model_fn_builder(model_config, num_train_steps, num_warmup_steps, flags):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info('*** Features ***')
    for name in sorted(features.keys()):
      tf.logging.info('  name = %s, shape = %s' % (name, features[name].shape))

    is_training = (mode == tf_estimator.ModeKeys.TRAIN)
    ngram_logits, extra_model_losses = _build_model(model_config, features,
                                                    is_training, flags)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if flags.init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = input_utils.get_assignment_map_from_checkpoint(
          tvars, flags.init_checkpoint)
      if flags.use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(flags.init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(flags.init_checkpoint, assignment_map)

    tf.logging.info('**** Trainable Variables ****')
    for var in tvars:
      if var.name in initialized_variable_names:
        init_string = ', *INIT_FROM_CKPT*'
      else:
        init_string = ', *RANDOM_INIT*'
      tf.logging.info('  name = %s, shape = %s%s', var.name, var.shape,
                      init_string)

    if mode == tf_estimator.ModeKeys.TRAIN:
      # [batch_size, kp_max_length * long_max_length]
      ngram_labels = make_ngram_labels(
          label_start_idx=features['label_start_idx'],
          label_phrase_len=features['label_phrase_len'],
          long_max_length=flags.long_max_length,
          kp_max_length=flags.kp_max_length,
          additive_smoothing_mass=flags.additive_smoothing_mass)

      reshaped_ngram_logits = tf.reshape(
          ngram_logits, shape=tf.shape(ngram_labels))
      per_example_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
          labels=ngram_labels, logits=reshaped_ngram_logits)
      total_loss = tf.reduce_mean(per_example_loss)
      if extra_model_losses:
        total_loss += tf.math.add_n(extra_model_losses)

      train_op = optimization.create_optimizer(
          total_loss, flags.learning_rate, num_train_steps, num_warmup_steps,
          flags.use_tpu, flags.optimizer, flags.poly_power,
          flags.start_warmup_step, flags.learning_rate_schedule)

      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)

    elif mode == tf_estimator.ModeKeys.PREDICT:
      predictions = {
          'url_code_points': tf.identity(features['url_code_points']),
          'ngram_logits': ngram_logits
      }
      output_spec = tf_estimator.tpu.TPUEstimatorSpec(
          mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    else:
      raise ValueError('Only TRAIN and PREDICT modes are supported: %s' %
                       (mode))

    return output_spec

  return model_fn


def _build_model(model_config, features, is_training, flags):
  """Build an ETC model for OpenKP."""

  global_embedding_adder = None
  long_embedding_adder = None

  # Create `global_embedding_adder` if using visual features.
  if flags.use_visual_features_in_global or flags.use_visual_features_in_long:
    global_embedding_adder = _create_global_visual_feature_embeddings(
        model_config, features, flags)

  if flags.use_visual_features_in_long:
    # Create `long_embedding_adder` based on `global_embedding_adder`
    long_embedding_adder = gather_global_embeddings_to_long(
        global_embedding_adder, features['long_vdom_idx'])

  if not flags.use_visual_features_in_global:
    global_embedding_adder = None

  model = modeling.EtcModel(
      config=model_config,
      is_training=is_training,
      use_one_hot_relative_embeddings=flags.use_tpu)

  model_inputs = dict(
      token_ids=features['long_token_ids'],
      global_token_ids=features['global_token_ids'],
      long_embedding_adder=long_embedding_adder,
      global_embedding_adder=global_embedding_adder)
  for field in attr.fields(input_utils.GlobalLocalTransformerSideInputs):
    model_inputs[field.name] = features[field.name]

  long_output, _ = model(**model_inputs)

  word_embeddings_unnormalized = batch_segment_sum_embeddings(
      long_embeddings=long_output,
      long_word_idx=features['long_word_idx'],
      long_input_mask=features['long_input_mask'])
  word_emb_layer_norm = tf.keras.layers.LayerNormalization(
      axis=-1, epsilon=1e-12, name='word_emb_layer_norm')
  word_embeddings = word_emb_layer_norm(word_embeddings_unnormalized)

  ngram_logit_list = []
  for i in range(flags.kp_max_length):
    conv = tf.keras.layers.Conv1D(
        filters=model_config.hidden_size,
        kernel_size=i + 1,
        padding='valid',
        activation=tensor_utils.get_activation('gelu'),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(
            stddev=0.02 / math.sqrt(i + 1)),
        name=f'{i + 1}gram_conv')
    layer_norm = tf.keras.layers.LayerNormalization(
        axis=-1, epsilon=1e-12, name=f'{i + 1}gram_layer_norm')

    logit_dense = tf.keras.layers.Dense(
        units=1,
        activation=None,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name=f'logit_dense{i}')
    # [batch_size, long_max_length - i]
    unpadded_logits = tf.squeeze(
        logit_dense(layer_norm(conv(word_embeddings))), axis=-1)

    # Pad to the right to get back to `long_max_length`.
    padded_logits = tf.pad(unpadded_logits, paddings=[[0, 0], [0, i]])

    # Padding logits should be ignored, so we make a large negative mask adder
    # for them.
    shifted_word_mask = tf.cast(
        tensor_utils.shift_elements_right(
            features['long_word_input_mask'], axis=-1, amount=-i),
        dtype=padded_logits.dtype)
    mask_adder = -10000.0 * (1.0 - shifted_word_mask)

    ngram_logit_list.append(padded_logits * shifted_word_mask + mask_adder)

  # [batch_size, kp_max_length, long_max_length]
  ngram_logits = tf.stack(ngram_logit_list, axis=1)

  extra_model_losses = model.losses

  return ngram_logits, extra_model_losses


def _create_global_visual_feature_embeddings(model_config, features,
                                             flags) -> tf.Tensor:
  """Creates global embeddings based on visual features."""
  initializer_range = 0.02

  indicator_cross_emb_lookup = etc_layers.EmbeddingLookup(
      vocab_size=2**len(flags.indicators_to_cross),
      embedding_size=model_config.hidden_size,
      initializer_range=initializer_range,
      use_one_hot_lookup=flags.use_tpu,
      name='indicator_cross_emb_lookup')
  global_embedding_adder = indicator_cross_emb_lookup(
      features['global_indicator_cross'])

  font_id_emb_lookup = etc_layers.EmbeddingLookup(
      vocab_size=generate_examples_lib.FONT_ID_VOCAB_SIZE,
      embedding_size=model_config.hidden_size,
      initializer_range=initializer_range,
      use_one_hot_lookup=flags.use_tpu,
      name='font_id_emb_lookup')
  global_embedding_adder += font_id_emb_lookup(features['global_font_ids'])

  parent_font_id_emb_lookup = etc_layers.EmbeddingLookup(
      vocab_size=generate_examples_lib.FONT_ID_VOCAB_SIZE,
      embedding_size=model_config.hidden_size,
      initializer_range=initializer_range,
      use_one_hot_lookup=flags.use_tpu,
      name='parent_font_id_emb_lookup')
  global_embedding_adder += parent_font_id_emb_lookup(
      features['global_parent_font_ids'])

  # Add transformation of dense features
  dense_feature_projection = tf.keras.layers.Dense(
      units=model_config.hidden_size,
      activation=tensor_utils.get_activation('gelu'),
      kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
      name='dense_feature_projection')
  dense_feature_embeddings = dense_feature_projection(
      features['global_dense_features'])
  if flags.extra_dense_feature_layers > 1:
    raise NotImplementedError('`extra_dense_feature_layers` must be at most 1.')
  elif flags.extra_dense_feature_layers == 1:
    dense_feature_layer2 = tf.keras.layers.Dense(
        units=model_config.hidden_size,
        activation=tensor_utils.get_activation('gelu'),
        kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
        name='dense_feature_layer2')
    dense_feature_embeddings = dense_feature_layer2(dense_feature_embeddings)
  global_embedding_adder += dense_feature_embeddings

  return global_embedding_adder


def gather_global_embeddings_to_long(global_embeddings: tf.Tensor,
                                     long_vdom_idx: tf.Tensor) -> tf.Tensor:
  """Gathers global embeddings to long positions based on VDOM index.

  Args:
    global_embeddings: <float32>[batch_size, global_max_length, hidden_size]
      Tensor of global embeddings.
    long_vdom_idx: <int32>[batch_size, long_max_length] Tensor representing the
      index of the global token each long token belongs to. Every value must be
      between 0 (inclusive) and `global_max_length` (exclusive).

  Returns:
    <float32>[batch_size, long_max_length, hidden_size] Tensor of embeddings
    copied from the corresponding global token.
  """
  return tf.gather(global_embeddings, long_vdom_idx, batch_dims=1)


def batch_segment_sum_embeddings(long_embeddings: tf.Tensor,
                                 long_word_idx: tf.Tensor,
                                 long_input_mask: tf.Tensor) -> tf.Tensor:
  """Sums wordpiece `long_embeddings` into word embeddings.

  Args:
    long_embeddings: <float32>[batch_size, long_max_length, hidden_size] Tensor
      of contextual embeddings for wordpieces, as output by ETC model.
    long_word_idx: <int32>[batch_size, long_max_length] Tensor representing the
      index of the word each wordpiece belongs to. The index for padding tokens
      can be any integer in the range [0, long_max_length) and will be ignored.
    long_input_mask: <int32>[batch_size, long_max_length] Tensor representing
      which *wordpiece* tokens in `long_embeddings` are present, with `1` for
      present tokens and `0` for padding.

  Returns:
    <float32>[batch_size, long_max_length, hidden_size] Tensor of embeddings
    for each word calculated by summing the embeddings of the wordpieces
    belonging to the word. The number of words is no greater than the number
    of wordpieces, but we keep `long_max_length`, so there may be an increase
    in padding. All padding embeddings will be 0.
  """
  # Zero out padding embeddings.
  long_embeddings *= tf.cast(
      long_input_mask, dtype=long_embeddings.dtype)[:, :, tf.newaxis]

  batch_size = tf.shape(long_embeddings)[0]
  example_idx = tf.broadcast_to(
      tf.range(batch_size)[:, tf.newaxis], shape=tf.shape(long_word_idx))
  scatter_indices = tf.stack([example_idx, long_word_idx], axis=-1)

  return tf.scatter_nd(
      indices=scatter_indices,
      updates=long_embeddings,
      shape=tf.shape(long_embeddings))


def make_ngram_labels(label_start_idx: tf.Tensor,
                      label_phrase_len: tf.Tensor,
                      long_max_length: int,
                      kp_max_length: int,
                      additive_smoothing_mass: float = 1e-6) -> tf.Tensor:
  """Makes ngram labels for `tf.nn.softmax_cross_entropy_with_logits`.

  Args:
    label_start_idx: <int32>[batch_size, num_labels] Tensor of the index of the
      first word in each correct key phrase. There must be at least 1 correct
      key phrase, and if there are less than `num_labels` then `-1` is used as
      right padding. All values must be less than `long_max_length`.
      `num_labels` is the maximum number of key phrase labels, which is 3 for
      OpenKP.
    label_phrase_len: <int32>[batch_size, num_labels] Tensor of the
      corresponding length of the key phrase, again using `-1` to pad.
      Non-padding values must be in the inclusive range [1, kp_max_length].
    long_max_length: Integer maximum number of words in the document.
    kp_max_length: Integer maximum number of words in a key phrase.
    additive_smoothing_mass: Total probability mass (on top of `1.0` mass for
      the actual label) to add for additive smoothing. We use a minimum of 1e-6
      to avoid any potential division by 0.

  Returns:
    <float32>[batch_size, kp_max_length * long_max_length] Tensor of label
    probabilities based on the inputs. Each row sums to 1.0, and the order
    of entries is compatible with reshaping `ngram_logits` from shape
    [batch_size, kp_max_length, long_max_length] to match these labels.
  """
  # [batch_size, num_labels, kp_max_length]
  phrase_len_one_hot = tf.one_hot(
      label_phrase_len - 1, depth=kp_max_length, dtype=tf.float32)

  # [batch_size, num_labels, long_max_length]
  start_idx_one_hot = tf.one_hot(
      label_start_idx, depth=long_max_length, dtype=tf.float32)

  # [batch_size, kp_max_length, long_max_length]
  combined_one_hot = tf.einsum('bnk,bnl->bkl', phrase_len_one_hot,
                               start_idx_one_hot)

  # [batch_size, kp_max_length * long_max_length]
  unnormalized_labels = tensor_utils.flatten_dims(combined_one_hot, first_dim=1)

  unsmoothed_labels = (
      unnormalized_labels /
      (tf.reduce_sum(unnormalized_labels, axis=-1, keepdims=True) + 1e-6))

  # Use at least 1e-6 smoothing mass to avoid divide by 0.
  additive_smoothing_mass = max(additive_smoothing_mass, 1e-6)

  num_classes = kp_max_length * long_max_length
  smoothed_labels = unsmoothed_labels + additive_smoothing_mass / num_classes

  return (smoothed_labels /
          tf.reduce_sum(smoothed_labels, axis=-1, keepdims=True))
