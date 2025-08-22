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

"""Feature converters for KL-guided temperature sampling.

We override EncDecFeatureConverter and PrefixLMFeatureConverter when context is
provided in inputs, represented by negative values. These feature converters
output most of the conventional model features the same as their base models,
except for:
  "encoder_input_tokens": all negative tokens are made positive;
  "decoder_input_tokens": all negative tokens are made positive;
  "decoder_target_tokens": all negative tokens are made positive;
additionally, with extra model features without contexts
  "encoder_input_tokens_wo":
      all negative tokens are remove and the feature is left shifted;
  "decoder_input_tokens_wo":
      all negative tokens are remove and the feature is left shifted;
  "decoder_target_tokens_wo":
      all negative tokens are remove and the feature is left shifted;
  "decoder_loss_weights_wo":
      updated according to "decoder_input_tokens_wo" and "decoder_target_tokens_wo";
  "decoder_causal_attention_wo":
      updated according to "decoder_input_tokens_wo" and "decoder_target_tokens_wo".
Note that packing is not supported.
"""
from typing import Mapping
import seqio
import tensorflow.compat.v2 as tf
FeatureConverter = seqio.FeatureConverter


class ContextualEncDecFeatureConverter(seqio.EncDecFeatureConverter):
  """Feature converter for an encoder-decoder architecture with contexts.

  The input dataset has "inputs" and "targets" field. These will be converted
  to a subset of standard features.

  Example:

  The input dataset has two examples each with "inputs" and "targets".

  ds = [{"inputs": [-7, 8, 5, 1], "targets": [3, 9, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 7}

  converted_ds = [{
       "encoder_input_tokens": [7, 8, 5, 1, 0, 0, 0, 0, 0, 0],
    "encoder_input_tokens_wo": [8, 5, 1, 0, 0, 0, 0, 0, 0, 0],
      "decoder_target_tokens": [3, 9, 1, 0, 0, 0, 0],
       "decoder_input_tokens": [0, 3, 9, 1, 0, 0, 0],
       "decoder_loss_weights": [1, 1, 1, 0, 0, 0, 0],
  }]

  Note that two examples are packed together into one example.
  """
  MODEL_FEATURES = {
      "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "encoder_input_tokens_wo": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {}

  def __init__(self,
               pack = False,
               use_custom_packing_ops = False,
               apply_length_check = True,
               bos_id = 0):
    assert not pack
    super().__init__(
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id)

  def _convert_example(
      self, features):
    """Convert a seq2seq example into an example with model features."""
    def fn_truncate(feature):
      return tf.roll(tf.maximum(feature, 0),
                     -tf.reduce_sum(tf.cast(feature < 0, tf.int32)), axis=0)

    feature_positive = {
        k: tf.abs(v) for k, v in features.items() if v.dtype != tf.string}
    d = super()._convert_example(feature_positive)
    feature_truncate = {
        k: fn_truncate(v) for k, v in features.items() if v.dtype != tf.string}
    d_wo = super()._convert_example(feature_truncate)
    d = {**d, **{"encoder_input_tokens_wo": d_wo["encoder_input_tokens"]}}
    return d

  def get_model_feature_lengths(self, task_feature_lengths
                                ):
    ret = super().get_model_feature_lengths(task_feature_lengths)
    assert "encoder_input_tokens" in ret
    return {**{"encoder_input_tokens_wo": ret["encoder_input_tokens"]}, **ret}


class ContextualPrefixLMFeatureConverter(seqio.PrefixLMFeatureConverter):
  """Feature converter for a prefix language model architecture with contexts.

  Example 2: unpacked dataset with extra long "inputs" `task_feature_lengths`
  ```
  ds = [{"inputs": [-9, 4, 6, 1], "targets": [3, 9, 1]}]

  task_feature_lengths = {"inputs": 10, "targets": 4}

  converted_ds = {
         "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
          "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
          "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
      "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      "decoder_target_tokens_wo": [4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0],
       "decoder_input_tokens_wo": [0, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
       "decoder_loss_weights_wo": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
   "decoder_causal_attention_wo": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  }

  Note that if the inputs length specified in `task_feature_lengths` is longer
  than the actual example length, the padding tokens are added after
  concatenation.
  ```

  Attributes:
    loss_on_targets_only: whether to compute loss on tokens which belonged to
      "targets" before concatenation.
  """
  MODEL_FEATURES = {
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_causal_attention": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens_wo": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens_wo": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights_wo": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_causal_attention_wo": FeatureConverter.FeatureSpec(
          dtype=tf.int32),
  }
  PACKING_FEATURE_DTYPES = {}

  def __init__(self,
               loss_on_targets_only = True,
               pack = False,
               use_custom_packing_ops = False,
               apply_length_check = True,
               bos_id = 0):
    assert not pack
    super().__init__(
        loss_on_targets_only=loss_on_targets_only,
        pack=pack,
        use_custom_packing_ops=use_custom_packing_ops,
        apply_length_check=apply_length_check,
        bos_id=bos_id)

  def _convert_example(
      self, features):
    def fn_truncate(feature, num_negatives):
      return tf.roll(tf.maximum(feature, 0), -num_negatives, axis=0)

    def fn_shift_and_reduce(feature, num_negatives):
      feature = tf.maximum(feature - num_negatives, 0)
      return tf.concat([feature[num_negatives:],
                        tf.zeros_like(feature[:num_negatives])], axis=0)
    feature_positive = {
        k: tf.abs(v) for k, v in features.items() if v.dtype != tf.string}
    d = super()._convert_example(feature_positive)

    num_negatives = tf.reduce_sum(tf.cast(features["targets"] < 0, tf.int32))
    inputs_width = fn_shift_and_reduce(features["inputs_width"], num_negatives)
    features_wo = {
        "targets": fn_truncate(features["targets"], num_negatives),
        "inputs_width": inputs_width,
        "inputs_width_add_pos": tf.where(inputs_width > 0, inputs_width + 1, 0)
    }
    d_wo = super()._convert_example(features_wo)
    d = {**d, **{f"{k}_wo": v for k, v in d_wo.items()}}
    return d

  def get_model_feature_lengths(
      self, task_feature_lengths):
    """Define the length relationship between task and model features."""
    ret = super().get_model_feature_lengths(task_feature_lengths)
    return {**ret, **{f"{k}_wo": v for k, v in ret.items()}}