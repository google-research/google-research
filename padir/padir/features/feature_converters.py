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

"""PaDIR feature converters."""

from typing import Mapping
import seqio
import tensorflow.compat.v2 as tf

EncDecFeatureConverter = seqio.feature_converters.EncDecFeatureConverter
FeatureConverter = seqio.feature_converters.FeatureConverter


class PadirEncDecFeatureConverter(EncDecFeatureConverter):
  """Feature converter for a NAR encoder-decoder architecture."""

  TASK_FEATURES = {
      "inputs": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "targets_masked": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "noise_mask": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }
  MODEL_FEATURES = {
      "encoder_input_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_target_tokens": FeatureConverter.FeatureSpec(dtype=tf.int32),
      "decoder_input_tokens_train": FeatureConverter.FeatureSpec(
          dtype=tf.int32
      ),
      "decoder_input_tokens_infer": FeatureConverter.FeatureSpec(
          dtype=tf.int32
      ),
      "decoder_loss_weights": FeatureConverter.FeatureSpec(dtype=tf.int32),
  }

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if self.pack:
      raise ValueError(
          "'pack=True' is not allowed in PadirEncDecFeatureConverter."
      )

  def _convert_example(
      self, features
  ):
    """Convert a seq2seq example into an example with model features."""
    decoder_input_tokens_infer = tf.zeros_like(features["targets"], tf.int32)
    return {
        "encoder_input_tokens": features["inputs"],
        "decoder_target_tokens": features["targets"],
        "decoder_input_tokens_train": features["targets_masked"],
        "decoder_input_tokens_infer": decoder_input_tokens_infer,
        "decoder_loss_weights": features["decoder_loss_weights"],
    }

  def get_model_feature_lengths(
      self, task_feature_lengths
  ):
    """Define the length relationship between input and output features."""
    encoder_length = task_feature_lengths["inputs"]
    decoder_length = task_feature_lengths["targets"]
    return {
        "encoder_input_tokens": encoder_length,
        "decoder_target_tokens": decoder_length,
        "decoder_input_tokens_train": decoder_length,
        "decoder_input_tokens_infer": decoder_length,
        "decoder_loss_weights": decoder_length,
    }
