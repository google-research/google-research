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

"""Tests for feature_converters."""
from seqio import test_utils
import tensorflow.compat.v2 as tf

from kl_guided_sampling import feature_converters


tf.compat.v1.enable_eager_execution()

assert_dataset = test_utils.assert_dataset
create_default_dataset = test_utils.create_default_dataset


class ContextualEncDecFeatureConverterTest(tf.test.TestCase):

  def test_encoder_decoder_unpacked_all_positive(self):
    x = [{"inputs": [9, 4, 3, 8, 1], "targets": [3, 9, 4, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 7, "targets": 5}

    converter = feature_converters.ContextualEncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [9, 4, 3, 8, 1, 0, 0],
        "encoder_input_tokens_wo": [9, 4, 3, 8, 1, 0, 0],
        "decoder_target_tokens": [3, 9, 4, 1, 0],
        "decoder_input_tokens": [0, 3, 9, 4, 1],
        "decoder_loss_weights": [1, 1, 1, 1, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_encoder_decoder_unpacked_some_negatives(self):
    x = [{"inputs": [-7, 8, 5, 1], "targets": [3, 9, 1]}]
    ds = create_default_dataset(x)
    task_feature_lengths = {"inputs": 10, "targets": 7}

    converter = feature_converters.ContextualEncDecFeatureConverter(pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "encoder_input_tokens": [7, 8, 5, 1, 0, 0, 0, 0, 0, 0],
        "encoder_input_tokens_wo": [8, 5, 1, 0, 0, 0, 0, 0, 0, 0],
        "decoder_target_tokens": [3, 9, 1, 0, 0, 0, 0],
        "decoder_input_tokens": [0, 3, 9, 1, 0, 0, 0],
        "decoder_loss_weights": [1, 1, 1, 0, 0, 0, 0],
    }
    assert_dataset(converted_ds, expected)


class ContextualPrefixLMFeatureConverter(tf.test.TestCase):

  def test_prefix_lm_unpacked_all_positive(self):
    x = [{"inputs": [9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 5, "targets": 4}
    converter = feature_converters.ContextualPrefixLMFeatureConverter(
        pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0],
        "decoder_target_tokens_wo": [9, 4, 6, 1, 3, 9, 1, 0, 0],
        "decoder_input_tokens_wo": [0, 9, 4, 6, 1, 3, 9, 1, 0],
        "decoder_loss_weights_wo": [0, 0, 0, 0, 1, 1, 1, 0, 0],
        "decoder_causal_attention_wo": [1, 1, 1, 1, 1, 0, 0, 0, 0],
    }
    assert_dataset(converted_ds, expected)

  def test_prefix_lm_unpacked_some_negatives(self):
    x = [{"inputs": [-9, 4, 6, 1], "targets": [3, 9, 1]}]
    ds = create_default_dataset(x)

    task_feature_lengths = {"inputs": 10, "targets": 4}
    converter = feature_converters.ContextualPrefixLMFeatureConverter(
        pack=False)
    converted_ds = converter(ds, task_feature_lengths)

    expected = {
        "decoder_target_tokens": [9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
        "decoder_input_tokens": [0, 9, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0],
        "decoder_loss_weights": [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        "decoder_causal_attention": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "decoder_target_tokens_wo": [4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "decoder_input_tokens_wo": [0, 4, 6, 1, 3, 9, 1, 0, 0, 0, 0, 0, 0, 0],
        "decoder_loss_weights_wo": [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "decoder_causal_attention_wo":
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    }
    assert_dataset(converted_ds, expected)


if __name__ == '__main__':
  tf.test.main()
