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

"""Utility functions for manipulating official Tensorflow BERT encoders."""
import tensorflow as tf
import tensorflow_models as tfm
from tensorflow_privacy.privacy.fast_gradient_clipping import gradient_clipping_utils


def dedup_bert_encoder(input_bert_encoder):
  """Deduplicates the layer names in a BERT encoder."""

  def make_get_new_config(layer_name):
    def get_new_config(sublayer, suffix):
      sublayer_config = sublayer.get_config()
      sublayer_config["name"] = layer_name + suffix
      return sublayer_config

    return get_new_config

  for layer in input_bert_encoder.layers:
    if isinstance(layer, tfm.nlp.layers.TransformerEncoderBlock):
      get_new_config = make_get_new_config(layer.name)
      # Inner Dropout deduplication.
      # pylint: disable=protected-access
      if layer._inner_dropout_layer is not None:
        new_config = get_new_config(
            layer._inner_dropout_layer, "/inner_dropout_layer"
        )
        layer._inner_dropout_layer = layer._inner_dropout_layer.from_config(
            new_config
        )
      # Attention's Outer Dropout deduplication.
      if layer._attention_dropout is not None:
        new_config = get_new_config(
            layer._attention_dropout, "/outer_dropout_layer"
        )
        layer._attention_dropout = layer._attention_dropout.from_config(
            new_config
        )
      # Attention deduplication.
      if layer._attention_layer is not None:  # pylint: disable=protected-access
        new_config = get_new_config(layer._attention_layer, "/attention_layer")
        layer._attention_layer = layer._attention_layer.from_config(new_config)
        # Attention's Inner Dropout deduplication.
        if layer._attention_layer._dropout_layer is not None:
          new_config = get_new_config(
              layer._attention_layer._dropout_layer,
              "/attention_inner_dropout_layer",
          )
          layer._attention_layer._dropout_layer = (
              layer._attention_layer._dropout_layer.from_config(new_config)
          )
      # Attention Layer Norm deduplication.
      if layer._attention_layer_norm is not None:
        new_config = get_new_config(
            layer._attention_layer_norm, "/attention_layer_norm"
        )
        layer._attention_layer_norm = layer._attention_layer_norm.from_config(
            new_config
        )
      # Intermediate Dense deduplication.
      if layer._intermediate_dense is not None:
        new_config = get_new_config(
            layer._intermediate_dense, "/intermediate_dense"
        )
        layer._intermediate_dense = layer._intermediate_dense.from_config(
            new_config
        )
      # Activation deduplication.
      if layer._intermediate_activation_layer is not None:
        # This is one of the few times that we cannot build from a config, due
        # to the presence of lambda functions.
        policy = tf.keras.mixed_precision.global_policy()
        if policy.name == "mixed_bfloat16":
          policy = tf.float32
        layer._intermediate_activation_layer = tf.keras.layers.Activation(
            layer._inner_activation,
            dtype=policy,
            name=layer.name + "/intermediate_activation_layer",
        )
      # Output deduplication.
      if layer._output_dense is not None:
        new_config = get_new_config(layer._output_dense, "/output_dense")
        layer._output_dense = layer._output_dense.from_config(new_config)
      # Output's Dropout deduplication.
      if layer._output_dropout is not None:
        new_config = get_new_config(layer._output_dropout, "/output_dropout")
        layer._output_dropout = layer._output_dropout.from_config(new_config)
      # Output's Layer Norm deduplication.
      if layer._output_layer_norm is not None:
        new_config = get_new_config(
            layer._output_layer_norm, "/output_layer_norm"
        )
        layer._output_layer_norm = layer._output_layer_norm.from_config(
            new_config
        )
      # pylint: enable=protected-access


def get_unwrapped_bert_encoder(
    input_bert_encoder,
):
  """Creates a new BERT encoder whose layers are core Keras layers."""
  dedup_bert_encoder(input_bert_encoder)
  core_test_outputs = (
      gradient_clipping_utils.generate_model_outputs_using_core_keras_layers(
          input_bert_encoder,
          custom_layer_set={tfm.nlp.layers.TransformerEncoderBlock},
      )
  )
  return tf.keras.Model(
      inputs=input_bert_encoder.inputs,
      outputs=core_test_outputs,
  )
