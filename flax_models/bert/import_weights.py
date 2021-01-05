# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Code for loading weights from a tensorflow checkpoint."""

from absl import logging
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf


def load_params_from_tf(init_checkpoint, d_model, num_heads,
                        num_classes=None, keep_masked_lm_head=False):
  """Return a jax parameter dict based on a tensorflow checkpoint."""

  print("Loading pre-trained weights from", init_checkpoint, flush=True)
  logging.info("Loading pre-trained weights from %s", init_checkpoint)
  ckpt = tf.train.load_checkpoint(init_checkpoint)
  tf_params = {k: ckpt.get_tensor(k) for k in ckpt.get_variable_to_dtype_map()}

  jax_params = {}
  # mapping between TF BERT and JAX model
  tf_key_to_jax_key = [
      # Output heads
      ("cls/seq_relationship/output_weights", "classification/kernel"),
      ("cls/seq_relationship/output_bias", "classification/bias"),
      ("cls/predictions/transform/LayerNorm",
       "predictions_transform_layernorm"),
      ("cls/predictions/transform/dense", "predictions_transform_dense"),
      ("cls/predictions/output_bias", "predictions_output/bias"),
      # Embeddings
      ("embeddings/word_embeddings", "word_embeddings/embedding"),
      ("embeddings/token_type_embeddings", "type_embeddings/embedding"),
      ("embeddings/position_embeddings", "position_embeddings/embedding"),
      ("embeddings/LayerNorm", "embeddings_layer_norm"),
      ("encoder/embedding_hidden_mapping_in", "embedding_hidden_mapping_in"),
      # Pooler
      ("pooler/dense/", "pooler/"),
      # Layers
      ("bert/encoder/transformer/group_0/inner_group_0",
       "bert/encoder_layer_0"),
      # ("bert/encoder/layer_", "bert/encoder_layer_"),
      ("attention_1/self", "self_attention/attn"),
      ("attention_1/output/dense", "self_attention/attn/output"),
      ("encoder_layer_0/LayerNorm/",
       "encoder_layer_0/self_attention_layer_norm/"),
      ("encoder_layer_0/LayerNorm_1/", "encoder_layer_0/output_layer_norm/"),
      ("ffn_1/intermediate/dense", "feed_forward/intermediate"),
      ("ffn_1/intermediate/output/dense", "feed_forward/output"),
      # Parameter names
      (":0", ""),
      ("beta", "bias"),
      ("gamma", "scale")
  ]
  for tf_key, val in tf_params.items():
    jax_key = tf_key
    for tf_name, jax_name in tf_key_to_jax_key:
      jax_key = jax_key.replace(tf_name, jax_name)

    # Reshape kernels if necessary
    jax_params[jax_key] = tf_params[tf_key]
    reshape_params = ["key", "query", "value"]
    for key in reshape_params:
      if "self_attention/attn/" + key + "/kernel" in jax_key:
        param = tf_params[tf_key]
        jax_params[jax_key] = np.swapaxes(
            param.reshape((d_model, num_heads, -1)), 0, 1)
      elif "self_attention/attn/" + key + "/bias" in jax_key:
        param = tf_params[tf_key]
        jax_params[jax_key] = param.reshape((num_heads, -1))
    if "self_attention/attn/output/kernel" in jax_key:
      param = tf_params[tf_key]
      jax_params[jax_key] = param.reshape((num_heads, -1, d_model))
    elif "self_attention/attn/output/bias" in jax_key:
      # The multihead attention implementation we use creates a bias vector for
      # each head, even though this is highly redundant.
      param = tf_params[tf_key]
      jax_params[jax_key] = np.stack(
          [param] + [np.zeros_like(param)] * (num_heads - 1), axis=0)

  # jax position embedding kernel has additional dimension
  pos_embedding = jax_params[
      "bert/position_embeddings/embedding"]
  jax_params[
      "bert/position_embeddings/embedding"] = pos_embedding[
          np.newaxis, ...]

  # this layer doesn't have parameters, but key is required to be present
  jax_params["GatherIndexes_0"] = {}

  # convert flat param dict into nested dict using `/` as delimeter
  outer_dict = {}
  for key, val in jax_params.items():
    tokens = key.split("/")
    inner_dict = outer_dict
    # each token except the very last should add a layer to the nested dict
    for token in tokens[:-1]:
      if token not in inner_dict:
        inner_dict[token] = {}
      inner_dict = inner_dict[token]
    inner_dict[tokens[-1]] = val

  if "global_step" in outer_dict:
    del outer_dict["global_step"]

  if not keep_masked_lm_head:
    del outer_dict["predictions_output"]
    del outer_dict["predictions_transform_dense"]
    del outer_dict["predictions_transform_layernorm"]

  # For some reason, using numpy arrays as weights doesn't cause a type error,
  # but instead leads to a shape discrepancy in some of the layers!
  outer_dict = jax.tree_map(jnp.asarray, outer_dict)

  if num_classes is not None:
    # Re-initialize the output head
    output_projection = outer_dict["classification"]
    output_projection["kernel"] = np.random.normal(
        scale=0.02,
        size=(num_classes, output_projection["kernel"].shape[1])).astype(
            output_projection["kernel"].dtype)
    output_projection["bias"] = np.zeros(
        num_classes, dtype=output_projection["bias"].dtype)

  return outer_dict
