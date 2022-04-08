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

"""Partial restore utils for pretraining or finetuning."""

import os
import pickle
import re

from absl import logging
import tensorflow as tf
import tensorflow_addons.image as tfa_image

_TEXT_EMBEDDINGS_DIR = "./misc/"

_KRS_TO_CKPT = [
    [r"^(.*)video_module/vit_(.*?)/(.*):0",
     (r"model/layer_with_weights-0/_base_layer/vid_backbone/_base/\3/.ATTRIBUTES/VARIABLE_VALUE")],

    [r"^(.*)audio_module/spt_(.*?)/(.*):0",
     (r"model/layer_with_weights-0/_base_layer/aud_backbone/_base/\3/.ATTRIBUTES/VARIABLE_VALUE")],

    [r"^(.*)/waveform_to_patch/(.*)",
     (r"\1/wave_to_patch/\2")],

    [r"^(.*)audio_module/wat_(.*?)/(.*):0",
     (r"model/layer_with_weights-0/_base_layer/aud_backbone/_base/\3/.ATTRIBUTES/VARIABLE_VALUE")],

    [r"^(.*)/spatio_temporal_embeddings/(.*)",
     r"\1/pos_embedding_lookup/\2"],

    [r"^(.*)/spectro_temporal_embeddings/(.*)",
     r"\1/pos_embedding_lookup/\2"],

    [r"^(.*)/temporal_embeddings/(.*)",
     r"\1/pos_embedding_lookup/\2"],

    [r"^(.*)/pos_embedding_lookup/layer_norm/(.*)",
     r"\1/pos_embedding_lookup/layernorm/\2"],

    [r"^(.*)/transformer/(.*)",
     r"\1/tx/\2"],

    [r"^(.*)/dense_relu_dense/(.*)",
     r"\1/mlp/\2"],

    [r"^(.*)/dense_gelu_dense/(.*)",
     r"\1/mlp/\2"],

    [r"^(.*)/dense_geglu_dense/(.*)",
     r"\1/mlp/\2"],

    [r"^(.*)/multi_head_attention/(.*)",
     r"\1/mha/\2"],

    [r"^(.*)/layer_([0-99]+)/(.*)",
     r"\1/layers/\2/\3"],
]

_KRS_TO_UT_CKPT = [
    [r"^(.*)/vit_(.*?)/agg_embedding:0",
     (r"\1/vit_\2/vid_agg_embedding:0")],

    [r"^(.*)/wat_(.*?)/agg_embedding:0",
     (r"\1/wat_\2/aud_agg_embedding:0")],

    [r"^(.*)/spt_(.*?)/agg_embedding:0",
     (r"\1/spt_\2/aud_agg_embedding:0")],

    [r"^(.*)/vit_(.*?)/pre_tx_projection/(.*?)",
     (r"\1/vit_\2/pre_proj/video/\3")],

    [r"^(.*)/vit_(.*?)/post_tx_projection/(.*?)",
     (r"\1/vit_\2/post_proj/video/\3")],

    [r"^(.*)/wat_(.*?)/pre_tx_projection/(.*?)",
     (r"\1/wat_\2/pre_proj/audio/\3")],

    [r"^(.*)/wat_(.*?)/post_tx_projection/(.*?)",
     (r"\1/wat_\2/post_proj/audio/\3")],

    [r"^(.*)/spt_(.*?)/pre_tx_projection/(.*?)",
     (r"\1/spt_\2/pre_proj/audio/\3")],

    [r"^(.*)/spt_(.*?)/post_tx_projection/(.*?)",
     (r"\1/spt_\2/post_proj/audio/\3")],

    [r"^(.*)video_module/vit_(.*?)/(.*):0",
     (r"model/layer_with_weights-0/_base_layer/unified_backbone/unified_transformer/\3/.ATTRIBUTES/VARIABLE_VALUE")],

    [r"^(.*)audio_module/spt_(.*?)/(.*):0",
     (r"model/layer_with_weights-0/_base_layer/unified_backbone/unified_transformer/\3/.ATTRIBUTES/VARIABLE_VALUE")],

    [r"^(.*)audio_module/wat_(.*?)/(.*):0",
     (r"model/layer_with_weights-0/_base_layer/unified_backbone/unified_transformer/\3/.ATTRIBUTES/VARIABLE_VALUE")],

    [r"^(.*)/voxel_to_patch/(.*)",
     (r"\1/raw_to_embeddings/video/\2")],

    [r"^(.*)/waveform_to_patch/(.*)",
     (r"\1/raw_to_embeddings/audio/\2")],

    [r"^(.*)/spectrum_to_patch/(.*)",
     (r"\1/raw_to_embeddings/audio/\2")],

    [r"^(.*)/spatio_temporal_embeddings/(.*)",
     r"\1/pos_embedding_lookup/video/\2"],

    [r"^(.*)/temporal_embeddings/(.*)",
     r"\1/pos_embedding_lookup/audio/\2"],

    [r"^(.*)/spectro_temporal_embeddings/(.*)",
     r"\1/pos_embedding_lookup/audio/\2"],

    [r"^(.*)/pos_embedding_lookup/(.*)/layer_norm/(.*)",
     r"\1/pos_embedding_lookup/\2/layernorm/\3"],

    [r"^(.*)/transformer/(.*)",
     r"\1/tx/\2"],

    [r"^(.*)/dense_relu_dense/(.*)",
     r"\1/mlp/\2"],

    [r"^(.*)/dense_gelu_dense/(.*)",
     r"\1/mlp/\2"],

    [r"^(.*)/dense_geglu_dense/(.*)",
     r"\1/mlp/\2"],

    [r"^(.*)/multi_head_attention/(.*)",
     r"\1/mha/\2"],

    [r"^(.*)/layer_([0-99]+)/(.*)",
     r"\1/layers/\2/\3"],
]

_KRS_TO_TSM_CKPT = [
    [r"^(.*)video_module/tsm/(.*):0",
     r"model/layer_with_weights-0/_base_layer/vid_backbone/_base/\2/.ATTRIBUTES/VARIABLE_VALUE"],

    [r"^(.*)/post_batch_norm/(.*)",
     r"\1/post_bn/\2"],

    [r"^(.*)/pre_batch_norm/(.*)",
     r"\1/pre_bn/\2"],

    [r"^(.*)/res_batch_norm_([0-99]+)/(.*)",
     r"\1/res_bn_\2/\3"],

    [r"^(.*)/shortcut_conv/(.*)",
     r"\1/projection/\2"],

    [r"^(.*)/tsm_block_([0-99]+)/(.*)",
     r"\1/tsm_blocks/\2/\3"],

    [r"^(.*)/unit_([0-99]+)/(.*)",
     r"\1/tsm_units/\2/\3"],
]


def interpolate_pos(source_weights, target_shape):
  """Interpolate missing points in the new pos embeddings."""
  source_buckets = source_weights.shape[0]
  lookup_keys = tf.range(source_buckets)
  available_buckets = lookup_keys / source_buckets
  available_buckets = tf.cast(available_buckets, tf.float32)[None, :, None]

  # define all possible target buckets
  # shape = [1, target_buckets, 1]
  target_buckets = target_shape[0]
  query_buckets = tf.range(target_buckets) / target_buckets
  query_buckets = tf.cast(query_buckets, tf.float32)[None, :, None]

  # fetch current available embeddings
  # shape = [1, source_buckets, embd_dim]
  available_embeddings = source_weights[None, Ellipsis]

  expanded_embeddings = tf.squeeze(tfa_image.interpolate_spline(
      train_points=available_buckets,
      train_values=available_embeddings,
      query_points=query_buckets,
      order=3), axis=0)
  logging.info("Positional embeddings interpolated from %s to %s",
               source_weights.shape, target_shape)
  return expanded_embeddings


def convert_keras_name_to_ckpt(krs_name):
  for source, dest in _KRS_TO_CKPT:
    krs_name = re.sub(source, dest, krs_name)
  return krs_name


def convert_keras_name_to_ut_ckpt(krs_name):
  for source, dest in _KRS_TO_UT_CKPT:
    krs_name = re.sub(source, dest, krs_name)
  return krs_name


def convert_keras_name_to_tsm_ckpt(krs_name):
  for source, dest in _KRS_TO_TSM_CKPT:
    krs_name = re.sub(source, dest, krs_name)
  return krs_name


def assign_weight_from_ckpt(layer, ckpt_path):
  """Convert Keras model name to saved checkpoint name and restore."""
  ckpt_reader = tf.train.load_checkpoint(ckpt_path)
  ckpt_names = [v[0] for v in tf.train.list_variables(ckpt_path)]
  is_unified = any(["unified_backbone" in name for name in ckpt_names])
  is_tsm = any(["tsm_blocks" in name for name in ckpt_names])

  skipped = []
  for krs_w in layer.weights:
    krs_name = krs_w.name
    if is_unified:
      ckpt_name = convert_keras_name_to_ut_ckpt(krs_name)
    elif is_tsm:
      ckpt_name = convert_keras_name_to_tsm_ckpt(krs_name)
    else:
      ckpt_name = convert_keras_name_to_ckpt(krs_name)
    if ckpt_name in ckpt_names:
      ckpt_weight = ckpt_reader.get_tensor(ckpt_name)
      if ckpt_weight.shape == krs_w.shape:
        krs_w.assign(ckpt_weight)
      elif "pos_embedding_lookup" in ckpt_name:
        krs_w.assign(interpolate_pos(ckpt_weight, krs_w.shape))
      else:
        skipped.append(krs_name)
    else:
      skipped.append(krs_name)

  return skipped


def assign_word_embeddings(embedding_layer, embedding_name):
  path = os.path.join(_TEXT_EMBEDDINGS_DIR, embedding_name + ".pkl")
  with open(path, "rb") as f:
    embedding_weights = pickle.load(f)["word_embeddings"]
  embedding_layer.set_weights([embedding_weights])
