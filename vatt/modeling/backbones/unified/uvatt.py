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

# Lint as: python3
"""Universal Video, Audio, and Text Transformer (UVATT)."""

import tensorflow as tf

from vatt.modeling.common import transformers


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


class UniversalVATT(tf.keras.layers.Layer):
  """The general Transformer for extracting different features for modalities."""

  def __init__(self,
               # pre-transformer parameters
               vid_temporal_patch_size=4,
               vid_spatial_patch_size=16,
               aud_temporal_patch_size=128,
               txt_vocab_size=2**16,
               txt_embedding_dim=300,
               txt_embedding_trainable=False,
               # video & audio input sampling
               random_patch_sampling=False,
               patch_sampling_rate=0.5,
               # transformer head parameters
               d_model=1024,
               d_kv=64,
               d_ff=4096,
               num_layers=24,
               num_heads=16,
               pre_norm=True,
               use_bias=True,
               activation="gelu",
               dropout_rate=0.1,
               layer_norm_epsilon=1e-6,
               # positional embedding parameters
               max_vid_temporal_buckets=8,
               max_vid_spatial_buckets=14,
               max_aud_temporal_buckets=1200,
               max_txt_temporal_buckets=16,
               # final head parameters
               d_post_proj=1024,
               post_proj_activation="gelu",
               name="unified_vat_transformer",
               **kwargs):
    super(UniversalVATT, self).__init__(name=name)
    self.d_model = d_model
    # define pre-tx projection
    self.raw_to_embeddings = {
        "video": tf.keras.layers.Conv3D(
            filters=d_model,
            kernel_size=(vid_temporal_patch_size,
                         vid_spatial_patch_size,
                         vid_spatial_patch_size),
            strides=(vid_temporal_patch_size,
                     vid_spatial_patch_size,
                     vid_spatial_patch_size),
            padding="valid",
            name="voxel_to_patch",
            ),
        "audio": tf.keras.layers.Conv1D(
            filters=d_model,
            kernel_size=aud_temporal_patch_size,
            strides=aud_temporal_patch_size,
            padding="valid",
            name="waveform_to_patch",
            ),
        "text": tf.keras.layers.Embedding(txt_vocab_size,
                                          txt_embedding_dim,
                                          trainable=txt_embedding_trainable,
                                          name="text_embedding")
    }
    self.pre_proj = {
        "video": tf.keras.layers.Dense(
            d_model,
            activation=activation,
            name="video_pre_tx_projection"
            ),
        "audio": tf.keras.layers.Dense(
            d_model,
            activation=activation,
            name="audio_pre_tx_projection"
            ),
        "text": tf.keras.layers.Dense(
            d_model,
            activation=activation,
            name="text_pre_tx_projection"
            ),}

    # define sampling-related params
    self.use_random_patches = random_patch_sampling
    self.patch_sampling_rate = patch_sampling_rate
    self.max_buckets = {
        "video": max_vid_temporal_buckets * (max_vid_spatial_buckets ** 2),
        "audio": max_aud_temporal_buckets,
    }
    self.max_num_patches = {
        "video": int(self.patch_sampling_rate * self.max_buckets["video"]),
        "audio": int(self.patch_sampling_rate * self.max_buckets["audio"]),
    }
    assert self.max_buckets["video"] > self.max_num_patches["video"], (
        "Max number of video positional buckets should be bigger than max"
        " number of video input patches"
        )
    assert self.max_buckets["audio"] > self.max_num_patches["audio"], (
        "Max number of audio positional buckets should be bigger than max"
        " number of audio input patches"
        )

    # define positional embedding module
    self.pos_embedding_lookup = {
        "video": transformers.SpatioTemporalEmbeddings(
            hidden_size=self.d_model,
            max_temporal_buckets=max_vid_temporal_buckets,
            max_vertical_buckets=max_vid_spatial_buckets,
            max_horizontal_buckets=max_vid_spatial_buckets,
            name="video_spatio_temporal_embeddings",
            ),
        "audio": transformers.TemporalEmbeddings(
            hidden_size=self.d_model,
            max_temporal_buckets=max_aud_temporal_buckets,
            name="audio_temporal_embeddings",
            ),
        "text": transformers.TemporalEmbeddings(
            hidden_size=self.d_model,
            max_temporal_buckets=max_txt_temporal_buckets,
            name="text_temporal_embeddings",
            ),
    }

    # define transformer head
    self.tx = transformers.TransformerEncoder(
        d_model=d_model,
        d_kv=d_kv,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        pre_norm=pre_norm,
        use_bias=use_bias,
        activation=activation,
        dropout_rate=dropout_rate,
        layer_norm_epsilon=layer_norm_epsilon,
        name="transformer",
        )

    # define post-tx projection head - it could be logits or embd space
    self.post_proj = {
        "video": tf.keras.layers.Dense(
            d_post_proj,
            activation=post_proj_activation,
            name="video_post_tx_projection"
            ),
        "audio": tf.keras.layers.Dense(
            d_post_proj,
            activation=post_proj_activation,
            name="audio_post_tx_projection"
            ),
        "text": tf.keras.layers.Dense(
            d_post_proj,
            activation=post_proj_activation,
            name="text_post_tx_projection"
            ),
    }

  def build(self, input_shapes):
    token_embds_kwargs = {
        "shape": (self.d_model,),
        "initializer": tf.keras.initializers.get("glorot_normal"),
        "trainable": True,
        "dtype": tf.float32,
    }

    # add modality-specific special token AGG
    self.agg_token = {
        "video": self.add_weight(
            name="vid_agg_embedding",
            **token_embds_kwargs,
            ),
        "audio": self.add_weight(
            name="aud_agg_embedding",
            **token_embds_kwargs,
            ),
        "text": self.add_weight(
            name="txt_agg_embedding",
            **token_embds_kwargs,
            ),
    }

  def _flatten_inputs(self,
                      inputs):

    input_shape = get_shape(inputs)
    bs = input_shape[0]
    d_embd = input_shape[-1]

    inputs = tf.reshape(inputs, [bs, -1, d_embd])
    return inputs, input_shape

  def _append_special_tokens(self,
                             inputs,
                             modality):

    batch_size = get_shape(inputs)[0]
    special_embd = self.agg_token[modality][None, None, :]

    # (batch_size, 1, d_model)
    special_embd = tf.tile(special_embd, [batch_size, 1, 1])

    return tf.concat([special_embd, inputs], axis=1)

  def _random_patch_selection(self,
                              inputs,
                              training,
                              input_shape,
                              modality):
    if training and modality != "text":
      # get inputs dimensions
      batch_size, seq_len, dim = get_shape(inputs)

      # shuffle on temporal axis and gather the first max_num_patches
      temporal_idx = tf.range(seq_len)
      temporal_idx = tf.random.shuffle(temporal_idx)[None, :]
      temporal_idx = tf.tile(temporal_idx, [batch_size, 1])

      batch_idx = tf.range(batch_size)[:, None]
      batch_idx = tf.tile(batch_idx, [1, seq_len])

      gather_idx = tf.stack([batch_idx, temporal_idx], axis=2)

      inputs = tf.gather_nd(inputs,
                            gather_idx)[:, :self.max_num_patches[modality], :]
      input_shape = [batch_size, self.max_num_patches[modality], dim]

    return inputs, input_shape

  def _extend_attn_mask(self,
                        attention_mask):
    attn_mask_shape = get_shape(attention_mask)
    if len(attn_mask_shape) > 2:
      raise NotImplementedError

    batch_size = attn_mask_shape[0]
    extention_mask = tf.ones((batch_size, 1), dtype=attention_mask.dtype)
    extended_attention_mask = tf.concat([extention_mask, attention_mask],
                                        axis=1)
    return extended_attention_mask

  def _modality_call(self,
                     inputs,
                     modality,
                     training=False,
                     attention_mask=None,
                     input_shape=None):

    # linear projection to d_model
    embeddings = self.raw_to_embeddings[modality](inputs)
    embeddings = self.pre_proj[modality](embeddings)

    # flatten inputs if not flattened already
    if input_shape is None:
      embeddings, input_shape = self._flatten_inputs(embeddings)
    else:
      is_flattened = len(get_shape(inputs)) == 3
      assert is_flattened, (
          "if input_shape provided, inputs should be flattened and have rank 3")

    # add modality-specific positional encoding embeddings
    embeddings = self.pos_embedding_lookup[modality](
        embeddings,
        input_shape,
        training
        )

    # randomly choose "max_num_patches" tokens
    if self.use_random_patches:
      embeddings, input_shape = self._random_patch_selection(
          embeddings,
          training,
          input_shape,
          modality,
          )

    # append modalities special tokens: [vid, aud, txt]
    tx_inputs = self._append_special_tokens(embeddings, modality)

    # extend attention_mask accordingly
    if attention_mask is not None:
      attention_mask = self._extend_attn_mask(attention_mask)

    # call Transformer
    tx_outputs = self.tx(tx_inputs,
                         attention_mask,
                         training)

    # get last hidden states and perform final linear projection
    last_hidden_states = tx_outputs["hidden_states"][-1]
    modality_outputs = self.post_proj[modality](last_hidden_states)
    output_shape = input_shape[:-1] + [get_shape(modality_outputs)[-1]]

    features_pooled = modality_outputs[:, 0, :]
    features = tf.reshape(modality_outputs[:, 1:, :], output_shape)

    # add token-level Transformer outputs
    outputs = {"features_pooled": features_pooled,
               "features": features}

    return outputs

  def call(self,
           inputs,
           training=False):
    outputs = {}

    for modality in ["video", "audio", "text"]:
      modality_inputs = inputs[modality]["data"]
      modality_attn_mask = inputs[modality].get("attention_mask", None)
      outputs[modality] = self._modality_call(inputs=modality_inputs,
                                              modality=modality,
                                              training=training,
                                              attention_mask=modality_attn_mask)

    return outputs
