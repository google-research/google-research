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

"""Vision Transformer model implemented in Jax/Flax."""

import jax
from jax import lax
import numpy as np

from imp.max.core import constants
from imp.max.core import utils
import imp.max.modeling as mnn
from imp.max.utils import typing

Modality = constants.Modality
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
VISION = Modality.VISION
INPUTS = DataFeatureType.INPUTS
OUTPUTS = DataFeatureType.OUTPUTS
ENCODER = DataFeatureRoute.ENCODER
LABEL_CLASSIFIER = DataFeatureRoute.LABEL_CLASSIFIER
TOKEN_RAW = DataFeatureName.TOKEN_RAW
TOKEN_MASK = DataFeatureName.TOKEN_MASK
TOKEN_COORDINATE = DataFeatureName.TOKEN_COORDINATE
LOGITS = DataFeatureName.LOGITS


class ViT(mnn.Model):
  """The standard Vision Transformer as in https://arxiv.org/abs/2010.11929."""

  # Input config
  batch_size: int
  image_size: tuple[int, int, int, int]  # (frames, height, width, channels)
  patch_size: tuple[int, int, int]  # (temporal, spatial, spatial)
  # Input sharding annotations
  pos_encode_embed_shardings: typing.ShardingAxes
  pos_encode_layernorm_shardings: typing.ShardingAxes
  token_raw_to_embed_kernel_shardings: typing.ShardingAxes
  tokens_shardings: typing.ShardingAxes
  # Encoder config
  d_model: int
  d_ff: int
  num_heads: int
  num_layers: int
  use_bias: bool
  dropout_rate: float
  remat: str
  scanned_layers: bool
  scan_axis: int
  dtype: jax.typing.DTypeLike
  qk_layernorm: bool
  precision: typing.Precision
  lora_rank: int
  lora_scale: float
  approximate_gelu: bool
  # Backbone sharding annotations
  scan_sharding_axis: str | None
  layernorm_shardings: typing.ShardingAxes
  mha_qkv_kernel_shardings: typing.ShardingAxes
  mha_out_kernel_shardings: typing.ShardingAxes
  mha_activation_shardings: typing.ShardingAxes
  ffn_inner_kernel_shardings: typing.ShardingAxes
  ffn_outer_kernel_shardings: typing.ShardingAxes
  ffn_intermediate_shardings: typing.ShardingAxes
  # Post-encoder config
  d_post_proj: int | None
  post_proj_position: str | None
  num_classes: int | None
  head_bias_init: float
  aggregation_type: str
  positional_embedding: str

  def setup(self):
    if self.positional_embedding not in ('learned_1d', 'learned_3d'):
      raise ValueError(
          f'Unsupported positional_embedding {self.positional_embedding}')
    # TODO(b/228698851): remove dependency on input_size
    self.pos_buckets = tuple([
        int(i/p) for i, p in zip(self.image_size, self.patch_size)
    ])

    self.raw_to_embeddings = mnn.TokenRawToEmbed(
        modality=VISION,
        d_model=self.d_model,
        pos_buckets=(np.prod(self.pos_buckets) if self.positional_embedding
                     == 'learned_1d' else self.pos_buckets),
        dropout_rate=self.dropout_rate,
        freeze_embeddings=False,
        raw_to_embed_dense_dot_general=lax.dot_general,
        pos_encode_lookup_dot_general=lax.dot_general,
        pos_encode_embed_shardings=self.pos_encode_embed_shardings,
        pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
        raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
        dtype=self.dtype,
        name='vision_raw_to_embed')

    if self.aggregation_type == constants.AggregationType.SPECIAL_TOKEN:
      self.add_cls_token = mnn.SpecialToken(
          features=self.d_model,
          extension=constants.Extension.PREPEND,
          activation_shardings=self.tokens_shardings,
          dtype=self.dtype,
          name='cls_token')

    self.encoder = mnn.TransformerEncoder(
        d_model=self.d_model,
        d_ff=self.d_ff,
        num_heads=self.num_heads,
        num_layers=self.num_layers,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        remat=self.remat,
        scanned_layers=self.scanned_layers,
        dtype=self.dtype,
        scan_axis=self.scan_axis,
        qk_layernorm=self.qk_layernorm,
        scan_sharding_axis=self.scan_sharding_axis,
        layernorm_shardings=self.layernorm_shardings,
        mha_qkv_kernel_shardings=self.mha_qkv_kernel_shardings,
        mha_out_kernel_shardings=self.mha_out_kernel_shardings,
        mha_activation_shardings=self.mha_activation_shardings,
        ffn_inner_kernel_shardings=self.ffn_inner_kernel_shardings,
        ffn_outer_kernel_shardings=self.ffn_outer_kernel_shardings,
        ffn_intermediate_shardings=self.ffn_intermediate_shardings,
        mha_qkv_dot_general=jax.lax.dot_general,
        mha_out_dot_general=jax.lax.dot_general,
        mha_einsum_dot_general=jax.lax.dot_general,
        ffn_dot_general=jax.lax.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        approximate_gelu=self.approximate_gelu,
        name='transformer_encoder',
    )

    self.post_encoder = mnn.VitPostEncoderHead(
        aggregation_type=self.aggregation_type,
        d_post_proj=self.d_post_proj,
        post_proj_position=self.post_proj_position,
        num_classes=self.num_classes,
        head_bias_init=self.head_bias_init,
        dtype=self.dtype,
        num_heads=self.num_heads,
        d_ff=self.d_ff,
        dropout_rate=self.dropout_rate,
        use_bias=self.use_bias,
        pre_logits_kernel_shardings=(),
        logits_kernel_shardings=(),
        post_proj_dot_general=jax.lax.dot_general,
        logits_dot_general=jax.lax.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        name='post_encoder')

  def get_rng_keys(self):
    """Returns keys of all rngs defined under this model."""

    keys = ()
    if self.dropout_rate > 0.:
      keys += ('dropout',)

    return keys

  def get_data_signature(self):
    """Returns the input signature to fully initialize this model."""
    num_pixels = np.prod(self.patch_size) * self.image_size[-1]
    num_tokens = np.prod(
        utils.get_patched_shape(self.image_size[:-1], self.patch_size))
    token_raw = jax.random.uniform(
        jax.random.key(0), (self.batch_size, 1, num_tokens, num_pixels))
    data = {
        INPUTS: {
            ENCODER: {
                VISION: {
                    TOKEN_RAW: token_raw
                },
            },
        },
    }
    return data

  def __call__(self,
               data,
               deterministic = True):

    inputs = data[INPUTS][ENCODER][
        VISION
    ]
    token_raw = inputs[TOKEN_RAW]
    token_coordinate = inputs.get(TOKEN_COORDINATE, None)
    token_mask = inputs.get(TOKEN_MASK, None)

    if token_raw.shape[2] == np.prod(self.pos_buckets):
      patched_shape = (
          token_raw.shape[:2] + self.pos_buckets + (token_raw.shape[-1],))
    else:
      # Either droptoken is applied or resolution has been changed
      patched_shape = None

    # Project raw tokens to a sequence of embedding vectors
    token_embds = self.raw_to_embeddings(token_raw, deterministic,
                                         token_coordinate)

    # We don't support relative biases yet
    attention_bias = None

    if self.aggregation_type == constants.AggregationType.SPECIAL_TOKEN:
      token_embds, token_mask, attention_bias = self.add_cls_token(
          token_embds, token_mask, attention_bias)

    if token_mask is not None:
      attention_mask = utils.create_attention_mask(
          token_mask, token_mask, dtype=token_mask.dtype)
    else:
      attention_mask = None

    encoder_outputs = self.encoder(
        inputs=token_embds,
        attention_mask=attention_mask,
        attention_bias=attention_bias,
        deterministic=deterministic)

    outputs = self.post_encoder(
        inputs=encoder_outputs,
        patched_shape=patched_shape,
        deterministic=deterministic)

    # TODO(b/309897474): Fix ViT head's outputs to avoid this boilerplate
    if LOGITS in outputs:
      data = utils.deep_update_data(
          data=data,
          update_data={
              OUTPUTS: {
                  LABEL_CLASSIFIER: {
                      VISION: {
                          TOKEN_RAW: {
                              LOGITS: outputs[LOGITS],
                          },
                      },
                  },
              },
          },
      )
    data = utils.deep_update_data(
        data=data,
        update_data={
            OUTPUTS: {
                ENCODER: {
                    VISION: {
                        TOKEN_RAW: outputs,
                    },
                },
            },
        },
    )
    return data
