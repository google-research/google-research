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

"""Model configs CoCa-VILA."""

import dataclasses
import typing
from typing import Any, Dict, Union

import numpy as np
from praxis import base_layer
from praxis import layers
from praxis import pax_fiddle
from praxis import py_utils
from praxis.layers import transformers

from vila import coca_vila
from vila import coca_vila_layers


NestedMap = py_utils.NestedMap
WeightInit = base_layer.WeightInit


@dataclasses.dataclass(slots=True)
class CocaVilaConfig:
  """Canonical params to build a CoCa-VILA model and its configs."""

  model_type: type[Any] = coca_vila.CoCaVilaPretrain

  model_dims: int = 768
  ff_hidden_dims: int = 768 * 4
  num_heads: int = 12
  atten_logit_cap: float = 50.0

  # Encoder
  image_size: int = 224
  patch_size: int = 16
  num_encoder_layers: int = 12

  # Decoder
  num_unimodal_layers: int = 6
  num_decoder_layers: int = 12
  text_vocab_size: int = 0
  decoding_max_len: int = 64

  # Loss
  temperature: float = 0.07
  generative_loss_weight: Union[float, Dict[str, float]] = 2.0
  contrastive_loss_weight: Union[float, Dict[str, float]] = 1.0

  def _init_transformer_layers(
      self, tfm_p
  ):
    """Initializes a transformer layer."""
    if isinstance(tfm_p.transformer_layer_params_tpl, (list, tuple)):
      tfm_xformer_p_lst = tfm_p.transformer_layer_params_tpl
    else:
      tfm_xformer_p_lst = [tfm_p.transformer_layer_params_tpl]
    for tfm_xformer_p in tfm_xformer_p_lst:
      tfm_xformer_p = typing.cast(
          pax_fiddle.Config[transformers.Transformer], tfm_xformer_p
      )
      tfm_xformer_p.tr_atten_tpl.set(  # pytype: disable=attribute-error
          atten_logit_cap=self.atten_logit_cap,
          internal_enable_per_dim_scale=False,
          params_init=WeightInit.Xavier(2 ** (-0.5)),
      )
      tfm_xformer_p.tr_fflayer_tpl.set(  # pytype: disable=attribute-error
          params_init=WeightInit.Xavier(2 ** (-0.5))
      )

  def encoder_transformer_layers(
      self,
  ):
    """Builds the transformer layer."""
    transformer_common_hparams = {
        'model_dims': self.model_dims,
        'hidden_dims': self.ff_hidden_dims,
        'num_heads': self.num_heads,
        'mask_self_attention': False,
        'use_cross_attention': False,
        'packed_input': False,
        'num_layers': self.num_encoder_layers,
    }

    p_stacked_tfm = pax_fiddle.Config(
        layers.StackedTransformer,
        **transformer_common_hparams,
    )

    p_tfm = typing.cast(
        pax_fiddle.Config[layers.Transformer],
        p_stacked_tfm.transformer_layer_params_tpl,
    )

    ln_tpl = pax_fiddle.Config(layers.LayerNorm, use_bias=True)

    p_tfm.ln_tpl = ln_tpl.clone()
    p_tfm.tr_fflayer_tpl.ln_tpl = ln_tpl.clone()
    p_tfm.norm_policy = 'pre'

    p_tfm.tr_fflayer_tpl.activation_tpl = pax_fiddle.Config(
        layers.GELU, approximate=False
    )

    p_tfm.tr_fflayer_tpl.use_gated_activation = False
    p_tfm.tr_fflayer_tpl.has_bias = True
    p_tfm.tr_fflayer_tpl.fflayer_tpl.has_bias = True

    p_tfm.tr_atten_tpl.atten_logit_cap = self.atten_logit_cap
    p_tfm.tr_atten_tpl.combine_qkv = False
    p_tfm.tr_atten_tpl.internal_enable_per_dim_scale = False
    p_tfm.tr_atten_tpl.use_bias = True
    p_tfm.tr_atten_tpl.use_rotary_position_emb = False

    return p_stacked_tfm

  def encoder(self):
    """Builds the encoder for CoCa."""
    pos_emb_shape = self.image_size // self.patch_size
    pos_emb_shapes = (pos_emb_shape, pos_emb_shape)
    num_patches = np.prod(pos_emb_shapes)
    pos_emb_tpl = pax_fiddle.Config(
        layers.TrainablePositionalEmbedding,
        max_seq_length=num_patches,
        embedding_dims=self.model_dims,
        params_init=base_layer.WeightInit.Gaussian(scale=0.02),
    )
    entry_layers_tpl = pax_fiddle.Config(
        layers.VitEntryLayers,
        name='entry',
        pos_emb_shapes=pos_emb_shapes,
        patch_size=self.patch_size,
        input_dims=self.patch_size**2 * 3,
        output_dims=self.model_dims,
        pos_emb_tpl=pos_emb_tpl,
    )

    exit_layers_tpl = pax_fiddle.Config(
        layers.VitExitLayers,
        name='exit',
        hidden_dim=self.model_dims,
        output_dim=self.model_dims,
        output_dropout_prob=0.0,
        pooled=False,
        output_fc_has_bias=True,
        output_fc_tanh=False,
    )
    p_vit = pax_fiddle.Config(
        layers.VisionTransformer,
        name='vit',
        entry_layers_tpl=entry_layers_tpl,
        transformer_layers_tpl=self.encoder_transformer_layers(),
        exit_layers_tpl=exit_layers_tpl,
    )

    return p_vit

  def decoder(self):
    """Builds the decoder for CoCa."""
    decoder_p = pax_fiddle.Config(
        coca_vila_layers.MultimodalDecoder,
        model_dims=self.model_dims,
        num_heads=self.num_heads,
        num_unimodal_layers=self.num_unimodal_layers,
        num_decoder_layers=self.num_decoder_layers,
        decoder_vocab_size=self.text_vocab_size,
    )

    # Init the unimodal and crossmodal decoder layers
    self._init_transformer_layers(decoder_p.unimodal_tr_tpl)
    self._init_transformer_layers(decoder_p.crossmodal_tr_tpl)
    decoder_p.num_class_tokens = 1
    return decoder_p

  def contrastive_img_pooler(
      self,
  ):
    """Builds the contrastive image pooler for CoCa."""
    pool_p = pax_fiddle.Config(
        coca_vila_layers.AttenTokenPoolingLayer,
        input_dims=self.model_dims,
        ff_hidden_dims=self.ff_hidden_dims,
        num_heads=self.num_heads,
        num_queries=1,
    )
    return pool_p

  def generative_img_pooler(
      self,
  ):
    """Builds the generative image pooler for CoCa."""
    num_img_generative_enc_queries = (self.image_size // self.patch_size) ** 2

    pool_p = pax_fiddle.Config(
        coca_vila_layers.AttenTokenPoolingLayer,
        input_dims=self.model_dims,
        ff_hidden_dims=self.ff_hidden_dims,
        num_heads=self.num_heads,
        num_queries=num_img_generative_enc_queries,
    )

    return pool_p

  def contrastive_loss_layer(
      self,
  ):
    """Builds the contrastive loss layer for CoCa."""
    loss_p = pax_fiddle.Config(
        coca_vila_layers.ContrastiveLossLayer,
        temperature_val=self.temperature,
    )
    return loss_p


def build_coca_vila_model(
    coca_config,
):
  """Builds the CoCa model."""
  coca_p = pax_fiddle.Config(coca_config.model_type, name='coca_vila')
  coca_p.generative_loss_weight = coca_config.generative_loss_weight
  coca_p.contrastive_loss_weight = coca_config.contrastive_loss_weight
  coca_p.decoding_max_len = coca_config.decoding_max_len

  coca_p.encoder_tpl = coca_config.encoder()
  coca_p.decoder_tpl = coca_config.decoder()
  coca_p.contrastive_img_pooler_tpl = coca_config.contrastive_img_pooler()
  coca_p.generative_img_pooler_tpl = coca_config.generative_img_pooler()
  coca_p.contrastive_loss_layer_tpl = coca_config.contrastive_loss_layer()

  return coca_p
