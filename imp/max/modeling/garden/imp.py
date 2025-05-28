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

"""IMP model implemented in Jax/Flax."""

import collections
import functools
from typing import Any

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from imp.max.core import constants
from imp.max.core import utils
from imp.max.data import config as data_config
import imp.max.modeling as mnn
from imp.max.utils import typing

AggregationType = constants.AggregationType
CommonSpace = constants.CommonSpace
DecodingMode = constants.DecodingMode
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
TaskFlowName = constants.TaskFlowName
Modality = constants.Modality


class BaseIntegratedMultimodalModel(mnn.Model):
  """Base class for Integrated Multimodal variants."""

  # Input params
  input_batch_size: int
  vision_input_size: tuple[int, int, int, int]
  vision_patch_size: tuple[int, int, int]
  vision_vocab_size: int
  vision_embed_size: int
  waveform_input_size: int
  waveform_patch_size: int
  waveform_vocab_size: int
  waveform_embed_size: int
  spectrogram_input_size: tuple[int, int]
  spectrogram_patch_size: tuple[int, int]
  spectrogram_vocab_size: int
  spectrogram_embed_size: int
  text_input_size: int
  text_vocab_size: int
  text_embed_size: int
  # Input sharding annotations
  pos_encode_embed_shardings: typing.ShardingAxes
  pos_encode_layernorm_shardings: typing.ShardingAxes
  token_raw_to_embed_kernel_shardings: typing.ShardingAxes
  token_id_to_embed_kernel_shardings: typing.ShardingAxes
  tokens_shardings: typing.ShardingAxes
  # Backbone params
  d_model: int
  d_ff: int
  num_heads: int
  use_bias: bool
  qk_layernorm: bool
  dropout_rate: int
  remat: str
  scanned_layers: bool
  scan_axis: int
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
  # Common space params
  common_space_type: str
  d_common: int
  aggregation_type: str
  # Classification heads on aggregated features
  vision_classes: typing.ClassificationHead
  waveform_classes: typing.ClassificationHead
  spectrogram_classes: typing.ClassificationHead
  text_classes: typing.ClassificationHead
  # Target heads on sequences of features
  vision_targets: typing.ClassificationHead
  waveform_targets: typing.ClassificationHead
  spectrogram_targets: typing.ClassificationHead
  text_targets: typing.ClassificationHead
  # Initialization Override
  init_override: str | None
  freeze_embeddings: tuple[str, Ellipsis]
  temperature_init: float
  # Misc
  precision: typing.Precision
  dtype: jax.typing.DTypeLike

  def setup_backbone(self):
    raise NotImplementedError

  def setup(self):
    # TODO(hassanak): remove dependency on input_size
    self.vision_pos_buckets = tuple([
        int(i/p) for i, p in zip(self.vision_input_size,
                                 self.vision_patch_size)
    ])
    self.waveform_pos_buckets = int(
        self.waveform_input_size/self.waveform_patch_size)
    self.spectrogram_pos_buckets = tuple([
        int(i/p) for i, p in zip(self.spectrogram_input_size,
                                 self.spectrogram_patch_size)
    ])
    self.text_pos_buckets = self.text_input_size

    self.raw_to_embeddings = {
        Modality.VISION: {
            DataFeatureName.TOKEN_RAW: mnn.TokenRawToEmbed(
                modality=Modality.VISION,
                d_model=self.d_model,
                pos_buckets=self.vision_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=Modality.VISION in self.freeze_embeddings,
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='vision_token_raw_to_embed',
            ),
            DataFeatureName.TOKEN_ID: mnn.TokenIdToEmbed(
                modality=Modality.VISION,
                d_model=self.d_model,
                vocab_size=self.vision_vocab_size,
                pos_buckets=self.vision_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=Modality.VISION in self.freeze_embeddings,
                id_to_embed_lookup_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                id_to_embed_kernel_shardings=self.token_id_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='vision_token_id_to_embed',
            ),
            DataFeatureName.TOKEN_EMBED: mnn.TokenRawToEmbed(
                modality=Modality.VISION,
                d_model=self.d_model,
                pos_buckets=self.vision_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=Modality.VISION in self.freeze_embeddings,
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='vision_token_embed_to_embed',
            ),
        },
        Modality.WAVEFORM: {
            DataFeatureName.TOKEN_RAW: mnn.TokenRawToEmbed(
                modality=Modality.WAVEFORM,
                d_model=self.d_model,
                pos_buckets=self.waveform_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=Modality.WAVEFORM in self.freeze_embeddings,
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='waveform_token_raw_to_embed',
            ),
            DataFeatureName.TOKEN_ID: mnn.TokenIdToEmbed(
                modality=Modality.WAVEFORM,
                d_model=self.d_model,
                vocab_size=self.waveform_vocab_size,
                pos_buckets=self.waveform_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=Modality.WAVEFORM in self.freeze_embeddings,
                id_to_embed_lookup_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                precision=self.precision,
                dtype=self.dtype,
                name='waveform_token_id_to_embed',
            ),
            DataFeatureName.TOKEN_EMBED: mnn.TokenRawToEmbed(
                modality=Modality.WAVEFORM,
                d_model=self.d_model,
                pos_buckets=self.waveform_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=Modality.WAVEFORM in self.freeze_embeddings,
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='waveform_token_embed_to_embed',
            ),
        },
        Modality.SPECTROGRAM: {
            DataFeatureName.TOKEN_RAW: mnn.TokenRawToEmbed(
                modality=Modality.SPECTROGRAM,
                d_model=self.d_model,
                pos_buckets=self.spectrogram_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=(
                    Modality.SPECTROGRAM in self.freeze_embeddings),
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='spectrogram_token_raw_to_embed',
            ),
            DataFeatureName.TOKEN_ID: mnn.TokenIdToEmbed(
                modality=Modality.SPECTROGRAM,
                d_model=self.d_model,
                vocab_size=self.spectrogram_vocab_size,
                pos_buckets=self.spectrogram_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=(
                    Modality.SPECTROGRAM in self.freeze_embeddings),
                id_to_embed_lookup_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                precision=self.precision,
                dtype=self.dtype,
                name='spectrogram_token_id_to_embed',
            ),
            DataFeatureName.TOKEN_EMBED: mnn.TokenRawToEmbed(
                modality=Modality.SPECTROGRAM,
                d_model=self.d_model,
                pos_buckets=self.spectrogram_pos_buckets,
                dropout_rate=self.dropout_rate,
                freeze_embeddings=(
                    Modality.SPECTROGRAM in self.freeze_embeddings),
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                dtype=self.dtype,
                name='spectrogram_token_embed_to_embed',
            ),
        },
        Modality.TEXT: {
            DataFeatureName.TOKEN_RAW: mnn.TokenRawToEmbed(
                modality=Modality.TEXT,
                d_model=self.d_model,
                pos_buckets=self.text_pos_buckets,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                freeze_embeddings=Modality.TEXT in self.freeze_embeddings,
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                name='text_token_raw_to_embed',
            ),
            DataFeatureName.TOKEN_ID: mnn.TokenIdToEmbed(
                modality=Modality.TEXT,
                d_model=self.text_embed_size,
                vocab_size=self.text_vocab_size,
                pos_buckets=self.text_pos_buckets,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                freeze_embeddings=Modality.TEXT in self.freeze_embeddings,
                id_to_embed_lookup_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                precision=self.precision,
                name='text_token_id_to_embed',
            ),
            DataFeatureName.TOKEN_EMBED: mnn.TokenRawToEmbed(
                modality=Modality.TEXT,
                d_model=self.d_model,
                pos_buckets=self.text_pos_buckets,
                dropout_rate=self.dropout_rate,
                dtype=self.dtype,
                freeze_embeddings=Modality.TEXT in self.freeze_embeddings,
                raw_to_embed_dense_dot_general=lax.dot_general,
                pos_encode_lookup_dot_general=lax.dot_general,
                pos_encode_embed_shardings=self.pos_encode_embed_shardings,
                pos_encode_layernorm_shardings=self.pos_encode_layernorm_shardings,
                raw_to_embed_kernel_shardings=self.token_raw_to_embed_kernel_shardings,
                precision=self.precision,
                name='text_token_embed_to_embed',
            ),
        },
    }

    if self.aggregation_type == AggregationType.SPECIAL_TOKEN:
      self.add_agg_token = utils.construct_per_modality_per_feature_modules(
          module=mnn.SpecialToken,
          modalities=sorted(self.supported_modalities),
          feature_keys=(DataFeatureName.TOKEN_RAW,
                        DataFeatureName.TOKEN_ID,
                        DataFeatureName.TOKEN_EMBED),
          common_kwargs={
              'features': self.d_model,
              'extension': constants.Extension.PREPEND,
              'activation_shardings': self.tokens_shardings,
              'dtype': self.dtype,
          },
          per_modality_per_feature_kwargs={},
          name='agg_token',
      )

    self.setup_backbone()

    self.aggregator_head = mnn.NonParametricAggregatorHead(
        aggregation_type=self.aggregation_type)

    common_space_kwargs = {
        'd_common': self.d_common,
        'd_hidden': self.d_common,
        'dot_general': lax.dot_general,
        'precision': self.precision,
        'lora_rank': self.lora_rank,
        'lora_scale': self.lora_scale,
        'target_modalities': sorted(self.supported_modalities),
        'dtype': self.dtype,
    }
    if self.common_space_type == CommonSpace.DISJOINT:
      self.cross_modal_head = utils.construct_per_modality_per_feature_modules(
          mnn.DisjointCommonSpace,
          modalities=sorted(self.supported_modalities),
          feature_keys=(DataFeatureName.TOKEN_RAW,
                        DataFeatureName.TOKEN_ID,
                        DataFeatureName.TOKEN_EMBED),
          common_kwargs=common_space_kwargs,
          per_modality_per_feature_kwargs={},
          name='disjoint_projection',
      )
    elif self.common_space_type == CommonSpace.JOINT:
      self.cross_modal_head = utils.construct_per_modality_per_feature_modules(
          mnn.JointCommonSpace,
          modalities=sorted(self.supported_modalities),
          feature_keys=(DataFeatureName.TOKEN_RAW,
                        DataFeatureName.TOKEN_ID,
                        DataFeatureName.TOKEN_EMBED),
          common_kwargs=common_space_kwargs,
          per_modality_per_feature_kwargs={},
          name='joint_projection',
      )
    else:
      raise ValueError(f'Unsupported common space {self.common_space_type}')

    # Construct the per-modality per-feature label and target classifiers
    def _fetch_modality_classes_kwargs(modality, classes):
      kwargs = {
          modality: {
              DataFeatureName.TOKEN_RAW: {'classes': classes},
              DataFeatureName.TOKEN_ID: {'classes': classes},
              DataFeatureName.TOKEN_EMBED: {'classes': classes},
          }
      }
      return kwargs
    modality_class_target = (
        (Modality.VISION,
         self.vision_classes,
         self.vision_targets),
        (Modality.WAVEFORM,
         self.waveform_classes,
         self.waveform_targets),
        (Modality.SPECTROGRAM,
         self.spectrogram_classes,
         self.spectrogram_targets),
        (Modality.TEXT,
         self.text_classes,
         self.text_targets),
    )
    lbl_cls_per_mod_per_feat_kwargs = {}
    tgt_cls_per_mod_per_feat_kwargs = {}
    for modality, classes, targets in modality_class_target:
      lbl_cls_per_mod_per_feat_kwargs.update(
          _fetch_modality_classes_kwargs(modality, classes)
      )
      tgt_cls_per_mod_per_feat_kwargs.update(
          _fetch_modality_classes_kwargs(modality, targets)
      )

    classifier_module = functools.partial(
        utils.construct_per_modality_per_feature_modules,
        module=mnn.Classifier,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'predictions_key': DataFeatureName.LOGITS,
            'dot_general': lax.dot_general,
            'precision': self.precision,
            'dtype': self.dtype,
        },
    )
    self.label_classifier = classifier_module(
        per_modality_per_feature_kwargs=lbl_cls_per_mod_per_feat_kwargs,
        name='label_classifier',
    )
    self.target_classifier = classifier_module(
        per_modality_per_feature_kwargs=tgt_cls_per_mod_per_feat_kwargs,
        name='target_classifier',
    )

    self.temperature = mnn.PerModalityTemperature(
        init_value=self.temperature_init,
        modalities=sorted(self.supported_modalities),
    )

  def get_rng_keys(self):
    """Returns keys of all rngs defined under this model."""

    keys = ()
    if self.dropout_rate > 0.:
      keys += ('dropout',)

    return keys

  def _prepare_encoder_inputs(
      self,
      inputdata,
      inputflow,
      deterministic,
  ):
    """Prepares inputs according to inputflow for the encoder.

    Args:
      inputdata: A nested dictionary containing the per-modality inputs,
        features, targets, etc. as leaves.
      inputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys as leaves.
      deterministic: A bool indicating whether the stochastic modules should
        operate deterministically or stochastically.

    Returns:
      concatenated_token_embed: An array containing token embeddings that are
        concatenated across modalities and their specific features as defined
        in `intputflow`.
      concatenated_token_mask: An array containing token masks that are
        concatenated across modalities and their specific features as defined
        in `intputflow`.
      seq_lengths: A nested dictionary that maps the modalities and their
        features (whose embeddings are present in `concatenated_token_embed`) to
        their corresponding sequence lengths.
      concatenation_order: A tuple of string pairs indicating the order in which
        different features of different modalities are concatenated. Each pair
        is presented as a tuple of strings as (modality, feature_name). These,
        along with `seq_lengths` are later used to split the features at the
        output of the encoder model.

    Raises:
      ValueError: if the modalities present in `inputflow` are not a subset of
        the modalities supported in this model.
    """
    inputflow_modalities = set(inputflow.keys())
    if not inputflow_modalities.issubset(self.supported_modalities):
      raise ValueError(
          'One or all of the modalities in the input metadata are not '
          f'supported by this model. Received {inputflow_modalities},'
          f'  while all supported modalities are {self.supported_modalities}.'
      )
    seq_lengths = collections.defaultdict(dict)
    concatenation_order = ()
    all_token_embeds = []
    all_token_masks = []
    for modality in inputflow:
      # Fetch proper inputs
      token_mask = inputdata[modality].get(DataFeatureName.TOKEN_MASK, None)
      token_coordinate = inputdata[modality].get(
          DataFeatureName.TOKEN_COORDINATE, None
      )

      # TODO(b/276944964): Move coordinate_scale to data pipeline.
      if modality == Modality.TEXT and token_coordinate is not None:
        # In text, we scale the normalized coordinates to max_seq_length to
        # avoid dilated pos_embedding gather.
        coordinate_scale = token_coordinate.shape[-1]
      else:
        coordinate_scale = None

      for token_feature_name in inputflow[modality]:
        # Fetch the pre-embed token data
        token_pre_embed = inputdata[modality][token_feature_name]

        # Project raw inputs to a sequence of embedding vectors
        token_embed = self.raw_to_embeddings[modality][token_feature_name](
            token_pre_embed, deterministic, token_coordinate, coordinate_scale)

        # Apply linear projection before feeding to Transformer
        token_embed = self.pre_encoder_proj[modality][token_feature_name](
            token_embed
        )

        if self.aggregation_type == AggregationType.SPECIAL_TOKEN:
          (token_embed,
           token_mask, _) = self.add_agg_token[modality][token_feature_name](
               token_embed, token_mask, None
           )

        # Fetch the sequence length to use later when separating cross-modal
        # output features
        token_seq_length = token_embed.shape[2]

        # Collect all necessary information
        seq_lengths[modality][token_feature_name] = token_seq_length
        concatenation_order += ((modality, token_feature_name),)
        all_token_embeds.append(token_embed)
        all_token_masks.append(token_mask)

    # Concatenate all token embeddings and masks
    (concatenated_token_embed,
     concatenated_token_mask) = utils.concatenate_cross_modal_tokens(
         token_embeds=all_token_embeds,
         token_masks=all_token_masks,
         concatenated_token_embed_shardings=self.tokens_shardings,
         contrarenated_token_mask_shardings=self.tokens_shardings[:-1],
     )

    return (
        concatenated_token_embed,
        concatenated_token_mask,
        seq_lengths,
        concatenation_order,
    )

  def _prepare_encoder_outputs(
      self,
      concatenated_features,
      concatenated_token_mask,
      seq_lengths,
      concatenation_order,
      outputflow,
  ):
    """Prepares outputs according to outputflow for the encoder.

    Args:
      concatenated_features: An array that contains the outputs of the encoder
        (given a concatenated multimodal inputs.)
      concatenated_token_mask: An optional array containing the concatenated
        token masks.
      seq_lengths: A nested dictionary that maps the modalities and their
        features (whose embeddings are present in `concatenated_token_embed`) to
        their corresponding sequence lengths.
      concatenation_order: A tuple of string pairs indicating the order in which
        different features of different modalities are concatenated. Each pair
        is presented as a tuple of strings as (modality, feature_name). These,
        along with `seq_lengths` is used to split the features.
      outputflow: A nested dictionary containing the per-modality feature keys
        as leaves.

    Returns:
      A nested dictionary with a structure similar to `outputflow` that contains
      the modality-specific (but still cross-modal) features as leaves.

    Raises:
      ValueError: if the modalities present in `inputflow` are not a subset of
        the modalities supported in this model.
    """
    outputdata = collections.defaultdict(dict)
    split_features = utils.split_cross_modal_tokens(
        concatenated_token_features=concatenated_features,
        seq_lengths=seq_lengths,
        concatenation_order=concatenation_order,
        split_token_features_shardings=self.tokens_shardings,
        dot_general=lax.dot_general,
    )
    if concatenated_token_mask is None:
      split_token_mask = jax.tree_util.tree_map(lambda x: None, split_features)
    else:
      split_token_mask = utils.split_cross_modal_tokens(
          concatenated_token_features=concatenated_token_mask,
          seq_lengths=seq_lengths,
          concatenation_order=concatenation_order,
          split_token_features_shardings=self.tokens_shardings[:-1],
          dot_general=lax.dot_general,
      )
    for modality in outputflow:
      for token_feature_name in outputflow[modality]:
        features = split_features[modality][token_feature_name]
        token_mask = split_token_mask[modality][token_feature_name]
        features = self.post_encoder_proj[modality][token_feature_name](
            features
        )
        # TODO(b/241931055): add feature_map reshape and MAP aggregation
        features = self.aggregator_head(inputs=features, token_mask=token_mask)

        # Add this feature to the outputdata
        outputdata[modality][token_feature_name] = features

    return {DataFeatureRoute.ENCODER: dict(outputdata)}

  def _encoder_heads_call(
      self,
      encoder_outputdata,
      outputflow,
      deterministic,
  ):
    raise NotImplementedError

  def _encoder_call(
      self,
      inputdata,
      inputflow,
      outputflow,
      deterministic,
  ):
    """A method to call the encoder based on specific sets of data.

    Args:
      inputdata: A nested dictionary containing the per-modality inputs,
        features, targets, etc. as leaves.
      inputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys as leaves.
      outputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys that are supposed to be
        included in the output data of the model.
      deterministic: A bool indicating whether the stochastic modules should
        operate deterministically or stochastically.

    Returns:
      A nested dictionary with a similar structure to the inputdata, but
      containing the model features as leaves.

    Raises:
      ValueError: if the modalities present in `inputflow` are not a subset of
        the modalities supported in this model.
    """

    # Prepare the encoder inputs based on the dataflow
    (token_embed,
     token_mask,
     seq_lengths,
     concatenation_order) = self._prepare_encoder_inputs(
         inputdata=inputdata,
         inputflow=inputflow,
         deterministic=deterministic)

    # We don't support relative biases yet
    attention_bias = None

    if token_mask is not None:
      attention_mask = utils.create_attention_mask(
          token_mask, token_mask, dtype=token_mask.dtype)
    else:
      attention_mask = None

    transformer_metadata = {'modality': '_'.join(inputflow.keys())}
    features = self.encoder(inputs=token_embed,
                            attention_mask=attention_mask,
                            attention_bias=attention_bias,
                            deterministic=deterministic,
                            metadata=transformer_metadata)

    outputdata = self._prepare_encoder_outputs(
        concatenated_features=features,
        concatenated_token_mask=token_mask,
        seq_lengths=seq_lengths,
        concatenation_order=concatenation_order,
        outputflow=outputflow,
    )
    return outputdata

  def _prepare_decoder_inputs(
      self,
      inputdata,
      inputflow,
      decoding_mode,
      deterministic,
      project_to_embeds,
  ):
    """Prepares inputs according to inputflow for the decoder.

    Args:
      inputdata: A nested dictionary containing the per-modality inputs,
        features, targets, etc. as leaves.
      inputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys as leaves.
      decoding_mode: A string indicating the decoding style. All supported
        decoding styles can be found in constants.DecodingMode.
      deterministic: A bool indicating whether the stochastic modules should
        operate deterministically or stochastically.
      project_to_embeds: A bool indicating whether to project the inputs to an
        embedding space or not.

    Returns:
      concatenated_token_embed: An array containing token embeddings that are
        concatenated across modalities and their specific features as defined
        in `intputflow`.
      concatenated_token_mask: An array containing token masks that are
        concatenated across modalities and their specific features as defined
        in `intputflow`.
      seq_lengths: A nested dictionary that maps the modalities and their
        features (whose embeddings are present in `concatenated_token_embed`) to
        their corresponding sequence lengths.
      concatenation_order: A tuple of string pairs indicating the order in which
        different features of different modalities are concatenated. Each pair
        is presented as a tuple of strings as (modality, feature_name). These,
        along with `seq_lengths` are later used to split the features at the
        output of the encoder model.
    """
    raise NotImplementedError

  def _prepare_decoder_outputs(
      self,
      concatenated_features,
      concatenated_token_mask,
      seq_lengths,
      concatenation_order,
      outputflow,
  ):
    """Prepares outputs according to outputflow for the decoder.

    Args:
      concatenated_features: An array that contains the outputs of the decoder
        (given a concatenated multimodal inputs.)
      concatenated_token_mask: An optional array containing the concatenated
        token masks.
      seq_lengths: A nested dictionary that maps the modalities and their
        features (whose embeddings are present in `concatenated_token_embed`) to
        their corresponding sequence lengths.
      concatenation_order: A tuple of string pairs indicating the order in which
        different features of different modalities are concatenated. Each pair
        is presented as a tuple of strings as (modality, feature_name). These,
        along with `seq_lengths` is used to split the features.
      outputflow: A nested dictionary containing the per-modality feature keys
        as leaves.

    Returns:
      A nested dictionary with a structure similar to `outputflow` that contains
      the modality-specific (but still cross-modal) features as leaves.
    """
    raise NotImplementedError

  def _decoder_heads_call(
      self,
      decoder_outputdata,
      outputflow,
      deterministic,
  ):
    raise NotImplementedError

  def _decoder_call(
      self,
      inputdata,
      cross_inputdata,
      inputflow,
      cross_inputflow,
      outputflow,
      decoding_mode,
      deterministic,
  ):
    """The abstract decoder call method.

    Args:
      inputdata: A nested dictionary containing the per-modality inputs,
        features, targets, etc. as leaves.
      cross_inputdata: A nested dictionary containing the per-modality cross-
        inputs, cross-features, etc. as leaves.
      inputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys as leaves.
      cross_inputflow: A nested dictionary with a similar structure to the
        cross_inputdata, but containing the per-modality feature keys as leaves.
      outputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys that are supposed to be
        included in the output data of the model.
      decoding_mode: A string indicating the decoding style. All supported
        decoding styles can be found in constants.DecodingMode.
      deterministic: A bool indicating whether the stochastic modules should
        operate deterministically or stochastically.
    """
    raise NotImplementedError

  def __call__(self,
               data,
               deterministic = True):
    raise NotImplementedError


class IMP(BaseIntegratedMultimodalModel):
  """Integrated Multimodal Percption (IMP)."""

  num_layers: int
  d_post_proj: int

  def setup_backbone(self):
    self.pre_encoder_proj = utils.construct_per_modality_per_feature_modules(
        mnn.Dense,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'features': self.d_model,
            'use_bias': True,
            'kernel_shardings': (),
            'dot_general': jax.lax.dot_general,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_scale': self.lora_scale,
            'dtype': self.dtype,
        },
        per_modality_per_feature_kwargs={},
        name='pre_encoder_projection',
    )

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

    self.post_encoder_proj = utils.construct_per_modality_per_feature_modules(
        mnn.Dense,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'features': self.d_post_proj,
            'use_bias': True,
            'kernel_shardings': (),
            'dot_general': jax.lax.dot_general,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_scale': self.lora_scale,
            'dtype': self.dtype,
        },
        per_modality_per_feature_kwargs={},
        name='post_encoder_projection',
    )

  def get_data_signature(self):
    """Returns the input signature required to fully initialize this model."""

    vision_patched_shape = utils.get_patched_shape(
        self.vision_input_size[:-1], self.vision_patch_size)
    waveform_patched_shape = utils.get_patched_shape(
        self.waveform_input_size, self.waveform_patch_size)
    spectrogram_patched_shape = utils.get_patched_shape(
        self.spectrogram_input_size, self.spectrogram_patch_size)

    num_vision_pixels = np.prod(
        self.vision_patch_size) * self.vision_input_size[-1]
    num_waveform_samples = self.waveform_patch_size
    num_spectrogram_samples = np.prod(self.spectrogram_patch_size)

    num_vision_tokens = np.prod(vision_patched_shape)
    num_waveform_tokens = np.prod(waveform_patched_shape)
    num_spectrogram_tokens = np.prod(spectrogram_patched_shape)
    num_text_tokens = self.text_input_size

    vision_token_id = jax.random.randint(
        jax.random.key(0),
        (self.input_batch_size, 1, num_vision_tokens),
        minval=0, maxval=self.vision_vocab_size)
    vision_token_raw = jax.random.uniform(
        jax.random.key(1),
        (self.input_batch_size, 1, num_vision_tokens, num_vision_pixels))
    vision_token_embed = jax.random.uniform(
        jax.random.key(2),
        (self.input_batch_size, 1, num_vision_tokens, self.vision_embed_size))
    waveform_token_id = jax.random.randint(
        jax.random.key(3),
        (self.input_batch_size, 1, num_waveform_tokens),
        minval=0, maxval=self.waveform_vocab_size)
    waveform_token_raw = jax.random.uniform(
        jax.random.key(4),
        (self.input_batch_size, 1, num_waveform_tokens, num_waveform_samples))
    waveform_token_embed = jax.random.uniform(
        jax.random.key(5),
        (self.input_batch_size, 1,
         num_waveform_tokens, self.waveform_embed_size))
    spectrogram_token_id = jax.random.randint(
        jax.random.key(6),
        (self.input_batch_size, 1, num_spectrogram_tokens),
        minval=0, maxval=self.spectrogram_vocab_size)
    spectrogram_token_raw = jax.random.uniform(
        jax.random.key(7),
        (self.input_batch_size, 1,
         num_spectrogram_tokens, num_spectrogram_samples))
    spectrogram_token_embed = jax.random.uniform(
        jax.random.key(8),
        (self.input_batch_size, 1,
         num_spectrogram_tokens, self.spectrogram_embed_size))
    text_token_id = jax.random.randint(
        jax.random.key(9),
        (self.input_batch_size, 1, num_text_tokens),
        minval=0, maxval=self.text_vocab_size)
    text_token_embed = jax.random.uniform(
        jax.random.key(10),
        (self.input_batch_size, 1, num_text_tokens, self.text_embed_size))
    text_token_mask = jax.random.randint(
        jax.random.key(11),
        (self.input_batch_size, 1, num_text_tokens),
        minval=0, maxval=2)

    vision_token_coordinate = jnp.tile(utils.construct_3d_positions(
        *vision_patched_shape), (self.input_batch_size, 1, 1, 1))
    waveform_token_coordinate = jnp.tile(utils.construct_1d_positions(
        *waveform_patched_shape), (self.input_batch_size, 1, 1))
    spectrogram_token_coordinate = jnp.tile(utils.construct_2d_positions(
        *spectrogram_patched_shape), (self.input_batch_size, 1, 1, 1))
    text_token_coordinate = jnp.tile(utils.construct_1d_positions(
        num_text_tokens), (self.input_batch_size, 1, 1))

    data = {
        DataFeatureType.INPUTS: {
            DataFeatureRoute.ENCODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_RAW:
                        vision_token_raw,
                    DataFeatureName.TOKEN_ID:
                        vision_token_id,
                    DataFeatureName.TOKEN_EMBED:
                        vision_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        vision_token_coordinate,
                },
                Modality.WAVEFORM: {
                    DataFeatureName.TOKEN_RAW:
                        waveform_token_raw,
                    DataFeatureName.TOKEN_ID:
                        waveform_token_id,
                    DataFeatureName.TOKEN_EMBED:
                        waveform_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        waveform_token_coordinate,
                },
                Modality.SPECTROGRAM: {
                    DataFeatureName.TOKEN_RAW:
                        spectrogram_token_raw,
                    DataFeatureName.TOKEN_ID:
                        spectrogram_token_id,
                    DataFeatureName.TOKEN_EMBED:
                        spectrogram_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        spectrogram_token_coordinate,
                },
                Modality.TEXT: {
                    DataFeatureName.TOKEN_ID:
                        text_token_id,
                    DataFeatureName.TOKEN_EMBED:
                        text_token_embed,
                    DataFeatureName.TOKEN_MASK:
                        text_token_mask,
                    DataFeatureName.TOKEN_COORDINATE:
                        text_token_coordinate,
                },
            },
        },
    }
    return data

  def get_default_metadata(self):
    """Constructs the default metadata with which this model can be called.

    The constructed `metadata` contains `dataflow` field that is not used in JAX
    transformations. Hence, only used for guiding the data consumption and
    objective function calculation later in the exeuction pipeline. This field
    contains a tuple of nested dictionaries each with a similar structure to
    data that the model consumes. Each dictionary contains the per-modality
    feature keys that the model is supposed to either consume or create (e.g.
    the outputdata). Each dictionary also represent a 'forward pass'.

    Returns:
      An instance of Metadata that contains the dataflow that could be used to
      fully initialize all of the model weights.
    """
    dataflow = ()
    for modality in self.supported_modalities:
      common_space_targets = tuple(
          sorted(self.supported_modalities - {modality})
      )
      if modality == Modality.TEXT:
        token_feature_names = (DataFeatureName.TOKEN_ID,
                               DataFeatureName.TOKEN_EMBED)
      else:
        token_feature_names = (DataFeatureName.TOKEN_RAW,
                               DataFeatureName.TOKEN_ID,
                               DataFeatureName.TOKEN_EMBED)

      for token_feature_name in token_feature_names:
        datapass = (
            {
                DataFeatureType.INPUTS: {
                    DataFeatureRoute.ENCODER: {
                        modality: {
                            token_feature_name: None,
                        },
                    },
                },
                DataFeatureType.OUTPUTS: {
                    DataFeatureRoute.ENCODER: {
                        modality: {
                            token_feature_name: DataFeatureName.FEATURES,
                        },
                    },
                    DataFeatureRoute.LABEL_CLASSIFIER: {
                        modality: {
                            token_feature_name: DataFeatureName.LOGITS,
                        },
                    },
                    DataFeatureRoute.TARGET_CLASSIFIER: {
                        modality: {
                            token_feature_name: DataFeatureName.LOGITS,
                        },
                    },
                    DataFeatureRoute.COMMON_SPACE: {
                        modality: {
                            token_feature_name: common_space_targets,
                        },
                    },
                },
                DataFeatureType.TARGETS: {
                    DataFeatureRoute.LABEL_CLASSIFIER: {
                        modality: DataFeatureName.LABEL,
                    },
                },
                DataFeatureType.HYPERPARAMS: {
                    DataFeatureRoute.ENCODER: {
                        modality: {
                            token_feature_name: DataFeatureName.TEMPERATURE,
                        },
                    },
                },
            },
        )
        dataflow += datapass
    default_metadata = data_config.Metadata(
        dataflow=dataflow,
        taskflow=(),
    )
    return default_metadata

  def _encoder_heads_call(
      self,
      encoder_outputdata,
      outputflow,
      deterministic,
  ):
    encoder_outputdata = encoder_outputdata[DataFeatureRoute.ENCODER]
    heads_outputdata = {}
    for route in outputflow:
      route_outputdata = collections.defaultdict(dict)
      for modality in outputflow[route]:
        for token_feature_name in outputflow[route][modality]:
          encoder_features = encoder_outputdata[modality][token_feature_name]
          if route == DataFeatureRoute.COMMON_SPACE:
            head_inputs = encoder_features[DataFeatureName.FEATURES_AGG]
            target_modalities = outputflow[route][modality][token_feature_name]
            route_outputdata[modality][token_feature_name] = (
                self.cross_modal_head[modality][token_feature_name](
                    inputs=head_inputs,
                    target_modalities=target_modalities,
                    deterministic=deterministic,
                )
            )

          elif route == DataFeatureRoute.LABEL_CLASSIFIER:
            head_inputs = encoder_features[DataFeatureName.FEATURES_AGG]
            route_outputdata[modality][token_feature_name] = (
                self.label_classifier[modality][token_feature_name](
                    inputs=head_inputs,
                )
            )

          elif route == DataFeatureRoute.TARGET_CLASSIFIER:
            head_inputs = encoder_features[DataFeatureName.FEATURES]
            route_outputdata[modality][token_feature_name] = (
                self.target_classifier[modality][token_feature_name](
                    inputs=head_inputs,
                )
            )

      if route_outputdata:
        heads_outputdata[route] = dict(route_outputdata)

    return heads_outputdata

  def __call__(self,
               data,
               deterministic = True):
    metadata = data.get(DataFeatureType.METADATA, self.get_default_metadata())
    outputdata = {}
    for dataflow in metadata.dataflow:
      # Fetch all the dataflow information
      inputflow = dataflow[DataFeatureType.INPUTS]
      outputflow = dataflow[DataFeatureType.OUTPUTS]
      encoder_inputflow = inputflow[DataFeatureRoute.ENCODER]
      encoder_outputflow = outputflow[DataFeatureRoute.ENCODER]

      # Fetch the inputs data
      encoder_inputdata = data[DataFeatureType.INPUTS][DataFeatureRoute.ENCODER]

      # Check if the dataflow exists in the inputdata
      utils.verify_flow_exists_in_data(
          flow=encoder_inputflow,
          data=encoder_inputdata,
      )

      # Call the Transformer encoder
      encoder_outputdata = self._encoder_call(
          inputdata=encoder_inputdata,
          inputflow=encoder_inputflow,
          outputflow=encoder_outputflow,
          deterministic=deterministic,
      )

      # Call the heads
      heads_outputdata = self._encoder_heads_call(
          encoder_outputdata=encoder_outputdata,
          outputflow=outputflow,
          deterministic=deterministic,
      )

      # Update the outputdata with encoder and heads outputs
      outputdata = utils.deep_update_data(outputdata, encoder_outputdata)
      outputdata = utils.deep_update_data(outputdata, heads_outputdata)

    hyperparams = ({
        DataFeatureRoute.ENCODER: {
            DataFeatureName.TEMPERATURE: self.temperature(),
        },
    })

    data.update({
        DataFeatureType.OUTPUTS: outputdata,
        DataFeatureType.HYPERPARAMS: hyperparams,
    })
    return data


class IMPeGe(BaseIntegratedMultimodalModel):
  """Integrated Multimodal Percption and Generation (IMPeGe)."""

  num_encoder_layers: int
  num_decoder_layers: int
  d_post_encoder_proj: int
  d_post_decoder_proj: int

  def setup_backbone(self):
    self.pre_encoder_proj = utils.construct_per_modality_per_feature_modules(
        mnn.Dense,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'features': self.d_model,
            'use_bias': True,
            'kernel_shardings': (),
            'dot_general': jax.lax.dot_general,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_scale': self.lora_scale,
            'dtype': self.dtype,
        },
        per_modality_per_feature_kwargs={},
        name='pre_encoder_projection',
    )
    self.encoder = mnn.TransformerEncoder(
        d_model=self.d_model,
        d_ff=self.d_ff,
        num_heads=self.num_heads,
        num_layers=self.num_encoder_layers,
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
    self.post_encoder_proj = utils.construct_per_modality_per_feature_modules(
        mnn.Dense,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'features': self.d_post_encoder_proj,
            'use_bias': True,
            'kernel_shardings': (),
            'dot_general': jax.lax.dot_general,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_scale': self.lora_scale,
            'dtype': self.dtype,
        },
        per_modality_per_feature_kwargs={},
        name='post_encoder_projection',
    )
    self.post_encoder_mask_filler = (
        utils.construct_per_modality_per_feature_modules(
            mnn.MaskFiller,
            modalities=sorted(self.supported_modalities),
            feature_keys=(
                DataFeatureName.TOKEN_RAW,
                DataFeatureName.TOKEN_ID,
                DataFeatureName.TOKEN_EMBED,
            ),
            common_kwargs={
                'dim': self.d_model,
                'embedding_shardings': (),
                'scatter_dot_general': jax.lax.dot_general,
                'precision': self.precision,
                'dtype': self.dtype,
            },
            per_modality_per_feature_kwargs={},
            name='post_encoder_mask_filler',
        )
    )

    self.pre_decoder_proj = utils.construct_per_modality_per_feature_modules(
        mnn.Dense,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'features': self.d_model,
            'use_bias': True,
            'kernel_shardings': (),
            'dot_general': jax.lax.dot_general,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_scale': self.lora_scale,
            'dtype': self.dtype,
        },
        per_modality_per_feature_kwargs={},
        name='pre_decoder_projection',
    )
    self.decoder = mnn.TransformerDecoder(
        d_model=self.d_model,
        d_ff=self.d_ff,
        num_heads=self.num_heads,
        num_layers=self.num_decoder_layers,
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
        name='transformer_decoder',
    )
    self.post_decoder_proj = utils.construct_per_modality_per_feature_modules(
        mnn.Dense,
        modalities=sorted(self.supported_modalities),
        feature_keys=(DataFeatureName.TOKEN_RAW,
                      DataFeatureName.TOKEN_ID,
                      DataFeatureName.TOKEN_EMBED),
        common_kwargs={
            'features': self.d_post_decoder_proj,
            'use_bias': True,
            'kernel_shardings': (),
            'dot_general': jax.lax.dot_general,
            'precision': self.precision,
            'lora_rank': self.lora_rank,
            'lora_scale': self.lora_scale,
            'dtype': self.dtype,
        },
        per_modality_per_feature_kwargs={},
        name='post_decoder_projection',
    )

  def get_data_signature(self):
    """Returns the input signature required to fully initialize this model."""

    vision_patched_shape = utils.get_patched_shape(
        self.vision_input_size[:-1], self.vision_patch_size
    )
    waveform_patched_shape = utils.get_patched_shape(
        self.waveform_input_size, self.waveform_patch_size
    )
    spectrogram_patched_shape = utils.get_patched_shape(
        self.spectrogram_input_size, self.spectrogram_patch_size
    )

    num_vision_pixels = (
        np.prod(self.vision_patch_size) * self.vision_input_size[-1]
    )
    num_waveform_samples = self.waveform_patch_size
    num_spectrogram_samples = np.prod(self.spectrogram_patch_size)

    num_vision_tokens = np.prod(vision_patched_shape)
    num_waveform_tokens = np.prod(waveform_patched_shape)
    num_spectrogram_tokens = np.prod(spectrogram_patched_shape)
    num_text_tokens = self.text_input_size

    vision_target_token_id = jax.random.randint(
        jax.random.key(0),
        (self.input_batch_size, 1, num_vision_tokens),
        minval=0, maxval=self.vision_vocab_size)
    vision_target_token_raw = jax.random.uniform(
        jax.random.key(1),
        (self.input_batch_size, 1, num_vision_tokens, num_vision_pixels))
    vision_target_token_embed = jax.random.uniform(
        jax.random.key(2),
        (self.input_batch_size, 1, num_vision_tokens, self.vision_embed_size))
    waveform_target_token_id = jax.random.randint(
        jax.random.key(3),
        (self.input_batch_size, 1, num_waveform_tokens),
        minval=0, maxval=self.waveform_vocab_size)
    waveform_target_token_raw = jax.random.uniform(
        jax.random.key(4),
        (self.input_batch_size, 1, num_waveform_tokens, num_waveform_samples))
    waveform_target_token_embed = jax.random.uniform(
        jax.random.key(5),
        (self.input_batch_size, 1,
         num_waveform_tokens, self.waveform_embed_size))
    spectrogram_target_token_id = jax.random.randint(
        jax.random.key(6),
        (self.input_batch_size, 1, num_spectrogram_tokens),
        minval=0, maxval=self.spectrogram_vocab_size)
    spectrogram_target_token_raw = jax.random.uniform(
        jax.random.key(7),
        (self.input_batch_size, 1,
         num_spectrogram_tokens, num_spectrogram_samples))
    spectrogram_target_token_embed = jax.random.uniform(
        jax.random.key(8),
        (self.input_batch_size, 1,
         num_spectrogram_tokens, self.spectrogram_embed_size))
    text_input_token_id = jax.random.randint(
        jax.random.key(9),
        (self.input_batch_size, 1, num_text_tokens),
        minval=0, maxval=self.text_vocab_size)
    text_input_token_embed = jax.random.uniform(
        jax.random.key(10),
        (self.input_batch_size, 1, num_text_tokens, self.text_embed_size))
    text_input_token_mask = jax.random.randint(
        jax.random.key(11),
        (self.input_batch_size, 1, num_text_tokens),
        minval=0, maxval=2)
    text_target_token_id = jax.random.randint(
        jax.random.key(12),
        (self.input_batch_size, 1, num_text_tokens),
        minval=0, maxval=self.text_vocab_size)
    text_target_token_embed = jax.random.uniform(
        jax.random.key(13),
        (self.input_batch_size, 1, num_text_tokens, self.text_embed_size))
    text_target_token_mask = jax.random.randint(
        jax.random.key(14),
        (self.input_batch_size, 1, num_text_tokens),
        minval=0, maxval=2)

    vision_target_coordinate = jnp.tile(utils.construct_3d_positions(
        *vision_patched_shape), (self.input_batch_size, 1, 1, 1))
    waveform_target_coordinate = jnp.tile(utils.construct_1d_positions(
        *waveform_patched_shape), (self.input_batch_size, 1, 1))
    spectrogram_target_coordinate = jnp.tile(utils.construct_2d_positions(
        *spectrogram_patched_shape), (self.input_batch_size, 1, 1, 1))
    text_coordinate = jnp.tile(utils.construct_1d_positions(
        num_text_tokens), (self.input_batch_size, 1, 1))

    def _drop_token_and_coordinate(
        token_id, token_raw, token_embed,
        token_coordinate, token_drop_rate=0.5):
      length = token_id.shape[2]
      keep_idx, drop_idx = utils.sample_drop_idx(
          length, token_drop_rate, jax.random.key(6))
      token_id = utils.take_along_axis(token_id, keep_idx, axis=2)
      token_raw = utils.take_along_axis(token_raw, keep_idx, axis=2)
      token_embed = utils.take_along_axis(token_embed, keep_idx, axis=2)
      token_coordinate = utils.take_along_axis(
          token_coordinate, keep_idx, axis=2)
      keep_idx = jnp.tile(keep_idx, token_raw.shape[:2] + (1,))
      drop_idx = jnp.tile(drop_idx, token_raw.shape[:2] + (1,))
      return (token_id, token_raw, token_embed,
              token_coordinate, keep_idx, drop_idx)

    (vision_input_token_id,
     vision_input_token_raw,
     vision_input_token_embed,
     vision_input_coordinate,
     vision_input_token_pos_id,
     vision_input_drop_pos_id) = _drop_token_and_coordinate(
         vision_target_token_id,
         vision_target_token_raw,
         vision_target_token_embed,
         vision_target_coordinate)

    (waveform_input_token_id,
     waveform_input_token_raw,
     waveform_input_token_embed,
     waveform_input_coordinate,
     waveform_input_token_pos_id,
     waveform_input_drop_pos_id) = _drop_token_and_coordinate(
         waveform_target_token_id,
         waveform_target_token_raw,
         waveform_target_token_embed,
         waveform_target_coordinate)

    (spectrogram_input_token_id,
     spectrogram_input_token_raw,
     spectrogram_input_token_embed,
     spectrogram_input_coordinate,
     spectrogram_input_token_pos_id,
     spectrogram_input_drop_pos_id) = _drop_token_and_coordinate(
         spectrogram_target_token_id,
         spectrogram_target_token_raw,
         spectrogram_target_token_embed,
         spectrogram_target_coordinate)

    data = {
        DataFeatureType.INPUTS: {
            DataFeatureRoute.ENCODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_ID:
                        vision_input_token_id,
                    DataFeatureName.TOKEN_RAW:
                        vision_input_token_raw,
                    DataFeatureName.TOKEN_EMBED:
                        vision_input_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        vision_input_coordinate,
                    DataFeatureName.TOKEN_POSITION_ID:
                        vision_input_token_pos_id,
                    DataFeatureName.DROP_POSITION_ID:
                        vision_input_drop_pos_id,
                },
                Modality.WAVEFORM: {
                    DataFeatureName.TOKEN_ID:
                        waveform_input_token_id,
                    DataFeatureName.TOKEN_RAW:
                        waveform_input_token_raw,
                    DataFeatureName.TOKEN_EMBED:
                        waveform_input_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        waveform_input_coordinate,
                    DataFeatureName.TOKEN_POSITION_ID:
                        waveform_input_token_pos_id,
                    DataFeatureName.DROP_POSITION_ID:
                        waveform_input_drop_pos_id,
                },
                Modality.SPECTROGRAM: {
                    DataFeatureName.TOKEN_ID:
                        spectrogram_input_token_id,
                    DataFeatureName.TOKEN_RAW:
                        spectrogram_input_token_raw,
                    DataFeatureName.TOKEN_EMBED:
                        spectrogram_input_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        spectrogram_input_coordinate,
                    DataFeatureName.TOKEN_POSITION_ID:
                        spectrogram_input_token_pos_id,
                    DataFeatureName.DROP_POSITION_ID:
                        spectrogram_input_drop_pos_id,
                },
                Modality.TEXT: {
                    DataFeatureName.TOKEN_ID:
                        text_input_token_id,
                    DataFeatureName.TOKEN_EMBED:
                        text_input_token_embed,
                    DataFeatureName.TOKEN_MASK:
                        text_input_token_mask,
                    DataFeatureName.TOKEN_COORDINATE:
                        text_coordinate,
                },
            },
            DataFeatureRoute.DECODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_ID:
                        vision_target_token_id,
                    DataFeatureName.TOKEN_RAW:
                        vision_target_token_raw,
                    DataFeatureName.TOKEN_EMBED:
                        vision_target_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        vision_target_coordinate,
                },
                Modality.WAVEFORM: {
                    DataFeatureName.TOKEN_ID:
                        waveform_target_token_id,
                    DataFeatureName.TOKEN_RAW:
                        waveform_target_token_raw,
                    DataFeatureName.TOKEN_EMBED:
                        waveform_target_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        waveform_target_coordinate,
                },
                Modality.SPECTROGRAM: {
                    DataFeatureName.TOKEN_ID:
                        spectrogram_target_token_id,
                    DataFeatureName.TOKEN_RAW:
                        spectrogram_target_token_raw,
                    DataFeatureName.TOKEN_EMBED:
                        spectrogram_target_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        spectrogram_target_coordinate,
                },
                Modality.TEXT: {
                    DataFeatureName.TOKEN_ID:
                        text_target_token_id,
                    DataFeatureName.TOKEN_EMBED:
                        text_target_token_embed,
                    DataFeatureName.TOKEN_COORDINATE:
                        text_coordinate,
                    DataFeatureName.TOKEN_MASK:
                        text_target_token_mask,
                },
            },
        },
    }
    return data

  def get_default_metadata(self):
    dataflow = ()
    taskflow = ()
    for modality in self.supported_modalities:
      common_space_targets = tuple(
          sorted(self.supported_modalities - {modality})
      )

      if modality == Modality.TEXT:
        token_feature_names = (DataFeatureName.TOKEN_ID,
                               DataFeatureName.TOKEN_EMBED)
        decoding_modes = (DecodingMode.AR,)
      else:
        token_feature_names = (DataFeatureName.TOKEN_RAW,
                               DataFeatureName.TOKEN_ID,
                               DataFeatureName.TOKEN_EMBED)
        decoding_modes = (DecodingMode.AR, DecodingMode.MAE)

      for token_feature_name in token_feature_names:
        for decoding_mode in decoding_modes:
          taskpass = (
              {
                  TaskFlowName.DECODING_MODE: decoding_mode,
              },
          )
          inputs_pass = {
              DataFeatureRoute.ENCODER: {
                  modality: {
                      token_feature_name: None,
                  },
              },
          }
          if decoding_mode == DecodingMode.AR:
            inputs_pass[DataFeatureRoute.DECODER] = {
                modality: {
                    token_feature_name: None,
                },
            }
          datapass = (
              {
                  DataFeatureType.INPUTS: inputs_pass,
                  DataFeatureType.OUTPUTS: {
                      DataFeatureRoute.ENCODER: {
                          modality: {
                              token_feature_name: DataFeatureName.FEATURES,
                          },
                      },
                      DataFeatureRoute.DECODER: {
                          modality: {
                              token_feature_name: DataFeatureName.FEATURES,
                          },
                      },
                      DataFeatureRoute.LABEL_CLASSIFIER: {
                          modality: {
                              token_feature_name: DataFeatureName.LOGITS,
                          },
                      },
                      DataFeatureRoute.TARGET_CLASSIFIER: {
                          modality: {
                              token_feature_name: DataFeatureName.LOGITS,
                          },
                      },
                      DataFeatureRoute.COMMON_SPACE: {
                          modality: {
                              token_feature_name: common_space_targets,
                          },
                      },
                  },
                  DataFeatureType.TARGETS: {
                      DataFeatureRoute.LABEL_CLASSIFIER: {
                          modality: DataFeatureName.LABEL,
                      },
                  },
                  DataFeatureType.HYPERPARAMS: {
                      DataFeatureRoute.ENCODER: {
                          modality: {
                              token_feature_name: DataFeatureName.TEMPERATURE,
                          },
                      },
                  },
              },
          )
          dataflow += datapass
          taskflow += taskpass
    default_metadata = data_config.Metadata(
        dataflow=dataflow,
        taskflow=taskflow,
    )
    return default_metadata

  def _prepare_encoder_ouputs_for_decoding(
      self,
      encoder_outputdata,
  ):
    """Fetches the output features of the encoder to be used for decoding.

    Args:
      encoder_outputdata: A nested dictionary that contains the
        modality-specific-feature-specific outputs of the encoder as leaves.

    Returns:
      A nested dictionary with a similar structure to `encoder_outputdata`, but
      w/o the last nesting layer which contains the fine-grained naming of the
      model's outputs.
    """
    encoder_outputdata = encoder_outputdata[DataFeatureRoute.ENCODER]
    decoder_inputdata = collections.defaultdict(dict)
    for modality in encoder_outputdata:
      for token_feature_name in encoder_outputdata[modality]:
        features = encoder_outputdata[modality][token_feature_name][
            DataFeatureName.FEATURES
        ]
        decoder_inputdata[modality][token_feature_name] = features
    return dict(decoder_inputdata)

  def _encoder_heads_call(
      self,
      encoder_outputdata,
      outputflow,
      deterministic,
  ):
    encoder_outputdata = encoder_outputdata[DataFeatureRoute.ENCODER]
    heads_outputdata = {}
    for route in outputflow:
      route_outputdata = collections.defaultdict(dict)
      for modality in outputflow[route]:
        for token_feature_name in outputflow[route][modality]:
          encoder_features = encoder_outputdata[modality][token_feature_name]
          if route == DataFeatureRoute.COMMON_SPACE:
            head_inputs = encoder_features[DataFeatureName.FEATURES_AGG]
            target_modalities = outputflow[route][modality][token_feature_name]
            route_outputdata[modality][token_feature_name] = (
                self.cross_modal_head[modality][token_feature_name](
                    inputs=head_inputs,
                    target_modalities=target_modalities,
                    deterministic=deterministic,
                )
            )

          elif route == DataFeatureRoute.LABEL_CLASSIFIER:
            head_inputs = encoder_features[DataFeatureName.FEATURES_AGG]
            route_outputdata[modality][token_feature_name] = (
                self.label_classifier[modality][token_feature_name](
                    inputs=head_inputs,
                )
            )

      if route_outputdata:
        heads_outputdata[route] = dict(route_outputdata)

    return heads_outputdata

  def _prepare_decoder_inputs(
      self,
      inputdata,
      inputflow,
      decoding_mode,
      deterministic,
      project_to_embeds,
  ):
    """Prepares inputs according to inputflow for the decoder.

    Args:
      inputdata: A nested dictionary containing the per-modality inputs,
        features, targets, etc. as leaves.
      inputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys as leaves.
      decoding_mode: A string indicating the decoding style. All supported
        decoding styles can be found in constants.DecodingMode.
      deterministic: A bool indicating whether the stochastic modules should
        operate deterministically or stochastically.
      project_to_embeds: A bool indicating whether to project the inputs to an
        embedding space or not.

    Returns:
      concatenated_token_embed: An array containing token embeddings that are
        concatenated across modalities and their specific features as defined
        in `intputflow`.
      concatenated_token_mask: An array containing token masks that are
        concatenated across modalities and their specific features as defined
        in `intputflow`.
      seq_lengths: A nested dictionary that maps the modalities and their
        features (whose embeddings are present in `concatenated_token_embed`) to
        their corresponding sequence lengths.
      concatenation_order: A tuple of string pairs indicating the order in which
        different features of different modalities are concatenated. Each pair
        is presented as a tuple of strings as (modality, feature_name). These,
        along with `seq_lengths` are later used to split the features at the
        output of the encoder model.

    Raises:
      NotImplementedError: if a special token is used as the aggregation method.
      ValueError: if the modalities present in `inputflow` are not a subset of
        the modalities supported in this model.
    """
    if self.aggregation_type == AggregationType.SPECIAL_TOKEN:
      raise NotImplementedError

    inputflow_modalities = set(inputflow.keys())
    if not inputflow_modalities.issubset(self.supported_modalities):
      raise ValueError(
          'One or all of the modalities in the input metadata are not '
          f'supported by this model. Received {inputflow_modalities},'
          f'  while all supported modalities are {self.supported_modalities}.'
      )
    seq_lengths = collections.defaultdict(dict)
    concatenation_order = ()
    all_token_embeds = []
    all_token_masks = []
    for modality in inputflow:
      # Fetch proper inputs
      token_mask = inputdata[modality].get(
          DataFeatureName.TOKEN_MASK, None)
      token_coordinate = inputdata[modality].get(
          DataFeatureName.TOKEN_COORDINATE, None)
      token_pos_id = inputdata[modality].get(
          DataFeatureName.TOKEN_POSITION_ID, None)
      drop_pos_id = inputdata[modality].get(
          DataFeatureName.DROP_POSITION_ID, None)

      if modality == Modality.TEXT and token_coordinate is not None:
        # In text, we scale the normalized coordinates to max_seq_length to
        # avoid dilated pos_embedding gather.
        coordinate_scale = token_coordinate.shape[-1]
      else:
        coordinate_scale = None

      for token_feature_name in inputflow[modality]:
        if project_to_embeds:
          # Fetch the pre-embed token data
          token_pre_embed = inputdata[modality][token_feature_name]

          # Project raw inputs to a sequence of embedding vectors
          token_embed = self.raw_to_embeddings[modality][token_feature_name](
              token_pre_embed,
              deterministic,
              token_coordinate,
              coordinate_scale,
          )

          # Apply linear projection before feeding to Transformer
          token_embed = self.pre_decoder_proj[modality][token_feature_name](
              token_embed
          )
        else:
          token_embed = inputdata[modality][token_feature_name]

        # Fill in mask embeddings if drop_pos_id is provided and decoder is in
        # the MAE mode.
        if decoding_mode == DecodingMode.MAE and drop_pos_id is not None:
          token_embed = (
              self.post_encoder_mask_filler[modality][token_feature_name](
                  inputs=token_embed,
                  mask_position_ids=drop_pos_id,
                  keep_position_ids=token_pos_id,
                  axis=-2,
              )
          )

        # Fetch the sequence length to use later when separating cross-modal
        # output features
        token_seq_length = token_embed.shape[2]

        # Collect all necessary information
        seq_lengths[modality][token_feature_name] = token_seq_length
        concatenation_order += ((modality, token_feature_name),)
        all_token_embeds.append(token_embed)
        all_token_masks.append(token_mask)

    # Concatenate all token embeddings and masks
    (concatenated_token_embed,
     concatenated_token_mask) = utils.concatenate_cross_modal_tokens(
         token_embeds=all_token_embeds,
         token_masks=all_token_masks,
         concatenated_token_embed_shardings=self.tokens_shardings,
         contrarenated_token_mask_shardings=self.tokens_shardings[:-1],
     )

    return (
        concatenated_token_embed,
        concatenated_token_mask,
        seq_lengths,
        concatenation_order,
    )

  def _prepare_decoder_outputs(
      self,
      concatenated_features,
      concatenated_token_mask,
      seq_lengths,
      concatenation_order,
      outputflow,
  ):
    """Prepares outputs according to outputflow for the decoder.

    Args:
      concatenated_features: An array that contains the outputs of the decoder
        (given a concatenated multimodal inputs.)
      concatenated_token_mask: An optional array containing the concatenated
        token masks.
      seq_lengths: A nested dictionary that maps the modalities and their
        features (whose embeddings are present in `concatenated_token_embed`) to
        their corresponding sequence lengths.
      concatenation_order: A tuple of string pairs indicating the order in which
        different features of different modalities are concatenated. Each pair
        is presented as a tuple of strings as (modality, feature_name). These,
        along with `seq_lengths` is used to split the features.
      outputflow: A nested dictionary containing the per-modality feature keys
        as leaves.

    Returns:
      A nested dictionary with a structure similar to `outputflow` that contains
      the modality-specific (but still cross-modal) features as leaves.

    Raises:
      ValueError: if the modalities present in `inputflow` are not a subset of
        the modalities supported in this model.
    """
    outputdata = collections.defaultdict(dict)
    split_features = utils.split_cross_modal_tokens(
        concatenated_token_features=concatenated_features,
        seq_lengths=seq_lengths,
        concatenation_order=concatenation_order,
        split_token_features_shardings=self.tokens_shardings,
        dot_general=lax.dot_general,
    )
    if concatenated_token_mask is None:
      split_token_mask = jax.tree_util.tree_map(lambda x: None, split_features)
    else:
      split_token_mask = utils.split_cross_modal_tokens(
          concatenated_token_features=concatenated_token_mask,
          seq_lengths=seq_lengths,
          concatenation_order=concatenation_order,
          split_token_features_shardings=self.tokens_shardings[:-1],
          dot_general=lax.dot_general,
      )
    for modality in outputflow:
      for token_feature_name in outputflow[modality]:
        features = split_features[modality][token_feature_name]
        token_mask = split_token_mask[modality][token_feature_name]
        features = self.post_encoder_proj[modality][token_feature_name](
            features
        )
        # TODO(b/241931055): add feature_map reshape and MAP aggregation
        features = self.aggregator_head(inputs=features, token_mask=token_mask)

        # Add this feature to the outputdata
        outputdata[modality][token_feature_name] = features

    return {DataFeatureRoute.DECODER: dict(outputdata)}

  def _decoder_heads_call(
      self,
      decoder_outputdata,
      outputflow,
      deterministic,
  ):
    decoder_outputdata = decoder_outputdata[DataFeatureRoute.DECODER]
    heads_outputdata = {}
    for route in outputflow:
      route_outputdata = collections.defaultdict(dict)
      for modality in outputflow[route]:
        for token_feature_name in outputflow[route][modality]:
          decoder_features = decoder_outputdata[modality][token_feature_name]
          if route == DataFeatureRoute.TARGET_CLASSIFIER:
            head_inputs = decoder_features[DataFeatureName.FEATURES]
            route_outputdata[modality][token_feature_name] = (
                self.target_classifier[modality][token_feature_name](
                    inputs=head_inputs,
                )
            )

      if route_outputdata:
        heads_outputdata[route] = dict(route_outputdata)

    return heads_outputdata

  def _decoder_call(
      self,
      inputdata,
      cross_inputdata,
      inputflow,
      cross_inputflow,
      outputflow,
      decoding_mode,
      deterministic,
  ):
    """A method to call the decoder based on specific sets of data.

    Args:
      inputdata: A nested dictionary containing the per-modality inputs,
        features, targets, etc. as leaves.
      cross_inputdata: A nested dictionary containing the per-modality cross-
        inputs, cross-features, etc. as leaves.
      inputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys as leaves.
      cross_inputflow: A nested dictionary with a similar structure to the
        cross_inputdata, but containing the per-modality feature keys as leaves.
      outputflow: A nested dictionary with a similar structure to the inputdata,
        but containing the per-modality feature keys that are supposed to be
        included in the output data of the model.
      decoding_mode: A string indicating the decoding style. All supported
        decoding styles can be found in constants.DecodingMode.
      deterministic: A bool indicating whether the stochastic modules should
        operate deterministically or stochastically.

    Returns:
      A nested dictionary with a similar structure to the inputdata, but
      containing the model features as leaves.
    """
    # It is assumed that cross-inputs are always coming from the encoder, hence
    # no raw-to-embed projection is performed. It is also assumed that only in
    # the AR decoding mode we need to project the inputdata into an embedding
    # space using the raw-to-embed layer.
    project_inputdata_to_embeds = decoding_mode == DecodingMode.AR
    project_cross_inputdata_to_embeds = False
    (token_embed,
     token_mask,
     seq_lengths,
     concatenation_order) = self._prepare_decoder_inputs(
         inputflow=inputflow,
         inputdata=inputdata,
         decoding_mode=decoding_mode,
         deterministic=deterministic,
         project_to_embeds=project_inputdata_to_embeds)

    if cross_inputdata is None:
      cross_token_embed = None
      cross_token_mask = None
    else:
      cross_token_embed, cross_token_mask, _, _ = self._prepare_decoder_inputs(
          inputflow=cross_inputflow,
          inputdata=cross_inputdata,
          decoding_mode=decoding_mode,
          deterministic=deterministic,
          project_to_embeds=project_cross_inputdata_to_embeds)

    # We don't support relative biases yet
    attention_bias = None

    # Set appropriate self-attention mask according to token_mask and causality
    if decoding_mode == DecodingMode.AR:
      if token_mask is None:
        # If token mask not provided, all tokens should be included causally
        token_mask = jnp.ones(token_embed.shape[:-1], dtype=jnp.int32)
      attention_mask = utils.create_causal_attention_mask(
          token_mask, dtype=token_mask.dtype)
    elif token_mask is not None:
      # Non-causal attention mask is created based on the token mask
      attention_mask = utils.create_attention_mask(
          query_token_mask=token_mask,
          key_token_mask=token_mask,
          dtype=token_mask.dtype)
    else:
      # Not causal and no token mask, hence no attention mask (full self-attn)
      attention_mask = None

    # Set cross-attention mask if cross_token_mask is provided
    if cross_token_mask is not None:
      if token_mask is None:
        # Input tokens are not padded, hence cross-inputs should attend on ALL
        # input tokens
        token_mask = jnp.ones(token_embed.shape[:-1],
                              dtype=cross_token_mask.dtype)
      cross_attention_mask = utils.create_attention_mask(
          query_token_mask=token_mask,
          key_token_mask=cross_token_mask,
          dtype=token_mask.dtype)
    elif token_mask is not None and cross_token_embed is not None:
      # Input tokens are padded while cross-inputs are not, hence ALL cross-
      # input tokens should attend on the unpadded input tokens
      if cross_token_mask is None:
        cross_token_mask = jnp.ones(cross_token_embed.shape[:-1],
                                    dtype=token_mask.dtype)
      cross_attention_mask = utils.create_attention_mask(
          query_token_mask=token_mask,
          key_token_mask=cross_token_mask,
          dtype=token_mask.dtype)
    else:
      # Either cross-inputs are not provided or none of inputs/cross-inputs are
      # padded, hence no need for cross-attention mask.
      cross_attention_mask = None

    features = self.decoder(inputs=token_embed,
                            cross_inputs=cross_token_embed,
                            deterministic=deterministic,
                            attention_mask=attention_mask,
                            attention_bias=attention_bias,
                            cross_attention_mask=cross_attention_mask,
                            max_decode_length=None,
                            decode=False)

    outputdata = self._prepare_decoder_outputs(
        concatenated_features=features,
        concatenated_token_mask=token_mask,
        seq_lengths=seq_lengths,
        concatenation_order=concatenation_order,
        outputflow=outputflow,
    )
    return outputdata

  def __call__(
      self, data, deterministic = True
  ):
    """Performs encoding and optionally decoding given a nested array.

    This model assumes that data is always fed to the encoder. Although, given
    the information in the data, an optional decoding step is performed. If any
    modality contains targets, it is assumed that the model should perform
    decoding on top of encoding. The style of this decoding is defined based on
    the given modality. Currently, we only perform autoregressive (AR) decoding
    for the text modality and MAE-style decoding for audio and vision modalities
    In the case of AR decoding, the outputs of the encoder are used as cross-
    inputs to the cross-attention layers of the decoder. However, in the case
    of MAE decoding, the output of the encoder are directly fed to the decoder
    inputs while all cross-attention layers inside the decoder are skipped.

    Args:
      data: A nested array containing modality-specific sequence of tokens
        and their corresponding masks. For the feature signature please refer to
        constants.DataFeatureName.
      deterministic: Whether the ops are performed deterministically.

    Returns:
      A nested dictionary of arrays containing the predictions and features.
    """
    metadata = data.get(DataFeatureType.METADATA, self.get_default_metadata())
    outputdata = {}
    for dataflow, taskflow in zip(metadata.dataflow, metadata.taskflow):
      # Fetch all the dataflow information
      inputflow = dataflow[DataFeatureType.INPUTS]
      outputflow = dataflow[DataFeatureType.OUTPUTS]
      encoder_inputflow = inputflow.get(DataFeatureRoute.ENCODER, {})
      encoder_outputflow = outputflow.get(DataFeatureRoute.ENCODER, {})
      decoder_inputflow = inputflow.get(DataFeatureRoute.DECODER, {})
      decoder_outputflow = outputflow.get(DataFeatureRoute.DECODER, {})
      decoding_mode = taskflow.get(TaskFlowName.DECODING_MODE, None)

      inputdata = data[DataFeatureType.INPUTS]
      encoding_performed = False
      decoding_performed = False
      if encoder_inputflow:
        # Fetch the inputs data
        encoder_inputdata = inputdata[DataFeatureRoute.ENCODER]

        # Check if the dataflow exists in the inputdata
        utils.verify_flow_exists_in_data(
            flow=encoder_inputflow,
            data=encoder_inputdata,
        )

        # Call the Transformer encoder
        encoder_outputdata = self._encoder_call(
            inputdata=encoder_inputdata,
            inputflow=encoder_inputflow,
            outputflow=encoder_outputflow,
            deterministic=deterministic,
        )
        encoding_performed = True

        # Call the heads
        encoder_heads_outputdata = self._encoder_heads_call(
            encoder_outputdata=encoder_outputdata,
            outputflow=outputflow,
            deterministic=deterministic,
        )

        # Update the outputdata with encoder and heads outputs
        outputdata = utils.deep_update_data(outputdata, encoder_outputdata)
        outputdata = utils.deep_update_data(
            outputdata, encoder_heads_outputdata
        )

        # Perform decoding based on the encoder outputs
        if decoding_mode == DecodingMode.MAE:
          decoder_inputdata = self._prepare_encoder_ouputs_for_decoding(
              encoder_outputdata
          )
          decoder_inputdata = utils.deep_update_data(
              inputdata[DataFeatureRoute.ENCODER], decoder_inputdata)
          utils.verify_flow_exists_in_data(
              flow=encoder_outputflow,
              data=decoder_inputdata,
          )
          decoder_outputdata = self._decoder_call(
              inputdata=decoder_inputdata,
              cross_inputdata=None,
              inputflow=encoder_outputflow,
              cross_inputflow={},
              outputflow=decoder_outputflow,
              decoding_mode=decoding_mode,
              deterministic=deterministic,
          )
          decoding_performed = True
        elif decoding_mode == DecodingMode.AR:
          if not decoder_inputflow:
            raise ValueError(
                f'When in {decoding_mode=}, an input for the decoder should be '
                'specified.')
          decoder_inputdata = inputdata[DataFeatureRoute.DECODER]
          decoder_cross_inputdata = self._prepare_encoder_ouputs_for_decoding(
              encoder_outputdata
          )
          utils.verify_flow_exists_in_data(
              flow=decoder_inputflow,
              data=decoder_inputdata,
          )
          utils.verify_flow_exists_in_data(
              flow=encoder_outputflow,
              data=decoder_cross_inputdata,
          )
          decoder_outputdata = self._decoder_call(
              inputdata=decoder_inputdata,
              cross_inputdata=decoder_cross_inputdata,
              inputflow=decoder_inputflow,
              cross_inputflow=encoder_outputflow,
              outputflow=decoder_outputflow,
              decoding_mode=decoding_mode,
              deterministic=deterministic,
          )
          decoding_performed = True

      elif decoder_inputflow:
        if decoding_mode == DecodingMode.AR:
          # If reached this point, a decoder-only pass will be performed
          # Fetch the inputs data
          decoder_inputdata = inputdata[DataFeatureRoute.DECODER]
          utils.verify_flow_exists_in_data(
              flow=decoder_inputflow,
              data=decoder_inputdata,
          )
          decoder_outputdata = self._decoder_call(
              inputdata=decoder_inputdata,
              cross_inputdata=None,
              inputflow=decoder_inputflow,
              cross_inputflow={},
              outputflow=decoder_outputflow,
              decoding_mode=decoding_mode,
              deterministic=deterministic,
          )
          decoding_performed = True
        else:
          raise ValueError(
              f'A decoder-only model with {decoding_mode=} is not supported.')

      else:
        raise NotImplementedError

      if decoding_performed:
        # Call the heads
        decoder_heads_outputdata = self._decoder_heads_call(
            decoder_outputdata=decoder_outputdata,  # pylint: disable=undefined-variable
            outputflow=outputflow,
            deterministic=deterministic,
        )

        # Update the outputdata with encoder and heads outputs
        outputdata = utils.deep_update_data(
            outputdata, decoder_outputdata)  # pylint: disable=undefined-variable
        outputdata = utils.deep_update_data(
            outputdata, decoder_heads_outputdata)

    if encoding_performed:  # pylint: disable=undefined-variable
      hyperparams = ({
          DataFeatureRoute.ENCODER: {
              DataFeatureName.TEMPERATURE: self.temperature(),
          },
      })
      data.update({
          DataFeatureType.HYPERPARAMS: hyperparams,
      })

    data.update({
        DataFeatureType.OUTPUTS: outputdata,
    })
    return data


# ----------------------------------------------------------------------
# -------------------- Mixture-of-Experts Variants ---------------------
# ----------------------------------------------------------------------


class SparseMoeIMP(IMP):
  """Sparse MoE version of IMP."""

  num_moe_layers: int
  moe_layers_distribution: str
  num_experts: int
  ignore_padding_tokens: bool
  jitter_noise: float
  comm_dtype: jax.typing.DTypeLike
  split_params: bool
  optimize_parallel_comms: bool
  router_kwargs: tuple[tuple[str, Any], Ellipsis]
  max_group_size: int
  capacity_factor: float
  min_expert_capacity: int
  router_type: str
  router_bias: bool
  strict_group_size: bool
  num_selected_experts: int
  batch_prioritized_routing: bool
  router_kernel_shardings: typing.ShardingAxes
  routed_ffn_intermediate_shardings: typing.ShardingAxes
  model_axis_size: int
  model_axis_name: str

  def setup(self):
    super().setup()
    self.encoder = mnn.SparseMoeTransformerEncoder(
        d_model=self.d_model,
        d_ff=self.d_ff,
        num_heads=self.num_heads,
        num_layers=self.num_layers,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        remat=self.remat,
        scanned_layers=self.scanned_layers,
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
        routed_ffn_intermediate_shardings=self.routed_ffn_intermediate_shardings,
        router_kernel_shardings=self.router_kernel_shardings,
        tokens_shardings=self.tokens_shardings,
        model_axis_size=self.model_axis_size,
        model_axis_name=self.model_axis_name,
        mha_qkv_dot_general=jax.lax.dot_general,
        mha_out_dot_general=jax.lax.dot_general,
        mha_einsum_dot_general=jax.lax.dot_general,
        ffn_dot_general=jax.lax.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        approximate_gelu=self.approximate_gelu,
        num_experts=self.num_experts,
        max_group_size=self.max_group_size,
        capacity_factor=self.capacity_factor,
        min_expert_capacity=self.min_expert_capacity,
        router_type=self.router_type,
        router_bias=self.router_bias,
        jitter_noise=self.jitter_noise,
        comm_dtype=self.comm_dtype,
        split_params=self.split_params,
        optimize_parallel_comms=self.optimize_parallel_comms,
        strict_group_size=self.strict_group_size,
        num_selected_experts=self.num_selected_experts,
        batch_prioritized_routing=self.batch_prioritized_routing,
        ignore_padding_tokens=self.ignore_padding_tokens,
        num_moe_layers=self.num_moe_layers,
        moe_layers_distribution=self.moe_layers_distribution,
        router_kwargs=self.router_kwargs,
        dtype=self.dtype,
        name='moe_transformer_encoder',
    )

  def get_rng_keys(self):
    """Returns keys of all rngs defined under this model."""

    keys = ()
    if self.dropout_rate > 0.:
      keys += ('dropout',)
    if self.jitter_noise > 0.:
      keys += ('jitter',)

    return keys


class SoftMoeIMP(IMP):
  """Soft MoE version of IMP."""

  num_moe_layers: int
  moe_layers_distribution: str
  num_experts: int
  ignore_padding_tokens: bool
  jitter_noise: float
  comm_dtype: jax.typing.DTypeLike
  split_params: bool
  optimize_parallel_comms: bool
  router_kwargs: tuple[tuple[str, Any], Ellipsis]
  expert_capacity: int
  router_kernel_shardings: typing.ShardingAxes
  routed_ffn_intermediate_shardings: typing.ShardingAxes
  model_axis_size: int
  model_axis_name: str

  def setup(self):
    super().setup()
    self.encoder = mnn.SoftMoeTransformerEncoder(
        d_model=self.d_model,
        d_ff=self.d_ff,
        num_heads=self.num_heads,
        num_layers=self.num_layers,
        use_bias=self.use_bias,
        dropout_rate=self.dropout_rate,
        remat=self.remat,
        scanned_layers=self.scanned_layers,
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
        routed_ffn_intermediate_shardings=self.routed_ffn_intermediate_shardings,
        router_kernel_shardings=self.router_kernel_shardings,
        tokens_shardings=self.tokens_shardings,
        model_axis_size=self.model_axis_size,
        model_axis_name=self.model_axis_name,
        mha_qkv_dot_general=jax.lax.dot_general,
        mha_out_dot_general=jax.lax.dot_general,
        mha_einsum_dot_general=jax.lax.dot_general,
        ffn_dot_general=jax.lax.dot_general,
        precision=self.precision,
        lora_rank=self.lora_rank,
        lora_scale=self.lora_scale,
        approximate_gelu=self.approximate_gelu,
        num_experts=self.num_experts,
        expert_capacity=self.expert_capacity,
        ignore_padding_tokens=self.ignore_padding_tokens,
        jitter_noise=self.jitter_noise,
        comm_dtype=self.comm_dtype,
        split_params=self.split_params,
        optimize_parallel_comms=self.optimize_parallel_comms,
        num_moe_layers=self.num_moe_layers,
        moe_layers_distribution=self.moe_layers_distribution,
        router_kwargs=self.router_kwargs,
        dtype=self.dtype,
        name='moe_transformer_encoder',
    )

  def get_rng_keys(self):
    """Returns keys of all rngs defined under this model."""

    keys = ()
    if self.dropout_rate > 0.:
      keys += ('dropout',)
    if self.jitter_noise > 0.:
      keys += ('jitter',)

    return keys
