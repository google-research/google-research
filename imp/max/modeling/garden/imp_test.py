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

"""Tests for MAX Garden."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax import traverse_util
import jax
import jax.numpy as jnp

from imp.max.core import constants
from imp.max.core import utils
from imp.max.modeling import garden
from imp.max.modeling.garden import config as garden_config

AggType = constants.AggregationType
Modality = constants.Modality
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
TaskFlowName = constants.TaskFlowName
DecodingMode = constants.DecodingMode
TOKEN_RAW = DataFeatureName.TOKEN_RAW
TOKEN_ID = DataFeatureName.TOKEN_ID
TOKEN_EMBED = DataFeatureName.TOKEN_EMBED
FEATURES = DataFeatureName.FEATURES
FEATURES_AGG = DataFeatureName.FEATURES_AGG
FEATURE_MAPS = DataFeatureName.FEATURE_MAPS
LOGITS = DataFeatureName.LOGITS
_MICRO_IMP_CONFIG = garden_config.IMP(
    num_layers=2,
    d_ff=8,
    num_heads=2,
    d_model=4,
    d_common=6,
    input_batch_size=2,
    vision_input_size=(2, 4, 4, 2),
    vision_patch_size=(1, 2, 2),
    vision_vocab_size=16,
    vision_embed_size=4,
    waveform_input_size=16,
    waveform_patch_size=4,
    waveform_vocab_size=16,
    waveform_embed_size=4,
    spectrogram_input_size=(4, 4),
    spectrogram_patch_size=(2, 2),
    spectrogram_vocab_size=16,
    spectrogram_embed_size=4,
    text_input_size=4,
    text_vocab_size=16,
    text_embed_size=4,
    d_post_proj=4,
    vision_classes=None,
    waveform_classes=None,
    spectrogram_classes=None,
    text_classes=None,
    aggregation_type=AggType.GLOBAL_AVERAGE_POOL,
    dtype=jnp.bfloat16)
_MICRO_IMPEGE_CONFIG = garden_config.IMPeGe(
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=8,
    num_heads=2,
    d_model=4,
    d_common=6,
    input_batch_size=2,
    vision_input_size=(2, 4, 4, 2),
    vision_patch_size=(1, 2, 2),
    vision_vocab_size=16,
    vision_embed_size=4,
    waveform_input_size=16,
    waveform_patch_size=4,
    waveform_vocab_size=16,
    waveform_embed_size=4,
    spectrogram_input_size=(4, 4),
    spectrogram_patch_size=(2, 2),
    spectrogram_vocab_size=16,
    spectrogram_embed_size=4,
    text_input_size=4,
    text_vocab_size=16,
    text_embed_size=4,
    d_post_encoder_proj=4,
    d_post_decoder_proj=4,
    vision_classes=None,
    waveform_classes=None,
    spectrogram_classes=None,
    text_classes=None,
    vision_targets=None,
    waveform_targets=None,
    spectrogram_targets=None,
    text_targets=None,
    aggregation_type=AggType.GLOBAL_AVERAGE_POOL,
    dtype=jnp.bfloat16)


def _maybe_apply_droptoken(inputs, drop_rate):
  if drop_rate == 0.:
    return inputs

  for modality in inputs:
    token_raw = inputs[modality].get(DataFeatureName.TOKEN_RAW, None)
    token_coordinate = inputs[modality].get(DataFeatureName.TOKEN_COORDINATE,
                                            None)

    if token_raw is not None:
      length = token_raw.shape[-1]
      keep_idx, _ = utils.sample_drop_idx(length, drop_rate,
                                          jax.random.key(0))

      token_raw = utils.take_along_axis(token_raw, keep_idx, axis=2)
      inputs[modality][DataFeatureName.TOKEN_RAW] = token_raw

      if token_coordinate is not None:
        token_coordinate = utils.take_along_axis(
            token_coordinate, keep_idx, axis=2)
        inputs[modality][DataFeatureName.TOKEN_COORDINATE] = token_coordinate

  return inputs


def _filter_data_modality(data, modalities):
  flattened_data = traverse_util.flatten_dict(data, sep='/')
  all_keys = list(flattened_data.keys())
  for k in all_keys:
    contains_modality = []
    for modality in modalities:
      contains_modality.append(modality in k)
    if not any(contains_modality):
      del flattened_data[k]
  data = traverse_util.unflatten_dict(flattened_data, sep='/')
  return data


class ImpTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'modalities': (Modality.VISION, Modality.TEXT),
          'num_classes': 7,
          'num_targets': 13,
          'remat': 'zero',
          'scanned_layers': False,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
      }, {
          'testcase_name': 'lora',
          'modalities': (Modality.VISION, Modality.TEXT),
          'num_classes': 7,
          'num_targets': 13,
          'remat': 'zero',
          'scanned_layers': False,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'lora_rank': 2,
          'lora_scale': 1.0,
      }
  )
  def test_end_to_end(self,
                      modalities,
                      num_classes,
                      num_targets,
                      remat,
                      scanned_layers,
                      aggregation_type,
                      droptoken_rate=0.,
                      dtype=jnp.bfloat16,
                      deterministic=True,
                      lora_rank=2,
                      lora_scale=0.):
    micro_imp_config = _MICRO_IMP_CONFIG.copy()
    micro_imp_config.vision_classes = num_classes
    micro_imp_config.waveform_classes = num_classes
    micro_imp_config.spectrogram_classes = num_classes
    micro_imp_config.text_classes = num_classes
    micro_imp_config.vision_targets = num_targets
    micro_imp_config.waveform_targets = num_targets
    micro_imp_config.spectrogram_targets = num_targets
    micro_imp_config.text_targets = num_targets
    micro_imp_config.remat = remat
    micro_imp_config.scanned_layers = scanned_layers
    micro_imp_config.aggregation_type = aggregation_type
    micro_imp_config.dtype = dtype
    micro_imp_config.lora_rank = lora_rank
    micro_imp_config.lora_scale = lora_scale

    d_model = micro_imp_config.d_model
    d_post_proj = micro_imp_config.d_post_proj

    class IMP(garden.IMP):
      def get_data_signature(self, modalities=modalities):
        data = super().get_data_signature()
        data = _filter_data_modality(data, modalities)
        return data

    imp_model = IMP(**micro_imp_config.as_dict())

    init_rngs = {'params': jax.random.key(0)}
    apply_rngs = {'dropout': jax.random.key(1)}

    @jax.jit
    def _run_forward(data):
      variables = imp_model.init(rngs=init_rngs, data=data)
      return imp_model.apply(
          variables=variables,
          rngs=apply_rngs,
          data=data,
          deterministic=deterministic)

    input_drop_rate = droptoken_rate if not deterministic else 0.
    data = imp_model.get_data_signature()
    metadata = imp_model.get_default_metadata()
    data[DataFeatureType.METADATA] = metadata
    inputs = data[DataFeatureType.INPUTS][DataFeatureRoute.ENCODER]
    inputs = _maybe_apply_droptoken(inputs, input_drop_rate)
    data[DataFeatureType.INPUTS][DataFeatureRoute.ENCODER] = inputs
    data = _run_forward(data)
    outputs = data[DataFeatureType.OUTPUTS]
    encoder_outputs = outputs[DataFeatureRoute.ENCODER]
    if num_classes is not None:
      label_cls_outputs = outputs[DataFeatureRoute.LABEL_CLASSIFIER]
    if num_targets is not None:
      target_cls_outputs = outputs[DataFeatureRoute.TARGET_CLASSIFIER]

    for modality in inputs:
      for feature_name in inputs[modality]:
        if feature_name not in (DataFeatureName.TOKEN_RAW,
                                DataFeatureName.TOKEN_ID,
                                DataFeatureName.TOKEN_EMBED):
          continue
        expected_features_agg_shape = inputs[modality][feature_name].shape[:2]
        expected_features_shape = inputs[modality][feature_name].shape[:3]

        if d_post_proj is None:
          expected_features_agg_shape += (d_model,)
          expected_features_shape += (d_model,)
        else:
          expected_features_agg_shape += (d_post_proj,)
          expected_features_shape += (d_post_proj,)

        chex.assert_shape(encoder_outputs[modality][feature_name][FEATURES],
                          expected_features_shape)
        chex.assert_shape(encoder_outputs[modality][feature_name][FEATURES_AGG],
                          expected_features_agg_shape)

        # pylint: disable=undefined-variable
        if num_classes is not None:
          expected_logits_shape = expected_features_agg_shape[:-1] + (
              num_classes,)
          chex.assert_shape(label_cls_outputs[modality][feature_name][LOGITS],
                            expected_logits_shape)
        if num_targets is not None:
          expected_logits_shape = expected_features_shape[:-1] + (
              num_targets,)
          chex.assert_shape(target_cls_outputs[modality][feature_name][LOGITS],
                            expected_logits_shape)
        # pylint: enable=undefined-variable


class IMPeGeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'modalities': (Modality.VISION, Modality.TEXT),
          'num_classes': None,
          'num_targets': None,
          'remat': 'zero',
          'scanned_layers': False,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
      }, {
          'testcase_name': 'lora',
          'modalities': (Modality.VISION, Modality.TEXT),
          'num_classes': None,
          'num_targets': None,
          'remat': 'zero',
          'scanned_layers': False,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'lora_rank': 2,
          'lora_scale': 1.0,
      }
  )
  def test_end_to_end(self,
                      modalities,
                      num_classes,
                      num_targets,
                      remat,
                      scanned_layers,
                      aggregation_type,
                      dtype=jnp.bfloat16,
                      deterministic=True,
                      lora_rank=2,
                      lora_scale=0.):
    micro_impege_config = _MICRO_IMPEGE_CONFIG.copy()
    micro_impege_config.vision_classes = num_classes
    micro_impege_config.waveform_classes = num_classes
    micro_impege_config.spectrogram_classes = num_classes
    micro_impege_config.text_classes = num_classes
    micro_impege_config.vision_targets = num_targets
    micro_impege_config.waveform_targets = num_targets
    micro_impege_config.spectrogram_targets = num_targets
    micro_impege_config.text_targets = num_targets
    micro_impege_config.remat = remat
    micro_impege_config.scanned_layers = scanned_layers
    micro_impege_config.aggregation_type = aggregation_type
    micro_impege_config.dtype = dtype
    micro_impege_config.lora_rank = lora_rank
    micro_impege_config.lora_scale = lora_scale

    d_model = micro_impege_config.d_model
    d_post_encoder_proj = micro_impege_config.d_post_encoder_proj
    d_post_decoder_proj = micro_impege_config.d_post_decoder_proj

    class IMPeGe(garden.IMPeGe):
      def get_data_signature(self, modalities=modalities):
        data = super().get_data_signature()
        data = _filter_data_modality(data, modalities)
        return data

    impege_model = IMPeGe(**micro_impege_config.as_dict())

    init_rngs = {'params': jax.random.key(0)}
    apply_rngs = {'dropout': jax.random.key(1)}

    @jax.jit
    def _run_forward(data):
      variables = impege_model.init(rngs=init_rngs, data=data)
      return impege_model.apply(
          variables=variables,
          rngs=apply_rngs,
          data=data,
          deterministic=deterministic)

    data = impege_model.get_data_signature()
    metadata = impege_model.get_default_metadata()
    data[DataFeatureType.METADATA] = metadata
    data = _run_forward(data)
    inputs = data[DataFeatureType.INPUTS]
    outputs = data[DataFeatureType.OUTPUTS]
    encoder_inputs = inputs[DataFeatureRoute.ENCODER]
    decoder_inputs = inputs[DataFeatureRoute.DECODER]
    encoder_outputs = outputs[DataFeatureRoute.ENCODER]
    decoder_outputs = outputs[DataFeatureRoute.DECODER]

    if num_classes is not None:
      label_cls_outputs = outputs[DataFeatureRoute.LABEL_CLASSIFIER]

    if num_targets is not None:
      target_cls_outputs = outputs[DataFeatureRoute.TARGET_CLASSIFIER]

    for modality in encoder_inputs:
      for feature_name in encoder_inputs[modality]:
        if feature_name not in (DataFeatureName.TOKEN_RAW,
                                DataFeatureName.TOKEN_ID,
                                DataFeatureName.TOKEN_EMBED):
          continue
        expected_features_agg_shape = (
            encoder_inputs[modality][feature_name].shape[:2]
        )
        expected_features_shape = (
            encoder_inputs[modality][feature_name].shape[:3]
        )
        expected_decode_features_agg_shape = (
            decoder_inputs[modality][feature_name].shape[:2]
        )
        expected_decode_features_shape = (
            decoder_inputs[modality][feature_name].shape[:3]
        )

        if d_post_encoder_proj is None:
          expected_features_agg_shape += (d_model,)
          expected_features_shape += (d_model,)
        else:
          expected_features_agg_shape += (d_post_encoder_proj,)
          expected_features_shape += (d_post_encoder_proj,)

        if d_post_decoder_proj is None:
          expected_decode_features_agg_shape += (d_model,)
          expected_decode_features_shape += (d_model,)
        else:
          expected_decode_features_agg_shape += (d_post_decoder_proj,)
          expected_decode_features_shape += (d_post_decoder_proj,)

        chex.assert_shape(encoder_outputs[modality][feature_name][FEATURES],
                          expected_features_shape)
        chex.assert_shape(encoder_outputs[modality][feature_name][FEATURES_AGG],
                          expected_features_agg_shape)
        chex.assert_shape(decoder_outputs[modality][feature_name][FEATURES],
                          expected_decode_features_shape)
        chex.assert_shape(decoder_outputs[modality][feature_name][FEATURES_AGG],
                          expected_decode_features_agg_shape)

        # pylint: disable=undefined-variable
        if num_classes is not None:
          expected_logits_shape = (
              expected_features_agg_shape[:-1] + (num_classes,))
          chex.assert_shape(label_cls_outputs[LOGITS],
                            expected_logits_shape)
        if num_targets is not None:
          expected_decode_logits_shape = (
              expected_decode_features_shape[:-1] + (num_targets,))
          chex.assert_shape(target_cls_outputs[LOGITS],
                            expected_decode_logits_shape)
        # pylint: enable=undefined-variable


if __name__ == '__main__':
  absltest.main()
