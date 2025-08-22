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
import jax
import jax.numpy as jnp
import numpy as np

from imp.max import modeling as mnn
from imp.max.core import constants
from imp.max.core import utils
from imp.max.modeling import garden
from imp.max.modeling.garden import config as garden_config

jax.config.update('jax_threefry_partitionable', False)

AggType = constants.AggregationType
Modality = constants.Modality
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
TOKEN_RAW = DataFeatureName.TOKEN_RAW
FEATURES = DataFeatureName.FEATURES
FEATURES_AGG = DataFeatureName.FEATURES_AGG
FEATURE_MAPS = DataFeatureName.FEATURE_MAPS
LOGITS = DataFeatureName.LOGITS
_MICRO_VIT_CONFIG = garden_config.ViT(
    num_layers=2,
    d_ff=16,
    num_heads=2,
    d_model=4,
    batch_size=2,
    image_size=(1, 16, 16, 3),
    patch_size=(1, 8, 8),
    d_post_proj=None,
    post_proj_position=None,
    num_classes=None,
    aggregation_type=AggType.GLOBAL_AVERAGE_POOL,
    positional_embedding='learned_1d',
    dtype=jnp.float32)


class VitTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'd_post_proj': None,
          'num_classes': None,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
      },
      {
          'testcase_name': 'special_token',
          'd_post_proj': None,
          'num_classes': None,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.SPECIAL_TOKEN,
          'positional_embedding': 'learned_1d',
      },
      {
          'testcase_name': 'with_temporal_tokens',
          'image_size': (4, 16, 16, 3),  # (t, h, w, c)
          'patch_size': (2, 4, 4),  # (p_t, p_h, p_w)
          'd_post_proj': None,
          'num_classes': None,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
      },
      {
          'testcase_name': 'drop_token',
          'd_post_proj': None,
          'num_classes': None,
          'droptoken_rate': 0.25,
          'aggregation_type': AggType.SPECIAL_TOKEN,
          'positional_embedding': 'learned_1d',
          'deterministic': False,
      },
      {
          'testcase_name': 'with_post_proj_pre_aggregation',
          'd_post_proj': 7,
          'post_proj_position': 'pre_aggregation',
          'num_classes': None,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
      },
      {
          'testcase_name': 'with_post_proj_post_aggregation',
          'd_post_proj': 7,
          'post_proj_position': 'post_aggregation',
          'num_classes': None,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
      },
      {
          'testcase_name': 'with_post_proj_and_classes',
          'd_post_proj': 7,
          'post_proj_position': 'pre_aggregation',
          'num_classes': 28,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
      },
      {
          'testcase_name': 'indeterministic',
          'd_post_proj': 7,
          'post_proj_position': 'pre_aggregation',
          'num_classes': 28,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
          'deterministic': False,
      },
      {
          'testcase_name': 'lora',
          'd_post_proj': 7,
          'post_proj_position': 'pre_aggregation',
          'num_classes': 28,
          'droptoken_rate': 0.,
          'lora_rank': 2,
          'lora_scale': 1.0,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
          'deterministic': False,
      },
      {
          'testcase_name': 'half-precision',
          'd_post_proj': 7,
          'post_proj_position': 'pre_aggregation',
          'num_classes': 28,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_1d',
          'dtype': jnp.bfloat16,
      },
      {
          'testcase_name': 'learned_3d_posemb',
          'd_post_proj': None,
          'num_classes': None,
          'droptoken_rate': 0.,
          'aggregation_type': AggType.GLOBAL_AVERAGE_POOL,
          'positional_embedding': 'learned_3d',
      })
  def test_end_to_end(self,
                      d_post_proj,
                      num_classes,
                      aggregation_type,
                      positional_embedding,
                      droptoken_rate,
                      image_size=(1, 16, 16, 3),
                      patch_size=(1, 8, 8),
                      dtype=jnp.float32,
                      lora_rank=2,
                      lora_scale=0.,
                      post_proj_position=None,
                      deterministic=True):
    micro_vit_config = _MICRO_VIT_CONFIG.copy()
    micro_vit_config.image_size = image_size
    micro_vit_config.patch_size = patch_size
    micro_vit_config.d_post_proj = d_post_proj
    micro_vit_config.post_proj_position = post_proj_position
    micro_vit_config.num_classes = num_classes
    micro_vit_config.aggregation_type = aggregation_type
    micro_vit_config.positional_embedding = positional_embedding
    micro_vit_config.dtype = dtype
    micro_vit_config.lora_rank = lora_rank
    micro_vit_config.lora_scale = lora_scale

    num_instances = 3
    batch_size = micro_vit_config.batch_size
    d_model = micro_vit_config.d_model

    init_rngs = {'params': jax.random.key(0)}
    apply_rngs = {'dropout': jax.random.key(1)}

    @jax.jit
    def _run_forward(data):
      vit_model = garden.ViT(**micro_vit_config.as_dict())
      variables = vit_model.init(rngs=init_rngs, data=data)
      return vit_model.apply(
          variables=variables,
          rngs=apply_rngs,
          data=data,
          deterministic=deterministic)

    input_shape = (batch_size, num_instances) + image_size
    inputs = jnp.ones(input_shape, dtype=jnp.float32)
    token_raw = mnn.extract_volume_patches(inputs, patch_size, flatten=True)
    if not deterministic and droptoken_rate > 0.:
      token_raw = mnn.DropToken(rate=droptoken_rate).apply(
          variables={},
          rngs={'droptoken': jax.random.key(2)},
          inputs=token_raw,
          deterministic=deterministic)
      token_coordinate = jnp.arange(0, token_raw.shape[2]) / token_raw.shape[2]
      token_coordinate = jnp.tile(token_coordinate, token_raw.shape[:2] + (1,))
    else:
      token_coordinate = None

    expected_feature_map_shape = (batch_size,
                                  num_instances) + utils.get_patched_shape(
                                      image_size[:-1], patch_size)
    expected_features_agg_shape = token_raw.shape[:2]
    expected_features_shape = token_raw.shape[:3]

    if d_post_proj is None:
      expected_feature_map_shape += (d_model,)
      expected_features_agg_shape += (d_model,)
      expected_features_shape += (d_model,)
    else:
      if post_proj_position == 'pre_aggregation':
        expected_feature_map_shape += (d_post_proj,)
        expected_features_shape += (d_post_proj,)
      elif post_proj_position == 'post_aggregation':
        expected_feature_map_shape += (d_model,)
        expected_features_shape += (d_model,)
      expected_features_agg_shape += (d_post_proj,)

    data = {
        DataFeatureType.INPUTS: {
            DataFeatureRoute.ENCODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_RAW: token_raw,
                    DataFeatureName.TOKEN_COORDINATE: token_coordinate,
                },
            },
        },
    }
    outputs = _run_forward(data)[DataFeatureType.OUTPUTS]
    outputs = outputs[DataFeatureRoute.ENCODER][Modality.VISION][
        DataFeatureName.TOKEN_RAW
    ]

    chex.assert_shape(outputs[FEATURES], expected_features_shape)
    chex.assert_shape(outputs[FEATURES_AGG], expected_features_agg_shape)

    has_feature_maps = deterministic or droptoken_rate == 0.0
    if has_feature_maps:
      chex.assert_shape(outputs[FEATURE_MAPS], expected_feature_map_shape)

    if num_classes is not None:
      expected_logits_shape = expected_features_agg_shape[:-1] + (num_classes,)
      chex.assert_shape(outputs[LOGITS], expected_logits_shape)

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline_1d',
          'positional_embedding': 'learned_1d',
      }, {
          'testcase_name': 'baseline_3d',
          'positional_embedding': 'learned_3d',
      })
  def test_values(self, positional_embedding='learned_1d'):
    image_size = (1, 4, 4, 3)  # (t, h, w, c)
    patch_size = (1, 2, 2)  # (p_t, p_h, p_w)

    micro_vit_config = _MICRO_VIT_CONFIG.copy()
    micro_vit_config.image_size = image_size
    micro_vit_config.patch_size = patch_size
    micro_vit_config.aggregation_type = AggType.MULTI_HEAD_ATTENTION_POOL
    micro_vit_config.d_post_proj = 4
    micro_vit_config.num_classes = 3
    micro_vit_config.post_proj_position = 'pre_aggregation'
    micro_vit_config.positional_embedding = positional_embedding

    num_instances = 2
    batch_size = micro_vit_config.batch_size

    init_rngs = {'params': jax.random.key(0)}
    apply_rngs = {'dropout': jax.random.key(1)}

    @jax.jit
    def _run_forward(data):
      vit_model = garden.ViT(**micro_vit_config.as_dict())
      variables = vit_model.init(rngs=init_rngs, data=data)
      return vit_model.apply(
          variables=variables,
          rngs=apply_rngs,
          data=data,
          deterministic=False)

    input_shape = (batch_size, num_instances) + image_size
    inputs = jnp.ones(input_shape, dtype=jnp.float32)
    token_raw = mnn.extract_volume_patches(inputs, patch_size, flatten=True)
    data = {
        DataFeatureType.INPUTS: {
            DataFeatureRoute.ENCODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_RAW: token_raw,
                },
            },
        },
    }
    outputs = _run_forward(data)[DataFeatureType.OUTPUTS]
    outputs = outputs[DataFeatureRoute.ENCODER][Modality.VISION][
        DataFeatureName.TOKEN_RAW
    ]
    features_agg = outputs[FEATURES_AGG]
    logits = outputs[LOGITS]

    if positional_embedding == 'learned_1d':
      expected_features_agg = jnp.array(
          [
              [
                  [0.629259, -0.7856853, 0.6408966, 0.42224324],
                  [0.28856313, -0.85049933, 0.27096546, 0.25470847],
              ],
              [
                  [0.64862305, -0.78223526, 0.65030825, 0.42140588],
                  [1.4799199, -0.17982268, 0.6097845, 0.3653572],
              ],
          ],
          dtype=jnp.float32,
      )
    elif positional_embedding == 'learned_3d':
      expected_features_agg = jnp.array(
          [
              [
                  [0.04829776, -0.9233873, 0.50381875, 0.3352753],
                  [0.01535177, -0.9131067, 0.50340354, 0.34045848],
              ],
              [
                  [0.14463246, -0.9023471, 0.4857359, 0.33032197],
                  [1.0436602, -0.29476857, 0.46156198, 0.2784783],
              ],
          ],
          dtype=jnp.float32,
      )
    else:
      raise ValueError(f'Unknown positional embedding: {positional_embedding}')

    expected_logits = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )

    np.testing.assert_array_almost_equal(features_agg, expected_features_agg, 1)
    np.testing.assert_array_almost_equal(logits, expected_logits, 1)

  def test_invalid_configuration(self):
    micro_vit_config = _MICRO_VIT_CONFIG.copy()
    micro_vit_config.d_post_proj = 4
    micro_vit_config.post_proj_position = None

    num_instances = 3
    batch_size = micro_vit_config.batch_size
    image_size = micro_vit_config.image_size
    patch_size = micro_vit_config.patch_size
    input_shape = (batch_size, num_instances) + image_size

    inputs = jnp.ones(input_shape, dtype=jnp.float32)
    token_raw = mnn.extract_volume_patches(inputs, patch_size, flatten=True)
    data = {
        DataFeatureType.INPUTS: {
            DataFeatureRoute.ENCODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_RAW: token_raw,
                },
            },
        },
    }

    vit_model = garden.ViT(**micro_vit_config.as_dict())
    init_rngs = {'params': jax.random.key(0)}
    expected_error_msg = 'Please provide a valid post projection position'
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      _ = vit_model.init(rngs=init_rngs, data=data)

    mismatched_patch_size = (1, 4, 4)
    token_raw = mnn.extract_volume_patches(
        inputs, mismatched_patch_size, flatten=True)
    data = {
        DataFeatureType.INPUTS: {
            DataFeatureRoute.ENCODER: {
                Modality.VISION: {
                    DataFeatureName.TOKEN_RAW: token_raw,
                },
            },
        },
    }
    micro_vit_config.d_post_proj = None
    vit_model = garden.ViT(**micro_vit_config.as_dict())
    expected_error_msg = (
        'The inputs do not contain the same number of tokens as the available '
        'buckets. num_input_tokens=16 while num_available_buckets=4. Please '
        'either provide `token_coordinate`, or configure this module with '
        'proper `pos_buckets`.')
    with self.assertRaisesRegex(ValueError, expected_error_msg):
      _ = vit_model.init(rngs=init_rngs, data=data)


if __name__ == '__main__':
  absltest.main()
