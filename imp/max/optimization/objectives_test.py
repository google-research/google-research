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

"""Tests for objectives."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from flax import traverse_util
import jax
from jax import numpy as jnp

from imp.max.core import constants
from imp.max.core import utils
from imp.max.optimization import config as opt_config
from imp.max.optimization import objectives

jax.config.update('jax_threefry_partitionable', False)


Modality = constants.Modality
DataFeatureType = constants.DataFeatureType
DataFeatureRoute = constants.DataFeatureRoute
DataFeatureName = constants.DataFeatureName
TOKEN_RAW = DataFeatureName.TOKEN_RAW
TOKEN_ID = DataFeatureName.TOKEN_ID
TOKEN_EMBED = DataFeatureName.TOKEN_EMBED
TOKEN_MASK = DataFeatureName.TOKEN_MASK
FEATURES = DataFeatureName.FEATURES
FEATURES_AGG = DataFeatureName.FEATURES_AGG
FEATURE_MAPS = DataFeatureName.FEATURE_MAPS
LABEL = DataFeatureName.LABEL
LOGITS = DataFeatureName.LOGITS
ENCODER = DataFeatureRoute.ENCODER
DECODER = DataFeatureRoute.DECODER
COMMON_SPACE = DataFeatureRoute.COMMON_SPACE
INPUTS = DataFeatureType.INPUTS
OUTPUTS = DataFeatureType.OUTPUTS
TARGETS = DataFeatureType.TARGETS
VISION = Modality.VISION
WAVEFORM = Modality.WAVEFORM
SPECTROGRAM = Modality.SPECTROGRAM
TEXT = Modality.TEXT


def _construct_cross_entropy_inputs_outputs(input_shape,
                                            num_classes,
                                            left_shift_targets,
                                            num_additional_pad,
                                            one_hot_targets,
                                            multiclass_targets,
                                            targets_key,
                                            targets_mask_key,
                                            predictions_key,
                                            route_key,
                                            token_key):
  if one_hot_targets and multiclass_targets:
    raise ValueError(
        'Targets cannot be one-hot and multi-class simultaneously.')

  def _construct_targets(rng):
    if multiclass_targets:
      targets = jax.random.randint(
          rng, shape=input_shape + (num_classes,), minval=0, maxval=2)
    else:
      targets = jax.random.randint(
          rng, shape=input_shape, minval=0, maxval=num_classes)
    if left_shift_targets:
      # Prepend a BOS token at the beginning of the targets.
      # We intentionally set it to one to reveal and edge cases (if any).
      if multiclass_targets:
        bos_token = jnp.ones(
            input_shape[:-1] + (1, num_classes), dtype=jnp.int32)
        concat_axis = -2
      else:
        bos_token = jnp.ones(input_shape[:-1] + (1,), dtype=jnp.int32)
        concat_axis = -1
      targets = jnp.concatenate([bos_token, targets], axis=concat_axis)
    if num_additional_pad > 0:
      if multiclass_targets:
        pad_token_shape = input_shape[:-1] + (num_additional_pad, num_classes)
        concat_axis = -2
      else:
        pad_token_shape = input_shape[:-1] + (num_additional_pad,)
        concat_axis = -1
      pad_token = jnp.ones(pad_token_shape, dtype=jnp.int32)
      targets = jnp.concatenate([targets, pad_token], axis=concat_axis)
    if one_hot_targets:
      targets = jax.nn.one_hot(targets, num_classes)
    return targets

  def _construct_predictions(rng):
    predictions = jax.random.normal(
        rng, shape=input_shape + (num_classes,))
    if left_shift_targets:
      # Append an EOS token at the end of the predictions
      eos_token = jax.random.normal(
          rng, shape=input_shape[:-1] + (1, num_classes))
      predictions = jnp.concatenate([predictions, eos_token], axis=-2)
    if num_additional_pad > 0:
      pad_token_shape = input_shape[:-1] + (num_additional_pad, num_classes)
      pad_token = jax.random.normal(rng, shape=pad_token_shape)
      predictions = jnp.concatenate([predictions, pad_token], axis=-2)
    return predictions

  def _construct_token_mask():
    target_token_mask = jnp.ones(input_shape, dtype=jnp.float32)
    if left_shift_targets:
      # Prepend a BOS token mask at the beginning of the targets_mask
      bos_token = jnp.ones(input_shape[:-1] + (1,), dtype=jnp.int32)
      target_token_mask = jnp.concatenate(
          [bos_token, target_token_mask], axis=-1)
    if num_additional_pad > 0:
      pad_mask_shape = input_shape[:-1] + (num_additional_pad,)
      pad_mask = jnp.zeros(pad_mask_shape, dtype=jnp.float32)
      target_token_mask = jnp.concatenate(
          [target_token_mask, pad_mask], axis=-1)
    return target_token_mask

  data = {
      TARGETS: {
          route_key: {
              VISION: {
                  targets_key: _construct_targets(rng=jax.random.key(0)),
                  targets_mask_key: _construct_token_mask(),
              },
              WAVEFORM: {
                  targets_key: _construct_targets(rng=jax.random.key(1)),
                  targets_mask_key: _construct_token_mask(),
              },
              TEXT: {
                  targets_key: _construct_targets(rng=jax.random.key(2)),
                  targets_mask_key: _construct_token_mask(),
              },
              SPECTROGRAM: {
                  targets_key: _construct_targets(rng=jax.random.key(3)),
                  targets_mask_key: _construct_token_mask(),
              },
          },
      },
      OUTPUTS: {
          route_key: {
              VISION: {
                  token_key: {
                      predictions_key: _construct_predictions(
                          rng=jax.random.key(3)),
                  },
              },
              WAVEFORM: {
                  token_key: {
                      predictions_key: _construct_predictions(
                          rng=jax.random.key(4)),
                  },
              },
              TEXT: {
                  token_key: {
                      predictions_key: _construct_predictions(
                          rng=jax.random.key(5)),
                  },
              },
              SPECTROGRAM: {
                  token_key: {
                      predictions_key: _construct_predictions(
                          rng=jax.random.key(6)),
                  },
              },
          },
      },
  }
  return data


class CrossEntropyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'left_shift_targets': False,
          'one_hot_targets': False,
      }, {
          'testcase_name': 'baseline_with_mask',
          'left_shift_targets': False,
          'one_hot_targets': False,
          'num_additional_pad': 3,
      }, {
          'testcase_name': 'baseline_bfloat16',
          'left_shift_targets': False,
          'one_hot_targets': False,
          'dtype': jnp.bfloat16,
      }, {
          'testcase_name': 'left_shift',
          'left_shift_targets': True,
          'one_hot_targets': False,
      }, {
          'testcase_name': 'one_hot_targets',
          'left_shift_targets': False,
          'one_hot_targets': True,
      }, {
          'testcase_name': 'left_shift_and_one_hot_targets',
          'left_shift_targets': True,
          'one_hot_targets': True,
      }, {
          'testcase_name': 'left_shift_and_one_hot_targets_with_mask',
          'left_shift_targets': True,
          'one_hot_targets': True,
          'num_additional_pad': 3,
      }, {
          'testcase_name': 'bfloat16_left_shift_and_one_hot_targets',
          'left_shift_targets': True,
          'one_hot_targets': True,
          'dtype': jnp.bfloat16,
      })
  def test_softmax_cross_entropy(self,
                                 left_shift_targets,
                                 one_hot_targets,
                                 num_additional_pad=0,
                                 dtype=None):
    input_shape = (2, 3, 5)
    num_classes = 7
    predictions_key = LOGITS
    targets_key = LABEL
    targets_mask_key = TOKEN_MASK
    route_key = ENCODER
    token_key = TOKEN_RAW

    @jax.jit
    def _run_forward(data):
      softmax_objective = objectives.SoftmaxCrossEntropy(
          modality_token_weights=utils.flatten_itemize_dict(
              {
                  VISION: {token_key: 1.},
                  WAVEFORM: {token_key: 1.},
                  SPECTROGRAM: {token_key: 1.},
                  TEXT: {token_key: 1.},
              }
          ),
          loss_weight=1.,
          route_key=route_key,
          predictions_key=predictions_key,
          targets_key=targets_key,
          left_shift_targets=left_shift_targets,
          one_hot_targets=one_hot_targets,
          dtype=dtype,
          name='test')
      return softmax_objective(data)

    data = _construct_cross_entropy_inputs_outputs(
        input_shape=input_shape,
        num_classes=num_classes,
        left_shift_targets=left_shift_targets,
        num_additional_pad=num_additional_pad,
        one_hot_targets=one_hot_targets,
        multiclass_targets=False,
        targets_key=targets_key,
        targets_mask_key=targets_mask_key,
        predictions_key=predictions_key,
        route_key=route_key,
        token_key=token_key,
    )
    loss, metrics = _run_forward(data)
    metrics = traverse_util.flatten_dict(metrics, sep='/')
    if dtype == jnp.bfloat16:
      decimal = 1
    else:
      decimal = 4
    self.assertAlmostEqual(loss, 2.2803879, decimal)
    self.assertAlmostEqual(
        metrics['test/vision/loss'], 2.123483, decimal)
    self.assertAlmostEqual(
        metrics[f'test/vision/{token_key}/accuracy/top_1'],
        0.23333335, decimal)
    self.assertAlmostEqual(
        metrics['test/waveform/loss'], 2.1301363, decimal)
    self.assertAlmostEqual(
        metrics[f'test/waveform/{token_key}/accuracy/top_1'],
        0.23333335, decimal)
    self.assertAlmostEqual(
        metrics['test/spectrogram/loss'], 2.4700904, decimal)
    self.assertAlmostEqual(
        metrics[f'test/spectrogram/{token_key}/accuracy/top_1'],
        0.13333334, decimal)
    self.assertAlmostEqual(
        metrics['test/text/loss'], 2.397834, decimal)
    self.assertAlmostEqual(
        metrics[f'test/text/{token_key}/accuracy/top_1'], 0.10000001, decimal)
    self.assertAlmostEqual(
        metrics['test/total_loss'], 2.2803879, decimal)

  def test_invalid_softmax_configuration(self):
    input_shape = (2, 3, 5)
    num_classes = 7
    num_additional_pad = 3
    left_shift_targets = True
    one_hot_targets = True
    predictions_key = LOGITS
    targets_key = LABEL
    targets_mask_key = TOKEN_MASK
    route_key = ENCODER
    token_key = TOKEN_RAW

    softmax_objective = objectives.SoftmaxCrossEntropy(
        modality_token_weights=utils.flatten_itemize_dict(
            {
                VISION: {token_key: 1.},
                WAVEFORM: {token_key: 1.},
                SPECTROGRAM: {token_key: 1.},
                TEXT: {token_key: 1.},
            }
        ),
        loss_weight=1.,
        route_key=route_key,
        predictions_key=predictions_key,
        targets_key=targets_key,
        left_shift_targets=left_shift_targets,
        one_hot_targets=one_hot_targets,
        name='test')

    data = _construct_cross_entropy_inputs_outputs(
        input_shape=input_shape,
        num_classes=num_classes,
        left_shift_targets=left_shift_targets,
        num_additional_pad=num_additional_pad,
        one_hot_targets=one_hot_targets,
        multiclass_targets=False,
        targets_key=targets_key,
        targets_mask_key=targets_mask_key,
        predictions_key=predictions_key,
        route_key=route_key,
        token_key=token_key,
    )

    # Invalid predictions shape
    data_invalid = copy.deepcopy(data)
    data_invalid[OUTPUTS][route_key][VISION][token_key][predictions_key] = (
        data[OUTPUTS][route_key][VISION][token_key][predictions_key].mean(
            axis=-1,
        )
    )
    with self.assertRaises(ValueError):
      softmax_objective(data_invalid)

    # Invalid targets shape
    data_invalid = copy.deepcopy(data)
    data_invalid[TARGETS][route_key][VISION][targets_key] = (
        data[TARGETS][route_key][VISION][targets_key].mean(axis=-1)
    )
    with self.assertRaises(ValueError):
      softmax_objective(data_invalid)

    # Invalid target mask
    data_invalid = copy.deepcopy(data_invalid)
    data_invalid[TARGETS][route_key][VISION][targets_mask_key] = (
        data_invalid[TARGETS][route_key][VISION][targets_mask_key].mean(axis=-1)
    )
    with self.assertRaises(ValueError):
      softmax_objective(data_invalid)

  @parameterized.named_parameters(
      {
          'testcase_name': 'baseline',
          'left_shift_targets': False,
      }, {
          'testcase_name': 'baseline_with_mask',
          'left_shift_targets': False,
          'num_additional_pad': 3,
      }, {
          'testcase_name': 'baseline_bfloat16',
          'left_shift_targets': False,
          'dtype': jnp.bfloat16,
      }, {
          'testcase_name': 'left_shift',
          'left_shift_targets': True,
      }, {
          'testcase_name': 'left_shift_with_mask',
          'left_shift_targets': True,
          'num_additional_pad': 3,
      }, {
          'testcase_name': 'bfloat16_left_shift_with_mask',
          'left_shift_targets': True,
          'num_additional_pad': 3,
          'dtype': jnp.bfloat16,
      })
  def test_sigmoid_cross_entropy(self,
                                 left_shift_targets,
                                 num_additional_pad=0,
                                 dtype=None):
    input_shape = (2, 3, 5)
    num_classes = 7
    predictions_key = LOGITS
    targets_key = LABEL
    targets_mask_key = TOKEN_MASK
    route_key = ENCODER
    token_key = TOKEN_RAW

    @jax.jit
    def _run_forward(data):
      sigmoid_objective = objectives.SigmoidBinaryCrossEntropy(
          modality_token_weights=utils.flatten_itemize_dict(
              {
                  VISION: {token_key: 1.},
                  WAVEFORM: {token_key: 1.},
                  SPECTROGRAM: {token_key: 1.},
                  TEXT: {token_key: 1.},
              }
          ),
          loss_weight=1.,
          route_key=route_key,
          predictions_key=predictions_key,
          targets_key=targets_key,
          left_shift_targets=left_shift_targets,
          dtype=dtype,
          name='test')
      return sigmoid_objective(data)

    data = _construct_cross_entropy_inputs_outputs(
        input_shape=input_shape,
        num_classes=num_classes,
        left_shift_targets=left_shift_targets,
        num_additional_pad=num_additional_pad,
        one_hot_targets=False,
        multiclass_targets=True,
        targets_key=targets_key,
        targets_mask_key=targets_mask_key,
        predictions_key=predictions_key,
        route_key=route_key,
        token_key=token_key,
    )
    loss, metrics = _run_forward(data)
    metrics = traverse_util.flatten_dict(metrics, sep='/')
    if dtype == jnp.bfloat16:
      decimal = 2
    else:
      decimal = 4
    self.assertAlmostEqual(loss, 0.8041295, decimal)
    self.assertAlmostEqual(metrics['test/vision/loss'], 0.8344437, decimal)
    self.assertAlmostEqual(metrics['test/waveform/loss'], 0.7984368, decimal)
    self.assertAlmostEqual(metrics['test/spectrogram/loss'], 0.7956387, decimal)
    self.assertAlmostEqual(metrics['test/text/loss'], 0.788005, decimal)
    self.assertAlmostEqual(metrics['test/total_loss'], 0.8041295, decimal)

  def test_invalid_sigmoid_configuration(self):
    input_shape = (2, 3, 5)
    num_classes = 7
    num_additional_pad = 3
    left_shift_targets = True
    predictions_key = LOGITS
    targets_key = LABEL
    targets_mask_key = TOKEN_MASK
    route_key = ENCODER
    token_key = TOKEN_RAW

    sigmoid_objective = objectives.SigmoidBinaryCrossEntropy(
        modality_token_weights=utils.flatten_itemize_dict(
            {
                VISION: {token_key: 1.},
                WAVEFORM: {token_key: 1.},
                SPECTROGRAM: {token_key: 1.},
                TEXT: {token_key: 1.},
            }
        ),
        loss_weight=1.,
        route_key=route_key,
        predictions_key=predictions_key,
        targets_key=targets_key,
        left_shift_targets=left_shift_targets,
        name='test')

    data = _construct_cross_entropy_inputs_outputs(
        input_shape=input_shape,
        num_classes=num_classes,
        left_shift_targets=left_shift_targets,
        num_additional_pad=num_additional_pad,
        one_hot_targets=False,
        multiclass_targets=True,
        targets_key=targets_key,
        targets_mask_key=targets_mask_key,
        predictions_key=predictions_key,
        route_key=route_key,
        token_key=token_key,
    )

    # Invalid predictions shape
    data_invalid = copy.deepcopy(data)
    data_invalid[OUTPUTS][route_key][VISION][token_key][predictions_key] = (
        data[OUTPUTS][route_key][VISION][token_key][predictions_key].mean(
            axis=-1
        )
    )
    with self.assertRaises(ValueError):
      sigmoid_objective(data_invalid)

    # Invalid targets shape
    data_invalid = copy.deepcopy(data)
    data_invalid[TARGETS][route_key][VISION][targets_key] = (
        data[TARGETS][route_key][VISION][targets_key].mean(axis=-1)
    )
    with self.assertRaises(ValueError):
      sigmoid_objective(data_invalid)

    # Invalid target mask
    data_invalid = copy.deepcopy(data)
    data_invalid[TARGETS][route_key][VISION][targets_mask_key] = (
        data[TARGETS][route_key][VISION][targets_mask_key].mean(axis=-1)
    )
    with self.assertRaises(ValueError):
      sigmoid_objective(data_invalid)


class MeanSquaredErrorTest(absltest.TestCase):

  def test_mean_squared_error(self):
    predictions_key = FEATURES
    targets_key = TOKEN_RAW
    targets_mask_key = TOKEN_MASK
    route_key = ENCODER
    token_key = TOKEN_RAW

    mse = objectives.fetch_objective_cls(
        constants.Objective.MEAN_SQUARED_ERROR)
    # pylint: disable=unexpected-keyword-arg
    objective = mse(
        modality_token_weights=(((VISION, token_key), 1.),),
        loss_weight=1.,
        route_key=route_key,
        predictions_key=predictions_key,
        targets_key=targets_key,
        dtype=None,
        name='test')
    # pylint: enable=unexpected-keyword-arg

    data = {
        TARGETS: {
            route_key: {
                VISION: {
                    targets_key: jnp.array([[[0., 1.], [1., 0.], [1., 1.]]]),
                    targets_mask_key: jnp.array([[1.0, 1.0, 0.0]])
                },
            },
        },
        OUTPUTS: {
            route_key: {
                VISION: {
                    token_key: {
                        predictions_key: jnp.array(
                            [[[0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]]),
                    },
                },
            },
        },
    }

    loss, metrics = objective(data)

    self.assertAlmostEqual(loss, 0.5, 5)
    self.assertAlmostEqual(metrics['test/total_loss'], 0.5, 5)


class CrossModalNceTest(absltest.TestCase):

  # TODO(b/234527705): split this test to multiple distinct scenarios
  def test_cross_modal_nce(self):
    objective = objectives.CrossModalNCE(
        modality_pair_weights=(
            ((SPECTROGRAM, TEXT), 1.),
            ((SPECTROGRAM, VISION), 0.2),
            ((SPECTROGRAM, WAVEFORM), 0.3),
            ((TEXT, VISION), 1.),
            ((TEXT, WAVEFORM), .8),
            ((VISION, WAVEFORM), .7),
        ),
        hparams_route_key=ENCODER,
        temperature=0.01,
        margin=0.01,
        loss_weight=1.,
        dtype='bfloat16',
        name='test')

    data = {
        OUTPUTS: {
            COMMON_SPACE: {
                VISION: {
                    TOKEN_RAW: {
                        TEXT: jnp.array(
                            [[[-1.51562, -0.375, 0.808594],
                              [0.261719, -0.96875, -0.192383]],
                             [[0.386719, -0.742188, 0.707031],
                              [0.451172, -0.0737305, -0.152344]],
                             [[-0.292969, -0.667969, 0.0245361],
                              [-0.597656, 1.28125, 0.730469]]],
                            dtype=jnp.bfloat16),
                        WAVEFORM: jnp.array(
                            [[[1.28125, 0.679688, 2.51562],
                              [-0.482422, 1.01562, -1.17969]],
                             [[-1.65625, 0.122559, 0.65625],
                              [1.19531, 1.54688, -0.503906]],
                             [[-1.65625, 0.894531, 0.0834961],
                              [1.69531, 0.808594, -0.375]]],
                            dtype=jnp.bfloat16),
                        SPECTROGRAM: jnp.array(
                            [[[1.69531, -1.74219, 0.925781],
                              [0.609375, 0.103027, 0.0834961]],
                             [[0.162109, -1.58594, 0.835938],
                              [1.08594, 0.408203, -0.769531]],
                             [[1.15625, -0.112793, 1.61719],
                              [0.585938, -0.9375, 1.01562]]],
                            dtype=jnp.bfloat16),
                    },
                },
                TEXT: {
                    TOKEN_ID: {
                        VISION: jnp.array(
                            [[[-0.824219, 0.181641, -0.0147095],
                              [-0.171875, 0.894531, 0.429688]],
                             [[-1.03125, -1.30469, -2.32812],
                              [-1.30469, -0.375, -1.95312]],
                             [[-0.0341797, -0.211914, -0.292969],
                              [1.24219, 1.48438, -1.07031]]],
                            dtype=jnp.bfloat16),
                        WAVEFORM: jnp.array(
                            [[[-0.796875, 0.5625, -1.95312],
                              [-0.333984, 0.142578, -0.273438]],
                             [[-0.96875, 0.679688, 1.08594],
                              [0.429688, 0.679688, -1.14062]],
                             [[1.78125, 0.181641, 0.429688],
                              [-0.621094, 1.42969, 0.609375]]],
                            dtype=jnp.bfloat16),
                        SPECTROGRAM: jnp.array(
                            [[[-1.17969, -0.769531, -1.21875],
                              [0.408203, -1.14062, 0.679688]],
                             [[0.429688, 1.89062, -1.30469],
                              [-1.03125, 0.539062, 0.202148]],
                             [[0.539062, 0.283203, 0.302734],
                              [-1.35156, -0.0932617, 1.08594]]],
                            dtype=jnp.bfloat16),
                    },
                },
                WAVEFORM: {
                    TOKEN_RAW: {
                        VISION: jnp.array(
                            [[[-0.96875, -0.211914, -0.355469],
                              [-0.396484, -1.95312, 1.28125]],
                             [[-0.644531, -0.292969, -1.03125],
                              [1.89062, -0.251953, -0.691406]],
                             [[0.0441895, -0.574219, -1.46094],
                              [0.808594, -0.152344, 1.24219]]],
                            dtype=jnp.bfloat16),
                        TEXT: jnp.array(
                            [[[-0.0147095, 0.953125, 0.757812],
                              [-0.691406, 0.78125, -0.621094]],
                             [[-0.667969, 0.953125, 0.0441895],
                              [0.202148, 1.24219, -0.112793]],
                             [[-0.667969, -0.769531, -0.769531],
                              [0.472656, -1.74219, 0.494141]]],
                            dtype=jnp.bfloat16),
                        SPECTROGRAM: jnp.array(
                            [[[0.142578, -0.527344, 0.202148],
                              [0.515625, -2.89062, -0.375]],
                             [[-0.0147095, -1.74219, 0.302734],
                              [1.61719, -0.851562, -0.192383]],
                             [[1.48438, -0.171875, -0.251953],
                              [0.632812, -0.0539551, -0.460938]]],
                            dtype=jnp.bfloat16),
                    },
                },
                SPECTROGRAM: {
                    TOKEN_RAW: {
                        VISION: jnp.array(
                            [[[-0.482422, -0.112793, -0.273438],
                              [-1.40625, 1.15625, 2.20312]],
                             [[-0.574219, 1.08594, 0.34375],
                              [0.707031, 0.757812, 0.386719]],
                             [[0.78125, 0.162109, 2.51562],
                              [0.609375, -0.0737305, 0.5625]]],
                            dtype=jnp.bfloat16),
                        TEXT: jnp.array(
                            [[[-1.46094, 0.472656, -0.621094],
                              [-0.273438, 0.0834961, 1.15625]],
                             [[0.302734, -0.482422, 1.19531],
                              [0.0634766, -1.21875, 0.408203]],
                             [[-1.58594, -0.132812, 2.03125],
                              [-0.171875, -0.691406, 0.867188]]],
                            dtype=jnp.bfloat16),
                        WAVEFORM: jnp.array(
                            [[[-0.742188, -1.07031, -0.667969],
                              [0.162109, -0.232422, 1.125]],
                             [[0.757812, 0.34375, -0.171875],
                              [0.122559, -0.824219, -0.211914]],
                             [[-0.621094, 0.429688, -1.03125],
                              [-1.03125, -1.58594, 1.61719]]],
                            dtype=jnp.bfloat16),
                    },
                },
            },
        },
    }

    loss, metrics = objective(data)

    decimal = 2

    self.assertAlmostEqual(loss, 23.00000, decimal)
    self.assertEqual(loss.dtype, jnp.bfloat16)
    self.assertAlmostEqual(metrics['test/total_loss'], 23.00000, decimal)

    self.assertAlmostEqual(
        metrics['test/spectrogram_vs_text'], 43.75000, decimal)
    self.assertAlmostEqual(
        metrics['test/text_vs_spectrogram'], 43.75000, decimal)

    self.assertAlmostEqual(
        metrics['test/spectrogram_vs_text/top_1'], 0.25000, decimal)
    self.assertAlmostEqual(
        metrics['test/text_vs_spectrogram/top_1'], 0.16699, decimal)

    self.assertAlmostEqual(
        metrics['test/spectrogram_vs_vision'], 7.34375, decimal)
    self.assertAlmostEqual(
        metrics['test/vision_vs_spectrogram'], 6.00000, decimal)

    self.assertAlmostEqual(
        metrics['test/spectrogram_vs_vision/top_1'], 0.41602, decimal)
    self.assertAlmostEqual(
        metrics['test/vision_vs_spectrogram/top_1'], 0.50000, decimal)

    self.assertAlmostEqual(
        metrics['test/spectrogram_vs_waveform'], 11.18750, decimal)
    self.assertAlmostEqual(
        metrics['test/waveform_vs_spectrogram'], 7.81250, decimal)

    self.assertAlmostEqual(
        metrics['test/spectrogram_vs_waveform/top_1'], 0.41602, decimal)
    self.assertAlmostEqual(
        metrics['test/waveform_vs_spectrogram/top_1'], 0.41602, decimal)

    self.assertAlmostEqual(
        metrics['test/text_vs_vision'], 33.50000, decimal)
    self.assertAlmostEqual(
        metrics['test/vision_vs_text'], 28.62500, decimal)

    self.assertAlmostEqual(
        metrics['test/text_vs_vision/top_1'], 0.25000, decimal)
    self.assertAlmostEqual(
        metrics['test/vision_vs_text/top_1'], 0.25000, decimal)

    self.assertAlmostEqual(
        metrics['test/waveform_vs_text'], 20.00000, decimal)
    self.assertAlmostEqual(
        metrics['test/text_vs_waveform'], 15.18750, decimal)

    self.assertAlmostEqual(
        metrics['test/waveform_vs_text/top_1'], 0.33398, decimal)
    self.assertAlmostEqual(
        metrics['test/text_vs_waveform/top_1'], 0.33398, decimal)

    self.assertAlmostEqual(
        metrics['test/waveform_vs_vision'], 29.75000, decimal)
    self.assertAlmostEqual(
        metrics['test/vision_vs_waveform'], 29.75000, decimal)

    self.assertAlmostEqual(
        metrics['test/waveform_vs_vision/top_1'], 0.25000, decimal)
    self.assertAlmostEqual(
        metrics['test/vision_vs_waveform/top_1'], 0.25000, decimal)


class ObjectiveAggregatorTest(absltest.TestCase):

  def test_objective_aggregator(self):
    loss_weight = 2.0

    softmax_config = opt_config.SoftmaxCrossEntropy(
        name=constants.Objective.SOFTMAX_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=None)
    sigmoid_config = opt_config.SigmoidBinaryCrossEntropy(
        name=constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=None)
    objective_wrapper_config = opt_config.ObjectiveAggregator(
        name=constants.Objective.OBJECTIVE_AGGREGATOR,
        loss_weight=loss_weight, objectives=(sigmoid_config, softmax_config),
        dtype=None)

    softmax_objective = objectives.get_objective([softmax_config])[0]
    sigmoid_objective = objectives.get_objective([sigmoid_config])[0]
    objective_wrapper = objectives.get_objective([objective_wrapper_config])[0]

    data = {
        TARGETS: {
            ENCODER: {
                VISION: {
                    LABEL: jnp.array([[[0., 1.], [1., 0.]]]),
                },
            },
        },
        OUTPUTS: {
            ENCODER: {
                VISION: {
                    TOKEN_RAW: {
                        LOGITS: jnp.array([[[0., 1.], [0., 1.]]]),
                    },
                },
            },
        },
    }

    softmax_loss, softmax_metrics = softmax_objective(data)
    sigmoid_loss, sigmoid_metrics = sigmoid_objective(data)
    total_loss, total_metrics = objective_wrapper(data)

    self.assertAlmostEqual(softmax_loss, 0.81326, 4)
    self.assertAlmostEqual(sigmoid_loss, 0.75320, 4)
    self.assertAlmostEqual(total_loss, loss_weight * (0.81326 + 0.75320), 4)
    self.assertDictEqual(total_metrics, {**softmax_metrics, **sigmoid_metrics})

  def test_objective_aggregator_mean(self):
    softmax_config = opt_config.SoftmaxCrossEntropy(
        name=constants.Objective.SOFTMAX_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=None)
    sigmoid_config = opt_config.SigmoidBinaryCrossEntropy(
        name=constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=None)
    objective_wrapper_config = opt_config.ObjectiveAggregator(
        name=constants.Objective.OBJECTIVE_AGGREGATOR,
        loss_weight=1., objectives=(sigmoid_config, softmax_config),
        aggregation_method='mean',
        dtype=None)

    objective_wrapper = objectives.get_objective([objective_wrapper_config])[0]

    data = {
        TARGETS: {
            ENCODER: {
                VISION: {
                    LABEL: jnp.array([[[0., 1.], [1., 0.]]]),
                },
            },
        },
        OUTPUTS: {
            ENCODER: {
                VISION: {
                    TOKEN_RAW: {
                        LOGITS: jnp.array([[[0., 1.], [0., 1.]]]),
                    },
                },
            },
        },
    }

    total_loss, total_metrics = objective_wrapper(data)

    self.assertAlmostEqual(total_loss, (0.81326 + 0.75320) / 2, 4)
    self.assertNotEmpty(total_metrics)

  def test_nested_objective_aggregator(self):
    sigmoid_config = opt_config.SigmoidBinaryCrossEntropy(
        name=constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=None)
    nested_config = opt_config.ObjectiveAggregator(
        name=constants.Objective.OBJECTIVE_AGGREGATOR,
        loss_weight=1.,
        objectives=(sigmoid_config,),
        dtype=None)
    objective_wrapper_config = opt_config.ObjectiveAggregator(
        name=constants.Objective.OBJECTIVE_AGGREGATOR,
        loss_weight=1.,
        objectives=(nested_config,),
        dtype=None)

    with self.assertRaises(ValueError):
      objectives.get_objective([objective_wrapper_config])

  def test_objective_aggregator_default_dtype(self):
    sigmoid_config = opt_config.SigmoidBinaryCrossEntropy(
        name=constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=jnp.bfloat16)
    objective_wrapper_config = opt_config.ObjectiveAggregator(
        name=constants.Objective.OBJECTIVE_AGGREGATOR,
        loss_weight=1.,
        objectives=(sigmoid_config,),
        dtype=None)

    objective_wrapper = objectives.get_objective([objective_wrapper_config])[0]

    data = {
        TARGETS: {
            ENCODER: {
                VISION: {
                    LABEL: jnp.array([[[0., 1.], [1., 0.]]]),
                },
            },
        },
        OUTPUTS: {
            ENCODER: {
                VISION: {
                    TOKEN_RAW: {
                        LOGITS: jnp.array([[[0., 1.], [0., 1.]]]),
                    },
                },
            },
        },
    }

    total_loss, _ = objective_wrapper(data)
    self.assertEqual(total_loss.dtype, jnp.float32.dtype)

  def test_objective_aggregator_dtype(self):
    sigmoid_config = opt_config.SigmoidBinaryCrossEntropy(
        name=constants.Objective.SIGMOID_BINARY_CROSS_ENTROPY,
        modality_token_weights=(((VISION, TOKEN_RAW), 1.),),
        loss_weight=1.,
        predictions_key=LOGITS,
        route_key=ENCODER,
        targets_key=LABEL,
        left_shift_targets=False, dtype=jnp.float32)
    objective_wrapper_config = opt_config.ObjectiveAggregator(
        name=constants.Objective.OBJECTIVE_AGGREGATOR,
        loss_weight=1.,
        objectives=(sigmoid_config,),
        dtype=jnp.bfloat16)

    objective_wrapper = objectives.get_objective([objective_wrapper_config])[0]

    data = {
        TARGETS: {
            ENCODER: {
                VISION: {
                    LABEL: jnp.array([[[0., 1.], [1., 0.]]]),
                },
            },
        },
        OUTPUTS: {
            ENCODER: {
                VISION: {
                    TOKEN_RAW: {
                        LOGITS: jnp.array([[[0., 1.], [0., 1.]]]),
                    },
                },
            },
        },
    }

    total_loss, _ = objective_wrapper(data)
    self.assertEqual(total_loss.dtype, jnp.bfloat16.dtype)

if __name__ == '__main__':
  absltest.main()
