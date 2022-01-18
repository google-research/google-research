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

"""Tests for imagenet.models."""

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from jax import random
import jax.numpy as jnp

from aqt.jax import flax_layers as aqt_flax_layers  # pylint: disable=unused-import
from aqt.jax import hlo_utils
from aqt.jax.imagenet import hparams_config
from aqt.jax.imagenet import models
from aqt.jax.imagenet.configs import resnet101_bfloat16
from aqt.jax.imagenet.configs import resnet50_w4
from aqt.jax.imagenet.configs import resnet50_w4_a4_fixed
from aqt.jax.imagenet.configs import resnet50_w8
from aqt.jax.imagenet.configs import resnet50_w8_a8_fixed
from aqt.jax.imagenet.configs.paper import resnet50_bfloat16
from aqt.jax.imagenet.configs.paper import resnet50_w4_a4_auto
from aqt.jax.imagenet.configs.paper import resnet50_w8_a8_auto
from aqt.jax.imagenet.train_utils import create_model
from aqt.utils import hparams_utils

FLAGS = flags.FLAGS


class ModelsTest(parameterized.TestCase):

  def setUp(self):
    super(ModelsTest, self).setUp()
    self.rng_key = random.PRNGKey(0)

  def _check_weight_and_act(self, weight_prec, quant_act):
    quant_ops = 0
    if weight_prec:
      quant_ops += 1
    if quant_act and quant_act.prec:
      quant_ops += 1
    return quant_ops

  def _num_dense_floors(self, hparams):
    # Each dense layer can have one weight quant op and one input quant op.
    dense_floors = 0
    # There is 1 dense layer in the ResNet model.
    dense_floors += self._check_weight_and_act(hparams.dense_layer.weight_prec,
                                               hparams.dense_layer.quant_act)

    return dense_floors

  def _num_conv_floors(self, hparams):
    # Each conv layer can have one weight quant op and one input quant op.
    conv_floors = 0
    # There is 1 conv layer at ResNet model input (conv_init).
    conv_floors += self._check_weight_and_act(hparams.conv_init.weight_prec,
                                              hparams.conv_init.quant_act)
    for block in hparams.residual_blocks:
      # There are 3 conv layers per ResNet block.
      conv_floors += self._check_weight_and_act(block.conv_1.weight_prec,
                                                block.conv_1.quant_act)
      conv_floors += self._check_weight_and_act(block.conv_2.weight_prec,
                                                block.conv_2.quant_act)
      conv_floors += self._check_weight_and_act(block.conv_3.weight_prec,
                                                block.conv_3.quant_act)
      # There is 1 projection conv layer per group of blocks.
      if block.conv_proj is not None:
        conv_floors += self._check_weight_and_act(block.conv_proj.weight_prec,
                                                  block.conv_proj.quant_act)

    return conv_floors

  @parameterized.named_parameters(
      # Unquantized
      dict(
          testcase_name='resnet50_quantization_none',
          base_config_filename=resnet50_bfloat16),
      # Weights only
      dict(
          testcase_name='resnet50_quantization_weights_only_8',
          base_config_filename=resnet50_w8),
      dict(
          testcase_name='resnet50_quantization_weights_only_4',
          base_config_filename=resnet50_w4),
      # Weights and activations (fixed)
      dict(
          testcase_name='resnet50_quantization_weights_8_fixed_acts_8',
          base_config_filename=resnet50_w8_a8_fixed),
      dict(
          testcase_name='resnet50_quantization_weights_4_fixed_acts_4',
          base_config_filename=resnet50_w4_a4_fixed),
      # Weights and activations (automatic)
      dict(
          testcase_name='resnet50_quantization_weights_8_auto_acts_8',
          base_config_filename=resnet50_w8_a8_auto),
      dict(
          testcase_name='resnet50_quantization_weights_4_auto_acts_4',
          base_config_filename=resnet50_w4_a4_auto),
  )  # pylint: disable=line-too-long
  def test_create_model_object(self, base_config_filename):
    hparams = hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams,
        base_config_filename.get_config())
    model, _ = create_model(
        self.rng_key, 32, 16, jnp.float32, hparams.model_hparams, train=True)
    self.assertIsNotNone(model)

  @parameterized.named_parameters(
      # Unquantized
      dict(
          testcase_name='resnet50_quantization_none',
          expected_layers=50,
          expected_layer_stages=[3, 4, 6, 3],
          base_config_filename=resnet50_bfloat16,
      ),
      dict(
          testcase_name='resnet101_quantization_none',
          expected_layers=101,
          expected_layer_stages=[3, 4, 23, 3],
          base_config_filename=resnet101_bfloat16,
      ),
      # Weights only
      dict(
          testcase_name='resnet50_quantization_weights_only_8',
          expected_layers=50,
          expected_layer_stages=[3, 4, 6, 3],
          base_config_filename=resnet50_w8),
      # Weights and activations (fixed)
      dict(
          testcase_name='resnet50_quantization_weights_8_fixed_acts_8',
          expected_layers=50,
          expected_layer_stages=[3, 4, 6, 3],
          base_config_filename=resnet50_w8_a8_fixed),
      # Weights and activations (automatic)
      dict(
          testcase_name='resnet50_quantization_weights_8_auto_acts_8',
          expected_layers=50,
          expected_layer_stages=[3, 4, 6, 3],
          base_config_filename=resnet50_w8_a8_auto),
  )  # pylint: disable=line-too-long
  def test_count_resnet_layers(self, expected_layers, expected_layer_stages,
                               base_config_filename):
    counted_layers = 0
    layer_stages = []
    hparams = hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams,
        base_config_filename.get_config())
    # Count the layers in ResNet hparams
    if hasattr(hparams.model_hparams, 'dense_layer'):
      counted_layers += 1
    if hasattr(hparams.model_hparams, 'conv_init'):
      counted_layers += 1
    if hasattr(hparams.model_hparams, 'residual_blocks'):
      stage_block_count = 0
      for block in hparams.model_hparams.residual_blocks:
        stage_block_count += 1
        if hasattr(block, 'conv_1'):
          counted_layers += 1
        if hasattr(block, 'conv_2'):
          counted_layers += 1
        if hasattr(block, 'conv_3'):
          counted_layers += 1
        if hasattr(block, 'conv_proj') and block.conv_proj is not None:
          layer_stages.append(stage_block_count)
          stage_block_count = 0
      layer_stages.append(stage_block_count + 1)

    # If the first layer in residual blocks is a projection layer,
    #   the first element in layer_stages represents no blocks.
    if layer_stages[0] == 1:
      layer_stages = layer_stages[1:]

    self.assertEqual(counted_layers, expected_layers)
    self.assertEqual(layer_stages, expected_layer_stages)

  @parameterized.named_parameters(
      # Unquantized
      dict(
          testcase_name='resnet50_quantization_none',
          expected_floor_count=0,
          base_config_filename=resnet50_bfloat16,
      ),
      # Weights only
      dict(
          testcase_name='resnet50_quantization_weights_only_8',
          expected_floor_count=54,
          base_config_filename=resnet50_w8),
      # Weights and activations (fixed)
      dict(
          testcase_name='resnet50_quantization_weights_8_fixed_acts_8',
          expected_floor_count=108,
          base_config_filename=resnet50_w8_a8_fixed),
      # Weights and activations (automatic)
      dict(
          testcase_name='resnet50_quantization_weights_8_auto_acts_8',
          expected_floor_count=108,
          base_config_filename=resnet50_w8_a8_auto),
  )  # pylint: disable=line-too-long
  def test_count_floor_ops(self, base_config_filename, expected_floor_count):
    hparams = hparams_utils.load_hparams_from_config_dict(
        hparams_config.TrainingHParams, models.ResNet.HParams,
        base_config_filename.get_config())
    input_shape = (32, 16, 16, 3)
    model, init_state = create_model(
        self.rng_key,
        input_shape[0],
        input_shape[1],
        jnp.float32,
        hparams.model_hparams,
        train=False)
    hlo_proto = hlo_utils.load_hlo_proto_from_model(model, init_state,
                                                    [input_shape])
    floor_count = hlo_utils.count_ops_in_hlo_proto(hlo_proto, r'floor')
    self.assertEqual(floor_count, expected_floor_count)
    # Expected floor count
    expected_floor_count_from_hparams = 0
    expected_floor_count_from_hparams += self._num_dense_floors(
        hparams.model_hparams)
    expected_floor_count_from_hparams += self._num_conv_floors(
        hparams.model_hparams)
    self.assertEqual(floor_count, expected_floor_count_from_hparams)

if __name__ == '__main__':
  FLAGS.metadata_enabled = True  # Passes quantization information to HLO
  absltest.main()
