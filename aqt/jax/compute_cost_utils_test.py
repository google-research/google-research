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

"""Tests for aqt.jax.compute_cost_utils."""

import logging

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from jax import random
from jax._src.lax import lax
from jax.nn import initializers
import jax.numpy as jnp
import numpy as onp

from aqt.jax import compute_cost_utils
from aqt.jax import flax_layers as aqt_flax_layers
from aqt.jax import get_bounds
from aqt.jax import hlo_utils
from aqt.jax import quant_config
from aqt.jax import quantization
from aqt.jax.quantization import QuantOps
from aqt.jax.quantization import QuantType

FLAGS = flags.FLAGS


class ComputeCostUtilsTest(parameterized.TestCase):

  def setUp(self):
    super(ComputeCostUtilsTest, self).setUp()
    self.rng_key = random.PRNGKey(0)

  def compare_hlo_instructions(self, hlo_no_annotation, hlo_w_annotation):
    """Compares two HLO models to check if they only differ in metadata info."""
    instrs_n = []
    instrs_w = []
    # gather instructions from both HLO models
    for computation in hlo_no_annotation.computations:
      for instr in computation.instructions:
        instrs_n.append(instr)
    for computation in hlo_w_annotation.computations:
      for instr in computation.instructions:
        instrs_w.append(instr)

    self.assertEqual(len(instrs_n), len(instrs_w))
    for i, _ in enumerate(instrs_n):
      # check instructions with the opcode 'convolution'
      # the metadata field for instrs_w and instrs_n should be different.
      if (instrs_n[i].opcode == 'convolution' and
          instrs_w[i].opcode == 'convolution'):
        self.assertNotEqual(instrs_n[i].metadata, instrs_w[i].metadata)

      # remove metadata op_type and op_name
      instrs_n[i].metadata.op_type = ''
      instrs_w[i].metadata.op_type = ''
      instrs_n[i].metadata.op_name = ''
      instrs_w[i].metadata.op_name = ''
      # compare the rest of the instructions.
      self.assertEqual(instrs_n[i], instrs_w[i])

  class TestModelWith1Dense(nn.Module):
    """Test model with a single DenseAqt layer."""

    @nn.compact
    def __call__(self, inputs, hparams, num_classes, dtype=jnp.float32):
      output = aqt_flax_layers.DenseAqt(
          features=num_classes,
          dtype=dtype,
          train=False,
          quant_context=quant_config.QuantContext(
              update_bounds=False, collect_acts_stats=False),
          paxis_name='batch',
          hparams=hparams,
      )(inputs, padding_mask=None)
      return output

  class TestModelWith1Conv(nn.Module):
    """Test model with a single ConvAqt layer."""

    @nn.compact
    def __call__(self,
                 inputs,
                 hparams,
                 kernel_size,
                 num_filters,
                 strides,
                 dtype=jnp.float32):
      output = aqt_flax_layers.ConvAqt(
          features=num_filters,
          kernel_size=kernel_size,
          strides=strides,
          use_bias=False,
          dtype=dtype,
          train=False,
          quant_context=quant_config.QuantContext(update_bounds=False),
          paxis_name='batch',
          hparams=hparams)(
              inputs)
      return output

  class TestModelWith1DynamicMatmul(nn.Module):
    """Test model with a single dynamic matmul."""

    @nn.compact
    def __call__(self, lhs_act, rhs_act, lhs_prec, rhs_prec):
      get_bounds_hyper = get_bounds.GetBounds.Hyper(
          initial_bound=10.0,
          stddev_coeff=0,
          absdev_coeff=0,
          mix_coeff=0,
          granularity=quant_config.QuantGranularity.per_tensor)
      lhs_act_hparams = QuantOps.ActHParams(
          input_distribution='symmetric',
          bounds=get_bounds_hyper,
          prec=lhs_prec)
      rhs_act_hparams = QuantOps.ActHParams(
          input_distribution='symmetric',
          bounds=get_bounds_hyper,
          prec=rhs_prec)
      lhs_get_bounds_params = get_bounds.GetBounds.Params(
          update_stats=False, update_bounds=False, module_name='lhs')
      rhs_get_bounds_params = get_bounds.GetBounds.Params(
          update_stats=False, update_bounds=False, module_name='rhs')
      output = quantization.quantized_dynamic_dot_general(
          lhs_act=lhs_act,
          rhs_act=rhs_act,
          lhs_act_hparams=lhs_act_hparams,
          rhs_act_hparams=rhs_act_hparams,
          dot_dimension_numbers=(((1,), (0,)), ((), ())),
          quant_type=QuantType.aqt,
          lhs_get_bounds_params=lhs_get_bounds_params,
          rhs_get_bounds_params=rhs_get_bounds_params)
      return output

  @parameterized.named_parameters(
      # TestModelWith1Dense
      dict(
          testcase_name='single_dense_layer_bfloat16',
          modelclass=TestModelWith1Dense,
          input_shapes=[(1, 8)],
          model_kwargs={
              'num_classes': 2,
              'hparams': aqt_flax_layers.DenseAqt.HParams(
                  weight_prec=None,
                  quant_type=QuantType.fake_quant,
                  quant_act=None,
                  weight_quant_granularity=quant_config.QuantGranularity.per_channel
              ),
          },
          expected_compute_cost=8 * 2 * (16 * 16),
          expected_compute_cost_ratio=1.0,
          expected_compute_cost_linear=8 * 2 * (16),
          expected_compute_cost_ratio_linear=1.0,
          expected_memory_cost=8 * 2 * (16),
          expected_memory_cost_ratio=1.0,
      ),
      dict(
          testcase_name='single_dense_layer_w8_a8',
          modelclass=TestModelWith1Dense,
          input_shapes=[(1, 8)],
          model_kwargs={
              'num_classes': 2,
              'hparams': aqt_flax_layers.DenseAqt.HParams(
                  weight_prec=8,
                  quant_type=QuantType.fake_quant,
                  quant_act=QuantOps.ActHParams(
                      input_distribution=QuantOps.ActHParams.InputDistribution.positive,
                      prec=8,
                      bounds=1.0,
                  ),
                  weight_quant_granularity=quant_config.QuantGranularity.per_channel
              ),
          },
          expected_compute_cost=8 * 2 * (8 * 8),
          expected_compute_cost_ratio=0.25,
          expected_compute_cost_linear=8 * 2 * (8),
          expected_compute_cost_ratio_linear=0.5,
          expected_memory_cost=8 * 2 * (8),
          expected_memory_cost_ratio=0.5,
      ),

      # TestModelWith1Conv
      dict(
          testcase_name='single_conv_layer_bfloat16',
          modelclass=TestModelWith1Conv,
          input_shapes=[(1, 8, 8, 3)],
          model_kwargs={
              'kernel_size': (3, 3),
              'num_filters': 16,
              'strides': (1, 1),
              'hparams': aqt_flax_layers.ConvAqt.HParams(
                  weight_prec=None,
                  quant_type=QuantType.fake_quant,
                  quant_act=None,
              ),
          },
          expected_compute_cost=(3 * 3) * (8 * 8) * 3 * 16 * (16 * 16),
          expected_compute_cost_ratio=1.0,
          expected_compute_cost_linear=(3 * 3) * (8 * 8) * 3 * 16 * (16),
          expected_compute_cost_ratio_linear=1.0,
          expected_memory_cost=(3 * 3) * 3 * 16 * (16),
          expected_memory_cost_ratio=1.0,
      ),
      dict(
          testcase_name='single_conv_layer_bfloat16_strided',
          modelclass=TestModelWith1Conv,
          input_shapes=[(1, 8, 8, 3)],
          model_kwargs={
              'kernel_size': (3, 3),
              'num_filters': 16,
              'strides': (4, 2),
              'hparams': aqt_flax_layers.ConvAqt.HParams(
                  weight_prec=None,
                  quant_type=QuantType.fake_quant,
                  quant_act=None,
              ),
          },
          expected_compute_cost=(3 * 3) * ((8 / 4) * (8 / 2)) * 3 * 16 * (16 * 16),
          expected_compute_cost_ratio=1.0,
          expected_compute_cost_linear=(3 * 3) * ((8 / 4) * (8 / 2)) * 3 * 16 * (16),
          expected_compute_cost_ratio_linear=1.0,
          expected_memory_cost=(3 * 3) * 3 * 16 * (16),
          expected_memory_cost_ratio=1.0,
      ),
      dict(
          testcase_name='single_conv_layer_bfloat16_3d',
          modelclass=TestModelWith1Conv,
          input_shapes=[(1, 8, 8, 8, 3)],
          model_kwargs={
              'kernel_size': (3, 3, 3),
              'num_filters': 16,
              'strides': (1, 1, 1),
              'hparams': aqt_flax_layers.ConvAqt.HParams(
                  weight_prec=None,
                  quant_type=QuantType.fake_quant,
                  quant_act=None,
              ),
          },
          expected_compute_cost=(3 * 3 * 3) * (8 * 8 * 8) * 3 * 16 * (16 * 16),
          expected_compute_cost_ratio=1.0,
          expected_compute_cost_linear=(3 * 3 * 3) * (8 * 8 * 8) * 3 * 16 * (16),
          expected_compute_cost_ratio_linear=1.0,
          expected_memory_cost=(3 * 3 * 3) * 3 * 16 * (16),
          expected_memory_cost_ratio=1.0,
      ),
      dict(
          testcase_name='single_conv_layer_w4_a2',
          modelclass=TestModelWith1Conv,
          input_shapes=[(1, 8, 8, 3)],
          model_kwargs={
              'kernel_size': (3, 3),
              'num_filters': 16,
              'strides': (1, 1),
              'hparams': aqt_flax_layers.ConvAqt.HParams(
                  weight_prec=4,
                  quant_type=QuantType.fake_quant,
                  quant_act=QuantOps.ActHParams(
                      input_distribution=QuantOps.ActHParams.InputDistribution.positive,
                      prec=2,
                      bounds=1.0,
                  ),
              ),
          },
          expected_compute_cost=(3 * 3) * (8 * 8) * 3 * 16 * (4 * 2),
          expected_compute_cost_ratio=0.03125,
          expected_compute_cost_linear=(3 * 3) * (8 * 8) * 3 * 16 * (4),
          expected_compute_cost_ratio_linear=0.25,
          expected_memory_cost=(3 * 3) * 3 * 16 * (4),
          expected_memory_cost_ratio=0.25,
      ),
      # TestModelWith1DynamicMatmul
      dict(
          testcase_name='single_dynamic_matmul_layer_bfloat16',
          modelclass=TestModelWith1DynamicMatmul,
          input_shapes=[(1, 8), (8, 1)],
          model_kwargs={'lhs_prec': None,
                        'rhs_prec': None},
          expected_compute_cost=8 * (16 * 16),
          expected_compute_cost_ratio=1.0,
          expected_compute_cost_linear=8 * (16),
          expected_compute_cost_ratio_linear=1.0,
          expected_memory_cost=0,
          expected_memory_cost_ratio=1.0,
      ),
      dict(
          testcase_name='single_dynamic_matmul_layer_l8_r8',
          modelclass=TestModelWith1DynamicMatmul,
          input_shapes=[(1, 8), (8, 1)],
          model_kwargs={'lhs_prec': 8,
                        'rhs_prec': 8},
          expected_compute_cost=8 * (8 * 8),
          expected_compute_cost_ratio=0.25,
          expected_compute_cost_linear=8 * 8,
          expected_compute_cost_ratio_linear=0.5,
          expected_memory_cost=0,
          expected_memory_cost_ratio=1.0,
      ),
      dict(
          testcase_name='single_dynamic_matmul_layer_l8_r4',
          modelclass=TestModelWith1DynamicMatmul,
          input_shapes=[(1, 8), (8, 1)],
          model_kwargs={'lhs_prec': 8,
                        'rhs_prec': 4},
          expected_compute_cost=8 * (8 * 4),
          expected_compute_cost_ratio=0.125,
          expected_compute_cost_linear=8 * (8),
          expected_compute_cost_ratio_linear=0.5,
          expected_memory_cost=0,
          expected_memory_cost_ratio=1.0,
      ),
  )  # pylint: disable=line-too-long
  def test_estimate_simple_model_cost(
      self, modelclass, input_shapes, model_kwargs, expected_compute_cost,
      expected_compute_cost_ratio, expected_compute_cost_linear,
      expected_compute_cost_ratio_linear, expected_memory_cost,
      expected_memory_cost_ratio):
    module = modelclass()
    input_shapes_with_type = [(sh, jnp.float32) for sh in input_shapes]
    dummy_inputs = [
        jnp.ones(input_shape, dtype=dtype)
        for (input_shape, dtype) in input_shapes_with_type
    ]
    init_state = module.init(random.PRNGKey(0), *dummy_inputs, **model_kwargs)

    hlo_proto = hlo_utils.load_hlo_proto_from_model(module, init_state,
                                                    input_shapes,
                                                    **model_kwargs)
    compute_result = compute_cost_utils.estimate_compute_cost(hlo_proto)
    memory_result = compute_cost_utils.estimate_memory_cost(hlo_proto)
    logging.info('compute cost result is %s', compute_result)
    logging.info('memory cost result is %s', memory_result)
    self.assertEqual(compute_result['compute_cost'], expected_compute_cost)
    self.assertEqual(memory_result['memory_cost'], expected_memory_cost)
    self.assertEqual(compute_result['compute_cost_ratio_to_bfloat16'],
                     expected_compute_cost_ratio)
    self.assertEqual(memory_result['memory_cost_ratio_to_bfloat16'],
                     expected_memory_cost_ratio)
    self.assertEqual(compute_result['compute_cost_linear'],
                     expected_compute_cost_linear)
    self.assertEqual(compute_result['compute_cost_ratio_to_bfloat16_linear'],
                     expected_compute_cost_ratio_linear)

  @parameterized.named_parameters(
      # TestModelWith1Dense
      dict(
          testcase_name='single_dense_layer_bfloat16_batch_size',
          modelclass=TestModelWith1Dense,
          input_shape_per_sample=(16,),
          model_kwargs={
              'num_classes':
                  20,
              'hparams':
                  aqt_flax_layers.DenseAqt.HParams(
                      weight_prec=None,
                      quant_act=None,
                      quant_type=QuantType.fake_quant,
                      weight_quant_granularity=quant_config.QuantGranularity
                      .per_channel)
          },
      ),
      # TestModelWith1Conv
      dict(
          testcase_name='single_conv_layer_bfloat16_batch_size',
          modelclass=TestModelWith1Conv,
          input_shape_per_sample=(16, 16, 3),
          model_kwargs={
              'kernel_size': (3, 3),
              'num_filters':
                  16,
              'strides': (2, 2),
              'hparams':
                  aqt_flax_layers.ConvAqt.HParams(
                      weight_prec=None,
                      quant_act=None,
                      quant_type=QuantType.fake_quant,
                  )
          },
      ),
  )
  def test_batch_size_has_no_effect_on_cost(self, modelclass,
                                            input_shape_per_sample,
                                            model_kwargs):
    expected_compute_cost = None
    expected_memory_cost = None
    batch_size_list = [32, 64, 128, 256, 512, 1024]

    module = modelclass()

    # Sweep over the batch size list
    for batch_size in batch_size_list:
      input_shape = (batch_size,) + input_shape_per_sample
      init_state = module.init(
          random.PRNGKey(0), jnp.ones(input_shape, jnp.float32), **model_kwargs)
      hlo_proto = hlo_utils.load_hlo_proto_from_model(module, init_state,
                                                      [input_shape],
                                                      **model_kwargs)
      del init_state
      compute_result = compute_cost_utils.estimate_compute_cost(hlo_proto)
      memory_result = compute_cost_utils.estimate_memory_cost(hlo_proto)
      # Save the first cost and compare it with the rest
      if expected_compute_cost is None:
        expected_compute_cost = compute_result['compute_cost']
      else:
        self.assertEqual(compute_result['compute_cost'], expected_compute_cost)
      if expected_memory_cost is None:
        expected_memory_cost = memory_result['memory_cost']
      else:
        self.assertEqual(memory_result['memory_cost'], expected_memory_cost)

  @parameterized.named_parameters(
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
  )
  def test_check_value_inside_and_outside_of_context_conv_general(
      self, weight_prec):
    original_op_name = 'conv_general_dilated'
    # The 'name' in primitive should change in the context in 'flax_layers'
    # if the context is enabled
    self.assertEqual(original_op_name, lax.conv_general_dilated_p.name)

    with compute_cost_utils.ConvMetadataMonkeyPatch(
        weight_prec=weight_prec, act_prec=None):
      self.assertNotEqual(original_op_name, lax.conv_general_dilated_p.name)
    self.assertEqual(original_op_name, lax.conv_general_dilated_p.name)

  @parameterized.named_parameters(
      dict(testcase_name='quant_8bit', weight_prec=8, acts_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4, acts_prec=4),
  )
  def test_annotation_only_changes_hlo_metadata_conv(self, weight_prec,
                                                     acts_prec):
    FLAGS.metadata_enabled = False
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        prec=acts_prec,
        bounds=1.0)
    input_shape = (1, 8, 8, 3)
    module_no_annotation = aqt_flax_layers.ConvAqt(
        features=4,
        kernel_size=(3, 3),
        padding='VALID',
        paxis_name='batch',
        quant_context=quant_config.QuantContext(update_bounds=False),
        train=False,
        hparams=aqt_flax_layers.ConvAqt.HParams(
            weight_prec=weight_prec,
            quant_act=quant_act,
            quant_type=QuantType.fake_quant),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        dtype=jnp.float32)

    init_state = module_no_annotation.init(self.rng_key,
                                           jnp.ones(input_shape, jnp.float32))
    output_no_annotation = module_no_annotation.apply(init_state,
                                                      jnp.ones(input_shape))

    hlo_no_annotation = hlo_utils.load_hlo_proto_from_model(
        module_no_annotation, init_state, [input_shape])
    del init_state

    FLAGS.metadata_enabled = True
    module_w_annotation = aqt_flax_layers.ConvAqt(
        features=4,
        kernel_size=(3, 3),
        padding='VALID',
        paxis_name='batch',
        quant_context=quant_config.QuantContext(update_bounds=False),
        train=False,
        hparams=aqt_flax_layers.ConvAqt.HParams(
            weight_prec=weight_prec,
            quant_act=quant_act,
            quant_type=QuantType.fake_quant),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        dtype=jnp.float32)

    init_state = module_w_annotation.init(self.rng_key,
                                          jnp.ones(input_shape, jnp.float32))
    output_w_annotation = module_w_annotation.apply(init_state,
                                                    jnp.ones(input_shape))

    hlo_w_annotation = hlo_utils.load_hlo_proto_from_model(
        module_w_annotation, init_state, [input_shape])
    del init_state

    onp.testing.assert_array_equal(output_no_annotation, output_w_annotation)
    self.compare_hlo_instructions(hlo_no_annotation, hlo_w_annotation)

  @parameterized.named_parameters(
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
  )
  def test_check_value_inside_and_outside_of_context_dot_general(
      self, weight_prec):
    original_op_name = 'dot_general'
    # The 'name' in primitive should change in the context in 'flax_layers'
    # if the context is enabled.
    self.assertEqual(original_op_name, lax.dot_general_p.name)

    with compute_cost_utils.DotMetadataMonkeyPatch(
        lhs_prec=None, rhs_prec=weight_prec, rhs_is_weight=True):
      self.assertNotEqual(original_op_name, lax.dot_general_p.name)
    self.assertEqual(original_op_name, lax.dot_general_p.name)

  @parameterized.named_parameters(
      dict(
          testcase_name='quant_8bit',
          weight_prec=8,
          acts_prec=8,
      ),)
  def test_annotation_only_changes_hlo_metadata_dense(self, weight_prec,
                                                      acts_prec):
    FLAGS.metadata_enabled = False
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        prec=acts_prec,
        bounds=1.0)
    input_shape = (1, 16)
    module_no_annotation = aqt_flax_layers.DenseAqt(
        features=4,
        use_bias=False,
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        paxis_name='batch',
        train=False,
        hparams=aqt_flax_layers.DenseAqt.HParams(
            weight_prec=weight_prec,
            quant_act=quant_act,
            quant_type=QuantType.fake_quant,
            weight_quant_granularity=quant_config.QuantGranularity.per_channel),
        dtype=jnp.float32)

    init_state = module_no_annotation.init(
        self.rng_key, jnp.ones(input_shape, jnp.float32), padding_mask=None)
    output_no_annotation = module_no_annotation.apply(
        init_state, jnp.ones(input_shape), padding_mask=None)

    hlo_no_annotation = hlo_utils.load_hlo_proto_from_model(
        module_no_annotation, init_state, [input_shape], padding_mask=None)
    del init_state

    FLAGS.metadata_enabled = True
    module_w_annotation = aqt_flax_layers.DenseAqt(
        features=4,
        use_bias=False,
        paxis_name='batch',
        train=False,
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        dtype=jnp.float32,
        hparams=aqt_flax_layers.DenseAqt.HParams(
            weight_prec=weight_prec,
            quant_act=quant_act,
            quant_type=QuantType.fake_quant,
            weight_quant_granularity=quant_config.QuantGranularity.per_channel),
    )

    init_state = module_w_annotation.init(
        self.rng_key, jnp.ones(input_shape, jnp.float32), padding_mask=None)
    output_w_annotation = module_w_annotation.apply(
        init_state, jnp.ones(input_shape), padding_mask=None)

    hlo_w_annotation = hlo_utils.load_hlo_proto_from_model(
        module_w_annotation, init_state, [input_shape], padding_mask=None)
    del init_state

    onp.testing.assert_array_equal(output_no_annotation, output_w_annotation)
    self.compare_hlo_instructions(hlo_no_annotation, hlo_w_annotation)


if __name__ == '__main__':
  FLAGS.metadata_enabled = True  # Passes quantization information to HLO
  absltest.main()
