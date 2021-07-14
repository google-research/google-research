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

"""Tests for aqt.jax.quantization."""

import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
import jax
from jax import random
import jax.numpy as jnp
import numpy as onp

from aqt.jax import fp_cast
from aqt.jax import get_bounds
from aqt.jax import primitives
from aqt.jax import quant_config
from aqt.jax import quantization
from aqt.jax import test_utils
from aqt.jax.get_bounds import GetBounds
from aqt.jax.quantization import QuantOps
from aqt.jax.quantization import QuantType
from aqt.jax.quantization import SCALE_DTYPE

fp32 = onp.float32
test_utils.configure_jax()


class QuantOpsTest(parameterized.TestCase):

  def setUp(self):
    super(QuantOpsTest, self).setUp()
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = True

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', bounds=6.0, prec=2),
      dict(testcase_name='prec_4', bounds=6.0, prec=4),
      dict(testcase_name='prec_8', bounds=6.0, prec=8),
      dict(
          testcase_name='2_features_prec_8',
          bounds=[6., 12.],
          prec=8),
  )
  def test_attributes_create_positive(self, bounds, prec):
    bounds = jnp.array(bounds)
    relu6 = QuantOps.create_positive(bounds=bounds, prec=prec)
    onp.testing.assert_array_equal(relu6._scale, 2**prec / bounds)
    self.assertEqual(relu6._symmetric, False)
    self.assertEqual(relu6._prec, prec)

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', bounds=6.0, prec=2),
      dict(testcase_name='prec_4', bounds=6.0, prec=4),
      dict(testcase_name='prec_8', bounds=6.0, prec=8),
      dict(
          testcase_name='2_features_prec_8',
          bounds=[6., 12.],
          prec=8),
  )
  def test_attributes_create_symmetric(self, bounds, prec):
    bounds = jnp.array(bounds)
    act_signed = QuantOps.create_symmetric(
        bounds=bounds, prec=prec, half_shift=False)
    onp.testing.assert_array_equal(act_signed._scale,
                                   (2**(prec - 1) - 1) / bounds)
    self.assertEqual(act_signed._symmetric, True)
    self.assertEqual(act_signed._prec, prec)

  @parameterized.named_parameters(
      dict(
          testcase_name='fp8_143',
          weight_range=[2.0, 64.0],
          weight_shape=(10, 1),
          fp_quant=QuantOps.FloatQuant(
              is_scaled=True,
              fp_spec=QuantOps.FloatQuant.FloatPrec(
                  exp_min=-11,
                  exp_max=4,
                  sig_bits=3,
              ),
          ),
      ),
      dict(
          testcase_name='fp8_152',
          weight_range=[2.0, 64.0],
          weight_shape=(10, 1),
          fp_quant=QuantOps.FloatQuant(
              is_scaled=True,
              fp_spec=QuantOps.FloatQuant.FloatPrec(
                  exp_min=-23,
                  exp_max=8,
                  sig_bits=2,
              ),
          ),
      ),
  )
  def test_attributes_create_weights_op_fp(
      self,
      weight_range,
      weight_shape,
      fp_quant,
  ):
    weights = jnp.array(
        fp32(onp.random.uniform(*weight_range, size=weight_shape)))
    axis = None if weight_shape[1] == 1 else 0
    weights_quant_op = QuantOps.create_weights_ops(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=fp_quant, axis=axis, half_shift=False))
    max_weight = onp.max(abs(weights), axis=0)
    onp.testing.assert_array_equal(
        jnp.squeeze(weights_quant_op._scale),
        jnp.exp2(-jnp.floor(jnp.log2(max_weight))))
    self.assertEqual(weights_quant_op._symmetric, True)
    self.assertIs(weights_quant_op._prec, fp_quant)
    weights_scaled = (weights * weights_quant_op._scale).astype(weights.dtype)
    weights_quant_expected = fp_cast.downcast_sat_ftz(
        weights_scaled,
        fp_quant.fp_spec.exp_min,
        fp_quant.fp_spec.exp_max,
        fp_quant.fp_spec.sig_bits,
    )
    weights_quant_calculated = weights_quant_op.to_quantized(
        weights, dtype=SCALE_DTYPE)
    onp.testing.assert_array_equal(weights_quant_expected,
                                   weights_quant_calculated)
    # Test the lower (23 - fp_quant.fp_spec.sig_bits) bits of the calculated
    # quantized weights are zero.
    sig_mask = jnp.int32((1 << (23 - fp_quant.fp_spec.sig_bits)) - 1)
    onp.testing.assert_array_equal(
        weights_quant_calculated.view(jnp.int32) & sig_mask,
        jnp.zeros_like(weights))

  @parameterized.named_parameters(
      dict(
          testcase_name='fp_act_symmetric',
          act_distribution='symmetric',
          use_hparams_bounds=False,
      ),
      # TODO(b/193561347): FP quantization with positive input distribution is
      # not supported yet
      dict(
          testcase_name='fp_act_positive',
          act_distribution='positive',
          use_hparams_bounds=False,
      ),
      dict(
          testcase_name='fp_act_symmetric_hyper_bounds',
          act_distribution='symmetric',
          use_hparams_bounds=True,
      ),
      dict(
          testcase_name='fp_act_positive_hyper_bounds',
          act_distribution='positive',
          use_hparams_bounds=True,
      ),
  )
  def test_attributes_create_acts_op_fp(
      self,
      act_distribution,
      use_hparams_bounds,
  ):
    inputs = jnp.array(fp32(2.0 * onp.random.uniform(0, 1.0, size=(10, 4))))
    fp_quant = QuantOps.FloatQuant(
        is_scaled=True,
        fp_spec=QuantOps.FloatQuant.FloatPrec(
            exp_min=-15,
            exp_max=15,
            sig_bits=2,
        ),
    )
    if use_hparams_bounds:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=1,
          absdev_coeff=0,
          mix_coeff=1,
          reset_stats=True,
          ema_coeff=None,
          use_cams=False,
          granularity=quant_config.QuantGranularity.per_tensor)
    else:
      bounds = 6.0

    hparams = QuantOps.ActHParams(
        input_distribution=act_distribution, bounds=bounds, prec=fp_quant,
        half_shift=False)

    class TestModule(nn.Module):
      hparams: QuantOps.ActHParams

      @nn.compact
      def __call__(self, inputs):
        return QuantOps.create_input_ops(
            inputs,
            hparams=hparams,
            get_bounds_params=GetBounds.Params(
                update_stats=False,
                update_bounds=False))

    test_module = TestModule(hparams=hparams)
    state = test_module.init(jax.random.PRNGKey(0), inputs=inputs)
    act_quant_op = test_module.apply(state, inputs=inputs)

    act_scaled = (inputs * act_quant_op._scale).astype(inputs.dtype)
    act_quant_expected = fp_cast.downcast_sat_ftz(
        act_scaled,
        fp_quant.fp_spec.exp_min,
        fp_quant.fp_spec.exp_max,
        fp_quant.fp_spec.sig_bits,
    )
    act_quant_calculated = act_quant_op.to_quantized(inputs, dtype=SCALE_DTYPE)
    onp.testing.assert_array_equal(act_quant_expected, act_quant_calculated)

  @parameterized.named_parameters(
      dict(
          testcase_name='pos_weight_prec_2',
          weight_range=[2.0, 10.0],
          weight_shape=(10, 1),
          prec=2),
      dict(
          testcase_name='pos_weight_prec_4',
          weight_range=[2.0, 10.0],
          weight_shape=(10, 1),
          prec=4),
      dict(
          testcase_name='pos_weight_prec_8',
          weight_range=[2.0, 10.0],
          weight_shape=(10, 1),
          prec=8),
      dict(
          testcase_name='neg_weight_prec_8',
          weight_range=[-12.0, 2.0],
          weight_shape=(10, 1),
          prec=8),
      dict(
          testcase_name='neg_weight_2_features_prec_8',
          weight_range=[-12.0, 2.0],
          weight_shape=(10, 2),
          prec=8),
  )
  def test_attributes_create_weights_ops(self, weight_range, weight_shape,
                                         prec):
    weights = jnp.array(
        fp32(
            onp.random.uniform(
                weight_range[0], weight_range[1], size=weight_shape)))
    axis = 0 if weight_shape[1] != 1 else None
    weights_quant = QuantOps.create_weights_ops(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=axis, half_shift=False))
    max_weight = onp.max(abs(weights), axis=0)
    onp.testing.assert_array_equal(
        jnp.squeeze(weights_quant._scale), (2**(prec - 1) - 1) / max_weight)
    self.assertEqual(weights_quant._symmetric, True)
    self.assertEqual(weights_quant._prec, prec)

  @parameterized.named_parameters(
      dict(testcase_name='per_layer_quant', axis=None),
      dict(testcase_name='per_channel_quant', axis=(0,)))
  def test_weight_scale_shape_is_expected(self, axis):
    # Tests if scale is as expected for weights quantization.

    num_features = 4
    expected_scale_shape = (1, 1) if axis is None else (1, num_features)

    # Weight Quantization
    weights = jnp.array(
        fp32(2.0 * onp.random.uniform(0, 1.0, size=(10, num_features))))
    _ = QuantOps.create_weights_fake_quant(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=8.0,
            axis=axis,
            expected_scale_shape=expected_scale_shape,
            half_shift=False))

  def test_inputs_scale_shape_is_expected(self):
    # Inputs quantization
    inputs = jnp.array(fp32(2.0 * onp.random.uniform(0, 1.0, size=(10, 4))))
    bounds = 6.0
    expected_inputs_scale_shape = ()

    _ = QuantOps.create_inputs_fake_quant(
        inputs=inputs,
        hparams=QuantOps.ActHParams(
            input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
            bounds=bounds,
            prec=8.0,
            half_shift=False),
        get_bounds_params=GetBounds.Params(
            update_stats=False,
            update_bounds=False,
            expected_bounds_shape=expected_inputs_scale_shape))

  @parameterized.named_parameters(
      dict(testcase_name='prec_2',
           prec=2), dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8))
  def test_positive_activation_quantization_clips_outside_bounds(self, prec):
    # Activation values less than 0 get clipped to 0, and values greater than
    # upper_bound get clipped to upper_bound
    relu6 = QuantOps.create_positive(bounds=6.0, prec=prec)
    activation = jnp.array(fp32([-0.5, 6.2, 3.141]))
    quantized_activations = relu6.to_quantized(activation, dtype=SCALE_DTYPE)
    onp.testing.assert_array_equal(quantized_activations[0:2],
                                   [0.0, 2**prec - 1])
    activations = relu6.from_quantized(quantized_activations, dtype=jnp.float32)
    max_clipped_val = (2**prec - 1) * (6.0 / 2**prec)
    onp.testing.assert_array_equal(activations[0:2], [0.0, max_clipped_val])

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8)
  )
  def test_per_feature_dim_unsigned_activation_quantization_clips_outside_bounds(
      self, prec):
    # Activation values less than -upper_bound get clipped to -upper_bound, and
    # values greater than upper_bound get clipped to upper_bound
    act_quant = QuantOps.create_symmetric(
        bounds=jnp.array([[6.0, 8.0]]), prec=prec, half_shift=False)
    activation = jnp.array(fp32([[-7, -8.9], [6.2, 9.4], [0, 0.]]))
    quantized_activations = act_quant.to_quantized(
        activation, dtype=SCALE_DTYPE)
    onp.testing.assert_array_equal(
        quantized_activations,
        jnp.array([[-2**(prec - 1.0) + 1.0], [2**(prec - 1.0) - 1.0], [0.0]]) *
        jnp.array([[1., 1.]]))
    activations = act_quant.from_quantized(
        quantized_activations, dtype=jnp.float32)
    onp.testing.assert_array_equal(activations,
                                   [[-6.0, -8.0], [6.0, 8.], [0, 0.]])

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8)
  )
  def test_scale_invariance_signed_activation_quantization(self, prec):
    # Scaling activation by power of 2 and bounds by same factor,
    # should scale the output by the same scale.
    activations = random.uniform(random.PRNGKey(0), (10, 1))
    act_scale = 8.
    scaled_activations = activations * act_scale

    bounds = 6.

    activations = QuantOps.create_inputs_fake_quant(
        inputs=activations,
        get_bounds_params=GetBounds.Params(
            update_stats=False, update_bounds=False),
        hparams=QuantOps.ActHParams(
            input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
            bounds=bounds,
            prec=prec,
            half_shift=False))

    scaled_activations = QuantOps.create_inputs_fake_quant(
        inputs=scaled_activations,
        get_bounds_params=GetBounds.Params(
            update_stats=False, update_bounds=False),
        hparams=QuantOps.ActHParams(
            input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
            bounds=bounds * act_scale,
            prec=prec,
            half_shift=False))
    onp.testing.assert_array_equal(activations * act_scale, scaled_activations)

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8)
  )
  def test_per_feature_dim_scale_invariance_pos_activation_quantization(
      self, prec):
    # Scaling each channel of activations by a different power of 2 and upper
    # bound with same scale, should scale the respective channel of output by
    # the same scale.
    activations = random.uniform(random.PRNGKey(0), (3, 4))
    act_scale = 2**jnp.arange(4)
    scaled_activations = activations * act_scale[jnp.newaxis, :]

    upper_bound = 6.0 * jnp.ones((3, 4), jnp.float32)

    act_quant_ops = QuantOps.create_positive(bounds=upper_bound, prec=prec)
    activations = act_quant_ops.fake_quant(
        activations, quantized_type=SCALE_DTYPE)

    scaled_act_quant_ops = QuantOps.create_positive(
        bounds=upper_bound * act_scale[jnp.newaxis, :], prec=prec)
    scaled_activations = scaled_act_quant_ops.fake_quant(
        scaled_activations, quantized_type=SCALE_DTYPE)
    onp.testing.assert_array_equal(activations * act_scale[jnp.newaxis, :],
                                   scaled_activations)

  @parameterized.named_parameters(
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8))
  def test_int_positive_act_quantization(self, prec):
    # Integer activations within upper_bound and upper_bound == 2^i s.t. i<prec
    # quantizes correctly.
    upper_bound = 2**(prec - 3)
    activations = random.randint(random.PRNGKey(0), (10, 1), 0, upper_bound)

    rescaled_activations = QuantOps.create_inputs_fake_quant(
        inputs=activations,
        get_bounds_params=GetBounds.Params(
            update_stats=False, update_bounds=False),
        hparams=QuantOps.ActHParams(
            input_distribution=QuantOps.ActHParams.InputDistribution.positive,
            bounds=upper_bound,
            prec=prec,
            half_shift=False))
    onp.testing.assert_array_equal(activations, rescaled_activations)

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8)
  )
  def test_int_symmetric_act_quantization(self, prec):
    # Integer activations within bounds and abs(bounds) == 2^(prec -1) - 1
    # quantizes correctly.
    bounds = 2**(prec - 1) - 1
    activations = random.randint(random.PRNGKey(0), (10, 1), -bounds, bounds)
    rescaled_activations = QuantOps.create_inputs_fake_quant(
        inputs=activations,
        get_bounds_params=GetBounds.Params(
            update_stats=False, update_bounds=False),
        hparams=QuantOps.ActHParams(
            input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
            bounds=bounds,
            prec=prec,
            half_shift=False))

    onp.testing.assert_array_equal(activations, rescaled_activations)

  @parameterized.named_parameters(
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8))
  def test_float_weights_quantization(self, prec):
    # Tests that quantized and rescaled float weights are close to original
    # weights.
    weights = jnp.array(fp32(2.0 * onp.random.uniform(0, 1.0, size=(10, 1))))
    rescaled_weights = QuantOps.create_weights_fake_quant(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=None, half_shift=False))
    test_utils.assert_all_close_prec(weights, rescaled_weights, prec=prec)

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8)
  )
  def test_full_range_int_weight_quantization(self, prec):
    # Integer weights in full range [-maxmin_signed_int, maxmin_signed_int]
    # quantizes correctly.
    minval = -2**(prec - 1) + 1
    maxval = 2**(prec - 1) - 1
    weights = random.randint(random.PRNGKey(0), (10, 1), minval, maxval + 1)
    weights = jax.ops.index_update(weights, jax.ops.index[0, :], maxval)
    weight_quant = QuantOps.create_weights_ops(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=None, half_shift=False))
    quantized_weights = weight_quant.to_quantized(weights, dtype=SCALE_DTYPE)
    onp.testing.assert_array_equal(quantized_weights[0],
                                   (2**(prec - 1.0) - 1.0))
    rescaled_weights = weight_quant.from_quantized(
        quantized_weights, dtype=jnp.float32)
    onp.testing.assert_array_equal(weights, rescaled_weights)

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8))
  def test_scale_invariance_weight_quantization(self, prec):
    # Scaling weights by power of 2, should scale the output by the same scale.
    weights = random.uniform(random.PRNGKey(0), (10, 1))
    weight_scale = 16
    scaled_weights = weights * weight_scale

    weights = QuantOps.create_weights_fake_quant(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=None, half_shift=False))

    scaled_weights = QuantOps.create_weights_fake_quant(
        w=scaled_weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=None, half_shift=False))

    onp.testing.assert_array_equal(weights * weight_scale, scaled_weights)

  @parameterized.named_parameters(
      dict(testcase_name='prec_2', prec=2),
      dict(testcase_name='prec_4', prec=4),
      dict(testcase_name='prec_8', prec=8)
  )
  def test_per_feature_dim_scale_invariance_weight_quantization(self, prec):
    # Scaling each channel of weights by a different power of 2, should scale
    # the respective channel of output by the same scale.
    weights = random.uniform(random.PRNGKey(0), (3, 4))
    weight_scale = 2**jnp.arange(4)[jnp.newaxis, :]
    scaled_weights = weights * weight_scale

    weights = quantization.QuantOps.create_weights_fake_quant(
        w=weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=0, half_shift=False))

    scaled_weights = quantization.QuantOps.create_weights_fake_quant(
        w=scaled_weights,
        weight_params=QuantOps.WeightParams(
            prec=prec, axis=0, half_shift=False))

    onp.testing.assert_array_equal(weights * weight_scale, scaled_weights)

  @parameterized.named_parameters(
      dict(
          testcase_name='fp_prec_scaled',
          prec=QuantOps.FloatQuant(
              is_scaled=True,
              fp_spec=QuantOps.FloatQuant.FloatPrec(
                  exp_min=-11,
                  exp_max=4,
                  sig_bits=3,
              ),
          ),
      ),
      dict(
          testcase_name='fp_prec_unscaled',
          prec=QuantOps.FloatQuant(
              is_scaled=False,
              fp_spec=QuantOps.FloatQuant.FloatPrec(
                  exp_min=-11,
                  exp_max=4,
                  sig_bits=3,
              ),
          ),
      ),
      dict(
          testcase_name='int_prec',
          prec=4.0,
      ),
  )
  def test_no_quantization(self, prec):
    # If initial_bound==-1 when using GetBounds, then create_inputs_fake_quant
    # should be a no-op.
    inputs = jnp.array([[.3, 1.4], [-5.2, 4.0]])
    bounds = get_bounds.GetBounds.Hyper(
        initial_bound=-1,
        stddev_coeff=1,
        absdev_coeff=0,
        mix_coeff=1,
        reset_stats=True,
        ema_coeff=None,
        use_cams=False,
        granularity=quant_config.QuantGranularity.per_tensor)
    hparams = quantization.QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=bounds,
        prec=prec,
        half_shift=False)

    # The call to create_inputs_fake_quant has to occur from within a Flax
    # module since it calls GetBounds, which is itself a Flax module.
    # Thus we create a wrapper module for testing.
    class TestModule(nn.Module):

      hparams: quantization.QuantOps.ActHParams

      @nn.compact
      def __call__(self, inputs):
        return quantization.QuantOps.create_inputs_fake_quant(
            inputs,
            hparams=hparams,
            get_bounds_params=GetBounds.Params(
                update_stats=True, update_bounds=False))

    test_module = TestModule(hparams=hparams)
    state = test_module.init(jax.random.PRNGKey(0), inputs=inputs)
    inputs_after_fake_quant, _ = test_module.apply(
        state, inputs=inputs, mutable=True)
    onp.testing.assert_array_equal(inputs, inputs_after_fake_quant)


# TODO(shivaniagrawal): Add tests for auto clip activation quantizations.


class AQTTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    key1, key2 = jax.random.split(jax.random.PRNGKey(0), 2)
    self.rhs = jax.random.normal(key1, (2, 4)) * 20
    self.lhs = jax.random.normal(key2, (3, 2)) * 2 + 3

  @parameterized.named_parameters(
      dict(
          testcase_name='per_layer_act_per_column_weight',
          act_bounds=4.0,
          weight_prec=16,
          weight_axis=(0,),
      ),
      dict(
          testcase_name='per_column_act_per_column_weight',
          act_bounds=[[3.0, 4.0]],
          weight_prec=16,
          weight_axis=(0,)),
      dict(
          testcase_name='per_layer_act_per_layer_weight',
          act_bounds=4.0,
          weight_prec=16,
          weight_axis=None),
      dict(
          testcase_name='per_column_act_per_layer_weight',
          act_bounds=[[3.0, 4.0]],
          weight_prec=16,
          weight_axis=None),
      dict(
          testcase_name='per_layer_act_no_weight_quant',
          act_bounds=4.0,
          weight_prec=None,
          weight_axis=None),
      dict(
          testcase_name='per_column_act_no_weight_quant',
          act_bounds=[[3.0, 4.0]],
          weight_prec=None,
          weight_axis=None),
      dict(
          testcase_name='no_act_quant_per_column_weight',
          act_bounds=None,
          weight_prec=16,
          weight_axis=(0,)),
      dict(
          testcase_name='no_act_quant_no_weight_quant',
          act_bounds=None,
          weight_prec=None,
          weight_axis=None),)
  def test_quantized_dot_aqt(self, act_bounds, weight_prec, weight_axis):
    # With a high enough precision, we expect results from fakequant and AQT to
    # be very similar.
    weight_params = QuantOps.WeightParams(
        prec=weight_prec, axis=weight_axis, half_shift=False)

    if act_bounds is None:
      act_params = None
    else:
      act_params = QuantOps.ActHParams(
          input_distribution='symmetric',
          bounds=jnp.array(act_bounds),
          prec=16,
          half_shift=False)

    def quantized_matmul(quant_type):
      return quantization.quantized_dot(
          w=self.rhs,
          act=self.lhs,
          weight_params=weight_params,
          act_hparams=act_params,
          get_bounds_params=None,
          quant_type=quant_type,
          prefer_int8_to_int32_dot=True)

    aqt_result = quantized_matmul(QuantType.aqt)
    fakequant_result = quantized_matmul(QuantType.fake_quant)
    onp.testing.assert_allclose(
        aqt_result,
        fakequant_result,
        rtol=1e-2,
        err_msg='AQT and fakequant significantly disagree')

  def assert_is_integer_in_range(self, x, *, prec, distribution):
    if distribution == 'symmetric':
      x_clipped = primitives.round_and_clip_to_signed_int(
          x, prec=prec, dtype=x.dtype, half_shift=False)
    elif distribution == 'positive':
      x_clipped = primitives.floor_and_clip_to_unsigned_int(
          x, prec=prec, dtype=x.dtype, half_shift=False)
    else:
      raise ValueError(f'Invalid distribution {distribution}')
    onp.testing.assert_array_equal(
        x, x_clipped,
        f'Array cannot be losslessly cast to integer with precision {prec} '
        f'and {distribution} distribution.')

  @parameterized.parameters(
      dict(act_distribution='symmetric', prefer_int8_to_int32_dot=True, prec=4),
      dict(act_distribution='symmetric', prefer_int8_to_int32_dot=True, prec=8),
      dict(act_distribution='positive', prefer_int8_to_int32_dot=True, prec=4),
      dict(act_distribution='positive', prefer_int8_to_int32_dot=True, prec=8),
      dict(
          act_distribution='symmetric', prefer_int8_to_int32_dot=False, prec=4))
  @mock.patch.object(jax.lax, 'dot_general')
  def test_lax_dot_has_integer_inputs_in_quantized_dot(self, mock_dot_general,
                                                       act_distribution,
                                                       prefer_int8_to_int32_dot,
                                                       prec):
    weight_params = QuantOps.WeightParams(
        prec=prec, axis=(0,), half_shift=False)
    act_params = QuantOps.ActHParams(
        input_distribution=act_distribution,
        bounds=jnp.array([[3.0, 1.5]]),
        prec=prec,
        half_shift=False)
    act = self.lhs
    if act_distribution == 'positive':
      act = jnp.abs(act)
    # We need this context manager to stop Jax from trying to compile the arms
    # of the `lax.cond` call in `dot_general_aqt`. By default, Jax will always
    # try to compile the functions passed to `lax.cond`, even if outside of a
    # JITed context. JIT compilation is incompatible with using a mock for the
    # call to 'dot_general' because during compilation Jax will expect
    # 'dot_general' to return a tracer and will throw an error if it returns a
    # mock instead. By explicily using jax.disable_jit, Jax will not try to
    # compile the arms to lax.cond and so using a mock will work fine.
    with jax.disable_jit():
      quantization.quantized_dot(
          w=self.rhs,
          act=act,
          weight_params=weight_params,
          act_hparams=act_params,
          get_bounds_params=None,
          quant_type=QuantType.aqt,
          prefer_int8_to_int32_dot=prefer_int8_to_int32_dot)
    act_inputs, weight_inputs = mock_dot_general.call_args[0]
    self.assert_is_integer_in_range(
        act_inputs, prec=prec, distribution=act_distribution)
    self.assert_is_integer_in_range(
        weight_inputs, prec=prec, distribution='symmetric')
    if prefer_int8_to_int32_dot and not (act_distribution == 'positive' and
                                         prec == 8):
      expected_input_dtype = jnp.int8
    else:
      expected_input_dtype = jnp.float32
    self.assertEqual(act_inputs.dtype, expected_input_dtype)
    self.assertEqual(weight_inputs.dtype, expected_input_dtype)

  @parameterized.parameters(
      itertools.product(
          (jnp.bfloat16, jnp.float32), (4, None),
          (quantization.QuantType.aqt, quantization.QuantType.fake_quant)))
  def test_quantized_dot_has_correct_dtype(self, input_dtype, act_prec,
                                           quant_type):
    weight_params = QuantOps.WeightParams(prec=4, axis=(0,), half_shift=False)
    act_params = QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=jnp.array([[3.0, 1.5]]),
        prec=act_prec,
        half_shift=False)
    act = self.lhs.astype(input_dtype)
    w = self.rhs.astype(input_dtype)
    output = quantization.quantized_dot(
        w=w,
        act=act,
        weight_params=weight_params,
        act_hparams=act_params,
        get_bounds_params=None,
        quant_type=quant_type,
        prefer_int8_to_int32_dot=True)
    self.assertEqual(output.dtype, input_dtype)

  @parameterized.parameters(
      dict(quant_type=quantization.QuantType.aqt),
      dict(quant_type=quantization.QuantType.fake_quant))
  def test_quantized_dot_raises_with_mixed_dtype(self, quant_type):
    weight_params = QuantOps.WeightParams(prec=4, axis=(0,), half_shift=False)
    act_params = QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=jnp.array([[3.0, 1.5]]),
        prec=4,
        half_shift=False)
    act = self.lhs.astype(jnp.bfloat16)
    w = self.rhs.astype(jnp.float32)
    with self.assertRaises(TypeError):
      quantization.quantized_dot(
          w=w,
          act=act,
          weight_params=weight_params,
          act_hparams=act_params,
          get_bounds_params=None,
          quant_type=quant_type,
          prefer_int8_to_int32_dot=True)

  @parameterized.parameters(
      itertools.product(
          (jnp.bfloat16, jnp.float32), (4, None),
          (quantization.QuantType.aqt, quantization.QuantType.fake_quant)))
  def test_dynamic_quantized_dot_general_has_correct_dtype(
      self, input_dtype, act_prec, quant_type):
    lhs_params = QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=2.0,
        prec=act_prec,
        half_shift=False)
    rhs_params = QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=1.5,
        prec=act_prec,
        half_shift=False)
    lhs_act = self.lhs.astype(input_dtype)
    rhs_act = self.rhs.astype(input_dtype)
    output = quantization.quantized_dynamic_dot_general(
        lhs_act=lhs_act,
        rhs_act=rhs_act,
        lhs_act_hparams=lhs_params,
        rhs_act_hparams=rhs_params,
        lhs_get_bounds_params=None,
        rhs_get_bounds_params=None,
        dot_dimension_numbers=(((1,), (0,)), ((), ())),
        quant_type=quant_type)
    self.assertEqual(output.dtype, input_dtype)

  def test_dynamic_quantized_dot_general_raises_with_mixed_dtype(self):
    lhs_params = QuantOps.ActHParams(
        input_distribution='symmetric', bounds=2.0, prec=4, half_shift=False)
    rhs_params = QuantOps.ActHParams(
        input_distribution='symmetric', bounds=1.5, prec=4, half_shift=False)
    lhs_act = self.lhs.astype(jnp.bfloat16)
    rhs_act = self.rhs.astype(jnp.float32)
    with self.assertRaises(TypeError):
      quantization.quantized_dynamic_dot_general(
          lhs_act=lhs_act,
          rhs_act=rhs_act,
          lhs_act_hparams=lhs_params,
          rhs_act_hparams=rhs_params,
          lhs_get_bounds_params=None,
          rhs_get_bounds_params=None,
          dot_dimension_numbers=(((1,), (0,)), ((), ())),
          quant_type=QuantType.aqt)

  @parameterized.parameters(
      dict(lhs_prec=16, rhs_prec=16), dict(lhs_prec=None, rhs_prec=16),
      dict(lhs_prec=16, rhs_prec=None), dict(lhs_prec=None, rhs_prec=None))
  def test_quantized_dynamic_dot_general(self, lhs_prec, rhs_prec):
    lhs_bounds = 2.0
    rhs_bounds = 1.5
    lhs_params = QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=lhs_bounds,
        prec=lhs_prec,
        half_shift=False)
    rhs_params = QuantOps.ActHParams(
        input_distribution='symmetric',
        bounds=rhs_bounds,
        prec=rhs_prec,
        half_shift=False)

    def quantized_matmul(quant_type):
      return quantization.quantized_dynamic_dot_general(
          lhs_act=self.lhs,
          rhs_act=self.rhs,
          lhs_act_hparams=lhs_params,
          rhs_act_hparams=rhs_params,
          lhs_get_bounds_params=None,
          rhs_get_bounds_params=None,
          dot_dimension_numbers=(((1,), (0,)), ((), ())),
          quant_type=quant_type)

    aqt_result = quantized_matmul(QuantType.aqt)
    fakequant_result = quantized_matmul(QuantType.fake_quant)
    onp.testing.assert_allclose(
        aqt_result,
        fakequant_result,
        rtol=1e-2,
        err_msg='AQT and fakequant significantly disagree')

  def test_quantized_dynamic_dot_general_get_bounds(self):

    class TestModule(nn.Module):

      @nn.compact
      def __call__(self, lhs, rhs):
        lhs_get_bounds = GetBounds.Hyper(
            initial_bound=10.0,
            stddev_coeff=0,
            absdev_coeff=0,
            mix_coeff=0,
            granularity=quant_config.QuantGranularity.per_tensor)
        rhs_get_bounds = GetBounds.Hyper(
            initial_bound=5.0,
            stddev_coeff=0,
            absdev_coeff=0,
            mix_coeff=0,
            granularity=quant_config.QuantGranularity.per_tensor)
        lhs_params = QuantOps.ActHParams(
            input_distribution='symmetric',
            bounds=lhs_get_bounds,
            prec=8,
            half_shift=False)
        rhs_params = QuantOps.ActHParams(
            input_distribution='symmetric',
            bounds=rhs_get_bounds,
            prec=8,
            half_shift=False)
        lhs_get_bounds_params = get_bounds.GetBounds.Params(
            update_stats=True, update_bounds=False, module_name='lhs')
        rhs_get_bounds_params = get_bounds.GetBounds.Params(
            update_stats=True, update_bounds=False, module_name='rhs')
        out = quantization.quantized_dynamic_dot_general(
            lhs_act=lhs,
            rhs_act=rhs,
            lhs_act_hparams=lhs_params,
            rhs_act_hparams=rhs_params,
            dot_dimension_numbers=(((1,), (0,)), ((), ())),
            quant_type=QuantType.aqt,
            lhs_get_bounds_params=lhs_get_bounds_params,
            rhs_get_bounds_params=rhs_get_bounds_params)
        return out

    lhs = jnp.array([[2.0]])
    rhs = jnp.array([[3.0]])
    module = TestModule()
    state = module.init(jax.random.PRNGKey(0), lhs, rhs)
    out, _ = module.apply(state, lhs, rhs, mutable=True)
    lhs_scale = 127.0 / 10.0
    rhs_scale = 127.0 / 5.0
    expected_out = (round(lhs_scale * 2.0) * round(rhs_scale * 3.0)) / (
        lhs_scale * rhs_scale)
    onp.testing.assert_allclose(out, [[expected_out]])

  @parameterized.parameters(
      dict(lhs_distribution='symmetric', rhs_distribution='symmetric'),
      dict(lhs_distribution='positive', rhs_distribution='symmetric'),
      dict(lhs_distribution='symmetric', rhs_distribution='positive'),
      dict(lhs_distribution='positive', rhs_distribution='positive'))
  @mock.patch.object(jax.lax, 'dot_general')
  def test_lax_dot_has_integer_inputs_in_dynamic_dot_general(
      self, mock_dot_general, lhs_distribution, rhs_distribution):
    lhs_params = QuantOps.ActHParams(
        input_distribution=lhs_distribution,
        bounds=2.0,
        prec=4,
        half_shift=False)
    rhs_params = QuantOps.ActHParams(
        input_distribution=rhs_distribution,
        bounds=1.5,
        prec=4,
        half_shift=False)
    lhs_act = self.lhs
    if lhs_distribution == 'positive':
      lhs_act = jnp.abs(lhs_act)
    rhs_act = self.rhs
    if rhs_distribution == 'positive':
      rhs_act = jnp.abs(rhs_act)
    quantization.quantized_dynamic_dot_general(
        lhs_act=lhs_act,
        rhs_act=rhs_act,
        lhs_act_hparams=lhs_params,
        rhs_act_hparams=rhs_params,
        lhs_get_bounds_params=None,
        rhs_get_bounds_params=None,
        dot_dimension_numbers=(((1,), (0,)), ((), ())),
        quant_type=QuantType.aqt)
    lhs_inputs, rhs_inputs = mock_dot_general.call_args[0]
    self.assert_is_integer_in_range(
        lhs_inputs, prec=4, distribution=lhs_distribution)
    self.assert_is_integer_in_range(
        rhs_inputs, prec=4, distribution=rhs_distribution)

  def test_quantized_dot_no_quant(self):
    act_hparams = QuantOps.ActHParams(
        input_distribution='symmetric', bounds=-1.0, prec=4, half_shift=False)
    weight_params = QuantOps.WeightParams(prec=4, axis=(0,), half_shift=False)
    act = jnp.array([[-5.0]])
    w = jnp.array([[-4.99]])
    res = quantization.quantized_dot(
        w=w,
        act=act,
        quant_type=quantization.QuantType.aqt,
        weight_params=weight_params,
        act_hparams=act_hparams,
        get_bounds_params=None,
        prefer_int8_to_int32_dot=True)
    onp.testing.assert_allclose(res, act * w)

  def test_quantized_dynamic_dot_general_no_quant(self):
    act_hparams = QuantOps.ActHParams(
        input_distribution='symmetric', bounds=-1.0, prec=4, half_shift=False)
    lhs_act = jnp.array([[-5.0]])
    rhs_act = jnp.array([[-4.99]])
    res = quantization.quantized_dynamic_dot_general(
        lhs_act=lhs_act,
        rhs_act=rhs_act,
        quant_type=quantization.QuantType.aqt,
        lhs_act_hparams=act_hparams,
        rhs_act_hparams=act_hparams,
        lhs_get_bounds_params=None,
        rhs_get_bounds_params=None,
        dot_dimension_numbers=(((1,), (0,)), ((), ())))
    onp.testing.assert_allclose(res, lhs_act * rhs_act)


class QuantizedDotFakeQuantTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.act = jnp.ones((3, 7))
    self.weight = jnp.ones((7, 4))

  @parameterized.named_parameters(
      dict(testcase_name='no_quantization', weight_prec=None, act_prec=None),
      dict(testcase_name='weight_only_quant', weight_prec=8., act_prec=None),
      dict(testcase_name='act_only_quant', weight_prec=None, act_prec=4),
      dict(testcase_name='both_quantized', weight_prec=4, act_prec=8),
      dict(
          testcase_name='both_quantized_fq_int',
          weight_prec=4,
          act_prec=8,
          strategy=QuantType.fake_quant_with_int),
  )
  @mock.patch.object(QuantOps, 'create_weights_fake_quant')
  @mock.patch.object(QuantOps, 'create_inputs_fake_quant')
  def test_quantized_dot_general_should_call_weights_and_inputs_quantization(
      self,
      mock_act_fq,
      mock_w_fq,
      weight_prec,
      act_prec,
      strategy=QuantType.fake_quant):
    mock_w_fq.side_effect = lambda inputs, **_: inputs
    mock_act_fq.side_effect = lambda inputs, **_: inputs

    weight_params = QuantOps.WeightParams(
        prec=weight_prec, axis=None, half_shift=False)
    act_hparams = QuantOps.ActHParams(  # pylint: disable=g-long-ternary
        bounds=6.,
        prec=act_prec,
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        half_shift=False) if act_prec else None
    get_bounds_params = GetBounds.Params(
        update_stats=False, update_bounds=False)

    quantization.quantized_dot(
        w=self.weight,
        act=self.act,
        quant_type=strategy,
        weight_params=weight_params,
        act_hparams=act_hparams,
        get_bounds_params=get_bounds_params,
        prefer_int8_to_int32_dot=True)

    quantized_type = strategy.to_jax_type()

    mock_w_fq.assert_called_with(
        mock.ANY,
        weight_params=weight_params,
        quantized_type=quantized_type,
        fake_dependency=mock.ANY)
    if act_hparams:
      mock_act_fq.assert_called_with(
          mock.ANY, hparams=act_hparams, get_bounds_params=get_bounds_params)
    else:
      mock_act_fq.assert_not_called()


class QuantizedDynamicDotGeneralTest(parameterized.TestCase):

  def setUp(self):
    super(QuantizedDynamicDotGeneralTest, self).setUp()
    self.lhs_act = jnp.ones((4, 2, 3, 7))
    self.rhs_act = jnp.ones((3, 7, 5, 6))
    self.dimension_numbers = (((2, 3), (0, 1)), ((), ()))

  @parameterized.named_parameters(
      dict(
          testcase_name='no_quantization', lhs_act_prec=None,
          rhs_act_prec=None),
      dict(testcase_name='lhs_only_quant', lhs_act_prec=8., rhs_act_prec=None),
      dict(testcase_name='rhs_only_quant', lhs_act_prec=None, rhs_act_prec=4),
      dict(testcase_name='both_quantized', lhs_act_prec=4, rhs_act_prec=8),
      dict(
          testcase_name='both_quantized_fq_int',
          lhs_act_prec=4,
          rhs_act_prec=8,
          strategy=QuantType.fake_quant_with_int),
  )
  @mock.patch.object(QuantOps, 'create_inputs_fake_quant')
  def test_quantized_dynamic_dot_general_should_call_inputs_quantization(
      self,
      mock_act_fq,
      lhs_act_prec,
      rhs_act_prec,
      strategy=QuantType.fake_quant):
    mock_act_fq.side_effect = lambda inputs, hparams, get_bounds_params: inputs

    # pylint: disable=g-long-ternary
    lhs_act_hparams = QuantOps.ActHParams(
        bounds=6.,
        prec=lhs_act_prec,
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        half_shift=False) if lhs_act_prec else None
    rhs_act_hparams = QuantOps.ActHParams(
        bounds=6.,
        prec=rhs_act_prec,
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        half_shift=False) if rhs_act_prec else None
    # pylint: enable=g-long-ternary

    get_bounds_params = GetBounds.Params(
        update_stats=False, update_bounds=False)

    quantization.quantized_dynamic_dot_general(
        lhs_act=self.lhs_act,
        rhs_act=self.rhs_act,
        quant_type=strategy,
        dot_dimension_numbers=self.dimension_numbers,
        lhs_act_hparams=lhs_act_hparams,
        lhs_get_bounds_params=get_bounds_params,
        rhs_act_hparams=rhs_act_hparams,
        rhs_get_bounds_params=get_bounds_params,
    )
    calls = []
    for prec in [lhs_act_prec, rhs_act_prec]:
      if prec is not None:
        act_hparams = QuantOps.ActHParams(
            bounds=6., prec=prec, input_distribution=mock.ANY, half_shift=False)
        calls.append(
            mock.call(
                mock.ANY,
                hparams=act_hparams,
                get_bounds_params=get_bounds_params))
    self.assertLen(calls, mock_act_fq.call_count)
    mock_act_fq.assert_has_calls(calls, any_order=True)


class QuantizedSumTest(parameterized.TestCase):

  @parameterized.parameters(
      # This roughly corresponds to float32, so we expect no difference vs a
      # float32 sum.
      dict(exp_min=-2**7, exp_max=2**7, sig_bits=23, expected_result=100.001),
      # In this low precision case, the addition of .001 to the accumulator will
      # have no effect after quantization
      dict(exp_min=-2**3, exp_max=2**3, sig_bits=1, expected_result=100.0))
  def test_quantized_sum(self, exp_min, exp_max, sig_bits, expected_result):
    x = jnp.array([0.001, 100.0])
    prec = QuantOps.FloatQuant.FloatPrec(exp_min, exp_max, sig_bits)
    x_quantized_sum, x_grad = jax.value_and_grad(quantization.quantized_sum)(
        x, axis=0, keepdims=False, prec=prec)
    onp.testing.assert_allclose(
        x_quantized_sum, onp.array(expected_result), rtol=1e-6)
    # This tests that the gradient is using the straight-through-estimator
    onp.testing.assert_equal(x_grad, onp.array([1.0, 1.0]))

  @parameterized.parameters(
      dict(keepdims=True, axis=(0, 1), expected_shape=(1, 1)),
      dict(keepdims=False, axis=(0, 1), expected_shape=()),
      dict(keepdims=True, axis=(0,), expected_shape=(1, 2)),
      dict(keepdims=False, axis=(1,), expected_shape=(3,)))
  def test_keepdims_and_axis(self, keepdims, axis, expected_shape):
    x = jnp.arange(6).reshape((3, 2)).astype(jnp.float32)
    prec = QuantOps.FloatQuant.FloatPrec(-2**7, 2**7, 23)
    x_quantized_sum = quantization.quantized_sum(
        x, keepdims=keepdims, axis=axis, prec=prec)
    self.assertEqual(x_quantized_sum.shape, expected_shape)


if __name__ == '__main__':
  absltest.main()
