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

"""Tests for aqt.jax.flax_layers."""

import itertools
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import linen as nn
import jax
from jax import lax
from jax import random
import jax.config as config
from jax.nn import initializers
import jax.numpy as jnp
import numpy as onp

from aqt.jax import flax_layers
from aqt.jax import fp_cast
from aqt.jax import get_bounds
from aqt.jax import primitives
from aqt.jax import quant_config
from aqt.jax import quantization
from aqt.jax import shape_utils
from aqt.jax import test_utils
from aqt.jax.quantization import QuantOps
from aqt.jax.quantization import QuantType

FLAGS = flags.FLAGS

# fp-1-4-3
#    1: sign
#    4: number of exponent-bits, (bias = 11), range: -11, ..., 4
#    3: number of significand-bits (excluding hidden-bit)
fp143_scaled = QuantOps.FloatQuant(
    is_scaled=True,
    fp_spec=QuantOps.FloatQuant.FloatPrec(
        exp_min=-11,
        exp_max=4,
        sig_bits=3,
    ),
)
fp143_unscaled = QuantOps.FloatQuant(
    is_scaled=False,
    fp_spec=QuantOps.FloatQuant.FloatPrec(
        exp_min=-11,
        exp_max=4,
        sig_bits=3,
    ),
)


class ConvAqtTest(parameterized.TestCase):
  """Tests for ConvAqt layer."""

  def setUp(self):
    super(ConvAqtTest, self).setUp()
    test_utils.configure_jax()
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = True
    self.rng_key = random.PRNGKey(0)

  def tearDown(self):
    config.update('jax_numpy_rank_promotion', 'warn')
    super(ConvAqtTest, self).tearDown()

  def init_model_with_1_layer(self,
                              inputs,
                              num_features,
                              kernel_size,
                              kernel_init=flax_layers.default_kernel_init,
                              weight_prec=None,
                              quant_act=None):
    """Create and initialize a flax model with a single ConvAqt layer."""
    layer_kwargs = {
        'kernel_init': kernel_init,
        'features': num_features,
        'use_bias': False,
        'quant_context': quant_config.QuantContext(update_bounds=False),
        'paxis_name': 'batch',
        'train': False,
        'kernel_size': kernel_size,
        'dtype': jnp.float32
    }
    layer_class = flax_layers.ConvAqt
    layer_kwargs['hparams'] = flax_layers.ConvAqt.HParams(
        weight_prec=weight_prec,
        quant_act=quant_act,
        quant_type=QuantType.fake_quant,
    )
    conv_module = layer_class(**layer_kwargs)
    initial_state = conv_module.init(self.rng_key, jnp.zeros(inputs.shape))
    return conv_module, initial_state

  # Following ConvAqt tests adapted from
  # Flax Conv tests.
  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
      dict(testcase_name='quant_2bit', weight_prec=2),
  )
  def test_conv(self, weight_prec=None):
    x = jnp.ones((1, 8, 8, 3))
    conv_module = flax_layers.ConvAqt(
        features=4,
        kernel_size=(3, 3),
        padding='VALID',
        paxis_name='batch',
        quant_context=quant_config.QuantContext(update_bounds=False),
        train=False,
        hparams=flax_layers.ConvAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.fake_quant),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        dtype=jnp.float32)

    y, state = conv_module.init_with_output(self.rng_key, x)
    self.assertEqual(state['params']['kernel'].shape, (3, 3, 3, 4))
    test_utils.assert_all_close_prec(y, onp.full((1, 6, 6, 4), 28.),
                                     weight_prec)

  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
      dict(testcase_name='quant_2bit', weight_prec=2),
  )
  def test_group_conv(self, weight_prec=None):
    x = jnp.ones((1, 8, 8, 4))
    conv_module = flax_layers.ConvAqt(
        features=4,
        kernel_size=(3, 3),
        feature_group_count=2,
        padding='VALID',
        paxis_name='batch',
        quant_context=quant_config.QuantContext(update_bounds=False),
        train=False,
        hparams=flax_layers.ConvAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.fake_quant),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
        dtype=jnp.float32)
    y, state = conv_module.init_with_output(self.rng_key, x)
    self.assertEqual(state['params']['kernel'].shape, (3, 3, 2, 4))
    test_utils.assert_all_close_prec(y, onp.full((1, 6, 6, 4), 19.),
                                     weight_prec)

  @parameterized.named_parameters(
      dict(testcase_name='conv_quant_8bit', weight_prec=8),
      dict(testcase_name='conv_quant_4bit', weight_prec=4),
      dict(testcase_name='conv_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_should_give_precise_output(
      self, weight_prec):
    # If weights are ints (already quantized) and
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel,
    # no quantization error should be introduced.

    num_features = 256
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 16, 16, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, kernel_size, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(
        self.rng_key, kernel_size + (input_dim, num_features), minval,
        maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = jax.ops.index_update(
        full_range_integer_weights, jax.ops.index[0, 0, :], maxval)
    state = state.unfreeze()
    state['params']['kernel'] = full_range_integer_weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)

    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint: disable=protected-access
    exp_outputs = lax.conv_general_dilated(
        inputs,
        jnp.asarray(state['params']['kernel'], jnp.float32), (1, 1),
        'SAME',
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=jax.lax.Precision.DEFAULT)

    onp.testing.assert_array_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      dict(testcase_name='conv_quant_4bit', weight_prec=4),
      dict(testcase_name='conv_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_with_float_scale_should_give_close_output(
      self, weight_prec):
    # If weights are ints (already quantized) with
    # max(abs(weights[..., ch])) == 2**(prec-1)-1 in each channel
    # and if these integer weights are multiplied by a float scale,
    # the resulting error should still be very small (just float rounding).

    num_features = 256
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 16, 16, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, kernel_size, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(
        self.rng_key, kernel_size + (input_dim, num_features), minval,
        maxval + 1)
    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = jax.ops.index_update(
        full_range_integer_weights, jax.ops.index[0, 0, :], maxval)

    # (batch_size, spatial_dim, spatial_dim, num_features)
    float_scale = jax.random.uniform(self.rng_key, (1, 1, 1, num_features))
    state = state.unfreeze()
    state['params']['kernel'] = full_range_integer_weights * float_scale
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs)
    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)  # pylint: disable=protected-access
    exp_outputs = lax.conv_general_dilated(
        inputs,
        jnp.asarray(state['params']['kernel'], jnp.float32), (1, 1),
        'SAME',
        lhs_dilation=None,
        rhs_dilation=None,
        dimension_numbers=dimension_numbers,
        feature_group_count=1,
        precision=jax.lax.Precision.DEFAULT)
    # We know that the noise should be proportional to the square root of
    # input_dim and inversely proportional to 2**weight_prec.
    # The following tol_const was obtained experimentally and should be derived
    # more systematically.
    tol_const = 5e-02
    onp.testing.assert_allclose(
        outputs,
        exp_outputs,
        rtol=jnp.sqrt(input_dim) * 2**(-weight_prec) * tol_const)

  @parameterized.named_parameters(
      dict(
          testcase_name='conv_quant_8bit',
          weight_prec=8,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='conv_quant_4bit',
          weight_prec=4,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='conv_quant_2bit',
          weight_prec=2,
          weight_scale=onp.array([1, 2, 4, 8])),
  )
  def test_weight_invariance_to_power_of_2_weight_scaling(
      self, weight_prec, weight_scale):
    # Scaling the weights before quantization by a power of 2 per channel should
    # also scale the output exactly by the same scale.

    num_features = 4
    assert num_features == weight_scale.shape[-1]
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 16, 16, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, 4, kernel_size, weight_prec=weight_prec)

    weights = random.uniform(
        self.rng_key, shape=kernel_size + (input_dim, num_features))
    weight_scale = weight_scale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    state = state.unfreeze()
    state['params']['kernel'] = weights
    outputs_without_scaling = model.apply(flax.core.freeze(state), inputs)
    state['params']['kernel'] = jnp.multiply(weights, weight_scale)
    outputs_with_scaling = model.apply(flax.core.freeze(state), inputs)

    onp.testing.assert_array_equal(outputs_without_scaling * weight_scale,
                                   outputs_with_scaling)

  def test_1_bit_makes_all_weight_equal_to_zero(self):
    num_features = 4
    input_dim = 3
    inputs = random.uniform(self.rng_key, shape=(1, 32, 32, input_dim))
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, kernel_size, weight_prec=1)
    weights = random.uniform(
        self.rng_key, shape=kernel_size + (input_dim, num_features))
    state = state.unfreeze()
    state['params']['kernel'] = weights
    outputs = model.apply(flax.core.freeze(state), inputs)
    onp.testing.assert_array_equal(outputs, onp.zeros(
        (1, 32, 32, num_features)))

  @parameterized.named_parameters(
      dict(
          testcase_name='conv_quant_8bit',
          weight_prec=8,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='conv_quant_4bit',
          weight_prec=4,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='conv_quant_2bit',
          weight_prec=2,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='conv_signed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='conv_signed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='conv_signed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='conv_signed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
      dict(
          testcase_name='conv_signed_input_quant_2bit',
          weight_prec=None,
          acts_prec=2,
          fixed_bounds=True),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_weights_and_symmetrics_acts_should_call_clip_and_round(
      self, floor_with_gradient, round_with_gradient, weight_prec, acts_prec,
      fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        prec=acts_prec,
        bounds=bounds)
    num_features = 4
    input_dim = 3
    inputs = jnp.ones((1, 32, 32, input_dim), dtype=jnp.float32)
    kernel_size = (3, 3)
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features,
        kernel_size,
        weight_prec=weight_prec,
        quant_act=quant_act)

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    outputs = model.apply(state, inputs)

    self.assertEqual(
        outputs.shape,
        (inputs.shape[0], inputs.shape[1], inputs.shape[2], num_features))
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_without_quantized_weights_should_not_call_quantization_ops(
      self, floor_with_gradient, round_with_gradient):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    inputs = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(inputs, 4, (3, 3))
    _ = model.apply(state, inputs)
    round_with_gradient.assert_not_called()
    floor_with_gradient.assert_not_called()


class DenseAqtTest(parameterized.TestCase):
  """Tests for DenseAqt layer."""

  def setUp(self):
    super(DenseAqtTest, self).setUp()
    test_utils.configure_jax()
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = True
    self.rng_key = random.PRNGKey(0)

  def tearDown(self):
    config.update('jax_numpy_rank_promotion', 'warn')
    super(DenseAqtTest, self).tearDown()

  def init_model_with_1_layer(self,
                              inputs,
                              num_features,
                              kernel_init=flax_layers.default_kernel_init,
                              weight_prec=None,
                              quant_act=None):
    """Create and initialize a flax model with a single DenseAqt layer."""
    quant_context = quant_config.QuantContext(
        update_bounds=False, collect_acts_stats=False)
    layer_kwargs = {
        'kernel_init': kernel_init,
        'features': num_features,
        'use_bias': False,
        'quant_context': quant_context,
        'paxis_name': 'batch',
        'train': False,
        'dtype': jnp.float32
    }
    layer_kwargs['hparams'] = flax_layers.DenseAqt.HParams(
        weight_prec=weight_prec,
        quant_act=quant_act,
        quant_type=QuantType.fake_quant,
        weight_quant_granularity=quant_config.QuantGranularity.per_channel)

    dense_module = flax_layers.DenseAqt(**layer_kwargs)
    initial_state = dense_module.init(
        self.rng_key, jnp.zeros(inputs.shape), padding_mask=None)
    return dense_module, initial_state

  def test_padding(self):
    """Test that padding results in the right statistics being collected."""
    # Exact values don't matter here, we just need code to think it's using
    # dynamic bounds so it gathers activation statistics
    bounds = get_bounds.GetBounds.Hyper(
        initial_bound=0.0,
        stddev_coeff=1.0,
        absdev_coeff=0.0,
        mix_coeff=1.0,
        reset_stats=False,
        granularity=quant_config.QuantGranularity.per_channel)
    quant_act = flax_layers.QuantOps.ActHParams(
        input_distribution=flax_layers.QuantOps.ActHParams.InputDistribution
        .symmetric,
        prec=8,
        bounds=bounds)
    hparams = flax_layers.DenseAqt.HParams(
        quant_type=flax_layers.QuantType.fake_quant,
        weight_prec=8,
        quant_act=quant_act,
        weight_quant_granularity=quant_config.QuantGranularity.per_channel)
    module = flax_layers.DenseAqt(
        hparams=hparams,
        features=1,
        paxis_name=None,
        quant_context=quant_config.QuantContext(
            update_bounds=True, collect_acts_stats=False),
        train=True,
        dtype=jnp.float32)

    # Simulate an input with a batch size of 2, three tokens per example, two
    # channels per token
    x = jnp.arange(12).astype(jnp.float32).reshape((2, 3, 2))
    # Reshape it to have dimensions [batch, feature]
    x = x.reshape(6, 2)

    initial_state = module.init(self.rng_key, x, padding_mask=None)

    # Check that the per-channel activation statistics are as expected with no
    # padding
    _, state_nopadding = module.apply(
        initial_state, x, padding_mask=None, mutable='get_bounds')
    expected_means = onp.array([[(0 + 2 + 4 + 6 + 8 + 10) / 6,
                                 (1 + 3 + 5 + 7 + 9 + 11) / 6]])
    actual_means = state_nopadding['get_bounds']['GetBounds_0']['stats'].mean
    onp.testing.assert_allclose(actual_means, expected_means)

    # Now we pad out some of the tokens (chosen arbitrarily) and check that the
    # computed per-channel stats are the means of the non-padding tokens only
    # Exclude the second and third tokens from the first batch and the first
    # token from the second batch.
    padding_mask = jnp.array([[True, False, False], [False, True, True]])
    # Reshape it to have dimensions [batch, feature]
    padding_mask = padding_mask.reshape(6, 1)
    _, state_padding = module.apply(
        initial_state, x, padding_mask=padding_mask, mutable='get_bounds')
    expected_means = onp.array([[(0 + 8 + 10) / 3, (1 + 9 + 11) / 3]])
    actual_means = state_padding['get_bounds']['GetBounds_0']['stats'].mean
    onp.testing.assert_allclose(actual_means, expected_means)

  @parameterized.named_parameters(
      dict(testcase_name='dense_float', weight_prec=None),
      dict(
          testcase_name='dense_quant_fp143_scaled',
          weight_prec=fp143_scaled,
      ),
      dict(
          testcase_name='dense_quant_fp143',
          weight_prec=fp143_unscaled,
      ),
      dict(testcase_name='dense_quant_8bit', weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_ones_weights_should_give_precise_output(self, weight_prec):
    """If all weights are 1, no quantization error should be introduced."""
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs,
        num_features=4,
        kernel_init=initializers.ones,
        weight_prec=weight_prec)
    outputs = model.apply(state, inputs, padding_mask=None)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    onp.testing.assert_array_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      dict(testcase_name='dense_quant_8bit', weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_should_give_precise_output(
      self, weight_prec):
    # If weights are ints (already quantized) and
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel,
    # no quantization error should be introduced.
    num_features = 256
    input_dim = 1024
    inputs = random.uniform(self.rng_key, shape=(2, input_dim))
    model, state = self.init_model_with_1_layer(
        inputs, num_features, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(self.rng_key,
                                                (input_dim, num_features),
                                                minval, maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = jax.ops.index_update(
        full_range_integer_weights, jax.ops.index[0, :], maxval)
    state = state.unfreeze()
    state['params']['kernel'] = full_range_integer_weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs, padding_mask=None)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    onp.testing.assert_array_equal(outputs, exp_outputs)

  @parameterized.named_parameters(
      # TODO(shivaniagrawal): this test is flaky and fails with rtol=0.0004
      # with given rtol=0.0001
      # dict(
      #     testcase_name='dense_quant_8bit',
      #     layer_class=flax_layers.DenseAqt,
      #     weight_prec=8),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_full_range_integer_weights_with_float_scale_should_give_close_output(
      self, weight_prec):
    # If weights are ints (already quantized) with
    # max(abs(weights[:, ch])) == 2**(prec-1)-1 in each channel
    # and if these integer weights are multiplied by a float scale,
    # the resulting error should still be very small (just float rounding).

    num_features = 256
    input_dim = 1024
    inputs = random.uniform(self.rng_key, shape=(2, input_dim))
    model, state = self.init_model_with_1_layer(
        inputs, num_features, weight_prec=weight_prec)
    minval = -2**(weight_prec - 1) + 1
    maxval = 2**(weight_prec - 1) - 1

    full_range_integer_weights = random.randint(self.rng_key,
                                                (input_dim, num_features),
                                                minval, maxval + 1)

    # manually set one value in each output dim of weights to be exactly maxval
    full_range_integer_weights = jax.ops.index_update(
        full_range_integer_weights, jax.ops.index[0, :], maxval)

    float_scale = jax.random.uniform(self.rng_key, (1, num_features))
    state = state.unfreeze()
    state['params']['kernel'] = full_range_integer_weights * float_scale
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs, padding_mask=None)
    exp_outputs = jnp.matmul(inputs, state['params']['kernel'])
    # TODO(wanglisa): Determine how much noise is expected for following test.
    # We know that the noise should be proportional to the square root of
    # input_dim and inversely proportional to 2**weight_prec.
    # The following tol_const was obtained experimentally and should be derived
    # more systematically.
    tol_const = 8e-04
    onp.testing.assert_allclose(
        outputs,
        exp_outputs,
        rtol=jnp.sqrt(input_dim) * 2**(-weight_prec) * tol_const)

  @parameterized.named_parameters(
      # dict(
      #     testcase_name='dense_quant_8bit',
      #     weight_prec=8),
      # TODO(shivaniagrawal): fix the above test, test above doesn't follow
      # the expected tolerance. Expected absolute difference = 0.188386,
      # actual absolute difference: 0.20296225
      dict(
          testcase_name='dense_quant_fp143_scaled',
          weight_prec=fp143_scaled,
      ),
      dict(testcase_name='dense_quant_fp143', weight_prec=fp143_unscaled),
      dict(testcase_name='dense_quant_4bit', weight_prec=4),
      dict(testcase_name='dense_quant_2bit', weight_prec=2),
  )
  def test_float_weights_should_give_close_output(self, weight_prec):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=weight_prec)
    float_weights = jnp.linspace(-1 / 3, 1 / 3, num=12).reshape((3, 4))

    exp_output_without_quant = jnp.matmul(inputs, float_weights)
    state = state.unfreeze()
    state['params']['kernel'] = float_weights
    state = flax.core.freeze(state)
    outputs_with_quant = model.apply(state, inputs, padding_mask=None)
    onp.testing.assert_raises(AssertionError, onp.testing.assert_array_equal,
                              outputs_with_quant, exp_output_without_quant)
    test_utils.assert_all_close_prec(exp_output_without_quant,
                                     outputs_with_quant, weight_prec)

  # TODO(wanglisa): Add tests with bigger matrices.

  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_fp143_scaled',
          weight_prec=fp143_scaled,
          weight_scale=onp.array([1, 2, 4, 8]),
      ),
      dict(
          testcase_name='dense_quant_fp143',
          weight_prec=fp143_unscaled,
          weight_scale=onp.array([1, 2, 4, 8]),
      ),
      dict(
          testcase_name='dense_quant_8bit',
          weight_prec=8,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='dense_quant_4bit',
          weight_prec=4,
          weight_scale=onp.array([1, 2, 4, 8])),
      dict(
          testcase_name='dense_quant_2bit',
          weight_prec=2,
          weight_scale=onp.array([1, 2, 4, 8])),
  )
  def test_weight_invariance_to_power_of_2_weight_scaling(
      self, weight_prec, weight_scale):
    # Scaling the weights before quantization by a power of 2 per channel should
    # also scale the output exactly by the same scale.

    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=weight_prec)
    weights = random.uniform(self.rng_key, shape=(3, 4))
    weight_scale = weight_scale[jnp.newaxis, :]
    state = state.unfreeze()
    state['params']['kernel'] = weights
    state = flax.core.freeze(state)
    outputs_without_scaling = model.apply(state, inputs, padding_mask=None)
    state = state.unfreeze()
    state['params']['kernel'] = jnp.multiply(weights, weight_scale)
    state = flax.core.freeze(state)
    outputs_with_scaling = model.apply(state, inputs, padding_mask=None)

    onp.testing.assert_array_equal(outputs_without_scaling * weight_scale,
                                   outputs_with_scaling)

  def test_1_bit_makes_all_weight_equal_to_zero(self):
    inputs = random.uniform(self.rng_key, shape=(2, 3))
    model, state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=1)
    weights = random.uniform(
        self.rng_key, shape=state['params']['kernel'].shape)
    state = state.unfreeze()
    state['params']['kernel'] = weights
    state = flax.core.freeze(state)
    outputs = model.apply(state, inputs, padding_mask=None)
    onp.testing.assert_array_equal(outputs, onp.zeros((2, 4)))

  # TODO(shivaniagrawal): change mock tests to check for QuantOps than
  # primitives.
  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_8bit',
          weight_prec=8,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='dense_quant_4bit',
          weight_prec=4,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='dense_quant_2bit',
          weight_prec=2,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='dense_signed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='dense_signed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='dense_signed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='dense_signed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
      dict(
          testcase_name='dense_signed_input_quant_2bit',
          weight_prec=None,
          acts_prec=2,
          fixed_bounds=True),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_weights_and_symmetrics_acts_should_call_clip_and_round(
      self, floor_with_gradient, round_with_gradient, weight_prec, acts_prec,
      fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        prec=acts_prec,
        bounds=bounds)
    num_features = 4
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(
        inputs, num_features, weight_prec=weight_prec, quant_act=quant_act)

    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    outputs = model.apply(state, inputs, padding_mask=None)

    self.assertEqual(outputs.shape, (inputs.shape[0], num_features))
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()

  # TODO(shivaniagrawal): change mock tests to check for QuantOps than
  # primitives.
  @parameterized.named_parameters(
      dict(
          testcase_name='dense_pos_quant_8bit',
          pos_inputs_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='dense_pos_quant_4bit',
          pos_inputs_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='dense_pos_quant_8bit_auto_clip',
          pos_inputs_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='dense_pos_quant_4bit_aut_clip',
          pos_inputs_prec=4,
          fixed_bounds=False),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_quantized_inputs_should_call_clip_and_round(self,
                                                       floor_with_gradient,
                                                       round_with_gradient,
                                                       pos_inputs_prec,
                                                       fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.positive,
        prec=pos_inputs_prec,
        bounds=bounds)
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, init_state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=None, quant_act=quant_act)
    floor_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(floor_with_gradient.call_count, 1)
    round_with_gradient.assert_not_called()

    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()

    model.apply(init_state, inputs, padding_mask=None)

    floor_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(floor_with_gradient.call_count, 1)
    round_with_gradient.assert_not_called()

  @parameterized.named_parameters(
      dict(
          testcase_name='dense_quant_fp143_scaled',
          inputs_prec=fp143_scaled,
          fixed_bounds=True,
      ),)
  @mock.patch.object(fp_cast, 'downcast_sat_ftz')
  def test_fp_quantized_inputs_should_call_downcast_sat_ftz(
      self, downcast_mock, inputs_prec, fixed_bounds):

    downcast_mock.side_effect = lambda x, *_: x
    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.positive,
        prec=inputs_prec,
        bounds=bounds)
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, init_state = self.init_model_with_1_layer(
        inputs, num_features=4, weight_prec=None, quant_act=quant_act)
    downcast_mock.assert_called_once_with(
        mock.ANY,
        inputs_prec.fp_spec.exp_min,
        inputs_prec.fp_spec.exp_max,
        inputs_prec.fp_spec.sig_bits,
    )
    downcast_mock.reset_mock()

    model.apply(init_state, inputs, padding_mask=None)

    downcast_mock.assert_called_once_with(
        mock.ANY,
        inputs_prec.fp_spec.exp_min,
        inputs_prec.fp_spec.exp_max,
        inputs_prec.fp_spec.sig_bits,
    )

  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_without_quantized_weights_should_not_call_quantization_ops(
      self, floor_with_gradient, round_with_gradient):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x
    inputs = jnp.ones((2, 3), dtype=jnp.float32)
    model, state = self.init_model_with_1_layer(inputs, num_features=4)
    _ = model.apply(state, inputs, padding_mask=None)
    round_with_gradient.assert_not_called()
    floor_with_gradient.assert_not_called()

  @parameterized.parameters(
      dict(granularity=quant_config.QuantGranularity.per_channel, axis=(0,)),
      dict(granularity=quant_config.QuantGranularity.per_tensor, axis=None))
  @mock.patch.object(quantization, 'quantized_dot')
  @mock.patch.object(shape_utils, 'assert_shapes_equal')
  def test_quant_granularity(self, _, mock_quantized_dot, granularity, axis):
    hparams = flax_layers.DenseAqt.HParams(
        weight_prec=8,
        quant_act=None,
        quant_type=quantization.QuantType.fake_quant,
        weight_quant_granularity=granularity)
    layer = flax_layers.DenseAqt(
        features=2,
        hparams=hparams,
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        paxis_name=None,
        train=False,
        dtype=jnp.float32)
    x = jnp.ones((2, 2))
    state = layer.init(self.rng_key, x, padding_mask=None)
    layer.apply(state, x, padding_mask=None)
    weight_params = mock_quantized_dot.call_args[1]['weight_params']
    self.assertEqual(weight_params.axis, axis)


class EmbedLayerTest(parameterized.TestCase):
  """Tests for AQT Embed layer."""

  # TODO(shivaniagrawal/malmaud): we are not raising error on jax rank
  # promotion. For EmbedAqt tests; in AQT style inputs and output are not be
  # of same shape; require more work to avoid rank promotion.
  @parameterized.named_parameters(
      dict(
          testcase_name='8_bit',
          weight_prec=8,
      ),
      dict(
          testcase_name='4_bit',
          weight_prec=4,
      ),
      dict(
          testcase_name='no_quantization',
          weight_prec=None,
      ),
  )
  def test_embed(self, weight_prec):
    # Since the dummy embedding matrix has a row of all zeros, we need 'epsilon'
    # to be added to it before calculating scale factors.
    quantization.DISABLE_EPSILON_IN_SCALE_FUN_FOR_TESTING = False
    rng = random.PRNGKey(0)
    x = jnp.arange(4)[None]
    dummy_embedding = jnp.broadcast_to(jnp.arange(4)[Ellipsis, None],
                                       (4, 3)).astype(jnp.float32)
    embed_module = flax_layers.EmbedAqt(
        num_embeddings=4,
        features=3,
        dtype=jnp.float32,
        hparams=flax_layers.EmbedAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.fake_quant),
        embedding_init=lambda _rng, _shape: dummy_embedding,
        train=False,
        paxis_name=None,
        quant_context=quant_config.QuantContext(update_bounds=False),
    )
    y, state = embed_module.init_with_output(rng, x)
    test_utils.assert_all_close_prec(dummy_embedding[None], y, weight_prec)

    z = embed_module.apply(
        state, jnp.ones((1, 3)), padding_mask=None, method=embed_module.attend)
    test_utils.assert_all_close_prec(3. * jnp.arange(4), z[0, Ellipsis], weight_prec)

  @parameterized.named_parameters(
      dict(
          testcase_name='8_bit',
          weight_prec=8,
      ),
      dict(
          testcase_name='4_bit',
          weight_prec=4,
      ),
      dict(
          testcase_name='no_quantization',
          weight_prec=None,
      ),
  )
  def test_embed_equality(self, weight_prec):
    rng = random.PRNGKey(0)
    x = 2 * jnp.ones(4, dtype=jnp.int32)[None]
    dummy_embedding = 2 * jnp.ones((4, 2)).astype(jnp.float32)
    embed_module = flax_layers.EmbedAqt(
        num_embeddings=4,
        features=2,
        dtype=jnp.float32,
        hparams=flax_layers.EmbedAqt.HParams(
            weight_prec=weight_prec,
            quant_act=None,
            quant_type=QuantType.fake_quant),
        embedding_init=lambda _rng, _shape: dummy_embedding,
        train=False,
        quant_context=quant_config.QuantContext(update_bounds=False),
        paxis_name=None)
    y, init_state = embed_module.init_with_output(rng, x)
    onp.testing.assert_array_equal(dummy_embedding[None], y)

    z = embed_module.apply(
        init_state,
        jnp.ones((1, 2)),
        padding_mask=None,
        method=embed_module.attend)
    onp.testing.assert_array_equal(2. * (2 * jnp.ones(4)), z[0, Ellipsis])

  @parameterized.named_parameters(
      dict(
          testcase_name='embed_quant_8bit',
          weight_prec=8,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='embed_quant_4bit',
          weight_prec=4,
          acts_prec=None,
          fixed_bounds=True),
      dict(
          testcase_name='embed_input_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=True),
      dict(
          testcase_name='embed_input_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=True),
      dict(
          testcase_name='embed_input_auto_quant_8bit',
          weight_prec=None,
          acts_prec=8,
          fixed_bounds=False),
      dict(
          testcase_name='embed_input_auto_quant_4bit',
          weight_prec=None,
          acts_prec=4,
          fixed_bounds=False),
  )
  @mock.patch.object(primitives, 'round_with_gradient')
  @mock.patch.object(primitives, 'floor_with_gradient')
  def test_embed_should_call_clip_and_round(self, floor_with_gradient,
                                            round_with_gradient, weight_prec,
                                            acts_prec, fixed_bounds):

    round_with_gradient.side_effect = lambda x: x
    floor_with_gradient.side_effect = lambda x: x

    if fixed_bounds:
      bounds = 6.0
    else:
      bounds = get_bounds.GetBounds.Hyper(
          initial_bound=6.0,
          stddev_coeff=3.0,
          absdev_coeff=2.0,
          mix_coeff=0.5,
          granularity=quant_config.QuantGranularity.per_tensor)
    quant_act = quantization.QuantOps.ActHParams(
        input_distribution=QuantOps.ActHParams.InputDistribution.symmetric,
        prec=acts_prec,
        bounds=bounds)
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))

    embed_module = flax_layers.EmbedAqt(
        num_embeddings=4,
        features=3,
        dtype=jnp.float32,
        hparams=flax_layers.EmbedAqt.HParams(
            weight_prec=weight_prec,
            quant_act=quant_act,
            quant_type=QuantType.fake_quant),
        quant_context=quant_config.QuantContext(update_bounds=False),
        paxis_name=None,
        train=False)
    init_state = embed_module.init(
        rng, x, method=embed_module.attend, padding_mask=None)
    round_with_gradient.reset_mock()
    floor_with_gradient.reset_mock()
    embed_module.apply(
        init_state, x, padding_mask=None, method=embed_module.attend)
    round_with_gradient.assert_called_with(mock.ANY)
    self.assertEqual(round_with_gradient.call_count, 1)
    floor_with_gradient.assert_not_called()


class LayerNormTest(parameterized.TestCase):

  @classmethod
  def make_hparams(cls, quantize_reductions, exp_min, exp_max, sig_bits):
    prec = QuantOps.FloatQuant.FloatPrec(
        exp_min=exp_min, exp_max=exp_max, sig_bits=sig_bits)
    reduction_prec = prec if quantize_reductions else None
    hparams = flax_layers.LayerNormAqt.HParams(
        quant_hparams=flax_layers.LayerNormAqt.QuantHParams(
            prec=prec,
            reduction_prec=reduction_prec,
        ))
    return hparams

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)

  @parameterized.parameters(itertools.product((False, True), (False, True)))
  def test_quantized_layer_norm_matches_unquantized_in_fp32(
      self, quantize_acts, quantize_reductions):
    # We 'quantize' to a custom floating-point format that is approximately
    # equivalent to IEEE float32 and test that results are the same as using
    # Flax's upstream unquantized LayerNorm.
    hparams = self.make_hparams(
        exp_min=-2**7,
        exp_max=2**7,
        sig_bits=23,
        quantize_reductions=quantize_reductions)
    quantized_layer_norm = flax_layers.LayerNormAqt(
        hparams=hparams,
        dtype=jnp.float32,
        quant_context=quant_config.QuantContext(
            update_bounds=False, quantize_acts=quantize_acts))
    x_rng, param_rng = jax.random.split(self.rng)
    x = jax.random.normal(x_rng, (3, 5))
    initial_params = quantized_layer_norm.init(param_rng, x)
    y_quantized = quantized_layer_norm.apply(initial_params, x)
    unquantized_layer_norm = nn.LayerNorm()
    y_unquantized = unquantized_layer_norm.apply(initial_params, x)
    onp.testing.assert_allclose(y_quantized, y_unquantized, rtol=1e-6)

  def test_epsilon_rounding(self):
    # We give LayerNorm a constant input. Since that input has a variance of
    # zero, we would expect layernorm to return NaN (0/0) unless the 'epsilon'
    # parameter which nudges the denominator away from zero was having an
    # effect. We test the case where the default epsilon value of 1e-6 would
    # ordinarily flush to zero after quantization with a high value of exp_min.
    # This test makes sure our code to round epsilon up to the smallest non-zero
    # representable value is wokring.
    hparams = self.make_hparams(
        exp_min=-2**2, exp_max=2**7, sig_bits=23, quantize_reductions=False)
    layer_norm = flax_layers.LayerNormAqt(
        hparams=hparams,
        use_bias=False,
        use_scale=False,
        epsilon=1e-6,
        dtype=jnp.float32,
        quant_context=quant_config.QuantContext(
            update_bounds=False, quantize_acts=True))
    x = jnp.ones((2, 5))
    y = layer_norm.apply({}, x)
    onp.testing.assert_equal(onp.array(y), onp.zeros(x.shape))


if __name__ == '__main__':
  absltest.main()
