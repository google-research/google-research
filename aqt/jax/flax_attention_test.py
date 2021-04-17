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

"""Tests for aqt.jax.flax_attention."""

import unittest

from absl.testing import absltest
from absl.testing import parameterized
from flax import jax_utils
import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp
import numpy as onp
import tensorflow as tf

from aqt.jax import flax_attention
from aqt.jax import flax_layers
from aqt.jax import get_bounds
from aqt.jax import quant_config
from aqt.jax import test_utils
from aqt.jax.flax_attention import DotProductAttnHParams
from aqt.jax.flax_attention import ExpHParams
from aqt.jax.flax_attention import ReciprocalHParams
from aqt.jax.flax_attention import SoftmaxHParams
from aqt.jax.quantization import QuantOps
from aqt.jax.quantization import QuantType

test_utils.configure_jax()


# Expressions to approximate Softmax
# Softmax with params sum_high_bound=1, low_bound=-200, clip_and_substract=True
def mod_softmax_clipped(sum_high_bound, low_bound, x, xs):
  exp_max_sub = lambda x: onp.exp(max(x, low_bound)) - onp.exp(low_bound)
  return exp_max_sub(x) / min(sum_high_bound, sum(exp_max_sub(x) for x in xs))


# Linearized exponential
def softmax_exp_lin(a, x, xs):
  exp_lin = lambda x: max(0, a * x + 1)
  return exp_lin(x) / sum(exp_lin(x) for x in xs)


# Linearized reciprocal
def softmax_recip_lin(a, x, xs):
  recip_lin = lambda x: a + 1 - a * x
  return onp.exp(x) * recip_lin(sum(onp.exp(x) for x in xs))


class SoftmaxTest(parameterized.TestCase, tf.test.TestCase):

  # Test mock input and output of modified versions of softmax.
  rand_2x2_tensor = onp.random.rand(2, 2)
  # Substract maximum from each column
  norm_dims = (0,)
  tensor_sub_max = rand_2x2_tensor - onp.max(rand_2x2_tensor, norm_dims[0])
  [[a11, a12], [a21, a22]] = tensor_sub_max

  @parameterized.named_parameters(
      dict(
          testcase_name='original_softmax',
          input_tensor=rand_2x2_tensor,
          norm_dims=norm_dims,
          softmax_hparams=SoftmaxHParams(None, None, None),
          expected_output=onp.array(
              [[
                  onp.exp(a11) / (onp.exp(a11) + onp.exp(a21)),
                  onp.exp(a12) / (onp.exp(a12) + onp.exp(a22))
              ],
               [
                   onp.exp(a21) / (onp.exp(a11) + onp.exp(a21)),
                   onp.exp(a22) / (onp.exp(a12) + onp.exp(a22))
               ]])),
      dict(
          testcase_name='exponential_clip=-200_clip_and_substract=True_' +
          'sum_high_bound=1.0',
          input_tensor=rand_2x2_tensor,
          norm_dims=norm_dims,
          softmax_hparams=SoftmaxHParams(
              exp_hparams=ExpHParams(
                  sum_high_bound=1.0,
                  low_bound=-200.0,
                  clip_and_subtract=True,
                  linear_gradient=None),
              reciprocal_hparams=None,
              quant_hparams=None),
          expected_output=onp.array(
              [[
                  mod_softmax_clipped(1.0, -200.0, a11, [a11, a21]),
                  mod_softmax_clipped(1.0, -200.0, a12, [a12, a22])
              ],
               [
                   mod_softmax_clipped(1.0, -200.0, a21, [a11, a21]),
                   mod_softmax_clipped(1.0, -200.0, a22, [a12, a22])
               ]])),
      dict(
          testcase_name='linear_gradient=1.0',
          input_tensor=rand_2x2_tensor,
          norm_dims=norm_dims,
          softmax_hparams=SoftmaxHParams(
              exp_hparams=ExpHParams(
                  sum_high_bound=0.0,
                  low_bound=0.0,
                  clip_and_subtract=False,
                  linear_gradient=1.0),
              reciprocal_hparams=None,
              quant_hparams=None),
          expected_output=onp.array([[
              softmax_exp_lin(1.0, a11, [a11, a21]),
              softmax_exp_lin(1.0, a12, [a12, a22])
          ],
                                     [
                                         softmax_exp_lin(1.0, a21, [a11, a21]),
                                         softmax_exp_lin(1.0, a22, [a12, a22])
                                     ]])),
      dict(
          testcase_name='linear_gradient=1.0.',
          input_tensor=rand_2x2_tensor,
          norm_dims=norm_dims,
          softmax_hparams=SoftmaxHParams(
              exp_hparams=ExpHParams(
                  sum_high_bound=0.0,
                  low_bound=0.0,
                  clip_and_subtract=False,
                  linear_gradient=0.0),
              reciprocal_hparams=ReciprocalHParams(
                  linear_gradient=1.0, low_bound=0.0),
              quant_hparams=None),
          expected_output=onp.array(
              [[
                  softmax_recip_lin(1.0, a11, [a11, a21]),
                  softmax_recip_lin(1.0, a12, [a12, a22])
              ],
               [
                   softmax_recip_lin(1.0, a21, [a11, a21]),
                   softmax_recip_lin(1.0, a22, [a12, a22])
               ]]),
      ),
  )
  def test_custom_softmax_vs_mock(self, input_tensor, norm_dims,
                                  softmax_hparams, expected_output):
    dtype = jax._src.numpy.lax_numpy.float32
    output = flax_attention.softmax(
        input_tensor, norm_dims, dtype, softmax_hparams,
        quant_config.QuantContext(update_bounds=False, quantize_acts=False))
    self.assertAllClose(expected_output, output, atol=1e-6)

  # # Test modified softmax vs original softmax.
  random_input = onp.random.rand(16, 16, 2, 2)

  @parameterized.named_parameters(
      dict(
          testcase_name='modified_softmax_with_exponential_low_bound_at_-10',
          input_tensor=random_input,
          softmax_hparams=SoftmaxHParams(
              exp_hparams=ExpHParams(
                  sum_high_bound=None,
                  low_bound=-10.0,
                  clip_and_subtract=False,
                  linear_gradient=None),
              reciprocal_hparams=None,
              quant_hparams=None,
          )),
      dict(
          testcase_name='modified_softmax_with_exponential_low_bound_=-200_' +
          'clip_substract=True',
          input_tensor=random_input,
          softmax_hparams=SoftmaxHParams(
              exp_hparams=ExpHParams(
                  sum_high_bound=None,
                  low_bound=-200.0,
                  clip_and_subtract=True,
                  linear_gradient=None),
              reciprocal_hparams=None,
              quant_hparams=None),
      ),
      # Test that 'downcasting' intermediate activations to a floating-point
      # format similar to IEEE fp32 produces almost the same results as
      # unquantized softmax.
      dict(
          testcase_name='fp_downcast_fp32',
          input_tensor=random_input,
          softmax_hparams=SoftmaxHParams(
              exp_hparams=None,
              reciprocal_hparams=None,
              quant_hparams=flax_attention.SoftmaxQuantHParams(
                  reduction_prec=None,
                  prec=QuantOps.FloatQuant.FloatPrec(
                      exp_min=-2**7, exp_max=2**7, sig_bits=23))),
      ))
  def test_softmax_vs_original(self, input_tensor, softmax_hparams):
    dtype = jax._src.numpy.lax_numpy.float32
    norm_dims = (0,)
    input_tensor = jnp.array(input_tensor)
    output = flax_attention.softmax(
        input_tensor, norm_dims, dtype, softmax_hparams,
        quant_config.QuantContext(update_bounds=False, quantize_acts=True))
    expected_output = flax_attention.softmax(
        input_tensor, norm_dims, dtype, SoftmaxHParams(None, None, None),
        quant_config.QuantContext(update_bounds=False, quantize_acts=True))
    self.assertAllClose(expected_output, output, atol=1e-8)


class AttentionTest(parameterized.TestCase):

  @classmethod
  def construct_hparams(cls, weight_prec):
    dense = flax_layers.DenseAqt.HParams(
        weight_prec=weight_prec,
        quant_act=None,
        quant_type=QuantType.fake_quant,
        weight_quant_granularity=quant_config.QuantGranularity.per_channel)
    return flax_attention.MultiHeadDotProductAttentionAqt.HParams(
        dense_kqv=dense,
        dense_out=dense,
        attn_acts=DotProductAttnHParams(
            attn_act_q=None,
            attn_act_k=None,
            attn_act_probs=None,
            attn_act_v=None,
            quant_type=QuantType.fake_quant,
            softmax=SoftmaxHParams(None, None, None)))

  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
  )
  def test_multihead_self_attention(self, weight_prec):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 3, 5))
    hparams = self.construct_hparams(weight_prec)
    sa_module = flax_attention.SelfAttentionAqt(
        hparams=hparams,
        num_heads=8,
        attention_axis=(1,),
        qkv_features=16,
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        train=False,
        paxis_name=None,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dtype=jnp.float32,
        causal_mask=False,
        dropout_rate=0.0,
        deterministic=False,
        decode=False)
    y, _ = sa_module.init_with_output(rng, x, padding_mask=None)
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
  )
  def test_multihead_encoder_decoder_attention(self, weight_prec):
    rng = random.PRNGKey(0)
    q = jnp.ones((4, 3, 5))
    kv = jnp.ones((4, 3, 5))
    sa_module = flax_attention.MultiHeadDotProductAttentionAqt(
        num_heads=8,
        hparams=self.construct_hparams(weight_prec),
        attention_axis=(1,),
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        train=False,
        paxis_name=None,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dtype=jnp.float32,
        causal_mask=False,
        dropout_rate=0.0,
        deterministic=False,
        decode=False)
    y, _ = sa_module.init_with_output(
        rng, q, kv, padding_mask=None, key_padding_mask=None)
    self.assertEqual(y.shape, q.shape)

  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
  )
  def test_multihead_self_attention_w_dropout(self, weight_prec):
    rng = random.PRNGKey(0)
    x = jnp.ones((4, 3, 5))
    sa_module = flax_attention.SelfAttentionAqt(
        num_heads=8,
        hparams=self.construct_hparams(weight_prec),
        attention_axis=(1,),
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        train=False,
        paxis_name=None,
        qkv_features=16,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        dropout_rate=0.1,
        dtype=jnp.float32,
        causal_mask=False,
        deterministic=False,
        decode=False)
    rng_dropout, rng_params = random.split(rng)
    y, _ = sa_module.init_with_output(
        {
            'dropout': rng_dropout,
            'params': rng_params
        }, x, padding_mask=None)
    self.assertEqual(y.shape, x.shape)

  @parameterized.named_parameters(
      dict(
          testcase_name='float_spatial_shape_5_attn_dim_1',
          weight_prec=None,
          spatial_shape=(5,),
          attn_dims=(1,)),
      dict(
          testcase_name='quant_8bit_spatial_shape_5_attn_dim_1',
          weight_prec=8,
          spatial_shape=(5,),
          attn_dims=(1,)),
      dict(
          testcase_name='quant_4bit_spatial_shape_5_attn_dim_1',
          weight_prec=4,
          spatial_shape=(5,),
          attn_dims=(1,)),
  )
  def test_decoding(self, weight_prec, spatial_shape, attn_dims):
    bs = 2
    num_heads = 3
    num_features = 4
    rng = random.PRNGKey(0)
    key1, key2 = random.split(rng)
    inputs = random.normal(key1,
                           (bs,) + spatial_shape + (num_heads * num_features,))
    module = flax_attention.SelfAttentionAqt(
        num_heads=num_heads,
        hparams=self.construct_hparams(weight_prec),
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        train=False,
        paxis_name=None,
        qkv_features=num_heads * num_features,
        attention_axis=attn_dims,
        decode=False,
        causal_mask=True,
        dtype=jnp.float32,
        dropout_rate=0.0,
        deterministic=False)

    initial_vars = module.init(key2, inputs, padding_mask=None)
    y_ref = module.apply(initial_vars, inputs, padding_mask=None)
    module.decode = True
    initial_vars_decode = module.init(key2, inputs, padding_mask=None)
    cache0 = initial_vars_decode['cache']

    def body_fn(cache, x):
      y, new_vars = module.apply({
          **initial_vars, 'cache': cache
      },
                                 x,
                                 mutable='cache',
                                 padding_mask=None)
      return new_vars['cache'], y

    # scan_in_dim supports scanning multiple dims
    _, y = jax_utils.scan_in_dim(
        body_fn, cache0, inputs, axis=attn_dims, keepdims=True)

    onp.testing.assert_allclose(y_ref, y, atol=1e-5)

  @parameterized.named_parameters(
      dict(testcase_name='float', weight_prec=None),
      dict(testcase_name='quant_8bit', weight_prec=8),
      dict(testcase_name='quant_4bit', weight_prec=4),
  )
  def test_autoregresive_receptive_field_1d(self, weight_prec):
    """Tests the autoregresive self-attention receptive field."""
    rng = random.PRNGKey(0)
    rng1, rng2 = random.split(rng, num=2)

    def model_loss(inputs, pos):
      out = module.apply(initial_vars, inputs, padding_mask=None)
      assert out.shape == input_shape
      assert len(out.shape) == 3
      return out[0, pos, :].sum()

    grad_fn = jax.jit(jax.grad(model_loss))

    def get_receptive_field_1d(pos):
      g = grad_fn(inputs, pos)[0, :, :]
      return jnp.any((jnp.abs(g) > 1e-5).astype(jnp.uint32), axis=-1)

    length = 10
    dim = 1
    num_heads = 1
    input_shape = (1, length, dim)
    inputs = random.normal(rng2, input_shape)

    module = flax_attention.SelfAttentionAqt(
        num_heads=num_heads,
        hparams=self.construct_hparams(weight_prec),
        quant_context=quant_config.QuantContext(
            update_bounds=False, collect_acts_stats=False),
        train=False,
        paxis_name=None,
        causal_mask=True,
        kernel_init=initializers.ones,
        dtype=jnp.float32,
        qkv_features=None,
        attention_axis=None,
        dropout_rate=0.0,
        deterministic=False,
        decode=False)
    initial_vars = module.init(
        rng1, jnp.ones((1,) + (length, dim), jnp.float32), padding_mask=None)
    # model = nn.Model(module, initial_params)

    for i in range(length):
      deps = get_receptive_field_1d(i)
      assert (deps[:i] == 1).all(), ('Receptive Field Error: Some of the '
                                     'previous positions are not reachable '
                                     'in autoregressive self-attention.')
      if i != length - 1:
        k = i + 1
        assert (deps[k:] == 0).all(), ('Receptive Field Error: Some of the '
                                       'future positions are reachable in '
                                       'autoregressive self-attention.')

  def test_padding_mask(self):
    """Test that the activation stats respect masking."""
    # This test's strategy is to change the value of a channels of a padding
    # token and make sure the stats don't change. Because the attention
    # calculation is fairly involved, this is more robust and less tedious than
    # trying to directly test numeric expected values.

    # Construct HParams with dynamic bounds.
    # Exact values don't matter, just need bounds to be dynamic so stats are
    # collected.
    bounds = get_bounds.GetBounds.Hyper(
        initial_bound=0.0,
        stddev_coeff=0.4,
        absdev_coeff=0.6,
        mix_coeff=0.4,
        reset_stats=False,
        granularity=quant_config.QuantGranularity.per_channel)
    quant_act = flax_layers.QuantOps.ActHParams(
        input_distribution=flax_layers.QuantOps.ActHParams.InputDistribution
        .symmetric,
        prec=8,
        bounds=bounds)
    attn_quant_act = flax_layers.QuantOps.ActHParams(
        input_distribution=flax_layers.QuantOps.ActHParams.InputDistribution
        .positive,
        prec=8,
        bounds=1.0)
    dense_hparams = flax_layers.DenseAqt.HParams(
        quant_type=flax_layers.QuantType.fake_quant,
        weight_prec=8,
        quant_act=quant_act,
        weight_quant_granularity=quant_config.QuantGranularity.per_channel)
    dotproduct_attn_hparams = flax_attention.DotProductAttnHParams(
        attn_act_q=quant_act,
        attn_act_k=quant_act,
        attn_act_v=quant_act,
        attn_act_probs=attn_quant_act,
        quant_type=QuantType.fake_quant,
        softmax=SoftmaxHParams(None, None, None))
    attn_hparams = flax_attention.MultiHeadDotProductAttentionAqt.HParams(
        dense_kqv=dense_hparams,
        dense_out=dense_hparams,
        attn_acts=dotproduct_attn_hparams)

    module = flax_attention.SelfAttentionAqt(
        hparams=attn_hparams,
        num_heads=2,
        paxis_name=None,
        train=True,
        quant_context=quant_config.QuantContext(
            update_bounds=True, collect_acts_stats=False),
        dtype=jnp.float32,
        qkv_features=None,
        attention_axis=None,
        causal_mask=False,
        dropout_rate=0.0,
        deterministic=False,
        decode=False)
    # Simulate an input of a batch size of 1 with two tokens, each with four
    # features
    x = onp.arange(8).astype(onp.float32).reshape((1, 2, 4))
    initial_state = module.init(random.PRNGKey(0), x, padding_mask=None)

    padding_mask = onp.full((1, 2, 1), True)
    padding_mask[0, 1, 0] = False  # Mask out the second token
    _, state1 = module.apply(
        initial_state, x, padding_mask=padding_mask, mutable=True)
    # Now we adjust the input for the masked token and recompute the mean. It
    # should be the same as before.
    x[0, 1, 0] = 100
    _, state2 = module.apply(
        initial_state, x, padding_mask=padding_mask, mutable=True)
    test_utils.assert_stats_are_equal(state1, state2)
    # Now we adjust the input for an unmasked token and verify that the stats
    # have changed.
    x[0, 0, 0] = 200
    _, state3 = module.apply(
        initial_state, x, padding_mask=padding_mask, mutable=True)
    test_utils.assert_stats_are_unequal(state1, state3)


class AttnActsMatmulQuantTest(parameterized.TestCase):

  def construct_hparams(self, attn_act_q, attn_act_k, attn_act_probs,
                        attn_act_v):
    dense = flax_layers.DenseAqt.HParams(
        weight_prec=None,
        quant_act=None,
        quant_type=QuantType.fake_quant,
        weight_quant_granularity=quant_config.QuantGranularity.per_channel)
    return flax_attention.MultiHeadDotProductAttentionAqt.HParams(
        dense_kqv=dense,
        dense_out=dense,
        attn_acts=flax_attention.DotProductAttnHParams(
            attn_act_q=attn_act_q,
            attn_act_k=attn_act_k,
            attn_act_probs=attn_act_probs,
            attn_act_v=attn_act_v,
            quant_type=QuantType.fake_quant,
            softmax=SoftmaxHParams(None, None, None)))

  @parameterized.named_parameters(
      dict(
          testcase_name='float',
          attn_act_q=None,
          attn_act_k=None,
          attn_act_probs=None,
          attn_act_v=None,
          update_bounds=False,
          paxis_name=None,
          train=False),
      dict(
          testcase_name='quant_q',
          attn_act_q=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=8,
              bounds=1),
          attn_act_k=None,
          attn_act_probs=None,
          attn_act_v=None,
          update_bounds=False,
          paxis_name='batch',
          train=True),
      dict(
          testcase_name='quant_qk',
          attn_act_q=None,
          attn_act_k=None,
          attn_act_probs=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=8,
              bounds=1.0),
          attn_act_v=None,
          update_bounds=False,
          paxis_name='batch',
          train=True),
      dict(
          testcase_name='quant_k',
          attn_act_q=None,
          attn_act_k=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=4,
              bounds=2),
          attn_act_probs=None,
          attn_act_v=None,
          update_bounds=False,
          paxis_name=None,
          train=True),
      dict(
          testcase_name='quant_v',
          attn_act_q=None,
          attn_act_k=None,
          attn_act_probs=None,
          attn_act_v=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=2,
              bounds=3),
          update_bounds=True,
          paxis_name='batch',
          train=False),
      dict(
          testcase_name='quant_all_aa',
          attn_act_q=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=8,
              bounds=1),
          attn_act_k=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=4,
              bounds=2),
          attn_act_probs=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=8,
              bounds=1.0),
          attn_act_v=QuantOps.ActHParams(
              input_distribution=QuantOps.ActHParams.InputDistribution
              .symmetric,
              prec=2,
              bounds=3),
          update_bounds=True,
          paxis_name=None,
          train=True),
  )
  @unittest.mock.patch.object(QuantOps, 'create_inputs_fake_quant')
  def test_self_attention_act_quant_should_call_quant_ops(
      self, mock_inputs_fake_quant, attn_act_q, attn_act_k, attn_act_probs,
      attn_act_v, update_bounds, paxis_name, train):

    mock_inputs_fake_quant.side_effect = (
        lambda inputs, hparams, get_bounds_params: inputs)

    rng = random.PRNGKey(0)
    x = jnp.ones((4, 3, 7))
    hparams = self.construct_hparams(attn_act_q, attn_act_k, attn_act_probs,
                                     attn_act_v)
    sa_module = flax_attention.SelfAttentionAqt(
        hparams=hparams,
        num_heads=4,
        quant_context=quant_config.QuantContext(
            update_bounds=update_bounds, collect_acts_stats=False),
        train=train,
        paxis_name=paxis_name,
        attention_axis=None,
        qkv_features=8,
        kernel_init=initializers.ones,
        bias_init=initializers.zeros,
        causal_mask=False,
        dtype=jnp.float32,
        dropout_rate=0.0,
        deterministic=False,
        decode=False)
    sa_module.init(rng, x, padding_mask=None)
    calls = []
    for hparam in [attn_act_q, attn_act_k, attn_act_probs, attn_act_v]:
      if hparam is not None:
        calls.append(
            unittest.mock.call(
                unittest.mock.ANY,
                hparams=hparam,
                get_bounds_params=get_bounds.GetBounds.Params(
                    update_stats=train,
                    update_bounds=update_bounds,
                    paxis_name=paxis_name,
                    mask=unittest.mock.ANY,
                    module_name=unittest.mock.ANY)))
    mock_inputs_fake_quant.assert_has_calls(calls, any_order=True)

    self.assertLen(calls, mock_inputs_fake_quant.call_count)

if __name__ == '__main__':
  absltest.main()
