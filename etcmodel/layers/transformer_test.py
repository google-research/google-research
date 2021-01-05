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

"""Tests for transformer layers."""

import collections

from absl.testing import parameterized
import mock
import numpy as np
import tensorflow as tf

from etcmodel import layers as etc_layers
from etcmodel.layers import recompute_grad as recompute_grad_lib


class TransformerLayersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(testcase_name='default_order_gather'),
      dict(
          testcase_name='pre_activation_order_gather',
          use_pre_activation_order=True),
      dict(testcase_name='default_order_one_hot', use_one_hot_lookup=True),
  )
  def test_global_local_transformer_layers(
      self,
      use_pre_activation_order: bool = False,
      use_one_hot_lookup: bool = False):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    long_seq_len = 64
    global_seq_len = 16
    long_hidden_size = 10
    global_hidden_size = 15
    num_layers = 2
    num_heads = 5
    local_radius = 7
    relative_vocab_size = 21

    global_intermediate_size = 2 * global_hidden_size
    long_intermediate_size = 2 * long_hidden_size

    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # the static `batch_size` is unknown.
    long_input = tf.compat.v1.placeholder_with_default(
        np.random.normal(size=[batch_size, long_seq_len, long_hidden_size]),
        shape=[None, long_seq_len, long_hidden_size])
    global_input = tf.compat.v1.placeholder_with_default(
        np.random.normal(size=[batch_size, global_seq_len, global_hidden_size]),
        shape=[None, global_seq_len, global_hidden_size])
    l2l_att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, long_seq_len, 2 * local_radius + 1]),
        shape=[None, long_seq_len, 2 * local_radius + 1])
    g2g_att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, global_seq_len, global_seq_len]),
        shape=[None, global_seq_len, global_seq_len])
    l2g_att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, long_seq_len, global_seq_len]),
        shape=[None, long_seq_len, global_seq_len])
    g2l_att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(
            n=1, p=0.9, size=[batch_size, global_seq_len, long_seq_len]),
        shape=[None, global_seq_len, long_seq_len])
    l2l_relative_att_ids = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, long_seq_len, 2 * local_radius + 1]),
        shape=[None, long_seq_len, 2 * local_radius + 1])
    g2g_relative_att_ids = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, global_seq_len, global_seq_len]),
        shape=[None, global_seq_len, global_seq_len])
    l2g_relative_att_ids = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, long_seq_len, global_seq_len]),
        shape=[None, long_seq_len, global_seq_len])
    g2l_relative_att_ids = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            relative_vocab_size,
            size=[batch_size, global_seq_len, long_seq_len]),
        shape=[None, global_seq_len, long_seq_len])

    layer = etc_layers.GlobalLocalTransformerLayers(
        long_hidden_size=long_hidden_size,
        global_hidden_size=global_hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        local_radius=local_radius,
        long_intermediate_size=long_intermediate_size,
        global_intermediate_size=global_intermediate_size,
        relative_vocab_size=relative_vocab_size,
        share_feed_forward_params=False,
        share_qkv_projections=False,
        use_pre_activation_order=use_pre_activation_order,
        use_one_hot_lookup=use_one_hot_lookup)

    long_output, global_output = layer(
        long_input,
        global_input,
        l2l_att_mask=l2l_att_mask,
        g2g_att_mask=g2g_att_mask,
        l2g_att_mask=l2g_att_mask,
        g2l_att_mask=g2l_att_mask,
        l2l_relative_att_ids=l2l_relative_att_ids,
        g2g_relative_att_ids=g2g_relative_att_ids,
        l2g_relative_att_ids=l2g_relative_att_ids,
        g2l_relative_att_ids=g2l_relative_att_ids,
        att_implementation='sparse')

    static_batch_size = global_input.shape.as_list()[0]
    self.assertAllEqual([static_batch_size, long_seq_len, long_hidden_size],
                        long_output.shape.as_list())
    self.assertAllEqual([static_batch_size, global_seq_len, global_hidden_size],
                        global_output.shape.as_list())

    # Make sure 'full' att_implementation gives the same output.
    long_output_full_att, global_output_full_att = layer(
        long_input,
        global_input,
        l2l_att_mask=l2l_att_mask,
        g2g_att_mask=g2g_att_mask,
        l2g_att_mask=l2g_att_mask,
        g2l_att_mask=g2l_att_mask,
        l2l_relative_att_ids=l2l_relative_att_ids,
        g2g_relative_att_ids=g2g_relative_att_ids,
        l2g_relative_att_ids=l2g_relative_att_ids,
        g2l_relative_att_ids=g2l_relative_att_ids,
        att_implementation='full')
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(long_output, long_output_full_att)
    self.assertAllClose(global_output, global_output_full_att)

    self.assertIsNot(layer.fused_att_layers[0], layer.fused_att_layers[1])
    self.assertIsNot(layer.long_feed_forward_layers[0],
                     layer.long_feed_forward_layers[1])
    self.assertIsNot(layer.global_feed_forward_layers[0],
                     layer.global_feed_forward_layers[1])

  @parameterized.named_parameters(
      dict(
          testcase_name='hidden_dropout_only',
          hidden_dropout_prob=0.5,
          attention_probs_dropout_prob=0.0),
      dict(
          testcase_name='attention_dropout_only',
          hidden_dropout_prob=0.0,
          attention_probs_dropout_prob=0.5),
      dict(
          testcase_name='both_dropout',
          hidden_dropout_prob=0.5,
          attention_probs_dropout_prob=0.5),
  )
  def test_global_local_transformer_layers_dropout(
      self, hidden_dropout_prob, attention_probs_dropout_prob):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 2
    long_seq_len = 16
    global_seq_len = 4
    long_hidden_size = 4
    global_hidden_size = 6
    num_layers = 2
    num_heads = 2
    local_radius = 3

    global_intermediate_size = 2 * global_hidden_size
    long_intermediate_size = 2 * long_hidden_size

    long_input = tf.constant(
        np.random.normal(size=[batch_size, long_seq_len, long_hidden_size]))
    global_input = tf.constant(
        np.random.normal(size=[batch_size, global_seq_len, global_hidden_size]))

    layer = etc_layers.GlobalLocalTransformerLayers(
        long_hidden_size=long_hidden_size,
        global_hidden_size=global_hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        local_radius=local_radius,
        long_intermediate_size=long_intermediate_size,
        global_intermediate_size=global_intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        share_feed_forward_params=False,
        share_qkv_projections=False)

    inference_long_output1, inference_global_output1 = layer(
        long_input, global_input, training=False)
    inference_long_output2, inference_global_output2 = layer(
        long_input, global_input, training=False)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(inference_long_output1, inference_long_output2)
    self.assertAllClose(inference_global_output1, inference_global_output2)

    # Dropout makes this non-deterministic.
    training_long_output1, training_global_output1 = layer(
        long_input, global_input, training=True)
    training_long_output2, training_global_output2 = layer(
        long_input, global_input, training=True)
    self.assertNotAllClose(training_long_output1, training_long_output2)
    self.assertNotAllClose(training_global_output1, training_global_output2)

  @parameterized.named_parameters(
      dict(
          testcase_name='default_order_gather',
          use_pre_activation_order=False,
          use_one_hot_lookup=False),
      dict(
          testcase_name='pre_activation_order_gather',
          use_pre_activation_order=True,
          use_one_hot_lookup=False),
      dict(
          testcase_name='default_order_one_hot',
          use_pre_activation_order=False,
          use_one_hot_lookup=True),
  )
  def test_relative_transformer_layers(self, use_pre_activation_order: bool,
                                       use_one_hot_lookup: bool):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 3
    seq_len = 16
    hidden_size = 10
    num_layers = 2
    num_heads = 5
    relative_vocab_size = 21

    # We use `placeholder_with_default` to simulate the TF v1 situation where
    # the static `batch_size` is unknown.
    inputs = tf.compat.v1.placeholder_with_default(
        np.random.normal(size=[batch_size, seq_len, hidden_size]).astype(
            np.float32),
        shape=[None, seq_len, hidden_size])
    att_mask = tf.compat.v1.placeholder_with_default(
        np.random.binomial(n=1, p=0.9, size=[batch_size, seq_len, seq_len]),
        shape=[None, seq_len, seq_len])
    relative_att_ids = tf.compat.v1.placeholder_with_default(
        np.random.randint(
            relative_vocab_size, size=[batch_size, seq_len, seq_len]),
        shape=[None, seq_len, seq_len])

    layer = etc_layers.RelativeTransformerLayers(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        relative_vocab_size=relative_vocab_size,
        use_pre_activation_order=use_pre_activation_order,
        use_one_hot_lookup=use_one_hot_lookup)

    result = layer(inputs, att_mask=att_mask, relative_att_ids=relative_att_ids)

    static_batch_size = inputs.shape.as_list()[0]
    self.assertAllEqual([static_batch_size, seq_len, hidden_size],
                        result.shape.as_list())

    self.assertNotEmpty(layer.feed_forward_layers)

  @parameterized.named_parameters(
      dict(
          testcase_name='hidden_dropout_only',
          hidden_dropout_prob=0.5,
          attention_probs_dropout_prob=0.0),
      dict(
          testcase_name='attention_dropout_only',
          hidden_dropout_prob=0.0,
          attention_probs_dropout_prob=0.5),
      dict(
          testcase_name='both_dropout',
          hidden_dropout_prob=0.5,
          attention_probs_dropout_prob=0.5),
  )
  def test_relative_transformer_layers_dropout(self, hidden_dropout_prob,
                                               attention_probs_dropout_prob):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 2
    seq_len = 16
    hidden_size = 6
    num_layers = 2
    num_heads = 2
    relative_vocab_size = 10

    inputs = tf.constant(
        np.random.normal(size=[batch_size, seq_len, hidden_size]), tf.float32)
    att_mask = tf.constant(
        np.random.binomial(n=1, p=0.9, size=[batch_size, seq_len, seq_len]))
    relative_att_ids = tf.constant(
        np.random.randint(
            relative_vocab_size, size=[batch_size, seq_len, seq_len]))

    layer = etc_layers.RelativeTransformerLayers(
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        relative_vocab_size=relative_vocab_size)

    inference_output1 = layer(
        inputs,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=False)
    inference_output2 = layer(
        inputs,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=False)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    self.assertAllClose(inference_output1, inference_output2)

    # Dropout makes this non-deterministic.
    training_output1 = layer(
        inputs,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=True)
    training_output2 = layer(
        inputs,
        att_mask=att_mask,
        relative_att_ids=relative_att_ids,
        training=True)
    self.assertNotAllClose(training_output1, training_output2)


class SeededDropout(object):
  """Dropout that uses NumPy for the local seed for determinism."""

  def __init__(self, seed):
    np.random.seed(seed)
    self.call_count = 0
    self._dropout = tf.nn.dropout

  def __call__(self, *args, **kwargs):
    self.call_count += 1
    if kwargs.get('seed') is None:
      kwargs['seed'] = np.random.randint(-2**31, 2**31, dtype=np.int32)
    return self._dropout(*args, **kwargs)


class GlobalLocalTransformerLayersGradientCheckpointingTest(
    tf.test.TestCase, parameterized.TestCase):
  """Tests for gradient checkpointing."""

  @parameterized.named_parameters(
      ('no_dropout',),
      ('dropout', 0.4, 0.5),
      ('dropout_period_2', 0.4, 0.5, 6, 2),
      ('dropout_period_3', 0.4, 0.5, 5, 3),
  )
  def test_correctness(self,
                       hidden_dropout_prob=0.0,
                       attention_probs_dropout_prob=0.0,
                       num_layers=2,
                       grad_checkpointing_period=1):
    tf.compat.v1.random.set_random_seed(1234)
    np.random.seed(1234)

    batch_size = 2
    long_seq_len = 16
    global_seq_len = 4
    long_hidden_size = 4
    global_hidden_size = 6
    num_heads = 2
    local_radius = 3

    global_intermediate_size = 2 * global_hidden_size
    long_intermediate_size = 2 * long_hidden_size

    long_input = tf.constant(
        np.random.normal(size=[batch_size, long_seq_len, long_hidden_size]))
    global_input = tf.constant(
        np.random.normal(size=[batch_size, global_seq_len, global_hidden_size]))
    # Deterministically generate a seed for the RecomputingDropout layers.
    np.random.seed(1235)
    recomputing_layers = etc_layers.GlobalLocalTransformerLayers(
        long_hidden_size=long_hidden_size,
        global_hidden_size=global_hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        local_radius=local_radius,
        long_intermediate_size=long_intermediate_size,
        global_intermediate_size=global_intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        share_feed_forward_params=False,
        share_qkv_projections=False,
        grad_checkpointing_period=grad_checkpointing_period)
    np.random.seed(1235)
    control_layers = etc_layers.GlobalLocalTransformerLayers(
        long_hidden_size=long_hidden_size,
        global_hidden_size=global_hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        local_radius=local_radius,
        long_intermediate_size=long_intermediate_size,
        global_intermediate_size=global_intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        share_feed_forward_params=False,
        share_qkv_projections=False)

    def compute_gradients(layer, inputs):
      (long_input, global_input) = inputs
      if tf.executing_eagerly():
        with tf.GradientTape() as tape:
          tape.watch(inputs)
          outputs = layer(long_input, global_input, training=True)
        return tape.gradient(outputs, inputs)
      else:
        outputs = layer(long_input, global_input, training=True)
        return tf.gradients(outputs, inputs)

    if not (tf.executing_eagerly() or tf.compat.v1.control_flow_v2_enabled()):
      compute_gradients = tf.function(compute_gradients)

    # Dropout calls depend on attention dropout and whether it's fused.
    dropout_calls_per_layer = 4
    if attention_probs_dropout_prob > 0:
      dropout_calls_per_layer += 2

    # Run layers to force building all variables.
    recomputing_layers(long_input, global_input)
    control_layers(long_input, global_input)
    self.evaluate(tf.compat.v1.global_variables_initializer())
    control_layers.set_weights(recomputing_layers.get_weights())
    # Set random state explicitly, so we can restore it and compare gradients.
    random_state = tf.random.experimental.create_rng_state(
        2022,
        tf.random.experimental.get_global_generator().algorithm)
    self.evaluate(tf.random.experimental.get_global_generator().state.assign(
        random_state))
    tf.compat.v1.random.set_random_seed(12345)

    with mock.patch.object(tf.nn, 'dropout',
                           SeededDropout(2020)) as mock_dropout:
      recompute_gradients = self.evaluate(
          compute_gradients(recomputing_layers, (long_input, global_input)))
    # The stateful dropout is called in the last chain of layers.
    self.assertEqual(
        mock_dropout.call_count,
        dropout_calls_per_layer * min(num_layers, grad_checkpointing_period))

    class GetRecomputeContextMock(object):
      """Creates a mock context to provide a seed to `RecomputingDropout`."""

      def __init__(self):
        self.call_count = 0

      def __call__(self):
        self.call_count += 1
        layer_idx = (self.call_count - 1) // dropout_calls_per_layer
        # The last chain of layers is not recomputed.
        if layer_idx >= num_layers - grad_checkpointing_period:
          return None
        # Remainder layers are the at the beginning.
        if layer_idx < num_layers % grad_checkpointing_period:
          seed_idx = 0
        else:
          # Shift the layers by the remainder.
          seed_idx = layer_idx - (num_layers % grad_checkpointing_period)
          seed_idx //= grad_checkpointing_period
          # Add 1 if there is a remainder.
          if num_layers % grad_checkpointing_period:
            seed_idx += 1
        seed = tf.constant(seeds[seed_idx])
        return recompute_grad_lib.RecomputeContext(
            is_recomputing=False, seed=seed, children=collections.deque())

    # Recreate random state to propagate the correct seed to dropout layers.
    self.evaluate(tf.random.experimental.get_global_generator().state.assign(
        random_state))
    num_seeds = (num_layers - 1) // grad_checkpointing_period
    seeds = [
        self.evaluate(
            tf.random.experimental.get_global_generator().uniform_full_int(
                [], tf.int32)) for _ in range(num_seeds)
    ]
    tf.compat.v1.random.set_random_seed(12345)

    with mock.patch.object(tf.nn, 'dropout', SeededDropout(2020)), \
        mock.patch.object(
            recompute_grad_lib, 'get_recompute_context',
            GetRecomputeContextMock()) as mock_get_recompute_context:
      control_gradients = self.evaluate(
          compute_gradients(control_layers, (long_input, global_input)))
    self.assertEqual(mock_get_recompute_context.call_count,
                     dropout_calls_per_layer * num_layers)
    for recompute_dx, control_dx in zip(recompute_gradients, control_gradients):
      # Gradients are tiny. Set a very small absolute tolerence. Increase
      # relative tolerence to deal with high number of layers.
      self.assertAllClose(recompute_dx, control_dx, rtol=1e-4, atol=1e-18)


if __name__ == '__main__':
  tf.test.main()
