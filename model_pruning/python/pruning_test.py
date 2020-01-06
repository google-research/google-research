# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Tests for the key functions in pruning library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf

from model_pruning.python import pruning


class PruningHParamsTest(tf.test.TestCase):
  PARAM_LIST = [
      "name=test", "threshold_decay=0.9", "pruning_frequency=10",
      "sparsity_function_end_step=100", "target_sparsity=0.9",
      "weight_sparsity_map=[conv1:0.8,conv2/kernel:0.8]",
      "block_dims_map=[dense1:4x4,dense2:1x4]"
  ]
  TEST_HPARAMS = ",".join(PARAM_LIST)

  def setUp(self):
    super(PruningHParamsTest, self).setUp()
    # Add global step variable to the graph
    self.global_step = tf.train.get_or_create_global_step()
    # Add sparsity
    self.sparsity = tf.Variable(0.5, name="sparsity")
    # Parse hparams
    self.pruning_hparams = pruning.get_pruning_hparams().parse(
        self.TEST_HPARAMS)

  def testInit(self):
    p = pruning.Pruning(self.pruning_hparams)
    self.assertEqual(p._spec.name, "test")
    self.assertAlmostEqual(p._spec.threshold_decay, 0.9)
    self.assertEqual(p._spec.pruning_frequency, 10)
    self.assertEqual(p._spec.sparsity_function_end_step, 100)
    self.assertAlmostEqual(p._spec.target_sparsity, 0.9)

  def testInitWithExternalSparsity(self):
    with self.cached_session():
      p = pruning.Pruning(spec=self.pruning_hparams, sparsity=self.sparsity)
      tf.global_variables_initializer().run()
      sparsity = p._sparsity.eval()
      self.assertAlmostEqual(sparsity, 0.5)

  def testInitWithVariableReuse(self):
    with self.cached_session():
      p = pruning.Pruning(spec=self.pruning_hparams, sparsity=self.sparsity)
      p_copy = pruning.Pruning(
          spec=self.pruning_hparams, sparsity=self.sparsity)
      tf.global_variables_initializer().run()
      sparsity = p._sparsity.eval()
      self.assertAlmostEqual(sparsity, 0.5)
      self.assertEqual(p._sparsity.eval(), p_copy._sparsity.eval())


class PruningTest(tf.test.TestCase):

  def setUp(self):
    super(PruningTest, self).setUp()
    self.global_step = tf.train.get_or_create_global_step()

  def testCreateMask2D(self):
    width = 10
    height = 20
    with self.cached_session():
      weights = tf.Variable(
          tf.random_normal([width, height], stddev=1), name="weights")
      masked_weights = pruning.apply_mask(weights, tf.get_variable_scope())
      tf.global_variables_initializer().run()
      weights_val = weights.eval()
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(weights_val, masked_weights_val)

  def testUpdateSingleMask(self):
    with self.cached_session() as session:
      weights = tf.Variable(tf.linspace(1.0, 100.0, 100), name="weights")
      masked_weights = pruning.apply_mask(weights)
      sparsity = tf.Variable(0.95, name="sparsity")
      p = pruning.Pruning(sparsity=sparsity)
      p._spec.threshold_decay = 0.0
      mask_update_op = p.mask_update_op()
      tf.global_variables_initializer().run()
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(np.count_nonzero(masked_weights_val), 100)
      session.run(mask_update_op)
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(np.count_nonzero(masked_weights_val), 5)

  def _blockMasking(self, hparams, weights, expected_mask):

    threshold = tf.Variable(0.0, name="threshold")
    sparsity = tf.Variable(0.5, name="sparsity")
    test_spec = ",".join(hparams)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)

    # Set up pruning
    p = pruning.Pruning(pruning_hparams, sparsity=sparsity)
    with self.cached_session():
      tf.global_variables_initializer().run()
      _, new_mask = p._maybe_update_block_mask(weights, threshold)
      # Check if the mask is the same size as the weights
      self.assertAllEqual(new_mask.get_shape(), weights.get_shape())
      mask_val = new_mask.eval()
      self.assertAllEqual(mask_val, expected_mask)

  def testBlockMaskingWithNonnegativeBlockDimensions(self):
    param_list = ["block_height=2", "block_width=2", "threshold_decay=0"]

    weights_avg = tf.constant([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                               [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]])
    weights_max = tf.constant([[0.1, 0.0, 0.2, 0.0], [0.0, -0.1, 0.0, -0.2],
                               [0.3, 0.0, 0.4, 0.0], [0.0, -0.3, 0.0, -0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(param_list + ["block_pooling_function=MAX"], weights_max,
                       expected_mask)
    self._blockMasking(param_list + ["block_pooling_function=AVG"], weights_avg,
                       expected_mask)

  def testBlockMaskingWithNegativeBlockDimensions(self):
    param_list = ["block_height=1", "block_width=-1", "threshold_decay=0"]

    weights_avg = tf.constant([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2],
                               [0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.4, 0.4]])
    weights_max = tf.constant([[0.1, 0.0, 0.1, 0.0], [0.0, 0.1, 0.0, 0.2],
                               [0.3, 0.0, 0.3, 0.0], [0.0, -0.3, 0.0, 0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    self._blockMasking(param_list + ["block_pooling_function=MAX"], weights_max,
                       expected_mask)
    self._blockMasking(param_list + ["block_pooling_function=AVG"], weights_avg,
                       expected_mask)

  def testBlockMaskingWithHigherDimensions(self):
    param_list = ["block_height=2", "block_width=2", "threshold_decay=0"]

    # Weights as in testBlockMasking, but with one extra dimension.
    weights_avg = tf.constant([[[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                                [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]]])
    weights_max = tf.constant([[[0.1, 0.0, 0.2, 0.0], [0.0, -0.1, 0.0, -0.2],
                                [0.3, 0.0, 0.4, 0.0], [0.0, -0.3, 0.0, -0.4]]])
    expected_mask = [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                      [1., 1., 1., 1.], [1., 1., 1., 1.]]]

    self._blockMasking(param_list + ["block_pooling_function=MAX"], weights_max,
                       expected_mask)
    self._blockMasking(param_list + ["block_pooling_function=AVG"], weights_avg,
                       expected_mask)

  def testPartitionedVariableMasking(self):
    partitioner = tf.variable_axis_size_partitioner(40)
    with self.cached_session() as session:
      with tf.variable_scope("", partitioner=partitioner):
        sparsity = tf.Variable(0.5, name="Sparsity")
        weights = tf.get_variable(
            "weights", initializer=tf.linspace(1.0, 100.0, 100))
        masked_weights = pruning.apply_mask(
            weights, scope=tf.get_variable_scope())
      p = pruning.Pruning(sparsity=sparsity)
      p._spec.threshold_decay = 0.0
      mask_update_op = p.mask_update_op()
      tf.global_variables_initializer().run()
      masked_weights_val = masked_weights.eval()
      session.run(mask_update_op)
      masked_weights_val = masked_weights.eval()
      self.assertAllEqual(np.count_nonzero(masked_weights_val), 50)

  def testConditionalMaskUpdate(self):
    param_list = [
        "pruning_frequency=2", "begin_pruning_step=1", "end_pruning_step=6",
        "nbins=100"
    ]
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)
    weights = tf.Variable(tf.linspace(1.0, 100.0, 100), name="weights")
    masked_weights = pruning.apply_mask(weights)
    sparsity = tf.Variable(0.00, name="sparsity")
    # Set up pruning
    p = pruning.Pruning(pruning_hparams, sparsity=sparsity)
    p._spec.threshold_decay = 0.0
    mask_update_op = p.conditional_mask_update_op()
    sparsity_val = tf.linspace(0.0, 0.9, 10)
    increment_global_step = tf.assign_add(self.global_step, 1)
    non_zero_count = []
    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      for i in range(10):
        session.run(tf.assign(sparsity, sparsity_val[i]))
        session.run(mask_update_op)
        session.run(increment_global_step)
        non_zero_count.append(np.count_nonzero(masked_weights.eval()))
    # Weights pruned at steps 0,2,4,and,6
    expected_non_zero_count = [100, 100, 80, 80, 60, 60, 40, 40, 40, 40]
    self.assertAllEqual(expected_non_zero_count, non_zero_count)

  def testWeightSpecificSparsity(self):
    param_list = [
        "begin_pruning_step=1", "pruning_frequency=1", "end_pruning_step=100",
        "target_sparsity=0.5",
        "weight_sparsity_map=[layer1:0.6,layer2/weights:0.75,.*kernel:0.6]",
        "threshold_decay=0.0"
    ]
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)

    with tf.variable_scope("layer1"):
      w1 = tf.Variable(tf.linspace(1.0, 100.0, 100), name="weights")
      _ = pruning.apply_mask(w1)
    with tf.variable_scope("layer2"):
      w2 = tf.Variable(tf.linspace(1.0, 100.0, 100), name="weights")
      _ = pruning.apply_mask(w2)
    with tf.variable_scope("layer3"):
      w3 = tf.Variable(tf.linspace(1.0, 100.0, 100), name="kernel")
      _ = pruning.apply_mask(w3)

    p = pruning.Pruning(pruning_hparams)
    mask_update_op = p.conditional_mask_update_op()
    increment_global_step = tf.assign_add(self.global_step, 1)

    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      for _ in range(110):
        session.run(mask_update_op)
        session.run(increment_global_step)

      self.assertAllClose(
          session.run(pruning.get_weight_sparsity()), [0.6, 0.75, 0.6])

  def testPerLayerBlockSparsity(self):
    param_list = [
        "block_dims_map=[layer1/weights:1x1,layer2/weights:1x2]",
        "block_pooling_function=AVG", "threshold_decay=0.0"
    ]

    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)

    with tf.variable_scope("layer1"):
      w1 = tf.Variable([[-0.1, 0.1], [-0.2, 0.2]], name="weights")
      pruning.apply_mask(w1)

    with tf.variable_scope("layer2"):
      w2 = tf.Variable([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4]],
                       name="weights")
      pruning.apply_mask(w2)

    sparsity = tf.Variable(0.5, name="sparsity")

    p = pruning.Pruning(pruning_hparams, sparsity=sparsity)
    mask_update_op = p.mask_update_op()
    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      session.run(mask_update_op)
      mask1_eval = session.run(pruning.get_masks()[0])
      mask2_eval = session.run(pruning.get_masks()[1])

      self.assertAllEqual(
          session.run(pruning.get_weight_sparsity()), [0.5, 0.5])

      self.assertAllEqual(mask1_eval, [[0.0, 0.0], [1., 1.]])
      self.assertAllEqual(mask2_eval, [[0, 0, 1., 1.], [0, 0, 1., 1.]])

  def testFirstOrderGradientCalculation(self):
    param_list = [
        "prune_option=first_order_gradient",
        "gradient_decay_rate=0.5",
    ]
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)
    tf.logging.info(pruning_hparams)

    w = tf.Variable(tf.linspace(1.0, 10.0, 10), name="weights")
    _ = pruning.apply_mask(w, prune_option="first_order_gradient")

    p = pruning.Pruning(pruning_hparams)
    old_weight_update_op = p.old_weight_update_op()
    gradient_update_op = p.gradient_update_op()

    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      session.run(gradient_update_op)
      session.run(old_weight_update_op)

      weights = pruning.get_weights()
      old_weights = pruning.get_old_weights()
      gradients = pruning.get_gradients()

      weight = weights[0]
      old_weight = old_weights[0]
      gradient = gradients[0]
      self.assertAllEqual(
          gradient.eval(),
          tf.math.scalar_mul(0.5,
                             tf.nn.l2_normalize(tf.linspace(1.0, 10.0,
                                                            10))).eval())
      self.assertAllEqual(weight.eval(), old_weight.eval())

  def testSecondOrderGradientCalculation(self):
    param_list = [
        "prune_option=second_order_gradient",
        "gradient_decay_rate=0.5",
    ]
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)
    tf.logging.info(pruning_hparams)

    w = tf.Variable(tf.linspace(1.0, 10.0, 10), name="weights")
    _ = pruning.apply_mask(w, prune_option="second_order_gradient")

    p = pruning.Pruning(pruning_hparams)
    old_weight_update_op = p.old_weight_update_op()
    old_old_weight_update_op = p.old_old_weight_update_op()
    gradient_update_op = p.gradient_update_op()

    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      session.run(old_weight_update_op)
      session.run(old_old_weight_update_op)
      session.run(tf.assign(w, tf.math.scalar_mul(2.0, w)))
      session.run(gradient_update_op)

      old_weights = pruning.get_old_weights()
      old_old_weights = pruning.get_old_old_weights()
      gradients = pruning.get_gradients()

      old_weight = old_weights[0]
      old_old_weight = old_old_weights[0]
      gradient = gradients[0]
      self.assertAllEqual(
          gradient.eval(),
          tf.math.scalar_mul(0.5,
                             tf.nn.l2_normalize(tf.linspace(1.0, 10.0,
                                                            10))).eval())
      self.assertAllEqual(old_weight.eval(), old_old_weight.eval())

  def testFirstOrderGradientBlockMasking(self):
    param_list = [
        "prune_option=first_order_gradient",
        "gradient_decay_rate=0.5",
        "block_height=2",
        "block_width=2",
        "threshold_decay=0",
        "block_pooling_function=AVG",
    ]
    threshold = tf.Variable(0.0, name="threshold")
    sparsity = tf.Variable(0.5, name="sparsity")
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)

    weights_avg = tf.constant([[0.1, 0.1, 0.2, 0.2], [0.1, 0.1, 0.2, 0.2],
                               [0.3, 0.3, 0.4, 0.4], [0.3, 0.3, 0.4, 0.4]])
    expected_mask = [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                     [1., 1., 1., 1.], [1., 1., 1., 1.]]

    w = tf.Variable(weights_avg, name="weights")
    _ = pruning.apply_mask(w, prune_option="first_order_gradient")

    p = pruning.Pruning(pruning_hparams, sparsity=sparsity)
    old_weight_update_op = p.old_weight_update_op()
    gradient_update_op = p.gradient_update_op()

    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      session.run(gradient_update_op)
      session.run(old_weight_update_op)

      weights = pruning.get_weights()
      _ = pruning.get_old_weights()
      gradients = pruning.get_gradients()

      weight = weights[0]
      gradient = gradients[0]

      _, new_mask = p._maybe_update_block_mask(weight, threshold, gradient)
      self.assertAllEqual(new_mask.get_shape(), weight.get_shape())
      mask_val = new_mask.eval()
      self.assertAllEqual(mask_val, expected_mask)

  def testWeightSparsityTiebreaker(self):
    param_list = [
        "begin_pruning_step=1", "pruning_frequency=1", "end_pruning_step=100",
        "target_sparsity=0.5",
        "threshold_decay=0.0"
    ]
    test_spec = ",".join(param_list)
    pruning_hparams = pruning.get_pruning_hparams().parse(test_spec)

    with tf.variable_scope("layer1"):
      w1 = tf.Variable(np.ones([100], dtype=np.float32),
                       name="weights")
      _ = pruning.apply_mask(w1)

    p = pruning.Pruning(pruning_hparams)
    mask_update_op = p.conditional_mask_update_op()
    increment_global_step = tf.assign_add(self.global_step, 1)

    with self.cached_session() as session:
      tf.global_variables_initializer().run()
      for _ in range(110):
        session.run(mask_update_op)
        session.run(increment_global_step)

      self.assertAllClose(
          session.run(pruning.get_weight_sparsity()), [0.5])


if __name__ == "__main__":
  tf.test.main()
