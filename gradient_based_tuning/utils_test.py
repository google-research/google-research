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
"""Tests for utils.py."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from gradient_based_tuning import utils


class ToHostTest(absltest.TestCase):

  def test_correct_shape(self):
    # shape = (4, 3, 2)
    input_array = np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]],
                            [[7, 7], [8, 8], [9, 9]], [[0, 0], [1, 1], [2, 2]]])
    output = utils.tohost(input_array)
    self.assertEqual(output.shape, (12, 2))

  def test_correct_values(self):
    # shape = (4, 3, 2)
    input_array = np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]],
                            [[7, 7], [8, 8], [9, 9]], [[0, 0], [1, 1], [2, 2]]])
    output = utils.tohost(input_array)
    self.assertEqual(list(input_array.flatten()), list(output.flatten()))

  def test_error_on_too_many_dims(self):
    input_array = np.array([1, 2, 3, 4, 5])
    with self.assertRaises(ValueError):
      _ = utils.tohost(input_array)

  def test_flattens_batch_dim(self):
    array = np.ones([2, 3, 4, 5])
    output = utils.tohost(array)
    self.assertEqual(output.shape, np.ones([6, 4, 5]).shape)

  def test_batch_dim_1(self):
    array = np.ones([1, 1, 4, 5])
    output = utils.tohost(array)
    self.assertEqual(output.shape, np.ones([1, 4, 5]).shape)

  def test_error_if_not_enough_dims(self):
    array = np.ones([3])
    with self.assertRaisesRegex(ValueError, '(?i)not enough values.*got 1'):
      _ = utils.tohost(array)


class ComputeWeightedAccuracyTest(absltest.TestCase):

  def test_accuracy(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 1],
        [1, 1],
    ])
    weights = np.array([
        [1., 1.],
        [1., 1.],
    ])
    acc, factor = utils.compute_weighted_accuracy(logits, targets, weights)
    self.assertEqual(acc, 4)
    self.assertEqual(factor, 4)

  def test_accuracy_no_weights(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 1],
        [1, 1],
    ])
    acc, factor = utils.compute_weighted_accuracy(logits, targets)
    self.assertEqual(acc, 16)
    self.assertEqual(factor, 4)

  def test_accuracy_loss_some_weights_zero(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 1],
        [1, 1],
    ])
    weights = np.array([
        [1, 0],
        [1, 0],
    ])
    acc, factor = utils.compute_weighted_accuracy(logits, targets, weights)
    self.assertEqual(acc, 2)
    self.assertEqual(factor, 2)

  def test_mismatched_shapes_raises_error(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([1])
    weights = np.array([
        [1, 0],
        [1, 0],
    ])
    with self.assertRaisesRegex(ValueError, '(?i)Incorrect shapes.*targets'):
      _, _ = utils.compute_weighted_accuracy(logits, targets, weights)


class ComputeMetricsTest(absltest.TestCase):

  def test_metrics(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    labels = np.array([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1],
    ])
    weights = np.array([
        [1., 1.],
        [1., 1.],
        [1., 1.],
        [1., 1.],
    ])
    output = utils.compute_metrics(logits, labels, weights)
    output = jax.tree.map(float, output)
    loss_out = output.pop('loss')
    self.assertAlmostEqual(loss_out, 2.737448, delta=1e-5)
    self.assertEqual(output, {'accuracy': 8, 'denominator': 8})


class ComputeWeightedCrossEntropyTest(absltest.TestCase):

  def test_ce_loss(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 1],
        [1, 1],
    ])
    weights = np.array([
        [1., 1.],
        [1., 1.],
    ])
    loss, factor = utils.compute_weighted_cross_entropy(logits, targets,
                                                        weights)
    self.assertEqual(loss, 1.3687246)
    self.assertEqual(factor, 4)

  def test_ce_loss_no_weights(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 1],
        [1, 1],
    ])
    loss, factor = utils.compute_weighted_cross_entropy(logits, targets)
    self.assertEqual(loss, 5.4748983)
    self.assertEqual(factor, 4)

  def test_ce_loss_some_weights_zero(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 1],
        [1, 1],
    ])
    weights = np.array([
        [1, 0],
        [1, 0],
    ])
    loss, factor = utils.compute_weighted_cross_entropy(logits, targets,
                                                        weights)
    self.assertEqual(loss, 0.62652326)
    self.assertEqual(factor, 2)

  def test_ce_loss_no_weights_targets_zero(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 0],
        [1, 0],
    ])
    loss, factor = utils.compute_weighted_cross_entropy(logits, targets)
    self.assertEqual(loss, 5.937449)
    self.assertEqual(factor, 2)

  def test_ce_loss_targets_zero(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([
        [1, 0],
        [1, 0],
    ])
    weights = np.array([
        [1, 0],
        [1, 0],
    ])
    loss, factor = utils.compute_weighted_cross_entropy(logits, targets,
                                                        weights)
    self.assertEqual(loss, 0.62652326)
    self.assertEqual(factor, 2)

  def test_ce_mismatched_shapes_raises_error(self):
    logits = np.array([
        [[-0.1, 0.9], [0.1, 0.9]],
        [[-0.1, 0.9], [0.1, 0.9]],
    ])
    targets = np.array([1])
    weights = np.array([
        [1, 0],
        [1, 0],
    ])
    with self.assertRaisesRegex(ValueError, '(?i)Incorrect shapes.*targets'):
      _, _ = utils.compute_weighted_cross_entropy(logits, targets, weights)


class CreateLearningRateSchedulerTest(absltest.TestCase):

  def test_default_vals(self):
    lr_fn = utils.create_learning_rate_scheduler()
    self.assertEqual(float(lr_fn(0)), 0)
    self.assertAlmostEqual(float(lr_fn(100)), 0.00158114, delta=1e-5)
    self.assertAlmostEqual(float(lr_fn(1000)), 0.01581139, delta=1e-5)
    self.assertAlmostEqual(float(lr_fn(10000)), 0.005, delta=1e-5)
    self.assertAlmostEqual(float(lr_fn(100000)), 0.00158114, delta=1e-5)

  def test_constant(self):
    lr_fn = utils.create_learning_rate_scheduler(
        factors='constant', base_learning_rate=3e-5)
    self.assertAlmostEqual(float(lr_fn(0)), 3e-5, delta=1e-5)
    self.assertAlmostEqual(float(lr_fn(100)), 3e-5, delta=1e-5)
    self.assertAlmostEqual(float(lr_fn(10000)), 3e-5, delta=1e-5)


class ApplyRegularizationLossTest(absltest.TestCase):

  def test_apply_l1norm(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss = utils.apply_regularization_loss('l1_norm', reg_vals)
    self.assertEqual(reg_loss, 15)

  def test_apply_l2norm(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss = utils.apply_regularization_loss('l2_norm', reg_vals)
    self.assertEqual(reg_loss, np.sqrt(55))

  def test_apply_lpnorm_1p5(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss = utils.apply_regularization_loss('lp_norm', reg_vals, 1.5)
    self.assertAlmostEqual(reg_loss, 9.2658, delta=1e-5)

  def test_apply_lpnorm_3(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss = utils.apply_regularization_loss('lp_norm', reg_vals, 3)
    self.assertAlmostEqual(reg_loss, 6.0822, delta=1e-5)

  def test_apply_lpnorm_5(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss = utils.apply_regularization_loss('lp_norm', reg_vals, 5)
    self.assertAlmostEqual(reg_loss, 5.36022, delta=1e-5)

  def test_lpnorm_1_equal_to_l1norm(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss_l1 = utils.apply_regularization_loss('l1_norm', reg_vals)
    reg_loss_lp = utils.apply_regularization_loss('lp_norm', reg_vals, 1)
    self.assertEqual(reg_loss_l1, reg_loss_lp)

  def test_lpnorm_2_equal_to_l2norm(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    reg_loss_l2 = utils.apply_regularization_loss('l2_norm', reg_vals)
    reg_loss_lp = utils.apply_regularization_loss('lp_norm', reg_vals, 2)
    self.assertEqual(reg_loss_l2, reg_loss_lp)

  def test_bad_norm_raise_error(self):
    reg_vals = np.array([1, 2, 3, 4, 5])
    with self.assertRaisesRegex(ValueError, '(?i)Unrecognized.*bad_norm_name'):
      _ = utils.apply_regularization_loss('bad_norm_name', reg_vals)


class GetTotalRegularizationLossTest(absltest.TestCase):

  def test_errant_tag_l1norm(self):
    guided_vars_dict = {
        'errant_tag': {
            'regularization_type': 'l1_norm',
            'regularization_norm': 1,
            'regularization_alpha': 1,
        },
    }
    # Just pass values, same as guided_parameters_utils.activation_self_fn.
    act_fn_dict = {'dp-errant_tag': lambda x: x}
    raw_errant_vars = np.array([0, 1, 4])
    reg_loss = utils.get_total_regularization_loss(
        guided_vars_dict, act_fn_dict, raw_errant_vars=raw_errant_vars)
    self.assertEqual(reg_loss, 5)

  def test_ex_index_l1norm(self):
    guided_vars_dict = {
        'ex_index': {
            'regularization_type': 'l1_norm',
            'regularization_norm': 1,
            'regularization_alpha': 1,
        },
    }
    # Just pass values, same as guided_parameters_utils.activation_self_fn.
    act_fn_dict = {'dp-ex_index': lambda x: x}
    raw_ex_vars = np.array([0, 1, 4])
    ex_index = np.array([1, 1, 1, 0, 2, 2, 2])
    reg_loss = utils.get_total_regularization_loss(guided_vars_dict,
                                                   act_fn_dict, raw_ex_vars,
                                                   ex_index)
    self.assertEqual(reg_loss, 15)

  def test_ex_index_l2norm(self):
    guided_vars_dict = {
        'ex_index': {
            'regularization_type': 'l2_norm',
            'regularization_norm': 1,
            'regularization_alpha': 1,
        },
    }
    # Just pass values, same as guided_parameters_utils.activation_self_fn.
    act_fn_dict = {'dp-ex_index': lambda x: x}
    raw_ex_vars = np.array([0, 1, 4])
    ex_index = np.array([1, 1, 1, 0, 2, 2, 2])
    reg_loss = utils.get_total_regularization_loss(guided_vars_dict,
                                                   act_fn_dict, raw_ex_vars,
                                                   ex_index)
    self.assertAlmostEqual(float(reg_loss), 7.14143, delta=1e-5)

  def test_ex_index_lpnorm(self):
    guided_vars_dict = {
        'ex_index': {
            'regularization_type': 'lp_norm',
            'regularization_norm': 1.5,
            'regularization_alpha': 1,
        },
    }
    # Just pass values, same as guided_parameters_utils.activation_self_fn.
    act_fn_dict = {'dp-ex_index': lambda x: x}
    raw_ex_vars = np.array([0, 1, 4])
    ex_index = np.array([1, 1, 1, 0, 2, 2, 2])
    reg_loss = utils.get_total_regularization_loss(guided_vars_dict,
                                                   act_fn_dict, raw_ex_vars,
                                                   ex_index)
    self.assertAlmostEqual(float(reg_loss), 9, delta=1e-5)

  def test_ex_index_l1norm_alpha_zero(self):
    guided_vars_dict = {
        'ex_index': {
            'regularization_type': 'l1_norm',
            'regularization_norm': 1,
            'regularization_alpha': 0,
        },
    }
    act_fn_dict = {'dp-ex_index': lambda x: x}
    raw_ex_vars = np.array([0, 1, 4])
    ex_index = np.array([1, 1, 1, 0, 2, 2, 2])
    reg_loss = utils.get_total_regularization_loss(guided_vars_dict,
                                                   act_fn_dict, raw_ex_vars,
                                                   ex_index)
    self.assertEqual(reg_loss, 0)

  def test_ex_index_l1norm_alpha_half(self):
    guided_vars_dict = {
        'ex_index': {
            'regularization_type': 'l1_norm',
            'regularization_norm': 1,
            'regularization_alpha': 0.5,
        },
    }
    act_fn_dict = {'dp-ex_index': lambda x: x}
    raw_ex_vars = np.array([0, 1, 4])
    ex_index = np.array([1, 1, 1, 0, 2, 2, 2])
    reg_loss = utils.get_total_regularization_loss(guided_vars_dict,
                                                   act_fn_dict, raw_ex_vars,
                                                   ex_index)
    self.assertEqual(reg_loss, 7.5)

  def test_both_regularized(self):
    guided_vars_dict = {
        'ex_index': {
            'regularization_type': 'l1_norm',
            'regularization_norm': 1,
            'regularization_alpha': 1,
        },
        'errant_tag': {
            'regularization_type': 'l1_norm',
            'regularization_norm': 1,
            'regularization_alpha': 1,
        },
    }
    act_fn_dict = {'dp-ex_index': lambda x: x, 'dp-errant_tag': lambda x: x}
    raw_ex_vars = np.array([0, 1, 4])
    ex_index = np.array([1, 1, 1, 0, 2, 2, 2])
    raw_errant_vars = np.array([0, 1, 4])
    reg_loss = utils.get_total_regularization_loss(guided_vars_dict,
                                                   act_fn_dict, raw_ex_vars,
                                                   ex_index, raw_errant_vars)
    self.assertEqual(reg_loss, 20)


class EntmaxLossTest(absltest.TestCase):

  def test_zero_loss(self):
    # this test checks the separation margin property
    # more details are in "Tsallis entmax losses" paragraph in
    # https://aclanthology.org/P19-1146.pdf for details
    alpha = 1.5
    margin = 1 / (alpha - 1)
    correct_logit = jnp.asarray(
        np.random.randint(8, 10, size=(5,)), dtype=float)
    incorrect_logit = correct_logit - margin
    logits = jnp.concatenate([correct_logit[:, None], incorrect_logit[:, None]],
                             axis=1)[None, :, :]
    targets = jnp.zeros((1, 5), dtype=int)[None, :, :]
    weights = jnp.ones((1, 5), dtype=int)[None, :, :]
    entmax_loss = utils.compute_entmax_loss(logits, targets, weights,
                                            alpha)[0].item()
    self.assertEqual(entmax_loss, 0.)

  def test_loss(self):
    alpha = 1.5
    incorrect_logit = jnp.asarray([9., 9., 8., 8., 8.])
    correct_logit = jnp.asarray([2., 2., 4., 2., 2.])
    logits = jnp.concatenate([incorrect_logit[:, None], correct_logit[:, None]],
                             axis=1)[None, :, :]
    targets = jnp.ones((1, 5), dtype=int)[None, :, :]
    weights = jnp.ones((1, 5), dtype=int)[None, :, :]
    entmax_loss = utils.compute_entmax_loss(logits, targets, weights,
                                            alpha)[0].item()
    self.assertEqual(entmax_loss, 30.)


if __name__ == '__main__':
  absltest.main()
