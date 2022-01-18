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

# Lint as: python3
"""Tests for forgetting_nuisance."""

import numpy as np
from numpy import linalg as la
import pandas as pd
import tensorflow.compat.v1 as tf

from correct_batch_effects_wdn import forgetting_nuisance


class ForgettingNuisanceTest(tf.test.TestCase):

  def R(self, theta):
    """Generate a 2x2 rotation matrix.

    Args:
      theta (float): angle

    Returns:
      R (numpy matrix): 2x2 matrix representing a rotation.
    """
    return np.matrix([[np.sin(theta), np.cos(theta)],
                      [np.cos(theta), -np.sin(theta)]])

  def C(self, eig1, eig2, theta):
    """Make covariance structure.

    Args:
      eig1 (float): First eigenvalue.
      eig2 (float): Second eigenvalue.
      theta (float): Angle to rotate by.

    Returns:
      C (numpy matrix): 2x2 matrix representing the covariance
        structure.
    """
    return self.R(theta) * np.matrix([[eig1**2, 0], [0, eig2**2]
                                     ]) * self.R(theta).T

  def setUp(self):
    super(ForgettingNuisanceTest, self).setUp()

    self.num = 10

    ## Make some data with controls versus drug.
    control = np.random.multivariate_normal([-10, 0], self.C(10, 2, np.pi / 6),
                                            self.num)
    drug = np.random.multivariate_normal([-20, 0], self.C(10, 2, np.pi / 6),
                                         self.num)
    inputs = np.vstack((control, drug))
    outputs = np.zeros((inputs.shape[0], 2))
    outputs[:self.num, 0] = 1
    outputs[self.num:, 1] = 1
    self.random_state = np.random.RandomState(seed=42)
    self.shuffler = forgetting_nuisance.DataShuffler(inputs, outputs,
                                                     self.random_state)

    ## Make data with control versus drug on two separate batches.
    controls_batch_1 = np.random.multivariate_normal([0, 0], self.C(1, 1, 0),
                                                     self.num)
    controls_batch_2 = np.random.multivariate_normal([10, 0], self.C(1, 1, 0),
                                                     self.num)
    drug_batch_1 = controls_batch_1 + [3, 2]
    drug_batch_2 = controls_batch_2 + [3, -2]
    labels = ([("control", "batch1")] * len(controls_batch_1) + [
        ("control", "batch2")
    ] * len(controls_batch_2) + [("drug", "batch1")] * len(drug_batch_1) +
              [("drug", "batch2")] * len(drug_batch_1))
    self.dummy_df = pd.DataFrame(
        np.vstack(
            [controls_batch_1, controls_batch_2, drug_batch_1, drug_batch_2]),
        index=pd.MultiIndex.from_tuples(labels, names=["compound", "batch"]))

  def tearDown(self):
    super(ForgettingNuisanceTest, self).tearDown()

    tf.reset_default_graph()

  def testDataShuffler(self):
    sample_inputs, sample_outputs = self.shuffler.next_batch(5)
    self.assertEqual(sample_inputs.shape, (5, 2))
    self.assertEqual(sample_outputs.shape, (5, 2))

  def testDataShuffler2(self):
    ## Ensure that the data shuffler is reproducible, i.e. that running it
    ## multiple times given a random_state produces the same results.
    random_state_1 = np.random.RandomState(30)
    random_state_2 = np.random.RandomState(30)
    inputs = np.arange(200)
    outputs = np.arange(500, 700)
    data_shuffler1 = forgetting_nuisance.DataShuffler(inputs, outputs,
                                                      random_state_1)
    data_shuffler2 = forgetting_nuisance.DataShuffler(inputs, outputs,
                                                      random_state_2)
    sample_inputs_1, sample_outputs_1 = data_shuffler1.next_batch(200)
    sample_inputs_2, sample_outputs_2 = data_shuffler2.next_batch(200)

    self.assertTrue(np.array_equal(sample_inputs_1, sample_inputs_2))
    self.assertTrue(np.array_equal(sample_outputs_1, sample_outputs_2))

  def testDataShuffler3(self):
    ## Ensure data shuffler works properly when batch size is larger than number
    ## of input/output pairs.
    inputs = np.arange(4)
    outputs = np.arange(10, 14)
    random_state = np.random.RandomState(30)
    data_shuffler = forgetting_nuisance.DataShuffler(inputs, outputs,
                                                     random_state)
    sample_inputs_1, sample_outputs_1 = data_shuffler.next_batch(200)
    sample_inputs_2, sample_outputs_2 = data_shuffler.next_batch(200)
    self.assertCountEqual(sample_inputs_1, sample_inputs_2)
    self.assertCountEqual(sample_outputs_1, sample_outputs_2)

  def testTensorflowSeed(self):
    """Make sure tf.set_random_seed results in reproducible outcomes."""
    tf.reset_default_graph()
    tf.set_random_seed(42)
    a = tf.random_uniform([1, 1],
                          minval=0,
                          maxval=1,
                          dtype=tf.float32)

    with tf.Session() as sess1:
      a_val_1 = sess1.run(a)

    with tf.Session() as sess2:
      a_val_2 = sess2.run(a)

    self.assertEqual(a_val_1, a_val_2)

  def testMakeHolder1(self):
    ## make a holder in which the dataframe is formatted by "multiplexed".
    ## In addition batch information is provided in the inputs.
    holder = forgetting_nuisance.DatasetHolder(
        self.dummy_df,
        input_category_level=["batch"],
        batch_input_info="multiplexed")
    ## add input/output shufflers for each batch, with compound label outputs.
    holder.add_shufflers(["batch"], ["compound"], None)
    ## Pick the shuffler from batch1 to compounds.
    test_shuffler = holder.data_shufflers[("batch1", "compound")]
    inputs, outputs = (forgetting_nuisance.get_dense_arr(test_shuffler.inputs),
                       forgetting_nuisance.get_dense_arr(test_shuffler.outputs))
    ## number of elements shoud be num_compounds * self.num.
    ## input dim should be (dim + 1) * num_batches for "multiplexed" format.
    ## output dim should be num_labels.
    self.assertEqual(inputs.shape, (2 * self.num, 6))
    self.assertEqual(outputs.shape, (2 * self.num, 2))

    ## Confirm the format is correct.
    zeros_chunck = np.zeros((2 * self.num, 3))
    ones_vector = np.ones((2 * self.num))
    self.assertTrue(
        np.array_equal(inputs[:, :3], zeros_chunck) or
        np.array_equal(inputs[:, 3:], zeros_chunck))
    self.assertTrue(
        np.array_equal(inputs[:, 2], ones_vector) or
        np.array_equal(inputs[:, 5], ones_vector))

    ## add input/output shufflers for each compound, with batch label outputs.
    holder.add_shufflers(["compound"], ["batch"], None)
    test_shuffler = holder.data_shufflers[("control", "batch")]
    inputs, outputs = (forgetting_nuisance.get_dense_arr(test_shuffler.inputs),
                       forgetting_nuisance.get_dense_arr(test_shuffler.outputs))
    self.assertEqual(inputs.shape, (2 * self.num, 6))
    self.assertEqual(outputs.shape, (2 * self.num, 2))

    ## Confirm the format is correct.
    inputs1, inputs2 = inputs[:self.num], inputs[self.num:]
    outputs1, outputs2 = outputs[:self.num], outputs[self.num:]
    zeros_chunck = np.zeros((self.num, 3))
    ones_vector = np.ones((self.num))
    self.assertTrue((np.array_equal(zeros_chunck, inputs1[:, :3]) and
                     (np.array_equal(zeros_chunck, inputs2[:, 3:]))) or
                    (np.array_equal(zeros_chunck, inputs1[:, 3:]) and
                     (np.array_equal(zeros_chunck, inputs2[:, :3]))))
    self.assertTrue((np.array_equal(ones_vector, inputs1[:, 2]) and
                     (np.array_equal(ones_vector, inputs2[:, 5]))) or
                    (np.array_equal(ones_vector, inputs1[:, 5]) and
                     (np.array_equal(ones_vector, inputs2[:, 2]))))
    self.assertTrue((np.array_equal(ones_vector, outputs1[:, 0]) and
                     (np.array_equal(ones_vector, outputs2[:, 1]))) or
                    (np.array_equal(ones_vector, outputs1[:, 1]) and
                     (np.array_equal(ones_vector, outputs2[:, 0]))))

    ## add input/output shufflers from groups specified by compound and batch
    ## to output labels specified by batch.
    ## Allow only shufflers that have control compounds only.
    holder.add_shufflers(["compound", "batch"], ["batch"],
                         [("control", "batch1"), ("control", "batch2")])
    test_shuffler = holder.data_shufflers[("control", "batch1"), "batch"]
    inputs, outputs = (forgetting_nuisance.get_dense_arr(test_shuffler.inputs),
                       forgetting_nuisance.get_dense_arr(test_shuffler.outputs))
    ## There should be 10 points in each set.
    self.assertEqual(inputs.shape, (self.num, 6))
    ## There is only one possible output label for this set since both batch and
    ## compound have been fixed.
    self.assertEqual(outputs.shape, (self.num, 1))

  def testMakeHolder2(self):
    ## make a holder in which the dataframe is formatted by "one_hot".
    ## In addition batch information is provided in the inputs.
    holder = forgetting_nuisance.DatasetHolder(
        self.dummy_df,
        input_category_level=["batch"],
        batch_input_info="one_hot")
    ## add input/output shufflers for each batch, with compound label outputs.
    holder.add_shufflers(["batch"], ["compound"], None)
    ## Pick the shuffler from batch1 to compounds.
    test_shuffler = holder.data_shufflers[("batch1", "compound")]
    inputs, outputs = (forgetting_nuisance.get_dense_arr(test_shuffler.inputs),
                       forgetting_nuisance.get_dense_arr(test_shuffler.outputs))
    ## number of elements shoud be num_compounds * self.num.
    ## input dim should be (dim + 1) * num_batches for "multiplexed" format.
    ## output dim should be num_labels.
    self.assertEqual(inputs.shape, (2 * self.num, 4))
    self.assertEqual(outputs.shape, (2 * self.num, 2))

    ## Confirm the format is correct.
    zeros_chunck = np.zeros((self.num))
    ones_chunck = np.ones((self.num))
    np.testing.assert_array_equal(inputs[:self.num, 2], ones_chunck)
    np.testing.assert_array_equal(inputs[:self.num, 3], zeros_chunck)
    np.testing.assert_array_equal(outputs[:self.num, 0], ones_chunck)
    np.testing.assert_array_equal(outputs[:self.num, 1], zeros_chunck)

    np.testing.assert_array_equal(inputs[self.num:, 2], ones_chunck)
    np.testing.assert_array_equal(inputs[self.num:, 3], zeros_chunck)
    np.testing.assert_array_equal(outputs[self.num:, 0], zeros_chunck)
    np.testing.assert_array_equal(outputs[self.num:, 1], ones_chunck)

    ## add input/output shufflers for each compound, with batch label outputs.
    holder.add_shufflers(["compound"], ["batch"], None)
    test_shuffler = holder.data_shufflers[("control", "batch")]
    inputs, outputs = (forgetting_nuisance.get_dense_arr(test_shuffler.inputs),
                       forgetting_nuisance.get_dense_arr(test_shuffler.outputs))
    self.assertEqual(inputs.shape, (2 * self.num, 4))
    self.assertEqual(outputs.shape, (2 * self.num, 2))

    ## Confirm the format is correct.
    zeros_chunck = np.zeros((self.num))
    ones_chunck = np.ones((self.num))
    np.testing.assert_array_equal(inputs[:self.num, 2], ones_chunck)
    np.testing.assert_array_equal(inputs[:self.num, 3], zeros_chunck)
    np.testing.assert_array_equal(outputs[:self.num, 0], ones_chunck)
    np.testing.assert_array_equal(outputs[:self.num, 1], zeros_chunck)

    np.testing.assert_array_equal(inputs[self.num:, 2], zeros_chunck)
    np.testing.assert_array_equal(inputs[self.num:, 3], ones_chunck)
    np.testing.assert_array_equal(outputs[self.num:, 0], zeros_chunck)
    np.testing.assert_array_equal(outputs[self.num:, 1], ones_chunck)

    ## add input/output shufflers from groups specified by compound and batch
    ## to output labels specified by batch.
    ## Allow only shufflers that have control compounds only.
    holder.add_shufflers(["compound", "batch"], ["batch"],
                         [("control", "batch1"), ("control", "batch2")])
    test_shuffler = holder.data_shufflers[("control", "batch1"), "batch"]
    inputs, outputs = (forgetting_nuisance.get_dense_arr(test_shuffler.inputs),
                       forgetting_nuisance.get_dense_arr(test_shuffler.outputs))
    ## There should be 10 points in each set.
    self.assertEqual(inputs.shape, (self.num, 4))
    ## There is only one possible output label for this set since both batch and
    ## compound have been fixed.
    self.assertEqual(outputs.shape, (self.num, 1))

  def testMakeTensorDict(self):
    ## Make sure make_tensor_dict works properly.
    sample_dict = {2: tf.constant(2), 3: tf.constant(3)}
    tensor_dict = forgetting_nuisance.make_tensor_dict(sample_dict)
    values = tf.map_fn(tensor_dict, tf.constant([2, 3]))
    sess = tf.InteractiveSession()
    values_from_sess = sess.run(values)
    self.assertNotEqual(values_from_sess[0], values_from_sess[1])

  def testDiscriminatorModel(self):
    input_dim = 8
    batch_n = 10
    layer_width = 2
    num_layers = 2
    discrminator_model = forgetting_nuisance.make_discriminator_model(
        input_dim, tf.nn.softplus, layer_width, num_layers)
    inputs = tf.Variable(np.ones((batch_n, input_dim)), dtype=tf.float32)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      outputs = sess.run(discrminator_model(inputs))
    self.assertEqual(outputs.shape, (batch_n, 1))

  def testWasserstineDistance(self):
    input_dim = 8
    batch_n = 10
    layer_width = 2
    num_layers = 2
    x_ = tf.Variable(np.ones((batch_n, input_dim)), dtype=tf.float32)
    y_ = tf.Variable(np.zeros((batch_n, input_dim)), dtype=tf.float32)
    disc_loss, gradient_penalty, _ = forgetting_nuisance.wasserstein_distance(
        x_, y_, layer_width, num_layers, batch_n)
    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      disc_loss_val, gradient_penalty_val = sess.run(
          [disc_loss, gradient_penalty])
    self.assertEqual(disc_loss_val.shape, ())
    self.assertEqual(gradient_penalty_val.shape, ())

  def testWassersteinNetwork(self):
    batch_n = 5
    holder = forgetting_nuisance.DatasetHolder(self.dummy_df)
    holder.add_shufflers(["compound", "batch"])
    input_dim = holder.input_dim
    feature_dim = input_dim
    network = forgetting_nuisance.WassersteinNetwork(holder, feature_dim,
                                                     batch_n, 0, 1)

    self.assertEqual(set(network._unique_targets), set([("drug",),
                                                        ("control",)]))
    self.assertEqual(set(network._unique_nuisances), set([("batch1",),
                                                          ("batch2",)]))
    keys_for_targets = {
        ("drug",): [(("drug", "batch1"), None), (("drug", "batch2"), None)],
        ("control",): [(("control", "batch1"), None),
                       (("control", "batch2"), None)]
    }
    self.assertSameElements(
        list(network._keys_for_targets.keys()), list(keys_for_targets.keys()))
    for key in keys_for_targets:
      self.assertSameElements(keys_for_targets[key],
                              network._keys_for_targets[key])

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      feed_dict = {}
      for key, shuffler in holder.data_shufflers.items():
        input_mini, _ = shuffler.next_batch(batch_n)
        feed_dict[network._x_vals[key]] = input_mini
      f_val, x_vals = sess.run(
          [network._features, network._x_vals], feed_dict=feed_dict)

      ## make sure each input used in every batch came from the actual inputs
      for key, vals in x_vals.items():
        for row in vals:  # iterate over every element
          ## identify distance from closest element in inputs
          differences = [la.norm(np.array(row) - np.array(candidates))
                         for candidates in holder.data_shufflers[key].inputs]
        self.assertAlmostEqual(min(differences), 0.0, places=5)
    self.assertEqual(list(f_val.values())[0].shape, (batch_n, input_dim))

    self.assertSameElements(
        list(network.wass_loss_target.keys()), [("drug",), ("control",)])


if __name__ == "__main__":
  tf.test.main()
