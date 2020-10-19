# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
"""Tests for supcon.losses."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v2 as tf

from supcon import enums
from supcon import losses


class LossesTest(tf.test.TestCase, parameterized.TestCase):

  def testKerasImplementation(self):
    features = np.random.uniform(0., 1., (12, 2, 20))
    labels = np.eye(12, 15, dtype=np.int32)
    loss = losses.ContrastiveLoss()(labels, features)
    self.assertEqual(loss.shape, ())
    self.assertFalse(np.isnan(loss.numpy()))

  def testKerasLossVsNonKerasLoss(self):
    features = np.random.uniform(0., 1., size=(12, 2, 20))
    labels = np.eye(12, 15, dtype=np.int32)
    loss_keras = losses.ContrastiveLoss()(labels, features)
    loss_direct = tf.reduce_mean(
        losses.contrastive_loss(features, labels=labels))
    self.assertFalse(np.isnan(loss_direct.numpy()))
    self.assertFalse(np.isnan(loss_keras.numpy()))
    self.assertEqual(loss_direct.numpy(), loss_keras.numpy())

  def testIncorrectFeaturesRank(self):
    features = np.zeros([1, 1])
    with self.assertRaisesRegex(ValueError, 'Invalid features rank'):
      losses.contrastive_loss(features)

  def testUnknownBatchSizeDimension(self):
    features = tf.keras.layers.Input(
        dtype=tf.float32, batch_size=None, shape=(2, 20))
    with self.assertRaisesRegex(ValueError, 'features has unknown batch_size'):
      losses.contrastive_loss(features)

  def testUnknownNumViewsDimension(self):
    features = tf.keras.layers.Input(
        dtype=tf.float32, batch_size=1, shape=(None, 20))
    with self.assertRaisesRegex(ValueError, 'features has unknown num_views'):
      losses.contrastive_loss(features)

  def testIncorrectLabelsShape(self):
    features = np.random.uniform(0., 1., size=(10, 3, 20))
    labels = np.random.randint(1, size=(5))
    with self.assertRaisesRegex(ValueError, 'Invalid labels shape'):
      losses.contrastive_loss(features, labels=labels)

  def testIncorrectLabelsRank(self):
    features = np.random.uniform(0., 1., size=(10, 3, 20))
    labels = np.random.randint(5, size=(4, 4))
    with self.assertRaisesRegex(ValueError, 'Invalid labels shape'):
      losses.contrastive_loss(features, labels=labels)

  def testUnknownContrastMode(self):
    features = np.random.uniform(size=(10, 3, 20))
    labels = np.eye(10, dtype=np.int32)
    with self.assertRaisesRegex(ValueError, 'Invalid contrast_mode'):
      losses.contrastive_loss(features, labels, contrast_mode='invalid')

  def testUnknownSummationLocation(self):
    features = np.random.uniform(size=(10, 3, 20))
    labels = np.eye(10, dtype=np.int32)
    with self.assertRaisesRegex(ValueError, 'Invalid summation_location'):
      losses.contrastive_loss(features, labels, summation_location='invalid')

  def testUnknownDenominatorMode(self):
    features = np.random.uniform(size=(10, 3, 20))
    labels = np.eye(10, dtype=np.int32)
    with self.assertRaisesRegex(ValueError, 'Invalid denominator_mode'):
      losses.contrastive_loss(features, labels, denominator_mode='invalid')

  def testDefaultBehaviourSameAsAllLabelsDifferent(self):
    features = np.random.uniform(size=(10, 3, 20))
    labels = np.eye(10, dtype=np.int64)
    loss = tf.reduce_mean(losses.contrastive_loss(features))
    loss_without_labels = tf.reduce_mean(
        losses.contrastive_loss(features, labels))
    self.assertFalse(np.isnan(loss.numpy()))
    self.assertFalse(np.isnan(loss_without_labels.numpy()))
    self.assertEqual(loss.numpy(), loss_without_labels.numpy())

  def testContrastModeOneVsAll(self):
    # shape (2, 2, 3)
    features = np.array([[[0, 0, 1], [0, 1, 0]], [[1., 0., 0.], [0., -1., 0.]]])
    loss_one = tf.reduce_mean(
        losses.contrastive_loss(
            features,
            contrast_mode=enums.LossContrastMode.ONE_VIEW,
            base_temperature=1.))
    self.assertFalse(np.isnan(loss_one.numpy()))
    expected_loss = 1.098612  # np.log(3.)
    self.assertAlmostEqual(np.mean(loss_one.numpy()), expected_loss, places=6)
    loss_all = tf.reduce_mean(
        losses.contrastive_loss(
            features,
            contrast_mode=enums.LossContrastMode.ALL_VIEWS,
            base_temperature=1.))
    self.assertFalse(np.isnan(loss_all.numpy()))
    self.assertNotAlmostEqual(
        np.mean(loss_all.numpy()), expected_loss, places=6)

  def testLossValue(self):
    sqrt2 = np.sqrt(2.)
    sqrt6 = np.sqrt(6.)
    features = np.array([[[0, 0, 1], [0, (2. * sqrt2) / 3., -1 / 3.]],
                         [[sqrt6 / 3., -sqrt2 / 3., -1. / 3],
                          [-sqrt6 / 3., -sqrt2 / 3., -1. / 3]]])
    loss = losses.contrastive_loss(features, base_temperature=1.)
    self.assertFalse(np.isnan(loss.numpy()).any())
    expected_loss = 1.098612  # np.log(3.)
    self.assertAlmostEqual(np.mean(loss.numpy()), expected_loss, places=6)

  def testLossValueWithLabels(self):
    sqrt2 = np.sqrt(2.)
    sqrt6 = np.sqrt(6.)
    features = np.array([[[0, 0, 1], [0, (2. * sqrt2) / 3., -1 / 3.]],
                         [[sqrt6 / 3., -sqrt2 / 3., -1. / 3],
                          [-sqrt6 / 3., -sqrt2 / 3., -1. / 3]]])
    labels = np.eye(2, dtype=np.int32)
    loss = losses.contrastive_loss(features, labels=labels, base_temperature=1.)
    self.assertFalse(np.isnan(loss.numpy()).any())
    expected_loss = 1.098612  # np.log(3.)
    self.assertAlmostEqual(np.mean(loss.numpy()), expected_loss, places=6)

  def testLossValueWithLabelsAndPositives(self):
    features = np.array([[[0, 0, 1], [0, 0, 1]], [[0, 1, 0], [0, 1, 0]],
                         [[1, 0, 0], [1, 0, 0]]])
    labels = np.eye(3, dtype=np.int32)
    # Make the label of sample 1 and 2 the same (= label 0)
    labels[1] = labels[0]
    loss = losses.contrastive_loss(
        features, labels, base_temperature=1.).numpy()
    self.assertFalse(np.isnan(loss).any())
    expected_loss = [
        1.57149910,  # (3. * np.log(np.e + 4) - 1) / 3.
        1.57149910,  # (3. * np.log(np.e + 4) - 1) / 3.
        0.90483244,  # np.log(np.e + 4) - 1
    ]
    self.assertAlmostEqual(loss[0], expected_loss[0], places=6)
    self.assertAlmostEqual(loss[1], expected_loss[1], places=6)
    self.assertAlmostEqual(loss[2], expected_loss[2], places=6)

  @parameterized.named_parameters(('1x1 features', (10, 3, 1, 1, 64)),
                                  ('3x3 features', (10, 3, 3, 3, 8)),
                                  ('16x16 features', (10, 3, 16, 16, 4)),
                                  ('rank-3 features', (10, 3, 16, 8)))
  def testConvFeatures(self, features_shape):
    features_shape = tf.TensorShape(features_shape)
    features = tf.random.uniform(shape=features_shape)
    # Normalize embeddings to ensure the Loss does not return NaN values
    #   for large feature sizes.
    normalization_axes = list(range(2, features_shape.rank))
    normalized_features = tf.nn.l2_normalize(features, axis=normalization_axes)
    loss = tf.reduce_mean(losses.contrastive_loss(normalized_features))
    self.assertFalse(np.isnan(loss.numpy()))

  @parameterized.named_parameters(
      # The following values have all been manually checked to be the correct
      # outputs given the inputs in the test.
      ('out_and_all', enums.LossSummationLocation.OUTSIDE,
       enums.LossDenominatorMode.ALL, -1, [23.936932, 25.676819, 22.638714]),
      ('out_and_one', enums.LossSummationLocation.OUTSIDE,
       enums.LossDenominatorMode.ONE_POSITIVE, -1,
       [16.832325, 19.193565, 22.638714]),
      ('out_and_none', enums.LossSummationLocation.OUTSIDE,
       enums.LossDenominatorMode.ONLY_NEGATIVES, -1,
       [11.378975, 14.507131, 19.327797]),
      ('out_and_large_cap', enums.LossSummationLocation.OUTSIDE,
       enums.LossDenominatorMode.ALL, 4, [23.936932, 25.676819, 22.638714]),
      ('out_and_small_cap', enums.LossSummationLocation.OUTSIDE,
       enums.LossDenominatorMode.ALL, 2, [16.26471, 16.89146, 16.51060],
       (0, 0, 0)),
      ('out_and_zero_cap', enums.LossSummationLocation.OUTSIDE,
       enums.LossDenominatorMode.ALL, 0, [19.67042, 15.408167, 22.638714]),
      ('in_and_all', enums.LossSummationLocation.INSIDE,
       enums.LossDenominatorMode.ALL, -1, [23.366682, 24.479816, 22.638714]),
      ('in_and_one', enums.LossSummationLocation.INSIDE,
       enums.LossDenominatorMode.ONE_POSITIVE, -1,
       [16.530962, 18.487953, 22.638714]),
      ('in_and_none', enums.LossSummationLocation.INSIDE,
       enums.LossDenominatorMode.ONLY_NEGATIVES, -1,
       [10.808728, 13.310129, 19.327797]),
      ('in_and_large_cap', enums.LossSummationLocation.INSIDE,
       enums.LossDenominatorMode.ALL, 4, [23.366682, 24.479816, 22.638714]),
      ('in_and_small_cap', enums.LossSummationLocation.INSIDE,
       enums.LossDenominatorMode.ALL, 2, [15.69446, 15.69446, 15.69446],
       (0, 0, 0)),
      ('in_and_zero_cap', enums.LossSummationLocation.INSIDE,
       enums.LossDenominatorMode.ALL, 0, [19.67042, 15.408167, 22.638714]))
  def testLossForSummationLocationsAndDenominatorModes(self,
                                                       summation_location,
                                                       denominator_mode,
                                                       positives_cap,
                                                       expected_loss,
                                                       labels=(0, 0, 1)):
    features = np.array([
        [[0.01, 0.02, 0.14], [0.38, 0.61, 0.50]],
        [[0.86, 0.97, 0.33], [0.26, 0.68, 0.45]],
        [[0.32, 0.64, 0.28], [0.45, 0.74, 0.73]],
    ])
    labels = tf.one_hot(labels, 2)
    loss = losses.contrastive_loss(
        features,
        labels=labels,
        summation_location=summation_location,
        denominator_mode=denominator_mode,
        positives_cap=positives_cap)
    self.assertTupleEqual(loss.numpy().shape, (len(expected_loss),))
    for index, (val1, val2) in enumerate(zip(loss.numpy(), expected_loss)):
      self.assertAlmostEqual(
          val1,
          val2,
          places=5,
          msg=f'Lists not almost equal at index {index}: '
          '{loss.numpy()} != {expected_loss}')

  def testLossForOneView(self):
    features = np.array([
        [[0.01, 0.02, 0.14]],
        [[0.86, 0.97, 0.33]],
        [[0.32, 0.64, 0.28]],
    ])
    labels = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                      dtype=np.int32)
    loss = losses.contrastive_loss(
        features, labels=labels, temperature=1.0, base_temperature=1.0)

    pos0 = np.exp(np.dot(features[0, 0, :], features[1, 0, :]))
    neg0 = np.exp(np.dot(features[0, 0, :], features[2, 0, :]))
    loss0 = -np.log(pos0 / (pos0 + neg0))
    pos1 = np.exp(np.dot(features[1, 0, :], features[0, 0, :]))
    neg1 = np.exp(np.dot(features[1, 0, :], features[2, 0, :]))
    loss1 = -np.log(pos1 / (pos1 + neg1))
    expected_loss = np.array([loss0, loss1, 0.0])

    self.assertTupleEqual(loss.numpy().shape, expected_loss.shape)
    for index, (val1, val2) in enumerate(zip(loss.numpy(), expected_loss)):
      self.assertAlmostEqual(
          val1,
          val2,
          places=5,
          msg=f'Lists not almost equal at index {index}: '
          f'{loss.numpy()} != {expected_loss}')

  def testLossOnTPU(self):
    # Calling tpu.replicate in Eager mode doesn't work. Wrapping in a graph
    # implicitly disables Eager mode within its scope.
    with tf.Graph().as_default():
      features = tf.constant([
          [[0.01, 0.02, 0.14], [0.38, 0.61, 0.50]],
          [[0.86, 0.97, 0.33], [0.26, 0.68, 0.45]],
          [[0.32, 0.64, 0.28], [0.45, 0.74, 0.73]],
          [[0.45, 0.62, 0.07], [0.13, 0.28, 0.91]],
      ])
      labels = tf.one_hot((0, 0, 1, 1), 2)

      tpu_result = tf.compat.v1.tpu.replicate(
          losses.contrastive_loss,
          [[features[:2], labels[:2]], [features[2:], labels[2:]]])
      # tpu_result should be a list of 2 lists, each containing a single float
      # Tensor with shape [2].
      self.assertLen(tpu_result, 2)
      self.assertLen(tpu_result[0], 1)
      self.assertLen(tpu_result[1], 1)
      self.assertEqual([2], tpu_result[0][0].shape.as_list())
      self.assertEqual([2], tpu_result[1][0].shape.as_list())
      tpu_loss = tf.reshape(tpu_result, [4])

      cpu_loss = losses.contrastive_loss(features, labels=labels)

      cpu_partial_loss_1 = losses.contrastive_loss(
          features[:2], labels=labels[:2])
      cpu_partial_loss_2 = losses.contrastive_loss(
          features[2:], labels=labels[2:])
      cpu_partial_loss = tf.concat([cpu_partial_loss_1, cpu_partial_loss_2],
                                   axis=0)

      with self.cached_session() as sess:
        sess.run(tf.compat.v1.tpu.initialize_system())

        tpu_loss, cpu_loss, cpu_partial_loss = sess.run(
            (tpu_loss, cpu_loss, cpu_partial_loss))
        print(tpu_loss)
        print(cpu_loss)
        # Numerical precision isn't so high on TPU.
        self.assertAllClose(tpu_loss, cpu_loss, atol=1e-2)
        # Verify that the TPU computation is different than independently
        # computing the two "local batches" on CPU, because of the internal
        # cross_replica_concat.
        self.assertNotAllClose(tpu_loss, cpu_partial_loss, atol=1e-2)


if __name__ == '__main__':
  tf.test.main()
