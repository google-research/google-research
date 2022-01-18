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
"""Tests for supcon.utils."""
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from supcon import blocks
from supcon import enums
from supcon import utils


class CreateTrainOpTest(tf.test.TestCase):

  def setUp(self):
    super(CreateTrainOpTest, self).setUp()
    np.random.seed(0)

    # Create an easy training set:
    self._inputs = np.random.rand(16, 4).astype(np.float32)
    self._labels = np.random.randint(0, 2, size=(16, 1)).astype(np.float32)

  def batchnorm_classifier(self, inputs):
    inputs = blocks.batch_norm()(inputs, True)
    return tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)(inputs)

  def testTrainOpInCollection(self):
    with tf.Graph().as_default():
      tf_inputs = tf.constant(self._inputs, dtype=tf.dtypes.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.dtypes.float32)

      tf_predictions = self.batchnorm_classifier(tf_inputs)
      loss = tf.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = utils.create_train_op(loss, optimizer)

      # Make sure the training op was recorded in the proper collection
      self.assertIn(train_op, tf.get_collection(tf.GraphKeys.TRAIN_OP))

  def testUseUpdateOps(self):
    with tf.Graph().as_default():
      tf_inputs = tf.constant(self._inputs, dtype=tf.dtypes.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.dtypes.float32)

      expected_mean = np.mean(self._inputs, axis=(0))
      expected_var = np.var(self._inputs, axis=(0))

      tf_predictions = self.batchnorm_classifier(tf_inputs)
      loss = tf.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)

      train_op = utils.create_train_op(loss, optimizer)

      moving_mean = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '.*moving_mean:')[0]
      moving_variance = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          '.*moving_variance:')[0]

      with self.cached_session() as session:
        # Initialize all variables
        session.run(tf.global_variables_initializer())
        mean, variance = session.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(200):
          session.run(train_op)

        mean = moving_mean.eval()
        variance = moving_variance.eval()
        # After 10 updates with decay 0.1 moving_mean == expected_mean and
        # moving_variance == expected_var.
        self.assertAllClose(mean, expected_mean)
        self.assertAllClose(variance, expected_var)

  def testEmptyUpdateOps(self):
    with tf.Graph().as_default():
      tf_inputs = tf.constant(self._inputs, dtype=tf.dtypes.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.dtypes.float32)

      tf_predictions = self.batchnorm_classifier(tf_inputs)
      loss = tf.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = utils.create_train_op(loss, optimizer, update_ops=[])

      moving_mean = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                      '.*moving_mean:')[0]
      moving_variance = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                          '.*moving_variance:')[0]

      with self.cached_session() as session:
        # Initialize all variables
        session.run(tf.global_variables_initializer())
        mean, variance = session.run([moving_mean, moving_variance])
        # After initialization moving_mean == 0 and moving_variance == 1.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

        for _ in range(10):
          session.run(train_op)

        mean = moving_mean.eval()
        variance = moving_variance.eval()

        # Since we skip update_ops the moving_vars are not updated.
        self.assertAllClose(mean, [0] * 4)
        self.assertAllClose(variance, [1] * 4)

  def testGlobalStepIsIncrementedByDefault(self):
    with tf.Graph().as_default():
      tf_inputs = tf.constant(self._inputs, dtype=tf.dtypes.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.dtypes.float32)

      tf_predictions = self.batchnorm_classifier(tf_inputs)
      loss = tf.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = utils.create_train_op(loss, optimizer)

      global_step = tf.train.get_or_create_global_step()

      with self.cached_session() as session:
        # Initialize all variables
        session.run(tf.global_variables_initializer())

        for _ in range(10):
          session.run(train_op)

        # After 10 updates global_step should be 10.
        self.assertAllClose(global_step.eval(), 10)

  def testGlobalStepNotIncrementedWhenSetToNone(self):
    with tf.Graph().as_default():
      tf_inputs = tf.constant(self._inputs, dtype=tf.dtypes.float32)
      tf_labels = tf.constant(self._labels, dtype=tf.dtypes.float32)

      tf_predictions = self.batchnorm_classifier(tf_inputs)
      loss = tf.losses.log_loss(tf_labels, tf_predictions)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
      train_op = utils.create_train_op(loss, optimizer, global_step=None)

      global_step = tf.train.get_or_create_global_step()

      with self.cached_session() as session:
        # Initialize all variables
        session.run(tf.global_variables_initializer())

        for _ in range(10):
          session.run(train_op)

        # Since train_op don't use global_step it shouldn't change.
        self.assertAllClose(global_step.eval(), 0)


def construct_tests_with_dtypes():
  dtypes_names = {
      tf.half: 'half',
      tf.float32: 'float32',
      tf.float64: 'float64',
  }
  return map(lambda x: (x[1], x[0]), dtypes_names.items())


class LARSOptimizerTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(*construct_tests_with_dtypes())
  def testBasic(self, dtype):
    with self.cached_session():
      var0 = tf.Variable([1.0, 2.0], dtype=dtype)
      var1 = tf.Variable([3.0, 4.0], dtype=dtype)
      grads0 = tf.constant([0.1, 0.1], dtype=dtype)
      grads1 = tf.constant([0.01, 0.01], dtype=dtype)
      optimizer = utils.LARSOptimizer(3.0)
      lars_op = optimizer.apply_gradients(
          zip([grads0, grads1], [var0, var1]))
      tf.global_variables_initializer().run()
      # Fetch params to validate initial values
      self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd
      lars_op.run()
      # Validate updated params
      self.assertAllCloseAccordingToType(
          [1.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.))),
           2.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.)))],
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          [3.0 - 3.0 * (0.001 * (5. / np.sqrt(2.))),
           4.0 - 3.0 * (0.001 * (5. / np.sqrt(2.)))],
          self.evaluate(var1))
      self.assertEmpty(list(optimizer.variables()))

  @parameterized.named_parameters(*construct_tests_with_dtypes())
  def testBasicCallableParams(self, dtype):
    with self.cached_session():
      var0 = tf.Variable([1.0, 2.0], dtype=dtype)
      var1 = tf.Variable([3.0, 4.0], dtype=dtype)
      grads0 = tf.constant([0.1, 0.1], dtype=dtype)
      grads1 = tf.constant([0.01, 0.01], dtype=dtype)
      lr = lambda: 3.0
      lars_op = utils.LARSOptimizer(lr).apply_gradients(
          zip([grads0, grads1], [var0, var1]))
      tf.global_variables_initializer().run()
      # Fetch params to validate initial values
      self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd
      lars_op.run()
      # Validate updated params
      self.assertAllCloseAccordingToType(
          [1.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.))),
           2.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.)))],
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          [3.0 - 3.0 * (0.001 * (5. / np.sqrt(2.))),
           4.0 - 3.0 * (0.001 * (5. / np.sqrt(2.)))],
          self.evaluate(var1))

  @parameterized.named_parameters(*construct_tests_with_dtypes())
  def testTensorLearningRate(self, dtype):
    with self.cached_session():
      var0 = tf.Variable([1.0, 2.0], dtype=dtype)
      var1 = tf.Variable([3.0, 4.0], dtype=dtype)
      grads0 = tf.constant([0.1, 0.1], dtype=dtype)
      grads1 = tf.constant([0.01, 0.01], dtype=dtype)
      lrate = tf.constant(3.0)
      lars_op = utils.LARSOptimizer(
          lrate).apply_gradients(zip([grads0, grads1], [var0, var1]))
      tf.global_variables_initializer().run()
      # Fetch params to validate initial values
      self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd
      lars_op.run()
      # Validate updated params
      self.assertAllCloseAccordingToType(
          [1.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.))),
           2.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.)))],
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          [3.0 - 3.0 * (0.001 * (5. / np.sqrt(2.))),
           4.0 - 3.0 * (0.001 * (5. / np.sqrt(2.)))],
          self.evaluate(var1))

  @parameterized.named_parameters(*construct_tests_with_dtypes())
  def testGradWrtRef(self, dtype):
    with self.cached_session():
      opt = utils.LARSOptimizer(3.0)
      values = [1.0, 3.0]
      vars_ = [tf.Variable([v], dtype=dtype) for v in values]
      grads_and_vars = opt.compute_gradients(vars_[0] + vars_[1], vars_)
      tf.global_variables_initializer().run()
      for grad, _ in grads_and_vars:
        self.assertAllCloseAccordingToType([1.0], self.evaluate(grad))

  @parameterized.named_parameters(*construct_tests_with_dtypes())
  def testWithGlobalStep(self, dtype):
    with self.cached_session():
      global_step = tf.Variable(0, trainable=False)
      var0 = tf.Variable([1.0, 2.0], dtype=dtype)
      var1 = tf.Variable([3.0, 4.0], dtype=dtype)
      grads0 = tf.constant([0.1, 0.1], dtype=dtype)
      grads1 = tf.constant([0.01, 0.01], dtype=dtype)
      lars_op = utils.LARSOptimizer(3.0).apply_gradients(
          zip([grads0, grads1], [var0, var1]), global_step=global_step)
      tf.global_variables_initializer().run()
      # Fetch params to validate initial values
      self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
      self.assertAllCloseAccordingToType([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd
      lars_op.run()
      # Validate updated params and global_step
      self.assertAllCloseAccordingToType(
          [1.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.))),
           2.0 - 3.0 * (0.001 * (np.sqrt(5.) / np.sqrt(2.)))],
          self.evaluate(var0))
      self.assertAllCloseAccordingToType(
          [3.0 - 3.0 * (0.001 * (5. / np.sqrt(2.))),
           4.0 - 3.0 * (0.001 * (5. / np.sqrt(2.)))],
          self.evaluate(var1))
      self.assertAllCloseAccordingToType(1, self.evaluate(global_step))


class CrossReplicaConcatTest(tf.test.TestCase, parameterized.TestCase):

  def testCrossReplicaConcat(self):
    # Test takes tensor_to_communicate,
    # splits it amongst the two cores (1x1 topology),
    # then asserts each core returns the complete combined tensor.
    total_input_length = 8
    num_cores = 2

    # Initialize variables
    numpy_tensor_to_communicate = np.arange(total_input_length, dtype=np.int32)
    tensor_to_communicate = tf.convert_to_tensor(numpy_tensor_to_communicate)
    replica0_input, replica1_input = tf.split(
        tensor_to_communicate, num_cores, 0)

    # Create TPU operations
    tpu_compute_function = tf.tpu.replicate(
        utils.cross_replica_concat,
        [[replica0_input], [replica1_input]])

    with self.cached_session() as session:
      session.run(tf.tpu.initialize_system())

      concat_result = session.run(tpu_compute_function)

      # Concatenation result has shape [num_cores, 1, total_input_length]
      self.assertAllEqual(concat_result[0][0], tensor_to_communicate)
      self.assertAllEqual(concat_result[1][0], tensor_to_communicate)
      session.run(tf.tpu.shutdown_system())

  @parameterized.named_parameters(
      ('Scalar', tuple()),
      ('1D', (8, 4)),
      ('2D', (6, 6)),
      ('3D', (8, 8, 3)))
  def testCrossReplicaConcatBatchedTensor(self, tensor_shape):
    # Test takes tensor_to_communicate,
    # splits it amongst the two cores (1x1 topology),
    # then asserts each core returns the complete combined tensor.
    batch_size_per_core = 4
    num_cores = 2

    # Initialize variables
    numpy_tensor_to_communicate = np.zeros(
        [batch_size_per_core * num_cores] + list(tensor_shape),
        dtype=np.int32)
    tensor_to_communicate = tf.convert_to_tensor(numpy_tensor_to_communicate)
    replica0_input, replica1_input = tf.split(
        tensor_to_communicate, num_cores, 0)

    # Create TPU operations
    tpu_compute_function = tf.tpu.replicate(
        utils.cross_replica_concat,
        [[replica0_input], [replica1_input]])

    with self.cached_session() as session:
      session.run(tf.tpu.initialize_system())

      concat_result = session.run(tpu_compute_function)

      # Concatenation result has shape [num_cores, 1] +
      # [batch_size * num_cores] + `tensor_shape`
      self.assertAllEqual(concat_result[0][0], tensor_to_communicate)
      self.assertAllEqual(concat_result[1][0], tensor_to_communicate)
      session.run(tf.tpu.shutdown_system())


class BuildLearningRateScheduleTest(tf.test.TestCase, parameterized.TestCase):

  def testZeroEpochSchedule(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.EXPONENTIAL,
        warmup_start_epoch=24,
        max_learning_rate_epoch=24,
        decay_end_epoch=24,
        global_step=global_step,
        steps_per_epoch=100)
    self.assertEqual(1., learning_rate)

  def testZeroEpochScheduleWithWarmup(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    with self.assertRaises(ValueError):
      utils.build_learning_rate_schedule(
          learning_rate=1.,
          decay_type=enums.DecayType.EXPONENTIAL,
          warmup_start_epoch=12,
          max_learning_rate_epoch=24,
          decay_end_epoch=24,
          global_step=global_step,
          steps_per_epoch=100)

  DEFAULT_EPOCHS_PER_DECAY = 2.4
  DEFAULT_DECAY_FACTOR = 0.97

  @parameterized.parameters(range(1, 10))
  def testExponentialDecayWithoutWarmup(self, decay_index):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.EXPONENTIAL,
        warmup_start_epoch=0,
        max_learning_rate_epoch=0,
        decay_end_epoch=24,
        global_step=global_step,
        steps_per_epoch=100)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(1., initial_learning_rate)

      staircase_drop_step = 100 * decay_index * self.DEFAULT_EPOCHS_PER_DECAY
      before_staircase_drop_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: staircase_drop_step})
      after_staircase_drop_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: staircase_drop_step + 1})

      self.assertAlmostEqual(
          self.DEFAULT_DECAY_FACTOR**(
              np.floor(staircase_drop_step /
                       (self.DEFAULT_EPOCHS_PER_DECAY * 100)) - 1),
          before_staircase_drop_learning_rate,
          places=5)
      self.assertAlmostEqual(
          self.DEFAULT_DECAY_FACTOR**(np.floor(
              staircase_drop_step / (self.DEFAULT_EPOCHS_PER_DECAY * 100))),
          after_staircase_drop_learning_rate,
          places=5)

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 2400})
      self.assertAlmostEqual(
          self.DEFAULT_DECAY_FACTOR**(np.floor(
              2400 / (self.DEFAULT_EPOCHS_PER_DECAY * 100)) - 1.),
          final_learning_rate,
          places=5)

  def testExponentialDecayWithWarmup(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.EXPONENTIAL,
        warmup_start_epoch=0,
        max_learning_rate_epoch=5,
        decay_end_epoch=10,
        global_step=global_step,
        steps_per_epoch=100)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(0., initial_learning_rate)

      max_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(1., max_learning_rate)

      intermediate_warmup_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 250})
      self.assertAlmostEqual(0.5, intermediate_warmup_learning_rate)

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000})
      self.assertAlmostEqual(
          self.DEFAULT_DECAY_FACTOR**np.floor(
              500 / (self.DEFAULT_EPOCHS_PER_DECAY * 100)),
          final_learning_rate,
          places=5)

  def testExponentialDecayWithEpochsPerDecay(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.EXPONENTIAL,
        warmup_start_epoch=0,
        max_learning_rate_epoch=5,
        decay_end_epoch=10,
        global_step=global_step,
        steps_per_epoch=100,
        epochs_per_decay=3.4)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(0., initial_learning_rate)

      max_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(1., max_learning_rate)

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000})
      self.assertAlmostEqual((0.97)**np.floor((500) / (3.4 * 100)),
                             final_learning_rate,
                             places=5)

  def testExponentialDecayWithDecayRate(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.EXPONENTIAL,
        warmup_start_epoch=0,
        max_learning_rate_epoch=5,
        decay_end_epoch=10,
        global_step=global_step,
        steps_per_epoch=100,
        decay_rate=0.99)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(0., initial_learning_rate)

      max_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(1., max_learning_rate)

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000})
      self.assertAlmostEqual((0.99)**np.floor((500) / (2.4 * 100)),
                             final_learning_rate,
                             places=5)

  def testCosineDecayWithoutWarmup(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.COSINE,
        warmup_start_epoch=0,
        max_learning_rate_epoch=0,
        decay_end_epoch=10,
        global_step=global_step,
        steps_per_epoch=100)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(1., initial_learning_rate)

      quarter_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 250})
      self.assertAlmostEqual((1. + np.cos(np.pi / 4.)) / 2.,
                             quarter_learning_rate)

      halfway_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(0.5, halfway_learning_rate)

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000})
      self.assertAlmostEqual(0., final_learning_rate)

  def testCosineDecayWithWarmup(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.COSINE,
        warmup_start_epoch=0,
        max_learning_rate_epoch=5,
        decay_end_epoch=10,
        global_step=global_step,
        steps_per_epoch=100)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(0., initial_learning_rate)

      intermediate_warmup_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 250})
      self.assertAlmostEqual(0.5, intermediate_warmup_learning_rate)

      max_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(1., max_learning_rate)

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000})
      self.assertAlmostEqual(0., final_learning_rate)

  DEFAULT_BOUNDARY_EPOCHS = [30, 60, 80, 90]
  DEFAULT_DECAY_RATES = [1, 0.1, 0.01, 0.001, 1e-4]

  def testPiecewiseLinearDecayWithoutWarmup(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    steps_per_epoch = 100
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.PIECEWISE_LINEAR,
        warmup_start_epoch=0,
        max_learning_rate_epoch=0,
        decay_end_epoch=1000,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1})
      self.assertAlmostEqual(1., initial_learning_rate)

      for i, epoch in enumerate(self.DEFAULT_BOUNDARY_EPOCHS):
        before_decay_learning_rate = sess.run(
            learning_rate, feed_dict={global_step: epoch * steps_per_epoch})
        after_decay_learning_rate = sess.run(
            learning_rate, feed_dict={global_step: epoch * steps_per_epoch + 1})

        self.assertAlmostEqual(before_decay_learning_rate,
                               self.DEFAULT_DECAY_RATES[i])
        self.assertAlmostEqual(after_decay_learning_rate,
                               self.DEFAULT_DECAY_RATES[i + 1])

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000 * steps_per_epoch})
      self.assertAlmostEqual(1e-4, final_learning_rate, places=7)

  def testPiecewiseLinearDecayWithWarmup(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    max_learning_rate_epoch = 5
    steps_per_epoch = 100
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.PIECEWISE_LINEAR,
        warmup_start_epoch=0,
        max_learning_rate_epoch=max_learning_rate_epoch,
        decay_end_epoch=1000,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(0., initial_learning_rate)

      max_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(1., max_learning_rate)

      for i, epoch in enumerate(self.DEFAULT_BOUNDARY_EPOCHS):
        before_decay_learning_rate = sess.run(
            learning_rate,
            feed_dict={
                global_step: (epoch + max_learning_rate_epoch) * steps_per_epoch
            })
        after_decay_learning_rate = sess.run(
            learning_rate,
            feed_dict={
                global_step:
                    ((epoch + max_learning_rate_epoch) * steps_per_epoch) + 1
            })

        self.assertAlmostEqual(before_decay_learning_rate,
                               self.DEFAULT_DECAY_RATES[i])
        self.assertAlmostEqual(after_decay_learning_rate,
                               self.DEFAULT_DECAY_RATES[i + 1])

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000 * steps_per_epoch})
      self.assertAlmostEqual(1e-4, final_learning_rate, places=7)

  def testPiecewiseLinearDecayWithBoundaries(self):
    global_step = tf.placeholder_with_default(0, [], 'global_step')
    boundary_epochs = [1, 2, 3, 4]
    decay_rate = 1e-1
    max_learning_rate_epoch = 5
    steps_per_epoch = 100
    learning_rate = utils.build_learning_rate_schedule(
        learning_rate=1.,
        decay_type=enums.DecayType.PIECEWISE_LINEAR,
        warmup_start_epoch=0,
        max_learning_rate_epoch=max_learning_rate_epoch,
        decay_end_epoch=1000,
        global_step=global_step,
        steps_per_epoch=steps_per_epoch,
        boundary_epochs=boundary_epochs,
        decay_rate=decay_rate)

    with self.cached_session() as sess:
      initial_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 0})
      self.assertAlmostEqual(0., initial_learning_rate)

      max_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 500})
      self.assertAlmostEqual(1., max_learning_rate)

      for i, epoch in enumerate(boundary_epochs):
        before_decay_learning_rate = sess.run(
            learning_rate,
            feed_dict={
                global_step: (epoch + max_learning_rate_epoch) * steps_per_epoch
            })
        after_decay_learning_rate = sess.run(
            learning_rate,
            feed_dict={
                global_step:
                    ((epoch + max_learning_rate_epoch) * steps_per_epoch) + 1
            })

        self.assertAlmostEqual(before_decay_learning_rate, decay_rate**i)
        self.assertAlmostEqual(after_decay_learning_rate, decay_rate**(i + 1))

      final_learning_rate = sess.run(
          learning_rate, feed_dict={global_step: 1000})
      self.assertAlmostEqual(1e-4,
                             final_learning_rate,
                             places=5)


if __name__ == '__main__':
  tf.test.main()
