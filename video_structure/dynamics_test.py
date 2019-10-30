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

# Lint as: python3
"""Tests for video_structure.vision."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from video_structure import dynamics
from video_structure import hyperparameters


class DynamicsTest(tf.test.TestCase):

  def setUp(self):

    # Hyperparameter config for test models:
    self.cfg = hyperparameters.get_config()
    self.cfg.batch_size = 4
    self.cfg.observed_steps = 2
    self.cfg.predicted_steps = 2
    self.cfg.num_keypoints = 3
    self.cfg.num_rnn_units = 4

    super().setUp()

  def testTrainingLossIsNotNan(self):
    """Tests a minimal Keras training loop for the dynamics model."""
    observed_keypoints = np.random.RandomState(0).normal(size=(
        self.cfg.batch_size, self.cfg.observed_steps + self.cfg.predicted_steps,
        self.cfg.num_keypoints, 3))
    model = dynamics.build_vrnn(self.cfg)
    model.add_loss(tf.nn.l2_loss(model.inputs[0] - model.outputs[0]))  # KP loss
    model.add_loss(tf.reduce_mean(model.outputs[1]))  # KL loss
    model.compile(tf.keras.optimizers.Adam(lr=1e-5))
    history = model.fit(x=observed_keypoints, steps_per_epoch=1, epochs=1)
    self.assertFalse(
        np.any(np.isnan(history.history['loss'])),
        'Loss contains nans: {}'.format(history.history['loss']))

  def testDecoderShapes(self):
    rnn_state = tf.zeros((self.cfg.batch_size, self.cfg.num_rnn_units))
    latent_code = tf.zeros((self.cfg.batch_size, self.cfg.latent_code_size))
    keypoints = dynamics.build_decoder(self.cfg)([rnn_state, latent_code])
    self.assertEqual(
        keypoints.shape.as_list(),
        [self.cfg.batch_size, self.cfg.num_keypoints * 3])

  def testPosteriorNetShapes(self):
    rnn_state = tf.zeros((self.cfg.batch_size, self.cfg.num_rnn_units))
    keypoints = tf.zeros((self.cfg.batch_size, self.cfg.num_keypoints * 3))
    means, stds = dynamics.build_posterior_net(self.cfg)([rnn_state, keypoints])
    self.assertEqual(
        means.shape.as_list(), [self.cfg.batch_size, self.cfg.latent_code_size])
    self.assertEqual(
        stds.shape.as_list(), [self.cfg.batch_size, self.cfg.latent_code_size])


class KLDivergenceTest(tf.test.TestCase):

  def testKLDivergenceIsZero(self):
    """Tests that KL divergence of identical distributions is zero."""
    with self.session() as sess:
      mean = tf.random.normal((3, 3, 3))
      std = tf.random.normal((3, 3, 3))
      kl_divergence = dynamics.KLDivergence()([mean, std, mean, std])
      result = sess.run([kl_divergence])[0]
    np.testing.assert_array_equal(result, result * 0.0)

  def testNonzeroKLDivergence(self):
    """Test that KL divergence layer provides correct result."""
    mu = 2.0
    sigma = 2.0
    n = 3

    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    result_np = 0.5 * (sigma ** 2 + mu ** 2 - np.log(sigma ** 2) - 1) * n

    with self.session() as sess:
      kl_divergence = dynamics.KLDivergence()(
          [tf.zeros(n), tf.ones(n), tf.zeros(n) + mu, tf.ones(n) * sigma])
      result_tf = sess.run(kl_divergence)

    np.testing.assert_almost_equal(result_tf, result_np, decimal=4)

  def testKLDivergenceAnnealing(self):
    inputs = tf.keras.Input(1)
    outputs = dynamics.KLDivergence(kl_annealing_steps=4)([
        inputs, inputs,
        tf.keras.layers.Lambda(lambda x: x + 1.0)(inputs), inputs
    ])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile('sgd', 'mse')
    obtained_kl = [model.predict(x=[1], steps=1)]  # Should be zero before fit.
    for _ in range(5):
      model.fit(x=[1], y=[1], epochs=1, steps_per_epoch=1)
      obtained_kl.append(model.predict(x=[1], steps=1))
    obtained_kl = np.array(obtained_kl).ravel()
    np.testing.assert_array_almost_equal(
        obtained_kl, [0, 0.125, 0.25, 0.375, 0.5, 0.5])


class TrainingStepCounterTest(tf.test.TestCase):

  def setUp(self):
    # Set up simple model in which the ground-truth data is a tensor of ones and
    # the predicted data is a tensor of zeros.
    super().setUp()  # Sets up the TensorFlow environment, so call it early.
    self.sess = tf.keras.backend.get_session()
    inputs = tf.keras.Input(1)
    outputs = dynamics.TrainingStepCounter()(inputs)
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
    self.model.compile('sgd', 'mse')

  def testTrainingPhaseStepsAreCounted(self):
    num_epochs = 2
    steps_per_epoch = 5
    self.model.fit(
        x=np.zeros(1),
        y=np.zeros(1),
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch)
    step_count = self.sess.run(self.model.layers[-1].weights[0])
    self.assertEqual(step_count, num_epochs * steps_per_epoch)

  def testTestingPhaseStepsAreNotCounted(self):
    self.model.predict(x=np.zeros(1), steps=10)
    step_count = self.sess.run(self.model.layers[-1].weights[0])
    self.assertEqual(step_count, 0.0)

  def testMultipleCallsAreCountedOnce(self):
    """Calling the same layer twice should not increase the counter twice."""

    # Create a model that calls the same TrainingStepCounter layer twice:
    counter = dynamics.TrainingStepCounter()
    inputs = tf.keras.Input(1)
    output1 = counter(inputs)
    output2 = counter(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
    model.compile('sgd', 'mse')

    # Train:
    num_epochs = 2
    steps_per_epoch = 5
    model.fit(
        x=np.zeros(1),
        y=[np.zeros(1), np.zeros(1)],
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch)

    step_count = self.sess.run(model.layers[-1].weights[0])
    self.assertEqual(step_count, num_epochs * steps_per_epoch)


class ScheduledSamplingTest(tf.test.TestCase):

  def setUp(self):
    # Set up simple model in which the ground-truth data is a tensor of ones and
    # the predicted data is a tensor of zeros.
    self.ramp_steps = 5000
    self.p_true_start = 1.0
    self.p_true_end = 0.0
    super().setUp()  # Sets up the TensorFlow environment, so call it early.
    self.sess = tf.keras.backend.get_session()
    inputs = tf.keras.Input(1)
    true = tf.keras.layers.Lambda(lambda x: x + 1.0)(inputs)
    pred = tf.keras.layers.Lambda(lambda x: x + 0.0)(inputs)
    outputs = dynamics.ScheduledSampling(
        p_true_start=self.p_true_start,
        p_true_end=self.p_true_end,
        ramp_steps=self.ramp_steps,
    )([true, pred])
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
    self.model.compile('sgd', 'mse')

  def testSamplingSchedule(self):
    num_epochs = 5
    steps_per_epoch = self.ramp_steps / num_epochs
    assert steps_per_epoch >= 1000, ('steps_per_epoch should be large enough to'
                                     'average out the sampling randomness.')
    history = self.model.fit(
        x=np.zeros(1),
        y=np.zeros(1),
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch)
    expected_schedule = np.linspace(
        self.p_true_start, self.p_true_end, 2 * num_epochs + 1)[1::2]
    # Note that the model is set up such that the "loss" variable contains the
    # average fraction of "true" samples obtained in each epoch:
    np.testing.assert_array_almost_equal(
        history.history['loss'],
        expected_schedule,
        decimal=1,
        err_msg='Observed schedule deviates from expected linear schedule.')


class SampleBestBeliefTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.num_samples = 5
    self.batch_size = 4
    self.latent_code_size = 3
    self.num_keypoints = 2

    # Create sampled_latent of shape [num_samples, batch_size, latent_code_size]
    # such that samples are numbered from 1 to num_samples:
    sampled_latent = np.arange(self.num_samples)[:, np.newaxis, np.newaxis]
    sampled_latent = np.tile(sampled_latent,
                             [1, self.batch_size, self.latent_code_size])
    self.sampled_latent = tf.convert_to_tensor(sampled_latent, dtype=tf.float32)

    # Create sampled_keypoints of shape [num_samples, batch_size,
    # 3 * num_keypoints] such that samples are numbered from 1 to num_samples:
    sampled_keypoints = np.arange(self.num_samples)
    sampled_keypoints = sampled_keypoints[:, np.newaxis, np.newaxis]
    sampled_keypoints = np.tile(sampled_keypoints,
                                [1, self.batch_size, 3 * self.num_keypoints])
    self.sampled_keypoints = tf.convert_to_tensor(
        sampled_keypoints, dtype=tf.float32)

    # Create sample_losses of shape [num_samples, batch_size] such that the best
    # sample varies for the different elements in the batch. For batch example
    # 0, sample 4 is best; for batch example 1, sample 2 is best, and so on...
    batch_example = [0, 1, 2, 3]
    self.best_samples = [4, 2, 1, 3]
    sample_losses = np.ones((self.num_samples, self.batch_size))
    sample_losses[self.best_samples, batch_example] = 0.0
    self.sample_losses = tf.convert_to_tensor(sample_losses, dtype=tf.float32)

  def testBestSampleIsReturnedDuringTraining(self):
    with self.session() as sess:
      tf.keras.backend.set_learning_phase(1)
      chosen_latent, chosen_keypoints = dynamics._choose_sample(
          self.sampled_latent, self.sampled_keypoints, self.sample_losses)
      chosen_latent, chosen_keypoints = sess.run(
          [chosen_latent, chosen_keypoints])

    # Check output shapes:
    self.assertEqual(chosen_latent.shape,
                     (self.batch_size, self.latent_code_size))
    self.assertEqual(chosen_keypoints.shape,
                     (self.batch_size, self.latent_code_size * 2))

    # Check that the correct sample is chosen for each example in the batch:
    self.assertEqual(list(chosen_latent[:, 0]), self.best_samples)
    self.assertEqual(list(chosen_keypoints[:, 0]), self.best_samples)

  def testFirstSampleIsReturnedDuringInference(self):
    with self.session() as sess:
      tf.keras.backend.set_learning_phase(0)
      chosen_latent, chosen_keypoints = dynamics._choose_sample(
          self.sampled_latent, self.sampled_keypoints, self.sample_losses)
      chosen_latent, chosen_keypoints = sess.run(
          [chosen_latent, chosen_keypoints])

    # Check that the 0th sample is chosen for each example in the batch:
    np.testing.assert_array_equal(chosen_latent, 0.0 * chosen_latent)
    np.testing.assert_array_equal(chosen_keypoints, 0.0 * chosen_keypoints)

  @parameterized.named_parameters(('_with_sampling', False),
                                  ('_use_mean_instead_of_sample', True))
  def testWholeLayerRuns(self, use_mean_instead_of_sample):

    def dummy_decoder(inputs):
      del inputs
      return tf.zeros((self.batch_size, 3 * self.num_keypoints))

    sampler = dynamics.SampleBestBelief(self.num_samples, dummy_decoder,
                                        use_mean_instead_of_sample)
    latent_mean = tf.zeros((self.batch_size, self.latent_code_size))
    latent_std = tf.zeros((self.batch_size, self.latent_code_size))
    rnn_state = tf.zeros((self.batch_size, 1))
    observed_keypoints_flat = tf.zeros(
        (self.batch_size, 3 * self.num_keypoints))
    chosen_latent, chosen_keypoints = sampler(
        [latent_mean, latent_std, rnn_state, observed_keypoints_flat])
    with self.session() as sess:
      chosen_latent, chosen_keypoints = sess.run(
          [chosen_latent, chosen_keypoints])
    self.assertEqual(chosen_latent.shape,
                     (self.batch_size, self.latent_code_size))
    self.assertEqual(chosen_keypoints.shape,
                     (self.batch_size, self.latent_code_size * 2))


if __name__ == '__main__':
  absltest.main()
