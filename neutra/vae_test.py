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

# Lint as: python2, python3
# pylint: disable=invalid-name,g-bad-import-order,g-long-lambda
"""NeuTra VAE tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

import gin
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from neutra import utils
from neutra import vae

tf.disable_v2_behavior()
tfb = tfp.bijectors


class NeutraTest(tf.test.TestCase):

  def setUp(self):
    super(NeutraTest, self).setUp()
    self.temp_dir = tempfile.mkdtemp()

  def tearDown(self):
    super(NeutraTest, self).tearDown()
    tf.gfile.DeleteRecursively(self.temp_dir)

  def testConv2DWN(self):
    x = vae.Conv2DWN("conv2d", num_filters=2)(tf.zeros([1, 3, 3, 1]))
    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([1, 3, 3, 2], x.shape)

  def testConvAR(self):
    shift, log_scale = vae.ConvAR(
        "conv_ar",
        real_event_shape=[1, 3, 3, 2],
        hidden_layers=[2],
        h=tf.zeros([1, 3, 3, 1]))(
            tf.zeros([1, 3, 3, 1]))
    self.evaluate(tf.global_variables_initializer())
    shift, log_scale = self.evaluate((shift, log_scale))
    self.assertAllEqual([1, 3, 3, 2], shift.shape)
    self.assertAllEqual([1, 3, 3, 2], log_scale.shape)

  def testDenseWN(self):
    x = vae.DenseWN("dense_wn", num_outputs=3)(tf.zeros([1, 1]))
    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([1, 3], x.shape)

  def testResConv2D(self):
    x = vae.ResConv2D(
        "conv2d", num_filters=2, kernel_size=[3, 3], stride=[1, 1])(
            tf.zeros([1, 3, 3, 1]))
    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([1, 3, 3, 2], x.shape)

  def testResDense(self):
    x = vae.ResDense("res_dense", num_dims=3)(tf.zeros([1, 1]))
    self.evaluate(tf.global_variables_initializer())
    x = self.evaluate(x)
    self.assertAllEqual([1, 3], x.shape)

  def testConvHier(self):
    encoder = vae.ConvHierEncoder("encoder", z_dims=2, h_dims=2)
    q = vae.ConvHierPriorPost(
        "q", encoder=encoder, z_dims=2, h_dims=2, image_width=4)
    p = vae.ConvHierPriorPost("p", z_dims=2, h_dims=2, image_width=4)
    z, q_z, _ = q(images=tf.zeros([2, 4, 4, 3]))
    _, p_z, x = p(images=tf.zeros([2, 4, 4, 1]), z=z)
    self.evaluate(tf.global_variables_initializer())
    z, q_z, p_z, x = self.evaluate([z, q_z, p_z, x])
    self.assertEqual(4, len(z))
    self.assertEqual(4, len(q_z))
    self.assertEqual(4, len(p_z))
    self.assertAllEqual([2, 4, 4, 3], x.shape)

  def testConv1(self):
    z = vae.ConvEncoder(
        "encoder", num_outputs=2, hidden_dims=2)(
            tf.zeros([1, 28, 28, 1]))
    x = vae.ConvDecoder("decoder", output_shape=[28, 28, 1], hidden_dims=2)(z)
    self.evaluate(tf.global_variables_initializer())
    x, z = self.evaluate([x, z])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1, 28, 28, 1], x.shape)

  def testConv2(self):
    z = vae.ConvEncoder2("encoder", num_outputs=2)(tf.zeros([1, 28, 28, 1]))
    x = vae.ConvDecoder2("decoder", output_shape=[28, 28, 1])(z)
    self.evaluate(tf.global_variables_initializer())
    x, z = self.evaluate([x, z])
    self.assertAllEqual([1, 4, 4, 2], z.shape)
    self.assertAllEqual([1, 28, 28, 1], x.shape)

  def testConv3(self):
    with tf.Graph().as_default():
      z = vae.ConvEncoder3("encoder", num_outputs=2)(tf.zeros([1, 28, 28, 1]))
      x = vae.ConvDecoder3("decoder", output_shape=[28, 28, 1])(z)
      self.evaluate(tf.global_variables_initializer())
      x, z = self.evaluate([x, z])
      self.assertAllEqual([1, 7, 7, 2], z.shape)
      self.assertAllEqual([1, 28, 28, 1], x.shape)

  def testConv4(self):
    with tf.Graph().as_default():
      z = vae.ConvEncoder4("encoder", num_outputs=2)(tf.zeros([1, 28, 28, 1]))
      x = vae.ConvDecoder4("decoder", output_shape=[28, 28, 1])(z)
      self.evaluate(tf.global_variables_initializer())
      x, z = self.evaluate([x, z])
      self.assertAllEqual([1, 2], z.shape)
      self.assertAllEqual([1, 28, 28, 1], x.shape)

  def testDense(self):
    z = vae.DenseEncoder(
        "encoder", num_outputs=2, hidden_layer_sizes=[2])(
            tf.zeros([1, 28, 28, 1]))
    x = vae.DenseDecoder(
        "decoder", output_shape=[28, 28, 1], hidden_layer_sizes=[2])(
            z)
    self.evaluate(tf.global_variables_initializer())
    x, z = self.evaluate([x, z])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1, 28, 28, 1], x.shape)

  def testIndependentBernouli3D(self):
    dist = vae.IndependentBernouli3D(tf.zeros([1, 28, 28, 1]))
    x = self.evaluate(dist.sample())
    lp = self.evaluate(dist.log_prob(x))
    self.assertAllEqual([1, 28, 28, 1], x.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertAllGreaterEqual(x, 0.)
    self.assertAllLessEqual(x, 1.)

  def testIndependentDiscreteLogistic3D(self):
    dist = vae.IndependentDiscreteLogistic3D(
        tf.zeros([1, 28, 28, 1]), tf.ones([1, 28, 28, 1]))
    x = self.evaluate(dist.sample())
    lp = self.evaluate(dist.log_prob(x))
    self.assertAllEqual([1, 28, 28, 1], x.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertAllGreaterEqual(x, 0.)
    self.assertAllLessEqual(x, 1.)

  def testIndependentDiscreteLogistic3D2(self):
    dist = vae.IndependentDiscreteLogistic3D2(
        tf.zeros([1, 28, 28, 1]), tf.ones([1, 28, 28, 1]))
    x = self.evaluate(dist.sample())
    lp = self.evaluate(dist.log_prob(x))
    self.assertAllEqual([1, 28, 28, 1], x.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertAllGreaterEqual(x, 0.)
    self.assertAllLessEqual(x, 1.)

  def testDenseRecognition(self):
    encoder = lambda _: tf.zeros([1, 4])
    [z], [lp], [b] = vae.DenseRecognition(
        "recog", encoder=encoder)(
            tf.zeros([1, 28, 28, 1]))
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testDenseRecognitionAffine(self):
    encoder = lambda _: tf.zeros([1, 2 + 3])
    [z], [lp], [b] = vae.DenseRecognitionAffine(
        "recog", encoder=encoder, z_dims=2)(
            tf.zeros([1, 28, 28, 1]))
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testDenseRecognitionAffineLR(self):
    encoder = lambda _: tf.zeros([1, 6])
    [z], [lp], [b] = vae.DenseRecognitionAffineLR(
        "recog", encoder=encoder, z_dims=2)(
            tf.zeros([1, 28, 28, 1]))
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testDenseRecognitionRNVP(self):
    encoder = lambda _: tf.zeros([1, 6])
    [z], [lp], [b] = vae.DenseRecognitionRNVP(
        "recog", encoder=encoder, condition_bijector=True, layer_sizes=[2])(
            tf.zeros([1, 28, 28, 1]))
    self.evaluate(tf.global_variables_initializer())
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testDenseRecognitionIAF(self):
    encoder = lambda _: tf.zeros([1, 6])
    [z], [lp], [b] = vae.DenseRecognitionIAF(
        "recog", encoder=encoder, condition_iaf=True, iaf_layer_sizes=[2])(
            tf.zeros([1, 28, 28, 1]))
    self.evaluate(tf.global_variables_initializer())
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testConvIAF(self):
    encoder = lambda _: tf.zeros([1, 2, 2, 6])
    [z], [lp], [b] = vae.ConvIAF(
        "recog", encoder=encoder, condition_iaf=True, iaf_layer_sizes=[2])(
            tf.zeros([1, 28, 28, 1]))
    self.evaluate(tf.global_variables_initializer())
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2, 2, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testConvShiftScale(self):
    encoder = lambda _: tf.zeros([1, 2, 2, 4])
    [z], [lp], [b] = vae.ConvShiftScale(
        "recog", encoder=encoder)(
            tf.zeros([1, 28, 28, 1]))
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([1, 2, 2, 2], z.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertIsInstance(b, tfb.Bijector)

  def testSimplePrior(self):
    [z], [lp] = vae.SimplePrior("prior", num_dims=2, batch=3)()
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([3, 2], z.shape)
    self.assertAllEqual([3], lp.shape)

  def testSimple3DPrior(self):
    [z], [lp] = vae.Simple3DPrior("prior", shape=[2, 2, 3], batch=3)()
    z, lp = self.evaluate([z, lp])
    self.assertAllEqual([3, 2, 2, 3], z.shape)
    self.assertAllEqual([3], lp.shape)

  def testDenseMNISTNoise(self):
    decoder = lambda _: tf.zeros([1, 4, 4, 1])
    x, lp = vae.DenseMNISTNoise("noise", decoder=decoder)()
    x, lp = self.evaluate([x, lp])
    self.assertAllEqual([1, 4, 4, 1], x.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertAllGreaterEqual(x, 0.)
    self.assertAllLessEqual(x, 1.)

  def testDenseCIFAR10Noise(self):
    decoder = lambda _: tf.zeros([1, 32, 32, 3])
    x, lp = vae.DenseCIFAR10TNoise("noise", decoder=decoder)()
    self.evaluate(tf.global_variables_initializer())
    x, lp = self.evaluate([x, lp])
    self.assertAllEqual([1, 32, 32, 3], x.shape)
    self.assertAllEqual([1], lp.shape)
    self.assertAllGreaterEqual(x, 0.)
    self.assertAllLessEqual(x, 1.)

  def testLearningRate(self):
    lr = vae.LearningRate(train_size=250, global_step=0)
    self.evaluate(lr)

  def testDLGM(self):
    gin.clear_config()
    gin.bind_parameter("dense_encoder.hidden_layer_sizes", [2, 2])
    gin.bind_parameter("ais.num_steps", 1)
    dataset = utils.FakeMNISTDataset()
    model_fn = lambda: vae.DLGM(
        z_dims=2, bijector_type="shift_scale", dataset=dataset)
    with tf.Graph().as_default():
      self.evaluate(tf.global_variables_initializer())
      vae.Train(
          model_fn(),
          dataset=dataset,
          train_dir=self.temp_dir,
          master=None,
          epochs=1)
    with tf.Graph().as_default():
      vae.Eval(
          model_fn(),
          dataset=dataset,
          train_dir=self.temp_dir,
          eval_dir=self.temp_dir,
          master=None,
          max_number_of_evaluations=1)
    with tf.Graph().as_default():
      writer = tf.summary.FileWriter(self.temp_dir)
      vae.AISEvalShard(
          shard=0,
          master=None,
          num_workers=1,
          num_chains=1,
          dataset=dataset,
          use_polyak_averaging=False,
          writer=writer,
          train_dir=self.temp_dir,
          model_fn=model_fn,
          batch=250)

  def testVAE(self):
    gin.clear_config()
    gin.bind_parameter("dense_encoder.hidden_layer_sizes", [2, 2])
    gin.bind_parameter("ais.num_steps", 1)
    dataset = utils.FakeMNISTDataset()
    model_fn = lambda: vae.VAE(
        z_dims=2, bijector_type="shift_scale", dataset=dataset)
    with tf.Graph().as_default():
      self.evaluate(tf.global_variables_initializer())
      vae.Train(
          model_fn(),
          dataset=dataset,
          train_dir=self.temp_dir,
          master=None,
          epochs=1)
    with tf.Graph().as_default():
      vae.Eval(
          model_fn(),
          dataset=dataset,
          train_dir=self.temp_dir,
          eval_dir=self.temp_dir,
          master=None,
          max_number_of_evaluations=1)
    with tf.Graph().as_default():
      writer = tf.summary.FileWriter(self.temp_dir)
      vae.AISEvalShard(
          shard=0,
          master=None,
          num_workers=1,
          num_chains=1,
          dataset=dataset,
          use_polyak_averaging=False,
          writer=writer,
          train_dir=self.temp_dir,
          model_fn=model_fn,
          batch=250)

if __name__ == "__main__":
  tf.test.main()
