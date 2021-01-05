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

"""Tests for util.py."""

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from robust_loss import util

tf.enable_v2_behavior()


class UtilTest(tf.test.TestCase):

  def setUp(self):
    super(UtilTest, self).setUp()
    np.random.seed(0)

  def testInvSoftplusIsCorrect(self):
    """Test that inv_softplus() is the inverse of tf.nn.softplus()."""
    x = np.float32(np.exp(np.linspace(-10., 10., 1000)))
    x_recon = tf.nn.softplus(util.inv_softplus(x))
    self.assertAllClose(x, x_recon)

  def testLogitIsCorrect(self):
    """Test that logit() is the inverse of tf.sigmoid()."""
    x = np.float32(np.linspace(1e-5, 1. - 1e-5, 1000))
    x_recon = tf.sigmoid(util.logit(x))
    self.assertAllClose(x, x_recon)

  def testAffineSigmoidSpansRange(self):
    """Check that affine_sigmoid()'s output is in [lo, hi]."""
    x = np.finfo(np.float32).max * np.array([-1, 1], dtype=np.float32)
    for _ in range(10):
      lo = np.random.uniform(0., 0.3)
      hi = np.random.uniform(0.5, 4.)
      y = util.affine_sigmoid(x, lo=lo, hi=hi)
      self.assertAllClose(y[0], lo)
      self.assertAllClose(y[1], hi)

  def testAffineSigmoidIsCentered(self):
    """Check that affine_sigmoid(0) == (lo+hi)/2."""
    for _ in range(10):
      lo = np.random.uniform(0., 0.3)
      hi = np.random.uniform(0.5, 4.)
      y = util.affine_sigmoid(np.array(0.), lo=lo, hi=hi)
      self.assertAllClose(y, (lo + hi) * 0.5)

  def testAffineSoftplusSpansRange(self):
    """Check that affine_softplus()'s output is in [lo, infinity]."""
    x = np.finfo(np.float32).max * np.array([-1, 1], dtype=np.float32)
    for _ in range(10):
      lo = np.random.uniform(0., 0.1)
      ref = np.random.uniform(0.2, 10.)
      y = util.affine_softplus(x, lo=lo, ref=ref)
      self.assertAllClose(y[0], lo)
      self.assertAllGreater(y[1], 1e10)

  def testAffineSoftplusIsCentered(self):
    """Check that affine_softplus(0) == 1."""
    for _ in range(10):
      lo = np.random.uniform(0., 0.1)
      ref = np.random.uniform(0.2, 10.)
      y = util.affine_softplus(np.array(0.), lo=lo, ref=ref)
      self.assertAllClose(y, ref)

  def testDefaultAffineSigmoidMatchesSigmoid(self):
    """Check that affine_sigmoid() matches tf.nn.sigmoid() by default."""
    x = np.float32(np.linspace(-10., 10., 1000))
    y = util.affine_sigmoid(x)
    y_true = tf.nn.sigmoid(x)
    self.assertAllClose(y, y_true, atol=1e-5, rtol=1e-3)

  def testDefaultAffineSigmoidRoundTrip(self):
    """Check that x = inv_affine_sigmoid(affine_sigmoid(x)) by default."""
    x = np.float32(np.linspace(-10., 10., 1000))
    y = util.affine_sigmoid(x)
    x_recon = util.inv_affine_sigmoid(y)
    self.assertAllClose(x, x_recon, atol=1e-5, rtol=1e-3)

  def testAffineSigmoidRoundTrip(self):
    """Check that x = inv_affine_sigmoid(affine_sigmoid(x)) in general."""
    x = np.float32(np.linspace(-10., 10., 1000))
    for _ in range(10):
      lo = np.random.uniform(0., 0.3)
      hi = np.random.uniform(0.5, 4.)
      y = util.affine_sigmoid(x, lo=lo, hi=hi)
      x_recon = util.inv_affine_sigmoid(y, lo=lo, hi=hi)
      self.assertAllClose(x, x_recon, atol=1e-5, rtol=1e-3)

  def testDefaultAffineSoftplusRoundTrip(self):
    """Check that x = inv_affine_softplus(affine_softplus(x)) by default."""
    x = np.float32(np.linspace(-10., 10., 1000))
    y = util.affine_softplus(x)
    x_recon = util.inv_affine_softplus(y)
    self.assertAllClose(x, x_recon, atol=1e-5, rtol=1e-3)

  def testAffineSoftplusRoundTrip(self):
    """Check that x = inv_affine_softplus(affine_softplus(x)) in general."""
    x = np.float32(np.linspace(-10., 10., 1000))
    for _ in range(10):
      lo = np.random.uniform(0., 0.1)
      ref = np.random.uniform(0.2, 10.)
      y = util.affine_softplus(x, lo=lo, ref=ref)
      x_recon = util.inv_affine_softplus(y, lo=lo, ref=ref)
      self.assertAllClose(x, x_recon, atol=1e-5, rtol=1e-3)

  def testStudentsTNllAgainstTfp(self):
    """Check that our Student's T NLL matches TensorFlow Probability."""
    for _ in range(10):
      x = np.random.normal()
      df = np.exp(4. * np.random.normal())
      scale = np.exp(4. * np.random.normal())
      nll = util.students_t_nll(x, df, scale)
      nll_true = -tfp.distributions.StudentT(
          df=df, loc=tf.zeros_like(scale), scale=scale).log_prob(x)
      self.assertAllClose(nll, nll_true)

  def testRgbToSyuvPreservesVolume(self):
    """Tests that rgb_to_syuv() is volume preserving."""
    for _ in range(4):
      im = np.float32(np.random.uniform(size=(1, 1, 3)))
      jacobian = util.compute_jacobian(util.rgb_to_syuv, im)
      # Assert that the determinant of the Jacobian is close to 1.
      det = np.linalg.det(jacobian)
      self.assertAllClose(det, 1., atol=1e-5, rtol=1e-5)

  def testRgbToSyuvRoundTrip(self):
    """Tests that syuv_to_rgb(rgb_to_syuv(x)) == x."""
    rgb = np.float32(np.random.uniform(size=(32, 32, 3)))
    syuv = util.rgb_to_syuv(rgb)
    rgb_recon = util.syuv_to_rgb(syuv)
    self.assertAllClose(rgb, rgb_recon)

  def testSyuvIsScaledYuv(self):
    """Tests that rgb_to_syuv is proportional to tf.image.rgb_to_yuv()."""
    rgb = np.float32(np.random.uniform(size=(32, 32, 3)))
    syuv = util.rgb_to_syuv(rgb)
    yuv = tf.image.rgb_to_yuv(rgb)
    # Check that the ratio between `syuv` and `yuv` is nearly constant.
    ratio = syuv / yuv
    self.assertAllClose(tf.reduce_min(ratio), tf.reduce_max(ratio))

  def testImageDctPreservesVolume(self):
    """Tests that image_dct() is volume preserving."""
    for _ in range(4):
      im = np.float32(np.random.uniform(size=(4, 4, 2)))
      jacobian = util.compute_jacobian(util.image_dct, im)
      # Assert that the determinant of the Jacobian is close to 1.
      det = np.linalg.det(jacobian)
      self.assertAllClose(det, 1., atol=1e-5, rtol=1e-5)

  def testImageDctIsOrthonormal(self):
    """Test that <im0, im1> = <image_dct(im0), image_dct(im1)>."""
    for _ in range(4):
      im0 = np.float32(np.random.uniform(size=(4, 4, 2)))
      im1 = np.float32(np.random.uniform(size=(4, 4, 2)))
      dct_im0 = util.image_dct(im0)
      dct_im1 = util.image_dct(im1)
      prod1 = tf.reduce_sum(im0 * im1)
      prod2 = tf.reduce_sum(dct_im0 * dct_im1)
      self.assertAllClose(prod1, prod2)

  def testImageDctRoundTrip(self):
    """Tests that image_idct(image_dct(x)) == x."""
    image = np.float32(np.random.uniform(size=(32, 32, 3)))
    image_recon = util.image_idct(util.image_dct(image))
    self.assertAllClose(image, image_recon)


if __name__ == '__main__':
  tf.test.main()
