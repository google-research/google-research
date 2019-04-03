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

"""Tests for wavelet.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import PIL.Image
import scipy.io
import tensorflow as tf
from robust_loss import util
from robust_loss import wavelet


class WaveletTest(tf.test.TestCase):

  def setUp(self):
    super(WaveletTest, self).setUp()
    np.random.seed(0)

  def _assert_pyramids_close(self, x0, x1, epsilon):
    """A helper function for assering that two wavelet pyramids are close."""
    if isinstance(x0, tuple) or isinstance(x0, list):
      assert isinstance(x1, (list, tuple))
      assert len(x0) == len(x1)
      for y0, y1 in zip(x0, x1):
        self._assert_pyramids_close(y0, y1, epsilon)
    else:
      assert not isinstance(x1, (list, tuple))
      self.assertAllEqual(x0.shape, x1.shape)
      self.assertAllClose(x0, x1, atol=epsilon, rtol=epsilon)

  def testPadWithOneReflectionIsCorrect(self):
    """Tests that pad_reflecting(p) matches tf.pad(p) when p is small."""
    for _ in range(4):
      n = int(np.ceil(np.random.uniform() * 8)) + 1
      x = np.random.uniform(size=(n, n, n))
      padding_below = int(np.round(np.random.uniform() * (n - 1)))
      padding_above = int(np.round(np.random.uniform() * (n - 1)))
      axis = int(np.floor(np.random.uniform() * 3.))

      if axis == 0:
        reference = tf.pad(x, [[padding_below, padding_above], [0, 0], [0, 0]],
                           'REFLECT')
      elif axis == 1:
        reference = tf.pad(x, [[0, 0], [padding_below, padding_above], [0, 0]],
                           'REFLECT')
      elif axis == 2:
        reference = tf.pad(x, [[0, 0], [0, 0], [padding_below, padding_above]],
                           'REFLECT')

      with self.session():
        result = wavelet.pad_reflecting(x, padding_below, padding_above,
                                        axis).eval()
        reference = reference.eval()
      self.assertAllEqual(result.shape, reference.shape)
      self.assertAllEqual(result, reference)

  def testPadWithManyReflectionsIsCorrect(self):
    """Tests that pad_reflecting(k * p) matches tf.pad(p) applied k times."""
    for _ in range(4):
      n = int(np.random.uniform() * 8.) + 1
      p = n - 1
      x = np.random.uniform(size=(n))
      with self.session():
        result1 = wavelet.pad_reflecting(x, p, p, 0).eval()
        result2 = wavelet.pad_reflecting(x, 2 * p, 2 * p, 0).eval()
        result3 = wavelet.pad_reflecting(x, 3 * p, 3 * p, 0).eval()
        reference1 = tf.pad(x, [[p, p]], 'REFLECT').eval()
        reference2 = tf.pad(reference1, [[p, p]], 'REFLECT').eval()
        reference3 = tf.pad(reference2, [[p, p]], 'REFLECT').eval()
      self.assertAllEqual(result1.shape, reference1.shape)
      self.assertAllEqual(result1, reference1)
      self.assertAllEqual(result2.shape, reference2.shape)
      self.assertAllEqual(result2, reference2)
      self.assertAllEqual(result3.shape, reference3.shape)
      self.assertAllEqual(result3, reference3)

  def testPadWithManyReflectionsGolden1IsCorrect(self):
    """Tests pad_reflecting() against a golden example."""
    n = 8
    p0 = 17
    p1 = 13
    x = np.arange(n)
    reference1 = np.concatenate((np.arange(3, 0, -1), np.arange(n),
                                 np.arange(n - 2, 0, -1), np.arange(n),
                                 np.arange(n - 2, 0, -1), np.arange(7)))
    with self.session():
      result1 = wavelet.pad_reflecting(x, p0, p1, 0).eval()
    self.assertAllEqual(result1.shape, reference1.shape)
    self.assertAllEqual(result1, reference1)

  def testPadWithManyReflectionsGolden2IsCorrect(self):
    """Tests pad_reflecting() against a golden example."""
    n = 11
    p0 = 15
    p1 = 7
    x = np.arange(n)
    reference1 = np.concatenate((np.arange(5, n), np.arange(n - 2, 0, -1),
                                 np.arange(n), np.arange(n - 2, 2, -1)))
    with self.session():
      result1 = wavelet.pad_reflecting(x, p0, p1, 0).eval()
    self.assertAllEqual(result1.shape, reference1.shape)
    self.assertAllEqual(result1, reference1)

  def testAnalysisLowpassFiltersAreNormalized(self):
    """Tests that the analysis lowpass filter doubles the input's magnitude."""
    for wavelet_type in wavelet.generate_filters():
      filters = wavelet.generate_filters(wavelet_type)
      # The sum of the outer product of the analysis lowpass filter with itself.
      magnitude = np.sum(filters.analysis_lo[:, np.newaxis] *
                         filters.analysis_lo[np.newaxis, :])
      self.assertAllClose(magnitude, 2., atol=1e-10, rtol=1e-10)

  def testWaveletTransformationIsVolumePreserving(self):
    """Tests that construct() is volume preserving when size is a power of 2."""
    for wavelet_type in wavelet.generate_filters():
      sz = (1, 4, 4)
      num_levels = 2
      # Construct the Jacobian of construct().
      im = np.float32(np.random.uniform(0., 1., sz))
      im_ph = tf.placeholder(tf.float32, im.shape)
      jacobian = []
      vec = lambda x: tf.reshape(x, [-1])
      for d in range(im.size):
        jacobian.append(
            vec(
                tf.gradients(
                    vec(
                        wavelet.flatten(
                            wavelet.construct(im_ph, num_levels,
                                              wavelet_type)))[d], im_ph)[0]))
      jacobian = tf.stack(jacobian, 1)
      with self.session() as sess:
        jacobian = sess.run(jacobian, {im_ph: im})
      # Assert that the determinant of the Jacobian is close to 1.
      det = np.linalg.det(jacobian)
      self.assertAllClose(det, 1., atol=1e-5, rtol=1e-5)

  def _load_golden_data(self):
    """Loads golden data: an RGBimage and its CDF9/7 decomposition.

    This golden data was produced by running the code from
    https://www.getreuer.info/projects/wavelet-cdf-97-implementation
    on a test image.

    Returns:
      A tuple containing and image, its decomposition, and its wavelet type.
    """
    with util.get_resource_as_file(
        'robust_loss/data/wavelet_golden.mat') as golden_filename:
      data = scipy.io.loadmat(golden_filename)
    im = np.float32(data['I_color'])
    pyr_true = data['pyr_color'][0, :].tolist()
    for i in range(len(pyr_true) - 1):
      pyr_true[i] = tuple(pyr_true[i].flatten())
    pyr_true = tuple(pyr_true)
    wavelet_type = 'CDF9/7'
    return im, pyr_true, wavelet_type

  def testConstructMatchesGoldenData(self):
    """Tests construct() against golden data."""
    im, pyr_true, wavelet_type = self._load_golden_data()
    pyr = wavelet.construct(im, len(pyr_true) - 1, wavelet_type)
    with self.session() as sess:
      pyr = sess.run(pyr)
    self._assert_pyramids_close(pyr, pyr_true, 1e-5)

  def testCollapseMatchesGoldenData(self):
    """Tests collapse() against golden data."""
    im, pyr_true, wavelet_type = self._load_golden_data()
    recon = wavelet.collapse(pyr_true, wavelet_type)
    with self.session() as sess:
      recon = sess.run(recon)
    self.assertAllClose(recon, im, atol=1e-5, rtol=1e-5)

  def testVisualizeMatchesGoldenData(self):
    """Tests visualize() (and implicitly flatten())."""
    _, pyr, _ = self._load_golden_data()
    vis = wavelet.visualize(pyr)
    with self.session() as sess:
      vis = sess.run(vis)
    golden_vis_filename = 'robust_loss/data/wavelet_vis_golden.png'
    vis_true = np.asarray(
        PIL.Image.open(util.get_resource_filename(golden_vis_filename)))
    # Allow for some slack as quantization may exaggerate some errors.
    self.assertAllClose(vis_true, vis, atol=1., rtol=0)

  def testAccurateRoundTripWithSmallRandomImages(self):
    """Tests that collapse(construct(x)) == x for x = [1, k, k], k in [1, 4]."""
    with self.session() as sess:
      for wavelet_type in wavelet.generate_filters():
        for width in range(1, 5):
          sz = [1, width, width]
          num_levels = sess.run(wavelet.get_max_num_levels(sz))
          im = np.random.uniform(size=sz)

          pyr = wavelet.construct(im, num_levels, wavelet_type)
          recon = wavelet.collapse(pyr, wavelet_type)
          pyr, recon = sess.run((pyr, recon))
          self.assertAllClose(recon, im, atol=1e-8, rtol=1e-8)

  def testAccurateRoundTripWithLargeRandomImages(self):
    """Tests that collapse(construct(x)) == x for large random x's."""
    with self.session() as sess:
      for wavelet_type in wavelet.generate_filters():
        for _ in range(4):
          num_levels = np.int32(np.ceil(4 * np.random.uniform()))
          sz_clamp = 2**(num_levels - 1) + 1
          sz = np.maximum(
              np.int32(
                  np.ceil(np.array([2, 32, 32]) * np.random.uniform(size=3))),
              np.array([0, sz_clamp, sz_clamp]))
          im = np.random.uniform(size=sz)
          pyr = wavelet.construct(im, num_levels, wavelet_type)
          recon = wavelet.collapse(pyr, wavelet_type)
          pyr, recon = sess.run([pyr, recon])
          self.assertAllClose(recon, im, atol=1e-8, rtol=1e-8)

  def testDecompositionIsNonRedundant(self):
    """Test that wavelet construction is not redundant.

    If the wavelet decompositon is not redundant, then we should be able to
    1) Construct a wavelet decomposition
    2) Alter a single coefficient in the decomposition
    3) Collapse that decomposition into an image and back
    and the two wavelet decompositions should be the same.
    """
    with self.session() as sess:
      for wavelet_type in wavelet.generate_filters():
        for _ in range(4):
          # Construct an image and a wavelet decomposition of it.
          num_levels = np.int32(np.ceil(4 * np.random.uniform()))
          sz_clamp = 2**(num_levels - 1) + 1
          sz = np.maximum(
              np.int32(
                  np.ceil(np.array([2, 32, 32]) * np.random.uniform(size=3))),
              np.array([0, sz_clamp, sz_clamp]))
          im = np.random.uniform(size=sz)
          pyr = wavelet.construct(im, num_levels, wavelet_type)
          pyr = sess.run(pyr)

        # Pick a coefficient at random in the decomposition to alter.
        d = np.int32(np.floor(np.random.uniform() * len(pyr)))
        v = np.random.uniform()
        if d == (len(pyr) - 1):
          if np.prod(pyr[d].shape) > 0:
            c, i, j = np.int32(
                np.floor(np.array(np.random.uniform(size=3)) *
                         pyr[d].shape)).tolist()
            pyr[d][c, i, j] = v
        else:
          b = np.int32(np.floor(np.random.uniform() * len(pyr[d])))
          if np.prod(pyr[d][b].shape) > 0:
            c, i, j = np.int32(
                np.floor(np.array(np.random.uniform(size=3)) *
                         pyr[d][b].shape)).tolist()
            pyr[d][b][c, i, j] = v

        # Collapse and then reconstruct the wavelet decomposition, and check
        # that it is unchanged.
        recon = wavelet.collapse(pyr, wavelet_type)
        pyr_again = wavelet.construct(recon, num_levels, wavelet_type)
        pyr_again = sess.run(pyr_again)
        self._assert_pyramids_close(pyr, pyr_again, 1e-8)

  def testUpsampleAndDownsampleAreTransposes(self):
    """Tests that _downsample() is the transpose of _upsample()."""
    n = 8
    x = np.random.uniform(size=(1, n, 1))
    for f_len in range(1, 5):
      f = np.random.uniform(size=f_len)
      for shift in [0, 1]:

        # We're only testing the resampling operators away from the boundaries,
        # as this test appears to fail in the presences of boundary conditions.
        # TODO(barron): Figure out what's happening and make this test more
        # thorough, and then set range1 = range(d), range2 = range(d//2) and
        # have this code depend on util.compute_jacobian().
        range1 = np.arange(f_len // 2 + 1, n - (f_len // 2 + 1))
        range2 = np.arange(f_len // 4, n // 2 - (f_len // 4))

        y = wavelet._downsample(x, f, 0, shift)

        # Construct the jacobian of _downsample().
        x_ph = tf.placeholder(tf.float32, x.shape)
        jacobian_down = []
        vec = lambda x: tf.reshape(x, [-1])
        for d in range2:
          jacobian_down.append(
              vec(
                  tf.gradients(
                      vec(wavelet._downsample(x_ph, f, 0, shift))[d], x_ph)[0]))
        jacobian_down = tf.stack(jacobian_down, 1)
        with tf.Session() as sess:
          jacobian_down, y = sess.run([jacobian_down, y], {x_ph: x})

        # Construct the jacobian of _upsample().
        y_ph = tf.placeholder(tf.float32, y.shape)
        jacobian_up = []
        vec = lambda x: tf.reshape(x, [-1])
        for d in range1:
          jacobian_up.append(
              vec(
                  tf.gradients(
                      vec(wavelet._upsample(y_ph, x.shape[1:], f, 0, shift))[d],
                      y_ph)[0]))
        jacobian_up = tf.stack(jacobian_up, 1)
        with tf.Session() as sess:
          jacobian_up = sess.run(jacobian_up, {y_ph: y})

        # Test that the jacobian of _downsample() is close to the transpose of
        # the jacobian of _upsample().
        self.assertAllClose(
            jacobian_down[range1, :],
            np.transpose(jacobian_up[range2, :]),
            atol=1e-6,
            rtol=1e-6)

  def _construct_preserves_dtype(self, float_dtype):
    """Checks that construct()'s output has the same precision as its input."""
    x = float_dtype(np.random.normal(size=(3, 16, 16)))
    for wavelet_type in wavelet.generate_filters():
      with self.session():
        y = wavelet.flatten(wavelet.construct(x, 3, wavelet_type)).eval()
      self.assertDTypeEqual(y, float_dtype)

  def testConstructPreservesDtypeSingle(self):
    self._construct_preserves_dtype(np.float32)

  def testConstructPreservesDtypeDouble(self):
    self._construct_preserves_dtype(np.float64)

  def _collapse_preserves_dtype(self, float_dtype):
    """Checks that collapse()'s output has the same precision as its input."""
    n = 16
    x = []
    for n in [8, 4, 2]:
      band = []
      for _ in range(3):
        band.append(float_dtype(np.random.normal(size=(3, n, n))))
      x.append(band)
    x.append(float_dtype(np.random.normal(size=(3, n, n))))
    for wavelet_type in wavelet.generate_filters():
      with self.session():
        y = wavelet.collapse(x, wavelet_type).eval()
      self.assertDTypeEqual(y, float_dtype)

  def testCollapsePreservesDtypeSingle(self):
    self._collapse_preserves_dtype(np.float32)

  def testCollapsePreservesDtypeDouble(self):
    self._collapse_preserves_dtype(np.float64)

  def testRescaleOneIsANoOp(self):
    """Tests that rescale(x, 1) = x."""
    im = np.random.uniform(size=(2, 32, 32))
    pyr = wavelet.construct(im, 4, 'LeGall5/3')
    pyr_rescaled = wavelet.rescale(pyr, 1.)
    with self.session() as sess:
      pyr, pyr_rescaled = sess.run([pyr, pyr_rescaled])
    self._assert_pyramids_close(pyr, pyr_rescaled, 1e-8)

  def testRescaleDoesNotAffectTheFirstLevel(self):
    """Tests that rescale(x, s)[0] = x[0] for any s."""
    im = np.random.uniform(size=(2, 32, 32))
    pyr = wavelet.construct(im, 4, 'LeGall5/3')
    pyr_rescaled = wavelet.rescale(pyr, np.exp(np.random.normal()))
    with self.session() as sess:
      pyr, pyr_rescaled = sess.run([pyr, pyr_rescaled])
    self._assert_pyramids_close(pyr[0:1], pyr_rescaled[0:1], 1e-8)

  def testRescaleOneHalfIsNormalized(self):
    """Tests that rescale(construct(k), 0.5)[-1] = k for constant image k."""
    for num_levels in range(5):
      k = np.random.uniform()
      im = k * np.ones((2, 32, 32))
      pyr = wavelet.construct(im, num_levels, 'LeGall5/3')
      pyr_rescaled = wavelet.rescale(pyr, 0.5)
      with self.session() as sess:
        pyr_rescaled = sess.run(pyr_rescaled)
      self.assertAllClose(
          pyr_rescaled[-1],
          k * np.ones_like(pyr_rescaled[-1]),
          atol=1e-8,
          rtol=1e-8)

  def testRescaleAndUnrescaleReproducesInput(self):
    """Tests that rescale(rescale(x, k), 1/k) = x."""
    im = np.random.uniform(size=(2, 32, 32))
    scale_base = np.exp(np.random.normal())
    pyr = wavelet.construct(im, 4, 'LeGall5/3')
    pyr_rescaled = wavelet.rescale(pyr, scale_base)
    pyr_recon = wavelet.rescale(pyr_rescaled, 1. / scale_base)
    self._assert_pyramids_close(pyr, pyr_recon, 1e-8)


if __name__ == '__main__':
  tf.test.main()
