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

"""Implements wavelets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

# The four filters used to define a wavelet decomposition:
# (analysis, synthesis) x (highpass, lowpass)
Filters = collections.namedtuple(
    'Filters', ['analysis_lo', 'analysis_hi', 'synthesis_lo', 'synthesis_hi'])

# How we're storing the non-redundant parts of a wavelet filter bank. The
# center of the filter is at the beginning, and the rest is symmetrized.
HalfFilters = collections.namedtuple('HalfFilters', ['lo', 'hi'])


def generate_filters(wavelet_type=None):
  """Generates the analysis and synthesis filters for a kind of wavelet.

  Currently only supports wavelet types where all filters have an odd length.
  TODO(barron): Generalize this to even filters as well, and support Haar and
  Debauchies wavelets.

  Args:
    wavelet_type: A string encoding the type of wavelet filters to return. This
      string is used as a key to the `supported_half_filters` dict in the code
      below, and so the string must be a valid key.

  Returns:
    If `wavelet_type` is not provided as input, this function returns a list of
    valid values for `wavelet_type`. If `wavelet_type` is a supported string,
    this function returns the wavelet type's `Filters` object.
  """
  supported_half_filters = {
      # CDF 9/7 filters from "Biorthogonal bases of compactly supported
      # wavelets", Cohen et al., Commun. Pure Appl. Math 1992.
      'CDF9/7':
          HalfFilters(
              lo=np.array([
                  +0.852698679009,
                  +0.377402855613,
                  -0.110624404418,
                  -0.023849465020,
                  +0.037828455507
              ]),
              hi=np.array([
                  +0.788485616406,
                  -0.418092273222,
                  -0.040689417609,
                  +0.064538882629
              ])),
      # Le Gall 5/3 filters (sometimes called CDF 5/3 filters).
      'LeGall5/3':
          HalfFilters(
              lo=np.array([0.75, 0.25, -0.125]) * np.sqrt(2.),
              hi=np.array([1., -0.5]) / np.sqrt(2.)),
  }  # pyformat: disable

  if wavelet_type is None:
    return list(supported_half_filters.keys())

  half_filters = supported_half_filters[wavelet_type]

  # Returns [f(n-1), ..., f(2), f(1), f(0), f(1), f(2), ... f(n-1)].
  mirror = lambda f: np.concatenate([f[-1:0:-1], f])
  # Makes an n-length vector containing [1, -1, 1, -1, 1, ... ].
  alternating_sign = lambda n: (-1)**np.arange(n)
  analysis_lo = mirror(half_filters.lo)
  analysis_hi = mirror(half_filters.hi)
  synthesis_lo = analysis_hi * mirror(alternating_sign(len(half_filters.hi)))
  synthesis_hi = analysis_lo * mirror(alternating_sign(len(half_filters.lo)))

  return Filters(
      analysis_lo=analysis_lo,
      analysis_hi=analysis_hi,
      synthesis_lo=synthesis_lo,
      synthesis_hi=synthesis_hi)


def pad_reflecting(x, padding_below, padding_above, axis):
  """Pads `x` with reflecting conditions above and/or below it along some axis.

  Pads `x` with reflecting conditions for `padding_below` entries below the
  tensor and `padding_above` entries above the tensor in the direction along
  `axis`. This is like using tf.pad(x, --, 'REFLECT'), except that this code
  allows for an unbounded number of reflections while tf.pad() only supports
  one reflection. Multiple reflections are necessary for for wavelet
  decompositions to guard against cases where the wavelet filters are larger
  than the input tensor along `axis`, which happens often at coarse scales.
  Note that "reflecting" boundary conditions are different from "symmetric"
  boundary conditions, in that it doesn't repeat the last element:
  reflect([A, B, C, D], 2) = [C, B, A, B, C, D, C, B]
  symmet.([A, B, C, D], 2) = [B, A, A, B, C, D, D, C]

  Args:
    x: The tensor to be padded with reflecting boundary conditions.
    padding_below: The number of elements being padded below the tensor.
    padding_above: The number of elements being padded above the tensor.
    axis: The axis in x in which padding will be performed.

  Returns:
    `x` padded according to `padding_below` and `padding_above` along `axis`
    with reflecting boundary conditions.
  """
  if not isinstance(padding_below, int):
    raise ValueError(
        'Expected `padding_below` of type int, but is of type {}'.format(
            type(padding_below)))
  if not isinstance(padding_above, int):
    raise ValueError(
        'Expected `padding_above` of type int, but is of type {}'.format(
            type(padding_above)))
  if not isinstance(axis, int):
    raise ValueError('Expected `axis` of type int, but is of type {}'.format(
        type(axis)))
  if not (axis >= 0 and axis < len(x.shape)):
    raise ValueError('Expected `axis` in [0, {}], but is = {}'.format(
        len(x.shape) - 1, axis))

  if padding_below == 0 and padding_above == 0:
    return tf.convert_to_tensor(x)
  n = tf.shape(x)[axis]
  # `i' contains the indices of the output padded tensor in the frame of
  # reference of the input tensor.
  i = tf.range(-padding_below, n + padding_above, dtype=tf.int32)
  # `j` contains the indices of the input tensor corresponding to the output
  # padded tensor.
  i_mod = tf.math.mod(i, tf.maximum(1, 2 * (n - 1)))
  j = tf.minimum(2 * (n - 1) - i_mod, i_mod)
  return tf.gather(x, j, axis=axis)


def _check_resample_inputs(x, f, direction, shift):
  """Checks the inputs to _downsample() and _upsample()."""
  if len(x.shape) != 3:
    raise ValueError('Expected `x` to have rank 3, but is of size {}'.format(
        x.shape))
  if len(f.shape) != 1:
    raise ValueError('Expected `f` to have rank 1, but is of size {}'.format(
        f.shape))
  if not (direction == 0 or direction == 1):
    raise ValueError(
        'Expected `direction` to be 0 or 1, but is {}'.format(direction))
  if not (shift == 0 or shift == 1):
    raise ValueError('Expected `shift` to be 0 or 1, but is {}'.format(shift))


def _downsample(x, f, direction, shift):
  """Downsample by a factor of 2 using reflecting boundary conditions.

  This function convolves `x` with filter `f` with reflecting boundary
  conditions, and then decimates by a factor of 2. This is usually done to
  downsample `x`, assuming `f` is some smoothing filter, but will also be used
  for wavelet transformations in which `f` is not a smoothing filter.

  Args:
    x: The input tensor (numpy or TF), of size (num_channels, width, height).
    f: The input filter, which must be an odd-length 1D numpy array.
    direction: The spatial direction in [0, 1] along which `x` will be convolved
      with `f` and then decimated. Because `x` has a batch/channels dimension,
      `direction` == 0 corresponds to downsampling along axis 1 in `x`, and
      `direction` == 1 corresponds to downsampling along axis 2 in `x`.
    shift: A shift amount in [0, 1] by which `x` will be shifted along the axis
      specified by `direction` before filtering.

  Returns:
    `x` convolved with `f` along the spatial dimension `direction` with
    reflection boundary conditions with an offset of `shift`.
  """
  _check_resample_inputs(x, f, direction, shift)
  assert_ops = [tf.Assert(tf.equal(tf.rank(f), 1), [tf.rank(f)])]
  with tf.control_dependencies(assert_ops):
    # The above and below padding amounts are different so as to support odd
    # and even length filters. An odd-length filter of length n causes a padding
    # of (n-1)/2 on both sides, while an even-length filter will pad by one less
    # below than above.
    x_padded = pad_reflecting(x, (len(f) - 1) // 2, len(f) // 2, direction + 1)
    if direction == 0:
      x_padded = x_padded[:, shift:, :]
      f_ex = f[:, tf.newaxis]
      strides = [1, 2, 1, 1]
    elif direction == 1:
      x_padded = x_padded[:, :, shift:]
      f_ex = f[tf.newaxis, :]
      strides = [1, 1, 2, 1]
    y = tf.nn.conv2d(x_padded[:, :, :, tf.newaxis],
                     tf.cast(f_ex, x.dtype)[:, :, tf.newaxis, tf.newaxis],
                     strides, 'VALID')[:, :, :, 0]
    return y


def _upsample(x, up_sz, f, direction, shift):
  """Upsample by a factor of 2 using transposed reflecting boundary conditions.

  This function undecimates `x` along the axis specified by `direction` and then
  convolves it with filter `f`, thereby upsampling it to have a size of `up_sz`.
  This function is a bit awkward, as it's written to be the transpose of
  _downsample(), which uses reflecting boundary conditions. As such, this
  function approximates *the transpose of reflecting boundary conditions*, which
  is not the same as reflecting boundary conditions.
  TODO(barron): Write out the true transpose of reflecting boundary conditions.

  Args:
    x: The input tensor (numpy or TF), of size (num_channels, width, height).
    up_sz: A tuple of ints of size (upsampled_width, upsampled_height). Care
      should be taken by the caller to match the upsampled_width/height with the
      input width/height along the axis that isn't being upsampled.
    f: The input filter, which must be an odd-length 1D numpy array.
    direction: The spatial direction in [0, 1] along which `x` will be convolved
      with `f` after being undecimated. Because `x` has a batch/channels
      dimension, `direction` == 0 corresponds to downsampling along axis 1 in
      `x`, and `direction` == 1 corresponds to downsampling along axis 2 in `x`.
    shift: A shift amount in [0, 1] by which `x` will be shifted along the axis
      specified by `direction` after undecimating.

  Returns:
    `x` undecimated and convolved with `f` along the spatial dimension
    `direction` with transposed reflection boundary conditions with an offset of
    `shift`, to match size `up_sz`.
  """
  _check_resample_inputs(x, f, direction, shift)
  assert_ops = [tf.Assert(tf.equal(tf.rank(f), 1), [tf.rank(f)])]
  with tf.control_dependencies(assert_ops):
    # Undecimate `x` by a factor of 2 along `direction`, by stacking it with
    # and tensor of all zeros along the right axis and then reshaping it such
    # that the zeros are interleaved.
    if direction == 0:
      sz_ex = tf.shape(x) * [1, 2, 1]
    elif direction == 1:
      sz_ex = tf.shape(x) * [1, 1, 2]
    if shift == 0:
      x_and_zeros = [x, tf.zeros_like(x)]
    elif shift == 1:
      x_and_zeros = [tf.zeros_like(x), x]
    x_undecimated = tf.reshape(tf.stack(x_and_zeros, direction + 2), sz_ex)
    # Ensure that `x_undecimated` has a size of `up_sz`, by slicing and padding
    # as needed.
    x_undecimated = x_undecimated[:, 0:up_sz[0], 0:up_sz[1]]
    x_undecimated = tf.pad(x_undecimated,
                           [[0, 0], [0, up_sz[0] - tf.shape(x_undecimated)[1]],
                            [0, up_sz[1] - tf.shape(x_undecimated)[2]]])

    # Pad `x_undecimated` with reflection boundary conditions.
    x_padded = pad_reflecting(x_undecimated,
                              len(f) // 2, (len(f) - 1) // 2, direction + 1)
    # Convolved x_undecimated with a flipped version of f.
    f_ex = tf.expand_dims(f[::-1], 1 - direction)
    y = tf.nn.conv2d(x_padded[:, :, :, tf.newaxis],
                     tf.cast(f_ex, x.dtype)[:, :, tf.newaxis, tf.newaxis],
                     [1, 1, 1, 1], 'VALID')[:, :, :, 0]
    return y


def get_max_num_levels(sz):
  """Returns the maximum number of levels that construct() can support.

  Args:
    sz: A tuple of ints representing some input size (batch, width, height).

  Returns:
    The maximum value for num_levels, when calling construct(im, num_levels),
    assuming `sz` is the shape of `im`.
  """
  min_sz = tf.minimum(sz[1], sz[2])
  log2 = lambda x: tf.math.log(tf.cast(x, tf.float32)) / tf.math.log(2.)
  max_num_levels = tf.cast(tf.math.ceil(log2(tf.maximum(1, min_sz))), tf.int32)
  return max_num_levels


def construct(im, num_levels, wavelet_type):
  """Constructs a wavelet decomposition of an image.

  Args:
    im: A numpy or TF tensor of single or double precision floats of size
      (batch_size, width, height)
    num_levels: The number of levels (or scales) of the wavelet decomposition to
      apply. A value of 0 returns a "wavelet decomposition" that is just the
      image.
    wavelet_type: The kind of wavelet to use, see generate_filters().

  Returns:
    A wavelet decomposition of `im` that has `num_levels` levels (not including
    the coarsest residual level) and is of type `wavelet_type`. This
    decomposition is represented as a tuple of 3-tuples, with the final element
    being a tensor:
      ((band00, band01, band02), (band10, band11, band12), ..., resid)
    Where band** and resid are TF tensors. Each element of these nested tuples
    is of shape [batch_size, width * 2^-(level+1), height * 2^-(level+1)],
    though the spatial dimensions may be off by 1 if width and height are not
    factors of 2. The residual image is of the same (rough) size as the last set
    of bands. The floating point precision of these tensors matches that of
    `im`.
  """
  if len(im.shape) != 3:
    raise ValueError(
        'Expected `im` to have a rank of 3, but is of size {}'.format(im.shape))
  if num_levels == 0:
    return (tf.convert_to_tensor(im),)
  max_num_levels = get_max_num_levels(tf.shape(im))
  assert_ops = [
      tf.Assert(
          tf.greater_equal(max_num_levels, num_levels),
          [tf.shape(im), num_levels, max_num_levels])
  ]
  with tf.control_dependencies(assert_ops):
    filters = generate_filters(wavelet_type)
    pyr = []
    for _ in range(num_levels):
      hi = _downsample(im, filters.analysis_hi, 0, 1)
      lo = _downsample(im, filters.analysis_lo, 0, 0)
      pyr.append((_downsample(hi, filters.analysis_hi, 1, 1),
                  _downsample(lo, filters.analysis_hi, 1, 1),
                  _downsample(hi, filters.analysis_lo, 1, 0)))
      im = _downsample(lo, filters.analysis_lo, 1, 0)
    pyr.append(im)
    pyr = tuple(pyr)
    return pyr


def collapse(pyr, wavelet_type):
  """Collapses a wavelet decomposition made by construct() back into an image.

  Args:
    pyr: A numpy or TF tensor of single or double precision floats containing a
      wavelet decomposition produced by construct().
    wavelet_type: The kind of wavelet to use, see generate_filters().

  Returns:
    A TF tensor of a reconstructed image, with the same floating point precision
    as the element of `pyr`, and the same size as the image that was used to
    create `pyr`.
  """
  if not isinstance(pyr, (list, tuple)):
    raise ValueError('Expected `pyr` to be a list or tuple, but is a {}'.format(
        type(pyr)))

  filters = generate_filters(wavelet_type)
  im = pyr[-1]
  num_levels = len(pyr) - 1
  for d in range(num_levels - 1, -1, -1):
    if not isinstance(pyr[d], (list, tuple)):
      raise ValueError(
          'Expected `pyr[{}]` to be a list or tuple, but is a {}'.format(
              d, type(pyr[d])))
    if len(pyr[d]) != 3:
      raise ValueError(
          'Expected `pyr[{}]` to have length 3, but has length {}'.format(
              d, len(pyr[d])))

    hi_hi, hi_lo, lo_hi = pyr[d]
    up_sz = (tf.shape(hi_lo)[1] + tf.shape(lo_hi)[1],
             tf.shape(lo_hi)[2] + tf.shape(hi_lo)[2])
    lo_sz = (tf.shape(im)[1], up_sz[1])
    hi_sz = (tf.shape(hi_hi)[1], up_sz[1])
    im = (
        _upsample(
            _upsample(im, lo_sz, filters.synthesis_lo, 1, 0) +
            _upsample(hi_lo, lo_sz, filters.synthesis_hi, 1, 1),
            up_sz, filters.synthesis_lo, 0, 0) +
        _upsample(
            _upsample(lo_hi, hi_sz, filters.synthesis_lo, 1, 0) +
            _upsample(hi_hi, hi_sz, filters.synthesis_hi, 1, 1),
            up_sz, filters.synthesis_hi, 0, 1))  # pyformat: disable
  return im


def rescale(pyr, scale_base):
  """Rescale a wavelet decomposition `pyr` by `scale_base`^level.

  Args:
    pyr: A wavelet decomposition produced by construct().
    scale_base: The base of the exponentiation used for the per-level scaling.

  Returns:
    pyr where each level has been scaled by `scale_base`^level. The first
    level is 0 and is therefore not scaled.
  """
  pyr_norm = []
  for d in range(len(pyr) - 1):
    level_norm = []
    scale = scale_base**d
    for b in range(3):
      level_norm.append(pyr[d][b] * scale)
    pyr_norm.append(level_norm)
  d = len(pyr) - 1
  scale = scale_base**d
  pyr_norm.append(pyr[d] * scale)
  return pyr_norm


def flatten(pyr):
  """Flattens a wavelet decomposition into an image-like single Tensor.

  construct() produces wavelet decompositions in the form of nested tuples,
  which is convenient for TensorFlow. But Wavelets are often formatted like:
  _____________________________________
  |        |        |                 |
  | Resid  | Band11 |                 |
  |________|________|      Band01     |
  |        |        |                 |
  | Band12 | Band10 |                 |
  |________|________|_________________|
  |                 |                 |
  |                 |                 |
  |     Band02      |      Band00     |
  |                 |                 |
  |                 |                 |
  |_________________|_________________|
  This function turns our internal representation into this more-standard
  representation. This is useful for visualization and for integration into
  loss functions.

  Args:
    pyr: A pyramid-formatted wavelet decomposition produced by construct()

  Returns:
    A (num_channels, width, height) representation of pyr, as described above.
  """
  flat = pyr[-1]
  for d in range(len(pyr) - 2, -1, -1):
    flat = tf.concat([
        tf.concat([flat, pyr[d][1]], axis=2),
        tf.concat([pyr[d][2], pyr[d][0]], axis=2)], axis=1)  # pyformat: disable
  return flat


def visualize(pyr, percentile=99.):
  """Visualizes a wavelet decomposition produced by construct().

  Args:
    pyr: A wavelet decomposition produced by construct(),
    percentile: The percentile of the deviation for each (non-residual) wavelet
      band to be clamped by before normalization. Seeting this to 100 causes
      visualization to clamp to the maximum deviation, which preserves the
      entire dynamic range but may make subtle details hard to see. A value of
      99 (the default) will clip away the 1% largest-magnitude values in each
      band.

  Returns:
    An image (a TF tensor of uint8's) of shape (width, height, num_channels).
    Note that the input wavelet decomposition was produced from an image of
    shape (num_channels, width, height) --- this function permutes the ordering
    to what is expected in a planar image.
  """
  vis_pyr = []
  for d in range(len(pyr) - 1):
    vis_band = []
    for b in range(3):
      band = pyr[d][b]
      max_mag = tfp.stats.percentile(tf.abs(band), percentile)
      vis_band.append(0.5 * (1. + tf.clip_by_value(band / max_mag, -1., 1.)))
    vis_pyr.append(vis_band)
  d = len(pyr) - 1
  resid = pyr[d]
  resid_norm = (resid - tf.reduce_min(resid)) / (
      tf.reduce_max(resid) - tf.reduce_min(resid))
  vis_pyr.append(resid_norm)
  vis = tf.cast(
      tf.math.round(255. * tf.transpose(flatten(vis_pyr), [1, 2, 0])), tf.uint8)
  return vis
