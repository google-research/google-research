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

"""Metrics utils files to compute certain similarity metrics."""

from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS


def VerifyCompatibleImageShapes(img1, img2):
  """Checks if two image tensors are compatible for metric computation.

  This function checks if two sets of images have ranks at least 3, and if the
  last three dimensions match.

  Args:
    img1: The first images tensor.
    img2: The second images tensor.

  Returns:
    A tuple of the first tensor shape, the second tensor shape, and a list of
    tf.Assert() implementing the checks.

  Raises:
    ValueError: when static shape check fails.
  """
  shape1 = img1.shape.with_rank_at_least(3)
  shape2 = img2.shape.with_rank_at_least(3)

  if shape1.ndims is not None and shape2.ndims is not None:
    for dim1, dim2 in zip(reversed(shape1[:-3]), reversed(shape2[:-3])):
      # For TF V1 compatibility.
      try:
        dim1 = dim1.value
        dim2 = dim2.value
      except AttributeError:
        pass

      if not (dim1 in (None, 1) or dim2 in (None, 1) or dim1 == dim2):
        raise ValueError('Two images are not compatible: %s and %s' %
                         (shape1, shape2))
  else:
    raise ValueError('The two images do not have a defined shape.')

  # Now assign shape tensors.
  shape1, shape2 = tf.shape_n([img1, img2])

  checks = []
  checks.append(
      tf.Assert(
          tf.greater_equal(tf.size(shape1), 3), [shape1, shape2], summarize=10))
  checks.append(
      tf.Assert(
          tf.reduce_all(tf.equal(shape1[-3:], shape2[-3:])), [shape1, shape2],
          summarize=10))
  return shape1, shape2, checks


def _SSIMHelper(x, y, reducer, max_val, compensation=1.0):
  r"""Helper function to SSIM.

  Arguments:
    x: first set of images.
    y: first set of images.
    reducer: Function that computes 'local' averages from set of images. For
      non-covolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
      for convolutional version, this is usually tf.nn.avg_pool or tf.nn.conv2d
      with weighted-sum kernel.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    compensation: Compensation factor. See above.

  Returns:
    A pair containing the luminance measure and the contrast-structure measure.
  """
  c1 = (0.01 * max_val)**2
  c2 = (0.03 * max_val)**2

  # SSIM luminance measure is
  # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
  mean0 = reducer(x)
  mean1 = reducer(y)
  num0 = mean0 * mean1 * 2.0
  den0 = tf.square(mean0) + tf.square(mean1)
  luminance = (num0 + c1) / (den0 + c1)

  # SSIM contrast-structure measure is
  #   (2 * cov_xy + c2) / (cov_xx + cov_yy + c2).
  # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
  #   cov_xy = \sum_i w_i (x_i - mu_x) (y_i - mu_y)
  #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
  num1 = reducer(x * y) * 2.0
  den1 = reducer(tf.square(x) + tf.square(y))
  c2 *= compensation
  cs = (num1 - num0 + c2) / (den1 - den0 + c2)

  # SSIM score is the product of the luminance and contrast-structure measures.
  return luminance, cs


def SSIMWithoutFilter(a,
                      b,
                      max_val=255.0,
                      filter_size=(8, 8),
                      strides=None,
                      spatial_average=True,
                      channel_average=True):
  """Computes unfiltered SSIM index between a and b per channel.

  Arguments:
    a: First set of patches.
    b: Second set of patches.
    max_val: The dynamic range (i.e., the difference between the maximum
      possible allowed value and the minimum allowed value).
    filter_size: Determines the moving average filter size to aggregate the SSIM
      over. Must be a sequence of length two: [filter_height, filter_width].
    strides: The strides of the moving average filter. Must be None or a
      sequence of length two: [row_stride, col_stride]. If None, defaults to
        `filter_size`.
    spatial_average: If True, return the mean value across space. Otherwise,
      return the full 2D spatial map.
    channel_average: If True, return the mean value across channels. Otherwise,
      return SSIM per channel.

  Returns:
    The SSIM index for each individual element in the batch.
    For color images, SSIM is averaged after computed in each channel
    separately.

  Raises:
    ValueError: if a and b don't have the broadcastable shapes, or the ranks of
      a and b are not at least 3.
  """
  # Enforce rank and shape checks.
  shape1, _, checks = VerifyCompatibleImageShapes(a, b)
  with tf.control_dependencies(checks):
    a = tf.identity(a)

  if strides is None:
    strides = filter_size

  n = float(np.prod(filter_size))
  kernel = tf.fill(
      dims=list(filter_size) + [shape1[-1], 1],
      value=tf.constant(1 / n, dtype=a.dtype))
  strides = [1] + list(strides) + [1]

  def reducer(x):  # pylint: disable=invalid-name
    shape = tf.shape(x)
    # DepthwiseConv2D takes rank 4 tensors. Flatten leading dimensions.
    x = tf.reshape(x, shape=tf.concat([[-1], shape[-3:]], 0))
    y = tf.nn.depthwise_conv2d(x, kernel, strides=strides, padding='VALID')
    return tf.reshape(y, tf.concat([shape[:-3], tf.shape(y)[1:]], 0))

  compensation = (n - 1) / n
  luminance, cs = _SSIMHelper(a, b, reducer, max_val, compensation)
  ssim = luminance * cs

  reduce_axis = [-3, -2] if spatial_average else []
  if channel_average:
    reduce_axis.append(-1)
  if reduce_axis:
    ssim = tf.reduce_mean(ssim, axis=reduce_axis)
  return ssim


def GradientDifferenceLoss(img1,
                           img2,
                           dist_func=tf.square,
                           reduce_func=tf.reduce_sum,
                           name=None):
  """Returns an op that calculates loss between image gradients.

  This function assumes that `img1` and `img2` are image batches,
  i.e. [batch_size, row, col, channels].

  Arguments:
    img1: First image batch.
    img2: Second image batch.
    dist_func: A TensorFlow op to apply to edge map differences (e.g. tf.square
      for L2 or tf.abs for L1).
    reduce_func: A TensorFlow op to reduce edge map distances into a single loss
      per image pair (e.g. tf.reduce_sum for a gradient or tf.reduce_mean for a
      per-pixel average score).
    name: Namespace in which to embed the computation.

  Returns:
    A tensor with size [batch_size] containing the finite difference edge loss
    for each image pair in the batch.
  """
  with tf.name_scope(name, 'GDL', [img1, img2]):
    _, _, checks = VerifyCompatibleImageShapes(img1, img2)
    dy1, dx1 = tf.image.image_gradients(img1)
    dy2, dx2 = tf.image.image_gradients(img2)
    diff = dist_func(dy1 - dy2) + dist_func(dx1 - dx2)
    loss = reduce_func(diff, list(range(-3, 0)))
    with tf.control_dependencies(checks):
      return tf.identity(loss)


def PSNR(a, b, max_val=255.0, name=None):
  """Returns the Peak Signal-to-Noise Ratio between a and b.

  Arguments:
    a: first set of images.
    b: second set of images.
    max_val: the dynamic range of the images (i.e., the difference between the
      maximum the and minimum allowed values).
    name: namespace to embed the computation in.

  Returns:
    The scalar PSNR between a and b. The shape of the returned tensor is
    [batch_size, 1].
  """
  with tf.name_scope(name, 'PSNR', [a, b]):
    psnr = tf.image.psnr(a, b, max_val=max_val, name=name)

    _, _, checks = VerifyCompatibleImageShapes(a, b)
    with tf.control_dependencies(checks):
      return tf.identity(psnr)


def ClippedPSNR(img1,
                img2,
                min_val=0.0,
                max_val=255.0,
                clip=True,
                quantize=True,
                max_psnr=100.0,
                name=None):
  """Return average Clipped PSNR between `a` and `b`.

  Arguments:
    img1: first set of images.
    img2: second set of images.
    min_val: smallest valid value for a pixel.
    max_val: largest valid value for a pixel.
    clip: If True, pixel values will be clipped to [`min_value`, `max_value`].
    quantize: If True, pixel values will be rounded before calculating PSNR.
    max_psnr: If not None, PSNR will be clipped by this value before rounding.
    name: namespace to embed the computation in.

  Returns:
    PSNR between img1 and img2 or average PSNR if input is a batch.
  """
  with tf.name_scope(name, 'clipped_psnr', [img1, img2]):
    if quantize:
      img1 = tf.round(img1)
      img2 = tf.round(img2)
    if clip:
      img1 = tf.clip_by_value(img1, min_val, max_val)
      img2 = tf.clip_by_value(img2, min_val, max_val)
    value_range = max_val - min_val
    psnr = PSNR(img1, img2, max_val=value_range)
    if max_psnr is not None:
      psnr = tf.minimum(psnr, max_psnr)
    return tf.reduce_mean(psnr)


def SobelEdgeLoss(img1, img2, dist_func=tf.square, reduce_func=tf.reduce_sum):
  """Returns an op that calculates Sobel edge loss between two images.

  Arguments:
    img1: First image batch.
    img2: Second image batch.
    dist_func: A TensorFlow op to apply to edge map differences (e.g. tf.square
      for L2 or tf.abs for L1).
    reduce_func: A TensorFlow op to reduce edge map distances into a single loss
      per image pair (e.g. tf.reduce_sum for a gradient or tf.reduce_mean for a
      per-pixel average score).

  Returns:
    A tensor with size [batch_size] containing the Sobel edge loss for each
    image pair in the batch.
  """

  _, _, checks = VerifyCompatibleImageShapes(img1, img2)

  # Sobel tensor has shape [batch_size, h, w, d, num_kernels].
  sobel1 = tf.image.sobel_edges(img1)
  sobel2 = tf.image.sobel_edges(img2)
  diff = dist_func(sobel1 - sobel2)

  # To match GDL, sum across dy and dx regardless of reduce_func.
  edge_maps = tf.reduce_sum(diff, axis=-1)

  # Reduce over all dimensions except batch_size.
  loss = reduce_func(edge_maps, list(range(-3, 0)))
  with tf.control_dependencies(checks):
    return tf.identity(loss)
