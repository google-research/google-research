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

r"""Implements the adaptive form of the loss.

You should only use this function if 1) you want the loss to change it's shape
during training (otherwise use general.py) or 2) you want to impose the loss on
a wavelet or DCT image representation, a only this function has easy support for
that.
"""

import numpy as np
import tensorflow.compat.v2 as tf
from robust_loss import distribution
from robust_loss import util
from robust_loss import wavelet


def _check_scale(scale_lo, scale_init):
  """Helper function for checking `scale_lo` and `scale_init`."""
  if not np.isscalar(scale_lo):
    raise ValueError('`scale_lo` must be a scalar, but is of type {}'.format(
        type(scale_lo)))
  if not np.isscalar(scale_init):
    raise ValueError('`scale_init` must be a scalar, but is of type {}'.format(
        type(scale_init)))
  if not scale_lo > 0:
    raise ValueError('`scale_lo` must be > 0, but is {}'.format(scale_lo))
  if not scale_init >= scale_lo:
    raise ValueError('`scale_init` = {} must be >= `scale_lo` = {}'.format(
        scale_init, scale_lo))


def _construct_scale(x, scale_lo, scale_init, float_dtype):
  """Helper function for constructing scale variables."""
  if scale_lo == scale_init:
    # If the difference between the minimum and initial scale is zero, then
    # we just fix `scale` to be a constant.
    scale = tf.tile(
        tf.cast(scale_init, float_dtype)[tf.newaxis, tf.newaxis],
        (1, x.shape[1]))
  else:
    # Otherwise we construct a "latent" scale variable and define `scale`
    # As an affine function of a softplus on that latent variable.
    latent_scale = tf.compat.v1.get_variable(
        'LatentScale', initializer=tf.zeros((1, x.shape[1]), float_dtype))
    scale = util.affine_softplus(latent_scale, lo=scale_lo, ref=scale_init)
  return scale


class AdaptiveLossFunction(tf.Module):
  """Implements the adaptive form of the general loss for matrix inputs.

  This loss behaves differently from general.lossfun() and
  distribution.nllfun(), which are "stateless", allow the caller to specify
  the shape and scale of the loss, and allow for arbitrary sized inputs. This
  loss only allows for rank-2 inputs, and expects those inputs to be of the form
  [batch_index, dimension_index]. This module constructs the free parameters
  (TF variables) that define the alpha and scale parameters for each
  of the `num_channels` dimension of the input, such that all alphas are
  in (`alpha_lo`, `alpha_hi`) and all scales are in (`scale_lo`, Infinity).
  The assumption is that the input residual `x` is, say, a matrix where x[i,j]
  corresponds to a pixel at location j for image i, with the idea being that
  all pixels at location j should be modeled with the same shape and scale
  parameters across all images in the batch. If the user wants to fix alpha or
  scale to be a constant, this can be done by setting `alpha_lo=alpha_hi` or
  `scale_lo=scale_init` respectively.
  """

  def __init__(self,
               num_channels,
               float_dtype,
               alpha_lo=0.001,
               alpha_hi=1.999,
               alpha_init=None,
               scale_lo=1e-5,
               scale_init=1.0,
               name=None):
    """Constructs the loss function.

    Args:
      num_channels: the number of different "channels" for the adaptive loss
        function, where each channel will be assigned its own shape (alpha) and
        scale parameters that are constructed as variables and can be optimized
        over.
      float_dtype: The expected numerical precision of the input, which will
        also determine the precision of the latent variables used to model scale
        and alpha internally.
      alpha_lo: The lowest possible value for loss's alpha parameters, must be
        >=0 and a scalar. Should probably be in (0, 2).
      alpha_hi: The highest possible value for loss's alpha parameters, must be
        >=alpha_lo and a scalar. Should probably be in (0, 2).
      alpha_init: The value that the loss's alpha parameters will be initialized
        to, must be in (`alpha_lo`, `alpha_hi`), unless `alpha_lo==alpha_hi` in
        which case this will be ignored. Defaults to (`alpha_lo+alpha_hi)/2`.
      scale_lo: The lowest possible value for the loss's scale parameters. Must
        be > 0 and a scalar. This value may have more of an effect than you
        think, as the loss is unbounded as scale approaches zero.
      scale_init: The initial value used for the loss's scale parameters. This
        also defines the zero-point of the latent representation of scales, so
        SGD may cause optimization to gravitate towards producing scales near
        this value.
      name: The name of the module.

    Raises:
      ValueError: If any of the arguments are invalid.
    """
    super(AdaptiveLossFunction, self).__init__(name=name)
    _check_scale(scale_lo, scale_init)
    if not np.isscalar(alpha_lo):
      raise ValueError('`alpha_lo` must be a scalar, but is of type {}'.format(
          type(alpha_lo)))
    if not np.isscalar(alpha_hi):
      raise ValueError('`alpha_hi` must be a scalar, but is of type {}'.format(
          type(alpha_hi)))
    if alpha_init is not None and not np.isscalar(alpha_init):
      raise ValueError(
          '`alpha_init` must be None or a scalar, but is of type {}'.format(
              type(alpha_init)))
    if not alpha_lo >= 0:
      raise ValueError('`alpha_lo` must be >= 0, but is {}'.format(alpha_lo))
    if not alpha_hi >= alpha_lo:
      raise ValueError('`alpha_hi` = {} must be >= `alpha_lo` = {}'.format(
          alpha_hi, alpha_lo))
    if alpha_init is not None and alpha_lo != alpha_hi:
      if not (alpha_init > alpha_lo and alpha_init < alpha_hi):
        raise ValueError(
            '`alpha_init` = {} must be in (`alpha_lo`, `alpha_hi`) = ({} {})'
            .format(alpha_init, alpha_lo, alpha_hi))

    if alpha_lo != alpha_hi:
      # If alpha isn't constant, construct a "latent" alpha variable.
      if alpha_init is None:
        alpha_init = (alpha_lo + alpha_hi) / 2.
      latent_alpha_init = (
          util.inv_affine_sigmoid(alpha_init, lo=alpha_lo, hi=alpha_hi))
      self._latent_alpha = tf.Variable(
          tf.fill((1, num_channels),
                  tf.cast(latent_alpha_init, dtype=float_dtype)),
          name='LatentAlpha')

    if scale_lo != scale_init:
      # If shape isn't constant, construct a "latent" scale variable.
      self._latent_scale = tf.Variable(
          tf.zeros((1, num_channels), float_dtype), name='LatentScale')

    self._num_channels = num_channels
    self._float_dtype = tf.dtypes.as_dtype(float_dtype)
    self._alpha_lo = alpha_lo
    self._alpha_hi = alpha_hi
    self._scale_lo = scale_lo
    self._scale_init = scale_init
    self._distribution = distribution.Distribution()

  def alpha(self):
    """Returns the loss's current alpha ("shape") parameters.

    Returns:
      a TF tensor of size (1, self._num_channels) and type self._float_dtype,
      containing the current estimated alpha parameter for each channel,
      which will presumably change during optimization. This tensor is a
      function of the latent alpha tensor being optimized over, and is not a
      TF variable itself.
    """
    if self._alpha_lo == self._alpha_hi:
      # If the range of alphas is a single item, then we just fix `alpha` to be
      # a constant.
      return tf.tile(
          tf.cast(self._alpha_lo, self._float_dtype)[tf.newaxis, tf.newaxis],
          (1, self._num_channels))
    else:
      return util.affine_sigmoid(
          self._latent_alpha, lo=self._alpha_lo, hi=self._alpha_hi)

  def scale(self):
    """Returns the loss's current scale parameters.

    Returns:
      a TF tensor of size (1, self._num_channels) and type self._float_dtype,
      containing the current estimated scale parameter for each channel,
      which will presumably change during optimization. This tensor is a
      function of the latent scale tensor being optimized over, and is not a
      TF variable itself.
    """
    if self._scale_lo == self._scale_init:
      # If the difference between the minimum and initial scale is zero, then
      # we just fix `scale` to be a constant.
      return tf.tile(
          tf.cast(self._scale_init, self._float_dtype)[tf.newaxis, tf.newaxis],
          (1, self._num_channels))
    else:
      return util.affine_softplus(
          self._latent_scale, lo=self._scale_lo, ref=self._scale_init)

  def __call__(self, x):
    """Evaluates the loss function on a matrix.

    Args:
      x: The residual for which the loss is being computed. Must be a rank-2
        tensor, where the innermost dimension is the batch index, and the
        outermost dimension corresponds to different "channels" and whose size
        must be equal to `num_channels'.

    Returns:
      A TF tensor of the same type and shape as input `x`, containing
      the loss at each element of `x` as a function of `x`, `alpha`, and
      `scale`. These "losses" are actually negative log-likelihoods (as produced
      by distribution.nllfun()) and so they are not actually bounded from below
      by zero --- it is okay if they go negative! You'll probably want to
      minimize their sum or mean.
    """
    x = tf.convert_to_tensor(x)
    tf.debugging.assert_rank(x, 2)
    tf.debugging.assert_same_float_dtype([x], self._float_dtype)
    with tf.control_dependencies([
        tf.Assert(
            tf.equal(x.shape[1], self._num_channels),
            [x.shape[1], self._num_channels])
    ]):
      return self._distribution.nllfun(x, self.alpha(), self.scale())


class StudentsTLossFunction(tf.Module):
  """Implements the NLL of a t-distribution for matrix inputs.

  This interface is similar to AdaptiveLossFunction, except that no
  functionality is provided to constrain the degrees-of-freedom parameter of the
  t-distribution, unlike the `alpha' parameter that governs the shape of the
  adaptive loss function above.
  """

  def __init__(self,
               num_channels,
               float_dtype,
               scale_lo=1e-5,
               scale_init=1.0,
               name=None):
    super(StudentsTLossFunction, self).__init__(name=name)
    _check_scale(scale_lo, scale_init)

    self._log_df = tf.Variable(
        tf.zeros((1, num_channels), float_dtype), name='LogDf')

    if scale_lo != scale_init:
      # Construct a "latent" scale variable.
      self._latent_scale = tf.Variable(
          tf.zeros((1, num_channels), float_dtype), name='LatentScale')

    self._num_channels = num_channels
    self._float_dtype = tf.dtypes.as_dtype(float_dtype)
    self._scale_lo = scale_lo
    self._scale_init = scale_init

  def df(self):
    """Returns the loss's current degrees-of-freedom parameters.

    Returns:
      a TF tensor of size (1, self._num_channels) and type self._float_dtype,
      containing the current estimated DF parameter for each channel,
      which will presumably change during optimization. This tensor is a
      function of the latent scale tensor being optimized over, and is not a
      TF variable itself.
    """
    return tf.math.exp(self._log_df)

  def scale(self):
    """Returns the loss's current scale parameters.

    Returns:
      a TF tensor of size (1, self._num_channels) and type self._float_dtype,
      containing the current estimated scale parameter for each channel,
      which will presumably change during optimization. This tensor is a
      function of the latent scale tensor being optimized over, and is not a
      TF variable itself.
    """
    if self._scale_lo == self._scale_init:
      # If the difference between the minimum and initial scale is zero, then
      # we just fix `scale` to be a constant.
      return tf.tile(
          tf.cast(self._scale_init, self._float_dtype)[tf.newaxis, tf.newaxis],
          (1, self._num_channels))
    else:
      return util.affine_softplus(
          self._latent_scale, lo=self._scale_lo, ref=self._scale_init)

  def __call__(self, x):
    """Evaluates the loss function on a matrix.

    Args:
      x: The residual for which the loss is being computed. Must be a rank-2
        tensor, where the innermost dimension is the batch index, and the
        outermost dimension corresponds to different "channels" and whose size
        must be equal to `num_channels'.

    Returns:
      A TF tensor of the same type and shape as input `x`, containing
      the loss at each element of `x` as a function of `x`, `df`, and
      `scale`. These "losses" are actually negative log-likelihoods.
    """
    x = tf.convert_to_tensor(x)
    tf.debugging.assert_rank(x, 2)
    tf.debugging.assert_same_float_dtype([x], self._float_dtype)
    with tf.control_dependencies([
        tf.Assert(
            tf.equal(x.shape[1], self._num_channels),
            [x.shape[1], self._num_channels])
    ]):
      return util.students_t_nll(x, self.df(), self.scale())


class AdaptiveImageLossFunction(tf.Module):
  """Implements the adaptive form of the general loss for image inputs.

  This class is a wrapper around AdaptiveLossFunction, but designed to for
  image inputs instead of matrix inputs. Like AdaptiveLossFunction, this loss
  is not "stateless" -- it requires inputs of a specific shape and size, and
  constructs variables describing each non-batch dimension in the input
  residuals. By default, this function uses a CDF9/7 wavelet decomposition in
  a YUV color space, which often works well.
  """

  def __init__(self,
               image_size,
               float_dtype,
               color_space='YUV',
               representation='CDF9/7',
               wavelet_num_levels=5,
               wavelet_scale_base=1,
               use_students_t=False,
               summarize_loss=True,
               name=None,
               **kwargs):
    """Constructs the a loss function instance.

    Args:
      image_size: The size (width, height, num_channels) of the input images.
      float_dtype: The expected numerical precision of the input, which will
        also determine the precision of the latent variables used to model scale
        and alpha internally.
      color_space: The color space that input images will be transformed into
        before computing the loss. Must be 'RGB' (in which case no
        transformation is applied) or 'YUV' (in which case we actually use a
        volume-preserving scaled YUV colorspace so that log-likelihoods still
        have meaning, see util.rgb_to_syuv()). If `image_size[2]` is anything
        other than 3, color_space must be 'RGB'.
      representation: The spatial image representation that inputs will be
        transformed into after converting the color space and before computing
        the loss. If this is a valid type of wavelet according to
        wavelet.generate_filters() then that is what will be used, but we also
        support setting this to 'DCT' which applies a 2D DCT to the images, and
        to 'PIXEL' which applies no transformation to the image, thereby causing
        the loss to be imposed directly on pixels.
      wavelet_num_levels: If `representation` is a kind of wavelet, this is the
        number of levels used when constructing wavelet representations.
        Otherwise this is ignored. Should probably be set to as large as
        possible a value that is supported by the input resolution, such as that
        produced by wavelet.get_max_num_levels().
      wavelet_scale_base: If `representation` is a kind of wavelet, this is the
        base of the scaling used when constructing wavelet representations.
        Otherwise this is ignored. For image_lossfun() to be volume preserving
        (a useful property when evaluating generative models) this value must be
        == 1. If the goal of this loss isn't proper statistical modeling, then
        modifying this value (say, setting it to 0.5 or 2) may significantly
        improve performance.
      use_students_t: If true, use the NLL of Student's T-distribution instead
        of the adaptive loss. This causes all `alpha_*` inputs to be ignored.
      summarize_loss: Whether or not to make TF summaries describing the latent
        state of the loss function. True by default.
      name: The name of the module.
      **kwargs: Arguments to be passed to the underlying lossfun().

    Raises:
      ValueError: if `color_space` of `representation` are unsupported color
        spaces or image representations, respectively.
    """
    super(AdaptiveImageLossFunction, self).__init__(name=name)
    color_spaces = ['RGB', 'YUV']
    if color_space not in color_spaces:
      raise ValueError('`color_space` must be in {}, but is {!r}'.format(
          color_spaces, color_space))
    representations = wavelet.generate_filters() + ['DCT', 'PIXEL']
    if representation not in representations:
      raise ValueError('`representation` must be in {}, but is {!r}'.format(
          representations, representation))

    assert len(image_size) == 3
    if image_size[2] != 3:
      assert color_space == 'RGB'

    # Set up the adaptive loss.
    num_channels = np.prod(image_size)
    if use_students_t:
      self._lossfun = StudentsTLossFunction(num_channels, float_dtype, **kwargs)
    else:
      self._lossfun = AdaptiveLossFunction(num_channels, float_dtype, **kwargs)

    self._image_size = image_size
    self._float_dtype = tf.dtypes.as_dtype(float_dtype)
    self._use_students_t = use_students_t
    self._color_space = color_space
    self._representation = representation
    self._wavelet_num_levels = wavelet_num_levels
    self._wavelet_scale_base = wavelet_scale_base
    self._summarize_loss = summarize_loss

  def alpha(self):
    """Returns an image of alphas.

    Returns: a tensor of size self._image_size (width, height, num_channnels).
    This contains the current estimated alpha parameter for each dimension,
    which may change during optimization. Not that these alpha variables may be
    not be in the native per-pixel RGB space of the inputs to the loss, but
    are visualized as images regardless.
    """
    assert not self._use_students_t
    return tf.reshape(self._lossfun.alpha(), self._image_size)

  def df(self):
    """Returns an image of degrees-of-freedom, for the t-distribution model.

    Returns: a tensor of size self._image_size (width, height, num_channnels).
    This contains the current estimated scale parameter for each dimension,
    which may change during optimization. Not that these scale variables may be
    not be in the native per-pixel RGB space of the inputs to the loss, but
    are visualized as images regardless.
    """
    assert self._use_students_t
    return tf.reshape(self._lossfun.df(), self._image_size)

  def scale(self):
    """Returns an image of scales.

    Returns: a tensor of size self._image_size (width, height, num_channnels).
    This contains the current estimated scale parameter for each dimension,
    which may change during optimization. Not that these scale variables may be
    not be in the native per-pixel RGB space of the inputs to the loss, but
    are visualized as images regardless.
    """
    return tf.reshape(self._lossfun.scale(), self._image_size)

  def __call__(self, x):
    """Evaluates the loss function on a batch of images.

    Args:
      x: The image residuals for which the loss is being computed, which is
        expected to be the differences between RGB images. Must be a rank-4
        tensor, where the innermost dimension is the batch index, and the
        remaining 3 dimension corresponds `self._image_size` (two spatial, one
        channel).

    Returns:
      A TF tensor of the same type and shape as input `x`, containing
      the loss at each element of `x` as a function of `x`, `alpha`, and
      `scale`. These "losses" are actually negative log-likelihoods (as produced
      by distribution.nllfun()) and so they are not actually bounded from below
      by zero --- it is okay if they go negative! You'll probably want to
      minimize their sum or mean.
    """
    x = tf.convert_to_tensor(x)
    tf.debugging.assert_rank(x, 4)
    with tf.control_dependencies([
        tf.Assert(
            tf.reduce_all(tf.equal(x.shape[1:], self._image_size)),
            [x.shape[1:], self._image_size])
    ]):
      if self._color_space == 'YUV':
        x = util.rgb_to_syuv(x)
      # If `color_space` == 'RGB', do nothing.

      # Reshape `x` from
      #   (num_batches, width, height, num_channels) to
      #   (num_batches * num_channels, width, height)
      width, height, num_channels = self._image_size
      x_stack = tf.reshape(
          tf.transpose(x, perm=(0, 3, 1, 2)), (-1, width, height))

      # Turn each channel in `x_stack` into the spatial representation specified
      # by `representation`.
      if self._representation in wavelet.generate_filters():
        x_stack = wavelet.flatten(
            wavelet.rescale(
                wavelet.construct(x_stack, self._wavelet_num_levels,
                                  self._representation),
                self._wavelet_scale_base))
      elif self._representation == 'DCT':
        x_stack = util.image_dct(x_stack)
      # If `representation` == 'PIXEL', do nothing.

      # Reshape `x_stack` from
      #   (num_batches * num_channels, width, height) to
      #   (num_batches, num_channels * width * height)
      x_mat = tf.reshape(
          tf.transpose(
              tf.reshape(x_stack, [-1, num_channels, width, height]),
              perm=[0, 2, 3, 1]), [-1, width * height * num_channels])

      # Set up the adaptive loss. Note, if `use_students_t` == True then
      # `alpha_mat` actually contains "log(df)" values.
      loss_mat = self._lossfun(x_mat)

      # Reshape the loss function's outputs to have the shapes as the input.
      loss = tf.reshape(loss_mat, [-1, width, height, num_channels])

      if self._summarize_loss:
        # Summarize the `alpha` and `scale` parameters as images (normalized to
        # [0, 1]) and histograms.
        # Note that these may look unintuitive unless the colorspace is 'RGB'
        # and the image representation is 'PIXEL', as the image summaries
        # (like most images) are rendered as RGB pixels.
        log_scale = tf.math.log(self.scale())
        log_scale_min = tf.reduce_min(log_scale)
        log_scale_max = tf.reduce_max(log_scale)
        tf.summary.image('robust/log_scale',
                         (log_scale[tf.newaxis] - log_scale_min) /
                         (log_scale_max - log_scale_min + 1e-10))
        tf.summary.histogram('robust/log_scale', log_scale)

        if not self._use_students_t:
          alpha = self.alpha()
          alpha_min = tf.reduce_min(alpha)
          alpha_max = tf.reduce_max(alpha)
          tf.summary.image('robust/alpha', (alpha[tf.newaxis] - alpha_min) /
                           (alpha_max - alpha_min + 1e-10))
          tf.summary.histogram('robust/alpha', alpha)

      return loss
