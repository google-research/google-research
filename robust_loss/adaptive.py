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

r"""Implements the adaptive form of the loss.

You should only use this function if 1) you want the loss to change it's shape
during training (otherwise use general.py) or 2) you want to impose the loss on
a wavelet or DCT image representation, a only this function has easy support for
that.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
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


def lossfun(x,
            alpha_lo=0.001,
            alpha_hi=1.999,
            alpha_init=None,
            scale_lo=1e-5,
            scale_init=1.,
            **kwargs):
  """Computes the adaptive form of the robust loss on a matrix.

  This function behaves differently from general.lossfun() and
  distribution.nllfun(), which are "stateless", allow the caller to specify the
  shape and scale of the loss, and allow for arbitrary sized inputs. This
  function only allows for rank-2 inputs for the residual `x`, and expects that
  `x` is of the form [batch_index, dimension_index]. This function then
  constructs free parameters (TF variables) that define the alpha and scale
  parameters for each dimension of `x`, such that all alphas are in
  (`alpha_lo`, `alpha_hi`) and all scales are in (`scale_lo`, Infinity).
  The assumption is that `x` is, say, a matrix where x[i,j] corresponds to a
  pixel at location j for image i, with the idea being that all pixels at
  location j should be modeled with the same shape and scale parameters across
  all images in the batch. This function also returns handles to the scale and
  shape parameters being optimized over, mostly for debugging and introspection.
  If the user wants to fix alpha or scale to be a constant, this can be done by
  setting alpha_lo=alpha_hi or scale_lo=scale_init respectively.

  Args:
    x: The residual for which the loss is being computed. Must be a rank-2
      tensor, where the innermost dimension is the batch index, and the
      outermost dimension corresponds to different "channels", where this
      function will assign each channel its own variable shape (alpha) and scale
      parameters that are constructed as TF variables and can be optimized over.
      Must be a TF tensor or numpy array of single or double precision floats.
      The precision of `x` will determine the precision of the latent variables
      used to model scale and alpha internally.
    alpha_lo: The lowest possible value for loss's alpha parameters, must be >=
      0 and a scalar. Should probably be in (0, 2).
    alpha_hi: The highest possible value for loss's alpha parameters, must be >=
      alpha_lo and a scalar. Should probably be in (0, 2).
    alpha_init: The value that the loss's alpha parameters will be initialized
      to, must be in (`alpha_lo`, `alpha_hi`), unless `alpha_lo` == `alpha_hi`
      in which case this will be ignored. Defaults to (`alpha_lo` + `alpha_hi`)
      / 2
    scale_lo: The lowest possible value for the loss's scale parameters. Must be
      > 0 and a scalar. This value may have more of an effect than you think, as
      the loss is unbounded as scale approaches zero (say, at a delta function).
    scale_init: The initial value used for the loss's scale parameters. This
      also defines the zero-point of the latent representation of scales, so SGD
      may cause optimization to gravitate towards producing scales near this
      value.
    **kwargs: Arguments to be passed to the underlying distribution.nllfun().

  Returns:
    A tuple of the form (`loss`, `alpha`, `scale`).

    `loss`: a TF tensor of the same type and shape as input `x`, containing
    the loss at each element of `x` as a function of `x`, `alpha`, and
    `scale`. These "losses" are actually negative log-likelihoods (as produced
    by distribution.nllfun()) and so they are not actually bounded from below
    by zero. You'll probably want to minimize their sum or mean.

    `scale`: a TF tensor of the same type as x, of size (1, x.shape[1]), as we
    construct a scale variable for each dimension of `x` but not for each
    batch element. This contains the current estimated scale parameter for
    each dimension, and will change during optimization.

    `alpha`: a TF tensor of the same type as x, of size (1, x.shape[1]), as we
    construct an alpha variable for each dimension of `x` but not for each
    batch element. This contains the current estimated alpha parameter for
    each dimension, and will change during optimization.

  Raises:
    ValueError: If any of the arguments are invalid.
  """
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

  float_dtype = x.dtype
  assert_ops = [tf.Assert(tf.equal(tf.rank(x), 2), [tf.rank(x)])]
  with tf.control_dependencies(assert_ops):
    if alpha_lo == alpha_hi:
      # If the range of alphas is a single item, then we just fix `alpha` to be
      # a constant.
      alpha = tf.tile(
          tf.cast(alpha_lo, float_dtype)[tf.newaxis, tf.newaxis],
          (1, x.shape[1]))
    else:
      # Otherwise we construct a "latent" alpha variable and define `alpha`
      # As an affine function of a sigmoid on that latent variable, initialized
      # such that `alpha` starts off as `alpha_init`.
      if alpha_init is None:
        alpha_init = (alpha_lo + alpha_hi) / 2.
      latent_alpha_init = util.inv_affine_sigmoid(
          alpha_init, lo=alpha_lo, hi=alpha_hi)
      latent_alpha = tf.compat.v1.get_variable(
          'LatentAlpha',
          initializer=tf.fill((1, x.shape[1]),
                              tf.cast(latent_alpha_init, dtype=float_dtype)))
      alpha = util.affine_sigmoid(latent_alpha, lo=alpha_lo, hi=alpha_hi)
    scale = _construct_scale(x, scale_lo, scale_init, float_dtype)
    loss = distribution.nllfun(x, alpha, scale, **kwargs)
    return loss, alpha, scale


def lossfun_students(x, scale_lo=1e-5, scale_init=1.):
  """A variant of lossfun() that uses the NLL of a Student's t-distribution.

  Args:
    x: The residual for which the loss is being computed. Must be a rank-2
      tensor, where the innermost dimension is the batch index, and the
      outermost dimension corresponds to different "channels", where this
      function will assign each channel its own variable shape (log-df) and
      scale parameters that are constructed as TF variables and can be optimized
      over. Must be a TF tensor or numpy array of single or double precision
      floats. The precision of `x` will determine the precision of the latent
      variables used to model scale and log-df internally.
    scale_lo: The lowest possible value for the loss's scale parameters. Must be
      > 0 and a scalar. This value may have more of an effect than you think, as
      the loss is unbounded as scale approaches zero (say, at a delta function).
    scale_init: The initial value used for the loss's scale parameters. This
      also defines the zero-point of the latent representation of scales, so SGD
      may cause optimization to gravitate towards producing scales near this
      value.

  Returns:
    A tuple of the form (`loss`, `log_df`, `scale`).

    `loss`: a TF tensor of the same type and shape as input `x`, containing
    the loss at each element of `x` as a function of `x`, `log_df`, and
    `scale`. These "losses" are actually negative log-likelihoods (as produced
    by distribution.nllfun()) and so they are not actually bounded from below
    by zero. You'll probably want to minimize their sum or mean.

    `scale`: a TF tensor of the same type as x, of size (1, x.shape[1]), as we
    construct a scale variable for each dimension of `x` but not for each
    batch element. This contains the current estimated scale parameter for
    each dimension, and will change during optimization.

    `log_df`: a TF tensor of the same type as x, of size (1, x.shape[1]), as we
    construct an log-DF variable for each dimension of `x` but not for each
    batch element. This contains the current estimated log(degrees-of-freedom)
    parameter for each dimension, and will change during optimization.

  Raises:
    ValueError: If any of the arguments are invalid.
  """
  _check_scale(scale_lo, scale_init)

  float_dtype = x.dtype
  assert_ops = [tf.Assert(tf.equal(tf.rank(x), 2), [tf.rank(x)])]
  with tf.control_dependencies(assert_ops):
    log_df = tf.compat.v1.get_variable(
        name='LogDf', initializer=tf.zeros((1, x.shape[1]), float_dtype))
    scale = _construct_scale(x, scale_lo, scale_init, float_dtype)
    loss = util.students_t_nll(x, tf.math.exp(log_df), scale)
    return loss, log_df, scale


def image_lossfun(x,
                  color_space='YUV',
                  representation='CDF9/7',
                  wavelet_num_levels=5,
                  wavelet_scale_base=1,
                  use_students_t=False,
                  summarize_loss=True,
                  **kwargs):
  """Computes the adaptive form of the robust loss on a set of images.

  This function is a wrapper around lossfun() above. Like lossfun(), this
  function is not "stateless" --- it requires inputs of a specific shape and
  size, and constructs TF variables describing each non-batch dimension in `x`.
  `x` is expected to be the difference between sets of RGB images, and the other
  arguments to this function allow for the color space and spatial
  representation of `x` to be changed before the loss is imposed. By default,
  this function uses a CDF9/7 wavelet decomposition in a YUV color space, which
  often works well. This function also returns handles to the scale and
  shape parameters (both in the shape of images) being optimized over,
  and summarizes both parameters in TensorBoard.

  Args:
    x: A set of image residuals for which the loss is being computed. Must be a
      rank-4 tensor of size (num_batches, width, height, color_channels). This
      is assumed to be a set of differences between RGB images.
    color_space: The color space that `x` will be transformed into before
      computing the loss. Must be 'RGB' (in which case no transformation is
      applied) or 'YUV' (in which case we actually use a volume-preserving
      scaled YUV colorspace so that log-likelihoods still have meaning, see
      util.rgb_to_syuv()). Note that changing this argument does not change the
      assumption that `x` is the set of differences between RGB images, it just
      changes what color space `x` is converted to from RGB when computing the
      loss.
    representation: The spatial image representation that `x` will be
      transformed into after converting the color space and before computing the
      loss. If this is a valid type of wavelet according to
      wavelet.generate_filters() then that is what will be used, but we also
      support setting this to 'DCT' which applies a 2D DCT to the images, and to
      'PIXEL' which applies no transformation to the image, thereby causing the
      loss to be imposed directly on pixels.
    wavelet_num_levels: If `representation` is a kind of wavelet, this is the
      number of levels used when constructing wavelet representations. Otherwise
      this is ignored. Should probably be set to as large as possible a value
      that is supported by the input resolution, such as that produced by
      wavelet.get_max_num_levels().
    wavelet_scale_base: If `representation` is a kind of wavelet, this is the
      base of the scaling used when constructing wavelet representations.
      Otherwise this is ignored. For image_lossfun() to be volume preserving (a
      useful property when evaluating generative models) this value must be ==
      1. If the goal of this loss isn't proper statistical modeling, then
      modifying this value (say, setting it to 0.5 or 2) may significantly
      improve performance.
    use_students_t: If true, use the NLL of Student's T-distribution instead
      of the adaptive loss. This causes all `alpha_*` inputs to be ignored.
    summarize_loss: Whether or not to make TF summaries describing the latent
      state of the loss function. True by default.
    **kwargs: Arguments to be passed to the underlying lossfun().

  Returns:
    A tuple of the form (`loss`, `alpha`, `scale`). If use_students_t == True,
    then `log(df)` is returned instead of `alpha`.

    `loss`: a TF tensor of the same type and shape as input `x`, containing
    the loss at each element of `x` as a function of `x`, `alpha`, and
    `scale`. These "losses" are actually negative log-likelihoods (as produced
    by distribution.nllfun()) and so they are not actually bounded from below
    by zero. You'll probably want to minimize their sum or mean.

    `scale`: a TF tensor of the same type as x, of size
      (width, height, color_channels),
    as we construct a scale variable for each spatial and color dimension of `x`
    but not for each batch element. This contains the current estimated scale
    parameter for each dimension, and will change during optimization.

    `alpha`: a TF tensor of the same type as x, of size
      (width, height, color_channels),
    as we construct an alpha variable for each spatial and color dimension of
    `x` but not for each batch element. This contains the current estimated
    alpha parameter for each dimension, and will change during optimization.

  Raises:
    ValueError: if `color_space` of `representation` are unsupported color
      spaces or image representations, respectively.
  """
  color_spaces = ['RGB', 'YUV']
  if color_space not in color_spaces:
    raise ValueError('`color_space` must be in {}, but is {!r}'.format(
        color_spaces, color_space))
  representations = wavelet.generate_filters() + ['DCT', 'PIXEL']
  if representation not in representations:
    raise ValueError('`representation` must be in {}, but is {!r}'.format(
        representations, representation))
  assert_ops = [tf.Assert(tf.equal(tf.rank(x), 4), [tf.rank(x)])]
  with tf.control_dependencies(assert_ops):
    if color_space == 'YUV':
      x = util.rgb_to_syuv(x)
    # If `color_space` == 'RGB', do nothing.

    # Reshape `x` from
    #   (num_batches, width, height, num_channels) to
    #   (num_batches * num_channels, width, height)
    _, width, height, num_channels = x.shape.as_list()
    x_stack = tf.reshape(tf.transpose(x, (0, 3, 1, 2)), (-1, width, height))

    # Turn each channel in `x_stack` into the spatial representation specified
    # by `representation`.
    if representation in wavelet.generate_filters():
      x_stack = wavelet.flatten(
          wavelet.rescale(
              wavelet.construct(x_stack, wavelet_num_levels, representation),
              wavelet_scale_base))
    elif representation == 'DCT':
      x_stack = util.image_dct(x_stack)
    # If `representation` == 'PIXEL', do nothing.

    # Reshape `x_stack` from
    #   (num_batches * num_channels, width, height) to
    #   (num_batches, num_channels * width * height)
    x_mat = tf.reshape(
        tf.transpose(
            tf.reshape(x_stack, [-1, num_channels, width, height]),
            [0, 2, 3, 1]), [-1, width * height * num_channels])

    # Set up the adaptive loss. Note, if `use_students_t` == True then
    # `alpha_mat` actually contains "log(df)" values.
    if use_students_t:
      loss_mat, alpha_mat, scale_mat = lossfun_students(x_mat, **kwargs)
    else:
      loss_mat, alpha_mat, scale_mat = lossfun(x_mat, **kwargs)

    # Reshape the loss function's outputs to have the shapes as the input.
    loss = tf.reshape(loss_mat, [-1, width, height, num_channels])
    alpha = tf.reshape(alpha_mat, [width, height, num_channels])
    scale = tf.reshape(scale_mat, [width, height, num_channels])

    if summarize_loss:
      # Summarize the `alpha` and `scale` parameters as images (normalized to
      # [0, 1]) and histograms.
      # Note that these may look unintuitive unless the colorspace is 'RGB' and
      # the image representation is 'PIXEL', as the image summaries (like most
      # images) are rendered as RGB pixels.
      alpha_min = tf.reduce_min(alpha)
      alpha_max = tf.reduce_max(alpha)
      tf.summary.image(
          'robust/alpha',
          (alpha[tf.newaxis] - alpha_min) / (alpha_max - alpha_min + 1e-10))
      tf.summary.histogram('robust/alpha', alpha)
      log_scale = tf.math.log(scale)
      log_scale_min = tf.reduce_min(log_scale)
      log_scale_max = tf.reduce_max(log_scale)
      tf.summary.image('robust/log_scale',
                       (log_scale[tf.newaxis] - log_scale_min) /
                       (log_scale_max - log_scale_min + 1e-10))
      tf.summary.histogram('robust/log_scale', log_scale)

    return loss, alpha, scale
