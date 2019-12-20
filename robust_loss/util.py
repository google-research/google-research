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

"""Helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.compat.v1 as tf



def log_safe(x):
  """The same as tf.math.log(x), but clamps the input to prevent NaNs."""
  return tf.math.log(tf.minimum(x, tf.cast(3e37, x.dtype)))


def log1p_safe(x):
  """The same as tf.math.log1p(x), but clamps the input to prevent NaNs."""
  return tf.math.log1p(tf.minimum(x, tf.cast(3e37, x.dtype)))


def exp_safe(x):
  """The same as tf.math.exp(x), but clamps the input to prevent NaNs."""
  return tf.math.exp(tf.minimum(x, tf.cast(87.5, x.dtype)))


def expm1_safe(x):
  """The same as tf.math.expm1(x), but clamps the input to prevent NaNs."""
  return tf.math.expm1(tf.minimum(x, tf.cast(87.5, x.dtype)))


def inv_softplus(y):
  """The inverse of tf.nn.softplus()."""
  return tf.where(y > 87.5, y, tf.math.log(tf.math.expm1(y)))


def logit(y):
  """The inverse of tf.nn.sigmoid()."""
  return -tf.math.log(1. / y - 1.)


def affine_sigmoid(real, lo=0, hi=1):
  """Maps reals to (lo, hi), where 0 maps to (lo+hi)/2."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  alpha = tf.sigmoid(real) * (hi - lo) + lo
  return alpha


def inv_affine_sigmoid(alpha, lo=0, hi=1):
  """The inverse of affine_sigmoid(., lo, hi)."""
  if not lo < hi:
    raise ValueError('`lo` (%g) must be < `hi` (%g)' % (lo, hi))
  real = logit((alpha - lo) / (hi - lo))
  return real


def affine_softplus(real, lo=0, ref=1):
  """Maps real numbers to (lo, infinity), where 0 maps to ref."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  shift = inv_softplus(tf.cast(1., real.dtype))
  scale = (ref - lo) * tf.nn.softplus(real + shift) + lo
  return scale


def inv_affine_softplus(scale, lo=0, ref=1):
  """The inverse of affine_softplus(., lo, ref)."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  shift = inv_softplus(tf.cast(1., scale.dtype))
  real = inv_softplus((scale - lo) / (ref - lo)) - shift
  return real


def students_t_nll(x, df, scale):
  """The NLL of a Generalized Student's T distribution (w/o including TFP)."""
  return 0.5 * ((df + 1.) * tf.math.log1p(
      (x / scale)**2. / df) + tf.math.log(df)) + tf.math.log(
          tf.abs(scale)) + tf.math.lgamma(
              0.5 * df) - tf.math.lgamma(0.5 * df + 0.5) + 0.5 * np.log(np.pi)


# A constant scale that makes tf.image.rgb_to_yuv() volume preserving.
_VOLUME_PRESERVING_YUV_SCALE = 1.580227820074


def rgb_to_syuv(rgb):
  """A volume preserving version of tf.image.rgb_to_yuv().

  By "volume preserving" we mean that rgb_to_syuv() is in the "special linear
  group", or equivalently, that the Jacobian determinant of the transformation
  is 1.

  Args:
    rgb: A tensor whose last dimension corresponds to RGB channels and is of
      size 3.

  Returns:
    A scaled YUV version of the input tensor, such that this transformation is
    volume-preserving.
  """
  return _VOLUME_PRESERVING_YUV_SCALE * tf.image.rgb_to_yuv(rgb)


def syuv_to_rgb(yuv):
  """A volume preserving version of tf.image.yuv_to_rgb().

  By "volume preserving" we mean that rgb_to_syuv() is in the "special linear
  group", or equivalently, that the Jacobian determinant of the transformation
  is 1.

  Args:
    yuv: A tensor whose last dimension corresponds to scaled YUV channels and is
      of size 3 (ie, the output of rgb_to_syuv()).

  Returns:
    An RGB version of the input tensor, such that this transformation is
    volume-preserving.
  """
  return tf.image.yuv_to_rgb(yuv / _VOLUME_PRESERVING_YUV_SCALE)


def image_dct(image):
  """Does a type-II DCT (aka "The DCT") on axes 1 and 2 of a rank-3 tensor."""
  dct_y = tf.transpose(tf.spectral.dct(image, type=2, norm='ortho'), [0, 2, 1])
  dct_x = tf.transpose(tf.spectral.dct(dct_y, type=2, norm='ortho'), [0, 2, 1])
  return dct_x


def image_idct(dct_x):
  """Inverts image_dct(), by performing a type-III DCT."""
  dct_y = tf.spectral.idct(tf.transpose(dct_x, [0, 2, 1]), type=2, norm='ortho')
  image = tf.spectral.idct(tf.transpose(dct_y, [0, 2, 1]), type=2, norm='ortho')
  return image


def compute_jacobian(f, x):
  """Computes the Jacobian of function `f` with respect to input `x`."""
  x_ph = tf.placeholder(tf.float32, x.shape)
  vec = lambda x: tf.reshape(x, [-1])
  jacobian = tf.stack(
      [vec(tf.gradients(vec(f(x_ph))[d], x_ph)[0]) for d in range(x.size)], 1)
  with tf.Session() as sess:
    jacobian = sess.run(jacobian, {x_ph: x})
  return jacobian


def get_resource_as_file(path):
  """A uniform interface for internal/open-source files."""

  class NullContextManager(object):

    def __init__(self, dummy_resource=None):
      self.dummy_resource = dummy_resource

    def __enter__(self):
      return self.dummy_resource

    def __exit__(self, *args):
      pass

  return NullContextManager('./' + path)


def get_resource_filename(path):
  """A uniform interface for internal/open-source filenames."""
  return './' + path
