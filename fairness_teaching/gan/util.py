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

import numpy as np
import tensorflow as tf
# pylint: skip-file

#####################  sample method #####################

def _sample_line(real, fake):
  shape = [tf.shape(real)[0]] + [1] * (real.shape.ndims - 1)
  alpha = tf.random.uniform(shape=shape, minval=0, maxval=1)
  sample = real + alpha * (fake - real)
  sample.set_shape(real.shape)
  return sample


def _sample_DRAGAN(real, fake):  # fake is useless
  beta = tf.random.uniform(shape=tf.shape(real), minval=0, maxval=1)
  fake = real + 0.5 * tf.math.reduce_std(real) * beta
  sample = _sample_line(real, fake)
  return sample


####################  gradient penalty  ####################

def _norm(x):
  norm = tf.norm(tf.reshape(x, [tf.shape(x)[0], -1]), axis=1)
  return norm


def _one_mean_gp(grad):
  norm = _norm(grad)
  gp = tf.reduce_mean((norm - 1)**2)
  return gp


def _zero_mean_gp(grad):
  norm = _norm(grad)
  gp = tf.reduce_mean(norm**2)
  return gp


def _lipschitz_penalty(grad):
  norm = _norm(grad)
  gp = tf.reduce_mean(tf.maximum(norm - 1, 0)**2)
  return gp


def gradient_penalty(f, real, fake, gp_mode, sample_mode):
  sample_fns = {
    'line': _sample_line,
    'real': lambda real, fake: real,
    'fake': lambda real, fake: fake,
    'dragan': _sample_DRAGAN,
  }

  gp_fns = {
    '1-gp': _one_mean_gp,
    '0-gp': _zero_mean_gp,
    'lp': _lipschitz_penalty,
  }

  if gp_mode == 'none':
    gp = tf.constant(0, dtype=real.dtype)
  else:
    x = sample_fns[sample_mode](real, fake)
    grad = tf.gradients(f(x), x)[0]
    gp = gp_fns[gp_mode](grad)

  return gp

####################  data format  ####################
def to_uint8(images, min_value=0, max_value=255, dtype=np.uint8):
  return ((images + 1.) / 2. * (max_value - min_value) + min_value).astype(dtype)

