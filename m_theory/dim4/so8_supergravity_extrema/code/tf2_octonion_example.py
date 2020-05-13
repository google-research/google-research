# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""TensorFlow optimization example.

Finding the transformation between octonion multiplication table and spin(8)
Gamma matrices in Green, Schwarz, Witten conventions.
"""

import numpy
import scipy.optimize
import tensorflow as tf


def get_gamma_vsc():
  """Computes SO(8) gamma-matrices."""
  # Conventions match Green, Schwarz, Witten's.
  entries = (
      '007+ 016- 025- 034+ 043- 052+ 061+ 070- '
      '101+ 110- 123- 132+ 145+ 154- 167- 176+ '
      '204+ 215- 226+ 237- 240- 251+ 262- 273+ '
      '302+ 313+ 320- 331- 346- 357- 364+ 375+ '
      '403+ 412- 421+ 430- 447+ 456- 465+ 474- '
      '505+ 514+ 527+ 536+ 541- 550- 563- 572- '
      '606+ 617+ 624- 635- 642+ 653+ 660- 671- '
      '700+ 711+ 722+ 733+ 744+ 755+ 766+ 777+')
  ret = numpy.zeros([8, 8, 8])
  for ijkc in entries.split():
    ijk = tuple(map(int, ijkc[:-1]))
    ret[ijk] = +1 if ijkc[-1] == '+' else -1
  return ret


def get_octonion_mult_table():
  """Computes the octonionic multiplication table"""
  # Cf. diagram at: http://math.ucr.edu/home/baez/octonions/
  ret = numpy.zeros([8, 8, 8])
  fano_lines = '124 156 137 235 267 346 457'
  for n in range(1, 8):
    ret[0, n, n] = -1
    ret[n, n, 0] = ret[n, 0, n] = 1
  ret[0, 0, 0] = 1
  for cijk in fano_lines.split():
    ijk = tuple(int(idx) for idx in cijk)
    for p, q, r in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
      # Note that we have to `go against the direction of the arrows'
      # to make the correspondence work.
      ret[ijk[r], ijk[p], ijk[q]] = -1
      ret[ijk[r], ijk[q], ijk[p]] = +1
  return ret


def f_fprime_from_tf_function(tf_f):
  def fprime(params):
    t_params = tf.constant(params, dtype=tf.float64)
    tape = tf.GradientTape()
    with tape:
      tape.watch(t_params)
      val = tf_f(t_params)
    grad = tape.gradient(val, t_params)
    return grad.numpy()
  def f(params):
    t_params = tf.constant(params, dtype=tf.float64)
    return tf_f(t_params).numpy()
  return f, fprime


def find_transforms():
  @tf.function
  def tf_get_loss(transform_params):
    t_transforms = tf.reshape(transform_params, (2, 8, 8))
    id8 = tf.constant(numpy.eye(8), dtype=tf.float64)
    gamma = tf.constant(get_gamma_vsc(), dtype=tf.float64)
    otable = tf.constant(get_octonion_mult_table(),
                         dtype=tf.float64)
    rotated_gamma = tf.einsum(
      'vab,aA,bB->vAB', gamma, t_transforms[0], t_transforms[1],
      optimize='greedy')
    delta_mult = rotated_gamma - otable
    delta_ortho_s = tf.einsum('ab,cb->ac',
                              t_transforms[0], t_transforms[0]) - id8
    delta_ortho_c = tf.einsum('ab,cb->ac',
                              t_transforms[1], t_transforms[1]) - id8
    # This 'loss' function punishes deviations of the rotated gamma matrices
    # from the octonionic multiplication table, and also deviations of the
    # spinor and cospinor transformation matrices from orthogonality.
    loss = (tf.nn.l2_loss(delta_mult) +
            tf.nn.l2_loss(delta_ortho_s) + tf.nn.l2_loss(delta_ortho_c))
    return loss
  f, fprime = f_fprime_from_tf_function(tf_get_loss)
  rng = numpy.random.RandomState(seed=0)
  x0 = rng.normal(size=2 * 8 * 8)
  opt = scipy.optimize.fmin_bfgs(f, x0, fprime=fprime)
  loss = f(opt)
  return loss, opt.reshape(2, 8, 8)


loss, transforms = find_transforms()
print('Loss: %.6g, Transforms:\n%r\n' % (
    loss, numpy.round(transforms, decimals=5)))
# Prints:
# Loss: 4.75556e-11, Transforms:
# array([[[-0.5, -0. ,  0. , -0.5, -0.5,  0. , -0.5,  0. ],
#         [ 0.5, -0. ,  0. , -0.5,  0.5,  0. , -0.5, -0. ],
#         [ 0. ,  0.5, -0.5, -0. , -0. , -0.5, -0. , -0.5],
#         [ 0. , -0.5,  0.5,  0. ,  0. , -0.5,  0. , -0.5],
#         [ 0.5,  0. ,  0. ,  0.5, -0.5, -0. , -0.5,  0. ],
#         [-0.5, -0. , -0. ,  0.5,  0.5, -0. , -0.5, -0. ],
#         [-0. ,  0.5,  0.5, -0. , -0. ,  0.5, -0. , -0.5],
#         [-0. , -0.5, -0.5, -0. , -0. ,  0.5,  0. , -0.5]],
#
#        [[ 0. , -0.5, -0.5,  0. , -0. ,  0.5,  0. , -0.5],
#         [-0. , -0.5, -0.5,  0. ,  0. , -0.5, -0. ,  0.5],
#         [-0.5, -0. , -0. , -0.5, -0.5,  0. ,  0.5,  0. ],
#         [-0.5,  0. ,  0. ,  0.5, -0.5,  0. , -0.5,  0. ],
#         [-0. ,  0.5, -0.5, -0. , -0. ,  0.5,  0. ,  0.5],
#         [-0. ,  0.5, -0.5, -0. , -0. , -0.5,  0. , -0.5],
#         [-0.5, -0. ,  0. , -0.5,  0.5,  0. , -0.5, -0. ],
#         [-0.5,  0. , -0. ,  0.5,  0.5,  0. ,  0.5, -0. ]]])
