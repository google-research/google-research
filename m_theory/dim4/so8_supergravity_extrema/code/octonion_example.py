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

"""TensorFlow optimization example.

Finding the transformation between octonion multiplication table and spin(8)
Gamma matrices in Green, Schwarz, Witten conventions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import tensorflow.compat.v1 as tf
from tensorflow.contrib import opt as contrib_opt


def get_gamma_vsc():
  """Computes SO(8) gamma-matrices."""
  # Conventions match Green, Schwarz, Witten's.
  entries = (
      "007+ 016- 025- 034+ 043- 052+ 061+ 070- "
      "101+ 110- 123- 132+ 145+ 154- 167- 176+ "
      "204+ 215- 226+ 237- 240- 251+ 262- 273+ "
      "302+ 313+ 320- 331- 346- 357- 364+ 375+ "
      "403+ 412- 421+ 430- 447+ 456- 465+ 474- "
      "505+ 514+ 527+ 536+ 541- 550- 563- 572- "
      "606+ 617+ 624- 635- 642+ 653+ 660- 671- "
      "700+ 711+ 722+ 733+ 744+ 755+ 766+ 777+")
  ret = numpy.zeros([8, 8, 8])
  for ijkc in entries.split():
    ijk = tuple(map(int, ijkc[:-1]))
    ret[ijk] = +1 if ijkc[-1] == '+' else -1
  return ret


def get_octonion_mult_table():
  """Computes the octonionic multiplication table"""
  # Cf. diagram at: http://math.ucr.edu/home/baez/octonions/
  ret = numpy.zeros([8, 8, 8])
  fano_lines = "124 156 137 235 267 346 457"
  for n in range(1, 8):
    ret[0, n, n] = -1
    ret[n, n, 0] = ret[n, 0, n] = 1
  ret[0, 0, 0] = 1
  for cijk in fano_lines.split():
    ijk = map(int, cijk)
    for p, q, r in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
      # Note that we have to `go against the direction of the arrows'
      # to make the correspondence work.
      ret[ijk[r], ijk[p], ijk[q]] = -1
      ret[ijk[r], ijk[q], ijk[p]] = +1
  return ret


def find_transforms():
  with tf.Graph().as_default():
    # Ensure reproducibility by seeding random number generators.
    tf.set_random_seed(0)
    transforms = tf.get_variable('transforms', shape=(2, 8, 8),
                                 dtype=tf.float64,
                                 trainable=True,
                                 initializer=tf.random_normal_initializer())
    id8 = tf.constant(numpy.eye(8), dtype=tf.float64)
    gamma = tf.constant(get_gamma_vsc(), dtype=tf.float64)
    otable = tf.constant(get_octonion_mult_table(),
                         dtype=tf.float64)
    # Transform gamma matrices step-by-step, since tf.einsum() does not
    # do SQL-like query planning optimization.
    rotated_gamma = tf.einsum(
        'vAb,bB->vAB', tf.einsum('vab,aA->vAb', gamma, transforms[0]),
        transforms[1])
    delta_mult = rotated_gamma - otable
    delta_ortho_s = tf.einsum('ab,cb->ac',
                              transforms[0], transforms[0]) - id8
    delta_ortho_c = tf.einsum('ab,cb->ac',
                              transforms[1], transforms[1]) - id8
    # This 'loss' function punishes deviations of the rotated gamma matrices
    # from the octonionic multiplication table, and also deviations of the
    # spinor and cospinor transformation matrices from orthogonality.
    loss = (tf.nn.l2_loss(delta_mult) +
            tf.nn.l2_loss(delta_ortho_s) + tf.nn.l2_loss(delta_ortho_c))
    opt = contrib_opt.ScipyOptimizerInterface(loss, options=dict(maxiter=1000))
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      opt.minimize(session=sess)
      return sess.run([loss, transforms])


loss, transforms = find_transforms()
print('Loss: %.6g, Transforms:\n%r\n' % (
    loss, numpy.round(transforms, decimals=5)))

## Prints:
# Loss: 2.19694e-11, Transforms:
# array([[[ 0.5, -0. , -0. ,  0.5,  0.5,  0. ,  0.5,  0. ],
#         [-0.5,  0. ,  0. ,  0.5, -0.5,  0. ,  0.5,  0. ],
#         [-0. , -0.5,  0.5, -0. , -0. ,  0.5,  0. ,  0.5],
#         [ 0. ,  0.5, -0.5, -0. , -0. ,  0.5, -0. ,  0.5],
#         [-0.5,  0. , -0. , -0.5,  0.5,  0. ,  0.5,  0. ],
#         [ 0.5, -0. ,  0. , -0.5, -0.5, -0. ,  0.5, -0. ],
#         [ 0. , -0.5, -0.5,  0. ,  0. , -0.5, -0. ,  0.5],
#         [-0. ,  0.5,  0.5,  0. , -0. , -0.5, -0. ,  0.5]],
#
#        [[-0. ,  0.5,  0.5,  0. ,  0. , -0.5,  0. ,  0.5],
#         [ 0. ,  0.5,  0.5, -0. , -0. ,  0.5,  0. , -0.5],
#         [ 0.5, -0. ,  0. ,  0.5,  0.5,  0. , -0.5,  0. ],
#         [ 0.5,  0. , -0. , -0.5,  0.5,  0. ,  0.5, -0. ],
#         [-0. , -0.5,  0.5, -0. ,  0. , -0.5, -0. , -0.5],
#         [-0. , -0.5,  0.5, -0. ,  0. ,  0.5,  0. ,  0.5],
#         [ 0.5,  0. ,  0. ,  0.5, -0.5, -0. ,  0.5, -0. ],
#         [ 0.5, -0. , -0. , -0.5, -0.5, -0. , -0.5, -0. ]]])
