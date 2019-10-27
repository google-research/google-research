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

"""Potential and Stationarity computation: tensorflow interface.

This module allows running the potential computation and also discovery
on GPU hardware.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy
import os
import pprint
import tensorflow as tf


from dim4.so8_supergravity_extrema.code import algebra
from dim4.so8_supergravity_extrema.code import scalar_sector
from m_theory_lib import tf_cexpm


# Must be wrapped up in a function, since we can only call this with
# TensorFlow 1.x graph-context.
def get_tf_scalar_evaluator():
  return scalar_sector.get_scalar_manifold_evaluator(
      to_scaled_constant=(
          lambda x, scale=1: tf.constant(numpy.array(x) * scale)),
      expm=tf_cexpm.cexpm,
      einsum=tf.einsum,
      eye=lambda n: tf.constant(numpy.eye(n), dtype=tf.complex128),
      trace=tf.linalg.trace,
      concatenate=lambda ts: tf.concat(ts, 0),
      complexify=lambda a: tf.cast(a, tf.complex128),
      re=tf.math.real,
      im=tf.math.imag,
      conjugate=tf.math.conj)


def S_id(v):
  """Fingerprints a potential value to a string identifier."""
  return 'S{:07d}'.format(int(round(-v * 1e5)))


def get_scanner(output_path,
                maxiter=1000,
                stationarity_threshold=1e-7):
  """Obtains a basic TensorFlow-based scanner for extremal points."""
  graph = tf.Graph()
  with graph.as_default():
    tf_scalar_evaluator = get_tf_scalar_evaluator()
    t_input = tf.compat.v1.placeholder(tf.float64, shape=[70])
    t_v70 = tf.Variable(
        initial_value=numpy.zeros([70]), trainable=True, dtype=tf.float64)
    op_assign_input = tf.compat.v1.assign(t_v70, t_input)
    sinfo = tf_scalar_evaluator(tf.cast(t_v70, tf.complex128))
    t_potential = sinfo.potential
    #
    t_stationarity = sinfo.stationarity
    op_opt = tf.contrib.opt.ScipyOptimizerInterface(
        tf.asinh(t_stationarity), options={'maxiter': maxiter})
    #
    def scanner(seed, scale=0.1, num_iterations=1):
      results = collections.defaultdict(list)
      rng = numpy.random.RandomState(seed)
      with graph.as_default():
        with tf.compat.v1.Session() as sess:
          sess.run([tf.compat.v1.global_variables_initializer()])
          for n in range(num_iterations):
            v70 = rng.normal(scale=scale, size=[70])
            sess.run([op_assign_input], feed_dict={t_input: v70})
            op_opt.minimize(sess)
            n_pot, n_stat, n_v70 = sess.run(
                [t_potential, t_stationarity, t_v70])
            if n_stat <= stationarity_threshold:
              results[S_id(n_pot)].append((n, n_pot, n_stat, list(n_v70)))
              # Overwrite output at every iteration.
              if output_path is not None:
                tmp_out = output_path + '.tmp'
                with open(tmp_out, 'w') as h:
                  h.write('n=%4d: p=%.12g s=%.12g\n' % (n, n_pot, n_stat))
                  h.write(pprint.pformat(dict(results)))
                os.rename(tmp_out, output_path)
      return dict(results)
    #
    return scanner
