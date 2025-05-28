# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Framework overhead timing measurements.

!!! This is not a software component to be used in larger engineering designs,
    but merely a script to validate some claims in the main article !!!
"""


import pickle
import pprint
import time

import matplotlib
from matplotlib import pyplot
import numpy
import tensorflow as tf


matplotlib.rc('font', family='normal', weight='bold', size=22)


# Helpers for stashing potentially-expensive results.
# Security concerns related to `pickle` are irrelevant here,
# since we have full control over the input in this use case.
def pkwrite(*, f, obj):
  with open(f, 'wb') as h:
    pickle.dump(obj, h)


def pkread(f):
  with open(f, 'rb') as h:
    return pickle.load(h)


def get_tf_call_overhead_timings(size_n=50,
                                 num_iter=200):
  """Measures call-overhead timings."""
  rng = numpy.random.RandomState(seed=0)
  measured = dict(size_n=size_n, num_iter=num_iter)
  # Note: there is a side effect in the list comprehension.
  # This is somewhat fishy.
  coeffs = rng.normal(size=(size_n, size_n, size_n))
  tc_coeffs = tf.constant(coeffs, dtype=tf.float64)
  y0 = rng.normal(size=size_n)
  tc_y0 = tf.constant(y0, dtype=tf.float64)
  @tf.function
  def tf_ydot(ys):
    return tf.einsum('abc,b,c->a', tc_coeffs, ys, ys)
  def np_ydot(ys):
    return numpy.einsum('abc,b,c->a', coeffs, ys, ys)
  def np_ydot_with_tf_wrapping(ys):
    # Adding the overhead to wrap and unwrap the data into a tf.Tensor.
    ys1 = tf.constant(ys, dtype=tf.float64).numpy()
    return numpy.einsum('abc,b,c->a', coeffs, ys1, ys1)
  def np_ydot_calling_tf(ys):
    return tf_ydot(tf.constant(ys, dtype=tf.float64)).numpy()
  tf_ydot_concrete = tf_ydot.get_concrete_function(tc_y0)
  def np_ydot_calling_concrete_tf(ys):
    return tf_ydot_concrete(tf.constant(ys, dtype=tf.float64)).numpy()
  #
  for _ in range(10):  # burn-in: trigger JIT compilation and fill caches.
    _ = tf_ydot(tc_y0)
  def add_measurement(tag, func, func_arg, ref):
    t_start = time.monotonic()
    for _ in range(num_iter):
      _ = func(func_arg)
    t_finish = time.monotonic()
    measured[tag] = (t_finish - t_start) / num_iter
    # Add relative-to-reference factor
    measured['rel_' + tag] = measured[tag] / measured[ref]
  #
  add_measurement('np_ydot', np_ydot, y0, ref='np_ydot')
  add_measurement('tf_ydot', tf_ydot, tc_y0, ref='np_ydot')
  add_measurement('np_ydot_with_tf_wrapping', np_ydot_with_tf_wrapping,
                  y0, ref='np_ydot')
  add_measurement('np_ydot_calling_tf', np_ydot_calling_tf, y0, ref='np_ydot')
  add_measurement('np_ydot_calling_concrete_tf', np_ydot_calling_concrete_tf,
                  y0, ref='np_ydot')
  return measured


all_measurements = []
for dim in range(10, 160, 10):
  measured_n = get_tf_call_overhead_timings(size_n=dim)
  all_measurements.append(measured_n)
  print('######')
  pprint.pprint(measured_n)
pkwrite(f='numpy_tf_basic_timings.pickle', obj=all_measurements)

fig = pyplot.figure(figsize=(16, 12))
ax = fig.gca()
for label, style in (('rel_np_ydot', '-k'),
                     ('rel_np_ydot_with_tf_wrapping', '-b'),
                     ('rel_np_ydot_calling_concrete_tf', '-c'),
                     ('rel_tf_ydot', '-g'),
                     ('rel_np_ydot_calling_tf', '-m')):
  ax.plot([m['size_n'] for m in all_measurements],
          [m[label] for m in all_measurements],
          style,
          label=label)

ax.set_xlabel('dim(y)')
ax.set_ylabel('Relative Effort')
ax.set_title('NumPy-TensorFlow call performance overhead')
ax.grid()
ax.legend()
fig.savefig('numpy_tf_basic_timings.pdf')
fig.show()
