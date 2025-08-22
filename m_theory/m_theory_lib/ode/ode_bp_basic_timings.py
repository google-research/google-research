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

"""Hessian-backpropagation timing measurements.

!!! This is not a software component to be used in larger engineering designs,
    but merely a script to validate some claims in the main article !!!
"""

import pickle
import pprint
import sys
import time

import matplotlib
from matplotlib import pyplot
import numpy
import ode_hessian
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


def gen_coeffs(rng, dim):
  return [rng.normal(size=[dim] * (order+1),
                     scale=dim**(-order / 2))
          for order in (1, 2)]


def get_ode_timings(dim=50,
                    seed=0,
                    t_max=0.2,
                    num_iter=10,
                    use_reconstructed_y0=False,
                    include_hessian_d2ydot_dy2_term=True,
                    odeint_args=(),
                    odeint_kwargs=(),
                    ):
  """Performs timing measurement."""
  rng = numpy.random.RandomState(seed=seed)
  # Note: there is a side effect in the list comprehension.
  # This is somewhat fishy.
  coeffs2, coeffs3 = gen_coeffs(rng, dim)
  tc_coeffs2 = tf.constant(coeffs2, dtype=tf.float64)
  tc_coeffs3 = tf.constant(coeffs3, dtype=tf.float64)
  y0 = rng.normal(size=dim)
  @tf.function
  def tf_ydot(y0):
    return (tf.einsum('ab,b->a', tc_coeffs2, y0) +
            tf.einsum('abc,b,c->a', tc_coeffs3, y0, y0))
  @tf.function
  def tf_loss(y0, y1):
    del y0  # Unused.
    return tf.math.reduce_sum(tf.math.square(y1))
  #
  bpp = ode_hessian.ODEBackpropProblem(
      dim_y=dim,
      with_timings=False,
      tf_dy_dt=tf_ydot,
      tf_L_y0y1=tf_loss)
  #
  timings_bp2 = []
  for n in range(num_iter):
    print(f'BP2 Iteration {n:3d}...')
    t0 = time.monotonic()
    result_bp2 = (
        bpp.backprop(y0=y0,
                     t0_to_t1=(0, t_max),
                     odeint_args=odeint_args,
                     odeint_kwargs=odeint_kwargs,
                     use_reconstructed_y0=use_reconstructed_y0))
    timings_bp2.append(time.monotonic() - t0)
  #
  timings_dp = []
  for n in range(num_iter):
    print(f'DP Iteration {n:3d}...')
    t0 = time.monotonic()
    result_dp = (
        bpp.dp_backprop(
            y0=y0,
            t0_to_t1=(0, t_max),
            odeint_args=odeint_args,
            odeint_kwargs=odeint_kwargs,
            include_hessian_d2ydot_dy2_term=include_hessian_d2ydot_dy2_term))
    timings_dp.append(time.monotonic() - t0)
  #
  t_bp2 = min(timings_bp2)
  t_dp = min(timings_dp)
  return dict(t_bp2=t_bp2,
              t_dp=t_dp,
              t_ratio_bp2_dp=t_bp2/t_dp,
              t_ratio_bp2_n_dp=t_bp2/(dim * t_dp),
              result_bp2=result_bp2,
              result_dp=result_dp)


if '--step1' in sys.argv:
  all_measurements = []
  for dim_y in range(10, 160, 10):
    print(f'### {dim_y=}')
    measured_n = get_ode_timings(
        dim=dim_y,
        num_iter=20,
        odeint_kwargs=(('method', 'RK45'), ('rtol', 1e-5), ('atol', 1e-5)))
    all_measurements.append(measured_n)
    print('######')
    pprint.pprint({k: v for k, v in measured_n.items() if k.startswith('t_')})
  pkwrite(f='bp_basic_timings.pickle', obj=all_measurements)
else:
  all_measurements = pkread('bp_basic_timings.pickle')


if '--step2' in sys.argv:
  ref_timing = next(m['t_dp']
                    for m in all_measurements
                    if m['result_dp'][0].size == 100)
  m_sizes = numpy.array([m['result_dp'][0].size for m in all_measurements],
                        dtype=numpy.float64)
  m_dp_timings = numpy.array([m['t_dp'] for m in all_measurements])
  m_bp2_timings = numpy.array([m['t_bp2'] for m in all_measurements])
  fig = pyplot.figure(figsize=(16, 12))
  ax = fig.gca()
  ax.plot(numpy.log10(m_sizes),
          numpy.log10(m_dp_timings / ref_timing),
          '-k', label=r'$\log10(T_{DP}/T_0)$')
  ax.plot(numpy.log10(m_sizes),
          numpy.log10(m_bp2_timings / ref_timing),
          '-b', label=r'$\log_{10}(T_{BP2}/T_0)$')
  ax.plot(numpy.log10(m_sizes),
          numpy.log10(m_bp2_timings / m_dp_timings),
          '-c', label=r'$\log_{10}(T_{BP2}/T_{DB})$')
  ax.plot(numpy.log10(m_sizes),
          numpy.log10(m_bp2_timings / m_dp_timings / m_sizes),
          '-m', label=r'$\log_{10}(T_{BP2}/(N \cdot T_{DB}))$')
  ax.set_xlabel(r'$\log_{10} N$')
  ax.grid()
  ax.legend()
  fig.savefig('ode_bp_basic_timings.pdf')
  fig.show()
