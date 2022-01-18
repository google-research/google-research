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

# Lint as: python3
r"""Approximate the distribution's partition function with a spline.

This script generates values for the distribution's partition function and then
fits a cubic hermite spline to those values, which is then stored to disk.
To run this script, assuming you're in this directory, run:
  python -m robust_loss.fit_partition_spline_test
This script will likely never have to be run again, and is provided here for
completeness and reproducibility, or in case someone decides to modify
distribution.partition_spline_curve() in the future in case they find a better
curve. If the user wants a more accurate spline approximation, this can be
obtained by modifying the `x_max`, `x_scale`, and `redundancy` parameters in the
code below, but this should only be done with care.
"""

from absl import app
import numpy as np
import tensorflow.compat.v2 as tf
from robust_loss import cubic_spline
from robust_loss import distribution
from robust_loss import general

tf.enable_v2_behavior()


def numerical_base_partition_function(alpha):
  """Numerically approximate the partition function Z(alpha)."""
  # Generate values `num_samples` values in [-x_max, x_max], with more samples
  # near the origin as `power` is set to larger values.
  num_samples = 2**24 + 1  # We want an odd value so that 0 gets sampled.
  x_max = 10**10
  power = 6
  t = t = tf.linspace(
      tf.constant(-1, tf.float64), tf.constant(1, tf.float64), num_samples)
  t = tf.sign(t) * tf.abs(t)**power
  x = t * x_max

  # Compute losses for the values, then exponentiate the negative losses and
  # integrate with the trapezoid rule to get the partition function.
  losses = general.lossfun(x, alpha, np.float64(1))
  y = tf.math.exp(-losses)
  partition = tf.reduce_sum((y[1:] + y[:-1]) * (x[1:] - x[:-1])) / 2.
  return partition


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Parameters governing how the x coordinate of the spline will be laid out.
  # We will construct a spline with knots at
  #   [0 : 1 / x_scale : x_max],
  # by fitting it to values sampled at
  #   [0 : 1 / (x_scale * redundancy) : x_max]
  x_max = 12
  x_scale = 1024
  redundancy = 4  # Must be >= 2 for the spline to be useful.

  spline_spacing = 1. / (x_scale * redundancy)
  x_knots = np.arange(
      0, x_max + spline_spacing, spline_spacing, dtype=np.float64)
  table = []
  # We iterate over knots, and for each knot recover the alpha value
  # corresponding to that knot with inv_partition_spline_curve(), and then
  # with that alpha we accurately approximate its partition function using
  # numerical_base_partition_function().
  for x_knot in x_knots:
    alpha = distribution.inv_partition_spline_curve(x_knot).numpy()
    partition = numerical_base_partition_function(alpha).numpy()
    table.append((x_knot, alpha, partition))
    print(table[-1])

  table = np.array(table)
  x = table[:, 0]
  alpha = table[:, 1]
  y_gt = np.log(table[:, 2])

  # We grab the values from the true log-partition table that correpond to
  # knots, by looking for where x * x_scale is an integer.
  mask = np.abs(np.round(x * x_scale) - (x * x_scale)) <= 1e-8
  values = y_gt[mask]

  # Initialize `tangents` using a central differencing scheme.
  values_pad = np.concatenate([[values[0] - values[1] + values[0]], values,
                               [values[-1] - values[-2] + values[-1]]], 0)
  tangents = (values_pad[2:] - values_pad[:-2]) / 2.

  # Construct the spline's value and tangent TF variables, constraining the last
  # knot to have a fixed value Z(infinity) and a tangent of zero.
  n = len(values)
  tangents = tf.Variable(tangents, tf.float64)
  values = tf.Variable(values, tf.float64)

  # Fit the spline.
  num_iters = 10001

  optimizer = tf.keras.optimizers.SGD(learning_rate=1e-9, momentum=0.99)

  trace = []
  for ii in range(num_iters):
    with tf.GradientTape() as tape:
      tape.watch([values, tangents])
      # Fix the endpoint to be a known constant with a zero tangent.
      i_values = tf.where(
          np.arange(n) == (n - 1),
          tf.ones_like(values) * 0.70526025442689566, values)
      i_tangents = tf.where(
          np.arange(n) == (n - 1), tf.zeros_like(tangents), tangents)
      i_y = cubic_spline.interpolate1d(x * x_scale, i_values, i_tangents)
      # We minimize the maximum residual, which makes for a very ugly
      # optimization problem but works well in practice.
      i_loss = tf.reduce_max(tf.abs(i_y - y_gt))
      grads = tape.gradient(i_loss, [values, tangents])
      optimizer.apply_gradients(zip(grads, [values, tangents]))
    trace.append(i_loss.numpy())
    if (ii % 200) == 0:
      print('{:5d}: {:e}'.format(ii, trace[-1]))

  mask = alpha <= 4
  max_error_a4 = np.max(np.abs(i_y[mask] - y_gt[mask]))
  max_error = np.max(np.abs(i_y - y_gt))
  print('Max Error (a <= 4): {:e}'.format(max_error_a4))
  print('Max Error: {:e}'.format(max_error))

  # Just a sanity-check on the error.
  assert max_error_a4 <= 5e-7
  assert max_error <= 5e-7

  # Save the spline to disk.
  np.savez(
      './data/partition_spline.npz',
      x_scale=x_scale,
      values=i_values.numpy(),
      tangents=i_tangents.numpy())


if __name__ == '__main__':
  app.run(main)
