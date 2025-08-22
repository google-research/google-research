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

"""GON experimental code on Rosenbrock and Griewank data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_lattice as tfl

_CGON_OR_GON = flags.DEFINE_string("cgon_or_gon", "BOTH", "CGON, GON or BOTH")
# Data generation hparams
_NUM_EXAMPLES = flags.DEFINE_integer("num_examples", 10000,
                                     "number of training examples.")
_INPUT_DIM = flags.DEFINE_integer("input_dim", 16, "number input dimensions.")
_BOUND = flags.DEFINE_float("bound", 2.0, "(soft) bound for sampling.")
_NOISE_SIGNAL_RATIO = flags.DEFINE_float(
    "noise_signal_ratio", 0.25,
    "standard deviation of the noise as a factor of the target function value "
    "in the target function.")
# GON/CGON hparams
_LATTICE_DIM = flags.DEFINE_integer("lattice_dim", 3,
                                    "number GON lattice input dimensions.")
# Training hparams
_STEPS = flags.DEFINE_integer("steps", 100000, "number of training steps.")


def sample_from_uniform_square(count=1000, dim=2):
  points = np.random.uniform(low=-_BOUND, high=_BOUND, size=(count, dim))
  return points


def rosenbrock(points, noise_signal_ratio=0.0):
  summation = 0
  for i in range(_INPUT_DIM - 1):
    summation += ((1.0 - points[:, i])**2 + 100.0 *
                  (points[:, i + 1] - points[:, i]**2)**2)
  noise = np.random.normal(
      scale=abs(summation * noise_signal_ratio), size=len(points))
  return summation + noise


def griewank(points, noise_signal_ratio=0.0):
  summation = 0
  product = 1
  for i in range(_INPUT_DIM):
    summation += (points[:, i] - 1.0)**2
    product *= np.cos((points[:, i] - 1.0) / np.sqrt(i + 1))
  y = 1 + summation / 4000 - product
  noise = np.random.normal(scale=abs(y * noise_signal_ratio), size=len(points))
  return y + noise


def parabola(points, noise_signal_ratio=0.0):
  summation = 0
  for i in range(_INPUT_DIM):
    summation += (points[:, i] - 1.0)**2
  noise = np.random.normal(
      scale=abs(summation * noise_signal_ratio), size=len(points))
  return summation + noise


# Global Optimization Networks (GON):
def build_rtl_gon(lattice_dim):
  """Build GON model."""
  input_layer = tf.keras.layers.Input(shape=(_INPUT_DIM,))
  calibrated_output = tfl.layers.PWLCalibration(
      input_keypoints=np.linspace(-_BOUND, _BOUND, 10),
      units=_INPUT_DIM,
      output_min=-1.0,
      output_max=1.0,
      clamp_min=True,
      clamp_max=True,
      monotonicity="increasing",
      name="input_calibration",
  )(
      input_layer)
  lattice_inputs = []
  for _ in range(_INPUT_DIM):
    indices = random.sample(range(_INPUT_DIM), lattice_dim)
    lattice_inputs.append(tf.gather(calibrated_output, indices, axis=1) + 1.0)
  lattice_input = tf.stack(lattice_inputs, axis=1)
  lattice_output_layer = tfl.layers.Lattice(
      lattice_sizes=[3] * lattice_dim,
      units=_INPUT_DIM,
      joint_unimodalities=(list(range(lattice_dim)), "valley"),
      kernel_initializer="random_uniform",
  )(
      lattice_input)
  output_layer = tf.keras.layers.Dense(
      units=1, kernel_constraint=tf.keras.constraints.NonNeg())(
          lattice_output_layer)
  keras_model = tf.keras.models.Model(
      inputs=[input_layer], outputs=output_layer)
  keras_model.compile(
      loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam())
  return keras_model


# Conditional Global Optimization Networks (CGON):
def build_rtl_cgon(lattice_dim, gon_dim):
  """Build GON model."""
  input_layer = tf.keras.layers.Input(shape=(_INPUT_DIM,))
  nongon_dim = _INPUT_DIM - gon_dim
  gon_input_layer, nongon_input_layer = tf.split(
      input_layer, [gon_dim, nongon_dim], axis=1)
  gon_calibrated_output = tfl.layers.PWLCalibration(
      input_keypoints=np.linspace(-_BOUND, _BOUND, 10),
      units=gon_dim,
      monotonicity="increasing",
      output_min=-1.0,
      output_max=1.0,
      clamp_min=True,
      clamp_max=True,
      name="gon_calibration",
  )(
      gon_input_layer)
  gon_rtl_input_layer = []
  for i in range(gon_dim):
    nongon_calibrated_output = tf.reduce_mean(
        tfl.layers.PWLCalibration(
            input_keypoints=np.linspace(-_BOUND, _BOUND, 10),
            units=nongon_dim,
            output_min=-1.0,
            output_max=1.0,
            name="nongon_pwl_" + str(i))(nongon_input_layer),
        axis=1,
        keepdims=True)
    gon_rtl_input_layer.append(nongon_calibrated_output +
                               tf.gather(gon_calibrated_output, [i], axis=1))
  gon_rtl_input_layer = tf.tanh(tf.concat(gon_rtl_input_layer, axis=1)) + 1.0
  lattice_inputs = []
  for _ in range(gon_dim):
    gon_indices = random.sample(range(gon_dim), lattice_dim)
    gon_lattice_input = tf.gather(gon_rtl_input_layer, gon_indices, axis=1)
    lattice_inputs.append(gon_lattice_input)
  lattice_input = tf.stack(lattice_inputs, axis=1)
  lattice_output_layer = tfl.layers.Lattice(
      lattice_sizes=[3] * lattice_dim,
      units=gon_dim,
      joint_unimodalities=(list(range(lattice_dim)), "valley"),
      kernel_initializer="random_uniform",
  )(
      lattice_input)
  output_layer = tf.keras.layers.Dense(
      units=1, kernel_constraint=tf.keras.constraints.NonNeg())(
          lattice_output_layer)
  keras_model = tf.keras.models.Model(
      inputs=[input_layer], outputs=output_layer)
  keras_model.compile(
      loss="mean_squared_error", optimizer=tf.keras.optimizers.Adam())
  return keras_model


def get_optim_from_cal(calibrator_weights, target=0.0):
  """Get Argmin Through Inverting the Calibrators."""
  keypoints_x = np.linspace(-_BOUND, _BOUND, 10)
  keypoints_y = np.cumsum(calibrator_weights)
  if keypoints_y[0] > target:
    return keypoints_x[0]
  for i in range(len(keypoints_x) - 1):
    if keypoints_y[i] < target and keypoints_y[i + 1] > target:
      w = (keypoints_y[i + 1] - target) / (keypoints_y[i + 1] - keypoints_y[i])
      return keypoints_x[i + 1] - w * (keypoints_x[i + 1] - keypoints_x[i])
  return keypoints_x[-1]


def main(_):
  data_generation_function = griewank  # rosenbrock / griewank / parabola
  # The global minimizer is at (1.0, 1.0, ..., 1.0).
  argmin_true = 1.0

  # Generate Data
  training_inputs = sample_from_uniform_square(
      count=_NUM_EXAMPLES, dim=_INPUT_DIM)
  labels = (
      data_generation_function(
          training_inputs, noise_signal_ratio=_NOISE_SIGNAL_RATIO) /
      data_generation_function(np.array([[-_BOUND] * _INPUT_DIM])))

  # Simulation for GON
  if _CGON_OR_GON != "CGON":
    # Global Optimization Networks
    keras_model_gon = build_rtl_gon(lattice_dim=_LATTICE_DIM)
    keras_model_gon.fit(
        training_inputs,
        labels,
        batch_size=100,
        epochs=int(_STEPS * 100 / _NUM_EXAMPLES),
        verbose=0,
    )
    result_gon = []
    for i in range(_INPUT_DIM):
      result_gon.append(
          get_optim_from_cal(
              keras_model_gon.get_layer("input_calibration").get_weights()[0]
              [:, i]))
    # f(argmin_hat)
    print("[metric] GON_val=" +
          str(data_generation_function(np.array([result_gon]))[0]))
    # ||argmin_hat - argmin_true||^2
    print("[metric] GON_dist=" + str(
        np.sum([(a - argmin_true) * (a - argmin_true) for a in result_gon])))

  # Simulation for CGON
  if _CGON_OR_GON != "GON":
    # Input features for each example consists of first optimization_dim for
    # optimiaztion, and then conditional_dim conditional inputs.
    optimization_dim = int(_INPUT_DIM / 4 * 3)
    conditional_dim = int(_INPUT_DIM / 4)
    # Conditional Global Optimization Networks
    keras_model_cgon = build_rtl_cgon(
        lattice_dim=_LATTICE_DIM, gon_dim=optimization_dim)
    keras_model_cgon.fit(
        training_inputs,
        labels,
        batch_size=100,
        epochs=int(_STEPS * 100 / _NUM_EXAMPLES),
        verbose=0,
    )
    result_cgon = []
    for i in range(optimization_dim):
      target = -tf.reduce_mean(
          keras_model_cgon.get_layer("nongon_pwl_" + str(i))(tf.zeros(
              shape=[1, conditional_dim], dtype=tf.float32))).numpy()
      result_cgon.append(
          get_optim_from_cal(
              keras_model_cgon.get_layer("gon_calibration").get_weights()[0][:,
                                                                             i],
              target))
    result_cgon = result_cgon + [0.0] * conditional_dim
    # f(argmin_hat)
    print("[metric] CGON_val=" +
          str(data_generation_function(np.array([result_cgon]))[0]))


if __name__ == "__main__":
  app.run(main)
