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

# pylint: skip-file
from abc import ABC, abstractmethod

from absl import flags
import numpy as np
import scipy.stats
import tensorflow as tf
from tensorflow import keras
import tensorflow_lattice as tfl

from quantile_regression import qr_lib
from quantile_regression import qr_lib_gasthaus

FLAGS = flags.FLAGS

flags.DEFINE_integer("lattice_sizes", None, "Lattice size used for x, q.")
flags.DEFINE_integer("x_keypoints", None, "Number of keypoints used for all x.")
flags.DEFINE_integer("q_keypoints", None, "Number of keypoints used for q.")
flags.DEFINE_string("q_monotonicity", "increasing", "increasing or none.")
flags.DEFINE_integer("q_lattice_size", None, "Lattice size used for q.")

flags.DEFINE_integer("num_hidden_layers", 4, "Number of DNN hidden layers.")
flags.DEFINE_integer("hidden_dim", 10, "Number of units in each hidden layer.")
flags.DEFINE_integer("gasthaus_keypoints", 10, "Number of keypoints in final Gasthaus CDF PWL.")

flags.DEFINE_integer("sin_left_skew", 1, "Left skew for SIN experiment.")
flags.DEFINE_integer("sin_right_skew", 7, "Right skew for SIN experiment.")


class Simulation(ABC):
  """Interface for defining a new simulation test case."""

  @abstractmethod
  def num_dims(self):
    """Returns the number of x-dimensions in this simulation case."""
    pass

  @abstractmethod
  def create_base_model(self, x_lattice_sizes, q_lattice_size, x_keypoints,
                        q_keypoints):
    """Creates and returns tfl.premade.CalibratedLattice model object."""
    pass

  @abstractmethod
  def generate_training_data(self, num_training_examples):
    """Generates training data for simulation in form ([x1,x2,...], y)."""
    pass

  # "Quantile MSE": L2 difference between true and estimated quantile curves,
  # estimated over a grid of x values.
  @abstractmethod
  def compute_quantile_mses(self, model):
    """Computes list of quantile MSEs for quantiles between 0.01 and 0.99."""
    pass


class SinTestCase(Simulation):
  """Test case 1 (1-dimensional) from Torossian (2020) survey paper."""

  def num_dims(self):
    return 1

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=10):
    if FLAGS.lattice_sizes is not None:
      x_lattice_sizes = FLAGS.lattice_sizes
      q_lattice_size = FLAGS.lattice_sizes
    if FLAGS.q_lattice_size is not None:
      q_lattice_size = FLAGS.q_lattice_size
    if FLAGS.x_keypoints is not None:
      x_keypoints = FLAGS.x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    config = tfl.configs.CalibratedLatticeConfig(
        feature_configs=[
            tfl.configs.FeatureConfig(
                name="x",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -1.0, 1.0, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="q",
                monotonicity=FLAGS.q_monotonicity,
                lattice_size=q_lattice_size,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 1.0, q_keypoints
                ),
            ),
        ],
        output_initialization=[-10, 10],
    )
    return tfl.premade.CalibratedLattice(config)

  def create_dnn_model(self):
    return qr_lib.build_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim)

  def create_gasthaus_model(self):
    return qr_lib_gasthaus.build_gasthaus_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim, FLAGS.gasthaus_keypoints)

  def generate_training_data(self, num_training_examples):
    x = np.random.uniform(-1, 1, num_training_examples)
    eta = np.random.normal(size=num_training_examples)
    eps = FLAGS.sin_left_skew * eta * (eta <= 0) + FLAGS.sin_right_skew * eta * (eta > 0)
    y = 5 * np.sin(8 * x) + (0.2 + 3 * x**3) * eps
    return [x], y

  def compute_quantile_mses(self, model, gasthaus=False):

    def true_y(x_test, q):

      def error_part(x):
        return 0.2 + 3 * x**3

      y50 = 5 * np.sin(8 * x_test)
      norm_inv_cdf = scipy.stats.norm.ppf(q)
      if q == 0.5:
        return y50
      elif q > 0.5:
        return y50 + error_part(x_test) * (
            (error_part(x_test) >= 0) * FLAGS.sin_right_skew * norm_inv_cdf +
            (error_part(x_test) < 0) * FLAGS.sin_left_skew * -norm_inv_cdf)
      elif q < 0.5:
        return y50 - error_part(x_test) * (
            (error_part(x_test) < 0) * FLAGS.sin_right_skew * norm_inv_cdf +
            (error_part(x_test) >= 0) * FLAGS.sin_left_skew * -norm_inv_cdf)

    TEST_SIZE = 1000
    x_test = np.linspace(-1, 1, num=TEST_SIZE)
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_test = true_y(x_test, q)
      if not gasthaus:
        y_pred = model([x_test, np.full(x_test.shape, q)])
      else:
        xs = tf.reshape(tf.convert_to_tensor(x_test, dtype=tf.float32), (-1, 1))
        qs = tf.convert_to_tensor(np.full(x_test.shape, q), dtype=tf.float32)
        y_pred = qr_lib_gasthaus.calculate_y_pred_tensor_gasthaus(model, xs, qs)
      q_mses.append(np.mean((y_pred.numpy().flatten() - y_test)**2))
    return q_mses

  def compute_monotonicity_violations(self, model, gasthaus=False):

    TEST_SIZE = 1000
    x_test = np.linspace(-1, 1, num=TEST_SIZE)
    old_preds = np.full(x_test.shape, -np.inf)
    mono_violations = np.full(x_test.shape, False)
    for q in np.linspace(0.01, 0.99, num=99):
      if not gasthaus:
        y_pred = model([x_test, np.full(x_test.shape, q)]).numpy().flatten()
      else:
        xs = tf.reshape(tf.convert_to_tensor(x_test, dtype=tf.float32), (-1, 1))
        qs = tf.convert_to_tensor(np.full(x_test.shape, q), dtype=tf.float32)
        y_pred = qr_lib_gasthaus.calculate_y_pred_tensor_gasthaus(model, xs, qs).numpy().flatten()
      mono_violations = mono_violations | (y_pred - old_preds < -1e-4)
      old_preds = y_pred
    return np.mean(mono_violations)


class GriewankTestCase(Simulation):
  """Test case 2 (2-dimensional) from Torossian (2020) survey paper."""

  def num_dims(self):
    return 2

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=10):
    if FLAGS.lattice_sizes is not None:
      x_lattice_sizes = FLAGS.lattice_sizes
      q_lattice_size = FLAGS.lattice_sizes
    if FLAGS.q_lattice_size is not None:
      q_lattice_size = FLAGS.q_lattice_size
    if FLAGS.x_keypoints is not None:
      x_keypoints = FLAGS.x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    config = tfl.configs.CalibratedLatticeConfig(
        feature_configs=[
            tfl.configs.FeatureConfig(
                name="x1",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -5.0, 5.0, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x2",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -3.0, 3.0, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="q",
                monotonicity=FLAGS.q_monotonicity,
                lattice_size=q_lattice_size,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 1.0, q_keypoints
                ),
            ),
        ],
        output_initialization=[0, 10],
    )
    return tfl.premade.CalibratedLattice(config)

  def create_dnn_model(self):
    return qr_lib.build_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim)

  def create_gasthaus_model(self):
    return qr_lib_gasthaus.build_gasthaus_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim, FLAGS.gasthaus_keypoints)

  def generate_training_data(self, num_training_examples):
    x1 = np.random.uniform(-5, 5, num_training_examples)
    x2 = np.random.uniform(-3, 3, num_training_examples)
    eta = np.random.normal(size=num_training_examples)
    eps = eta * (eta <= 0) + 5 * eta * (eta > 0)
    y = ((1 / 4000) *
         (x1**2 + x2**2) - np.cos(x1) * np.cos(x2 / np.sqrt(2)) + 1) * eps
    return [x1, x2], y

  def compute_quantile_mses(self, model, gasthaus=False):

    def true_y(x1, x2, q):
      norm_inv_cdf = scipy.stats.norm.ppf(q)
      if q < 0.5:
        return ((1 / 4000) * (x1**2 + x2**2) -
                np.cos(x1) * np.cos(x2 / np.sqrt(2)) + 1) * norm_inv_cdf
      else:
        return ((1 / 4000) * (x1**2 + x2**2) -
                np.cos(x1) * np.cos(x2 / np.sqrt(2)) + 1) * norm_inv_cdf * 5

    SQRT_TEST_SIZE = 100
    x1_test = np.repeat(np.linspace(-5, 5, num=SQRT_TEST_SIZE), SQRT_TEST_SIZE)
    x2_test = np.tile(np.linspace(-3, 3, num=SQRT_TEST_SIZE), SQRT_TEST_SIZE)
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_test = true_y(x1_test, x2_test, q)
      if not gasthaus:
        y_pred = model([x1_test, x2_test, np.full(x1_test.shape, q)])
      else:
        xs = [x1_test.astype(np.float32), x2_test.astype(np.float32)]
        qs = tf.convert_to_tensor(np.full(x1_test.shape, q), dtype=tf.float32)
        y_pred = qr_lib_gasthaus.calculate_y_pred_tensor_gasthaus(model, xs, qs)
      q_mses.append(np.mean((y_pred.numpy().flatten() - y_test)**2))
    return q_mses

  def compute_monotonicity_violations(self, model):

    SQRT_TEST_SIZE = 100
    x1_test = np.repeat(np.linspace(-5, 5, num=SQRT_TEST_SIZE), SQRT_TEST_SIZE)
    x2_test = np.tile(np.linspace(-3, 3, num=SQRT_TEST_SIZE), SQRT_TEST_SIZE)

    old_preds = np.full(x1_test.shape, -np.inf)
    mono_violations = np.full(x1_test.shape, False)
    for q in np.linspace(0.01, 0.99, num=99):
      y_pred = model([x1_test, x2_test, np.full(x1_test.shape, q)]).numpy().flatten()
      mono_violations = mono_violations | (y_pred - old_preds < -1e-4)
      old_preds = y_pred
    return np.mean(mono_violations)


class MichalewiczTestCase(Simulation):
  """Test case 3 from Torossian (2020) survey paper."""

  def num_dims(self):
    return 1

  def create_base_model(self, x_lattice_sizes=3, q_lattice_size=3,
                        x_keypoints=10, q_keypoints=10):
    if FLAGS.lattice_sizes is not None:
      x_lattice_sizes = FLAGS.lattice_sizes
      q_lattice_size = FLAGS.lattice_sizes
    if FLAGS.q_lattice_size is not None:
      q_lattice_size = FLAGS.q_lattice_size
    if FLAGS.x_keypoints is not None:
      x_keypoints = FLAGS.x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    config = tfl.configs.CalibratedLatticeConfig(
        feature_configs=[
            tfl.configs.FeatureConfig(
                name="x1",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(0, 4, x_keypoints),
            ),
            tfl.configs.FeatureConfig(
                name="q",
                monotonicity=FLAGS.q_monotonicity,
                lattice_size=q_lattice_size,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 1.0, q_keypoints
                ),
            ),
        ],
        output_initialization=[-5, 1],
    )
    return tfl.premade.CalibratedLattice(config)

  def generate_training_data(self, num_training_examples):
    x1 = np.random.uniform(0, 4, num_training_examples)
    eta = np.random.normal(size=num_training_examples)
    eps = 3 * eta * (eta <= 0) + 6 * eta * (eta > 0)
    sin_part = np.sin(x1)*np.power(np.sin(1*x1**2/np.pi), 30)
    cos_part = np.power(np.cos(np.pi*1*x1/10), 3)
    y = -2 * sin_part - (0.1*cos_part / np.abs(-sin_part + 2)) * eps**2
    return [x1], y

  def create_dnn_model(self):
    return qr_lib.build_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim)

  def create_gasthaus_model(self):
    return qr_lib_gasthaus.build_gasthaus_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim, FLAGS.gasthaus_keypoints)

  def compute_quantile_mses(self, model, gasthaus=False):

    TEST_SIZE = 4000

    def true_y(x1, q, eps_sq):
      eps_sq_inv_cdf = eps_sq[int(q*TEST_SIZE)]
      reverse_inv_cdf = eps_sq[int((1-q)*TEST_SIZE)]
      sin_part = np.sin(x1)*np.power(np.sin(1*x1**2/np.pi), 30)
      cos_part = np.power(np.cos(np.pi*1*x1/10), 3)
      return (cos_part <= 0) * (
          -2 * sin_part
          - (0.1 * cos_part / np.abs(-sin_part + 2)) * eps_sq_inv_cdf
      ) + (cos_part > 0) * (
          -2 * sin_part
          - (0.1 * cos_part / np.abs(-sin_part + 2)) * reverse_inv_cdf
      )

    x1 = np.linspace(0, 4, TEST_SIZE)
    eta = np.random.normal(size=TEST_SIZE)
    eps = 3 * eta * (eta <= 0) + 6 * eta * (eta > 0)
    eps_sq = np.sort(eps**2)
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_test = true_y(x1, q, eps_sq)
      if not gasthaus:
        y_pred = model([x1, np.full(x1.shape, q)])
      else:
        xs = tf.reshape(tf.convert_to_tensor(x1, dtype=tf.float32), (-1, 1))
        qs = tf.convert_to_tensor(np.full(x1.shape, q), dtype=tf.float32)
        y_pred = qr_lib_gasthaus.calculate_y_pred_tensor_gasthaus(model, xs, qs)
      q_mses.append(np.mean((y_pred.numpy().flatten() - y_test)**2))
    return q_mses

  def compute_monotonicity_violations(self, model):

    TEST_SIZE = 4000
    x1 = np.linspace(0, 4, TEST_SIZE)
    old_preds = np.full(x1.shape, -np.inf)
    mono_violations = np.full(x1.shape, False)
    for q in np.linspace(0.01, 0.99, num=99):
      y_pred = model([x1, np.full(x1.shape, q)]).numpy().flatten()
      mono_violations = mono_violations | (y_pred - old_preds < -1e-4)
      old_preds = y_pred
    return np.mean(mono_violations)


class AckleyTestCase(Simulation):
  """Test case 4 (9-dimensional) from Torossian (2020) survey paper."""

  def num_dims(self):
    return 9

  def create_base_model(self, x_lattice_sizes=3, q_lattice_size=3,
                        x_keypoints=10, q_keypoints=10):
    if FLAGS.lattice_sizes is not None:
      x_lattice_sizes = FLAGS.lattice_sizes
      q_lattice_size = FLAGS.lattice_sizes
    if FLAGS.q_lattice_size is not None:
      q_lattice_size = FLAGS.q_lattice_size
    if FLAGS.x_keypoints is not None:
      x_keypoints = FLAGS.x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    config = tfl.configs.CalibratedLatticeConfig(
        feature_configs=[
            tfl.configs.FeatureConfig(
                name="x1",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -1.0, -0.7, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x2",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 1.0, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x3",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -0.7, -0.3, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x4",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    0.5, 1, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x5",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -1.0, -0.5, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x6",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -3.0, -2.6, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x7",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    -0.1, 0.0, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x8",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 0.1, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="x9",
                lattice_size=x_lattice_sizes,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 0.8, x_keypoints
                ),
            ),
            tfl.configs.FeatureConfig(
                name="q",
                monotonicity=FLAGS.q_monotonicity,
                lattice_size=q_lattice_size,
                pwl_calibration_input_keypoints=np.linspace(
                    0.0, 1.0, q_keypoints
                ),
            ),
        ],
        output_initialization=[0, 10],
    )
    return tfl.premade.CalibratedLattice(config)

  def create_dnn_model(self):
    return qr_lib.build_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim)

  def create_gasthaus_model(self):
    return qr_lib_gasthaus.build_gasthaus_dnn_model(self.num_dims(), FLAGS.num_hidden_layers, FLAGS.hidden_dim, FLAGS.gasthaus_keypoints)

  def generate_training_data(self, num_training_examples):
    x1 = np.random.uniform(-1, -0.7, num_training_examples)
    x2 = np.random.uniform(0, 1, num_training_examples)
    x3 = np.random.uniform(-0.7, -0.3, num_training_examples)
    x4 = np.random.uniform(0.5, 1, num_training_examples)
    x5 = np.random.uniform(-1, -0.5, num_training_examples)
    x6 = np.random.uniform(-3, -2.6, num_training_examples)
    x7 = np.random.uniform(-0.1, 0, num_training_examples)
    x8 = np.random.uniform(0, 0.1, num_training_examples)
    x9 = np.random.uniform(0, 0.8, num_training_examples)
    X = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9], axis=1)
    a = 10
    b = 2 * 10**-4
    c = 0.9 * np.pi
    eps = np.random.lognormal(size=num_training_examples)
    A = a * np.exp(-b * np.sqrt((1/9)*(np.sum(X**2, axis=1))) -
                   np.exp((1/9)*(np.sum(np.cos(c*X), axis=1)))
                  ) + a + np.exp(1)
    R = 3 * A
    y = 30 * A + R*eps
    return [x1, x2, x3, x4, x5, x6, x7, x8, x9], y

  def compute_quantile_mses(self, model, gasthaus=False):

    def true_y(X, q):
      lognorm_inv_cdf = scipy.stats.lognorm.ppf(q, s=1)
      a = 10
      b = 2 * 10**-4
      c = 0.9 * np.pi
      A = a * np.exp(-b * np.sqrt((1/9)*(np.sum(X**2, axis=1))) -
                     np.exp((1/9)*(np.sum(np.cos(c*X), axis=1)))
                    ) + a + np.exp(1)
      R = 3 * A
      return 30 * A + R*lognorm_inv_cdf

    # Use 10k random samples as full grid search is prohibitive.
    TEST_SIZE = 10000
    x1 = np.random.uniform(-1, -0.7, TEST_SIZE)
    x2 = np.random.uniform(0, 1, TEST_SIZE)
    x3 = np.random.uniform(-0.7, -0.3, TEST_SIZE)
    x4 = np.random.uniform(0.5, 1, TEST_SIZE)
    x5 = np.random.uniform(-1, -0.5, TEST_SIZE)
    x6 = np.random.uniform(-3, -2.6, TEST_SIZE)
    x7 = np.random.uniform(-0.1, 0, TEST_SIZE)
    x8 = np.random.uniform(0, 0.1, TEST_SIZE)
    x9 = np.random.uniform(0, 0.8, TEST_SIZE)
    X = np.stack([x1, x2, x3, x4, x5, x6, x7, x8, x9], axis=1)
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_test = true_y(X, q)
      if not gasthaus:
        y_pred = model([x1, x2, x3, x4, x5, x6, x7, x8, x9, np.full(x1.shape, q)])
      else:
        xs = [x1.astype(np.float32), x2.astype(np.float32),
              x3.astype(np.float32), x4.astype(np.float32),
              x5.astype(np.float32), x6.astype(np.float32),
              x7.astype(np.float32), x8.astype(np.float32),
              x9.astype(np.float32)]
        qs = tf.convert_to_tensor(np.full(x1.shape, q), dtype=tf.float32)
        y_pred = qr_lib_gasthaus.calculate_y_pred_tensor_gasthaus(model, xs, qs)
      q_mses.append(np.mean((y_pred.numpy().flatten() - y_test)**2))
    return q_mses

  def compute_monotonicity_violations(self, model):

    TEST_SIZE = 10000
    x1 = np.random.uniform(-1, -0.7, TEST_SIZE)
    x2 = np.random.uniform(0, 1, TEST_SIZE)
    x3 = np.random.uniform(-0.7, -0.3, TEST_SIZE)
    x4 = np.random.uniform(0.5, 1, TEST_SIZE)
    x5 = np.random.uniform(-1, -0.5, TEST_SIZE)
    x6 = np.random.uniform(-3, -2.6, TEST_SIZE)
    x7 = np.random.uniform(-0.1, 0, TEST_SIZE)
    x8 = np.random.uniform(0, 0.1, TEST_SIZE)
    x9 = np.random.uniform(0, 0.8, TEST_SIZE)
    old_preds = np.full(x1.shape, -np.inf)
    mono_violations = np.full(x1.shape, False)
    for q in np.linspace(0.01, 0.99, num=99):
      y_pred = model([x1, x2, x3, x4, x5, x6, x7, x8, x9, np.full(x1.shape, q)]).numpy().flatten()
      mono_violations = mono_violations | (y_pred - old_preds < -1e-4)
      old_preds = y_pred
    return np.mean(mono_violations)


class UniformTestCase(Simulation):
  """Uniform distribution (no x-features)."""

  def num_dims(self):
    return 0

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=2):
    del x_lattice_sizes, q_lattice_size, x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    inputs = keras.Input((1,))
    output = tfl.layers.PWLCalibration(
        np.linspace(0, 1, q_keypoints), monotonicity=FLAGS.q_monotonicity
    )(inputs)
    return keras.Model(inputs, output)

  def generate_training_data(self, num_training_examples):
    return [], np.random.uniform(size=num_training_examples)

  def compute_quantile_mses(self, model):
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = q
      y_pred = model(np.reshape(q, (-1, 1)))
      q_mses.append((y_pred.numpy().flatten() - y_true)**2)
    return q_mses

  def compute_sample_mses(self, y):
    sample_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = q
      y_sample = scipy.stats.mstats.mquantiles(y, q)[0]
      sample_mses.append((y_sample - y_true)**2)
    return sample_mses

  def compute_harrell_davis_mses(self, y):
    hd_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = q
      y_hd = scipy.stats.mstats.hdquantiles(y, q)[0]
      hd_mses.append((y_hd - y_true)**2)
    return hd_mses


class ExponentialTestCase(Simulation):
  """Exponential distribution (no x-features)."""

  def num_dims(self):
    return 0

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=2):
    del x_lattice_sizes, q_lattice_size, x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    inputs = keras.Input((1,))
    output = tfl.layers.PWLCalibration(
        np.linspace(0, 1, q_keypoints), monotonicity=FLAGS.q_monotonicity
    )(inputs)
    return keras.Model(inputs, output)

  def generate_training_data(self, num_training_examples):
    return [], np.random.exponential(size=num_training_examples)

  def compute_quantile_mses(self, model):
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.expon.ppf(q)
      y_pred = model(np.reshape(q, (-1, 1)))
      q_mses.append((y_pred.numpy().flatten() - y_true)**2)
    return q_mses

  def compute_validation_pinballs(self, model, y_val):
    val_pinballs = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_pred = model(np.reshape(q, (-1, 1))).numpy().flatten()
      diff = y_val - y_pred
      val_pinballs.append(tf.reduce_mean(
          tf.maximum(diff, 0.0) * q + tf.minimum(diff, 0.0) * (q - 1.0)))
    return val_pinballs

  def compute_sample_mses(self, y):
    sample_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.expon.ppf(q)
      y_sample = scipy.stats.mstats.mquantiles(y, q)[0]
      sample_mses.append((y_sample - y_true)**2)
    return sample_mses

  def compute_harrell_davis_mses(self, y):
    hd_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.expon.ppf(q)
      y_hd = scipy.stats.mstats.hdquantiles(y, q)[0]
      hd_mses.append((y_hd - y_true)**2)
    return hd_mses


class GammaTestCase(Simulation):
  """Gamma distribution (no x-features)."""

  def num_dims(self):
    return 0

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=2):
    del x_lattice_sizes, q_lattice_size, x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    inputs = keras.Input((1,))
    output = tfl.layers.PWLCalibration(
        np.linspace(0, 1, q_keypoints), monotonicity=FLAGS.q_monotonicity
    )(inputs)
    return keras.Model(inputs, output)

  def generate_training_data(self, num_training_examples):
    return [], np.random.gamma(2, 2, size=num_training_examples)

  def compute_quantile_mses(self, model):
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.gamma.ppf(q, 2, scale=2)
      y_pred = model(np.reshape(q, (-1, 1)))
      q_mses.append((y_pred.numpy().flatten() - y_true)**2)
    return q_mses

  def compute_sample_mses(self, y):
    sample_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.gamma.ppf(q, 2, scale=2)
      y_sample = scipy.stats.mstats.mquantiles(y, q)[0]
      sample_mses.append((y_sample - y_true)**2)
    return sample_mses

  def compute_harrell_davis_mses(self, y):
    hd_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.gamma.ppf(q, 2, scale=2)
      y_hd = scipy.stats.mstats.hdquantiles(y, q)[0]
      hd_mses.append((y_hd - y_true)**2)
    return hd_mses


class GaussianTestCase(Simulation):
  """Gaussian distribution (no x-features)."""

  def num_dims(self):
    return 0

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=2):
    del x_lattice_sizes, q_lattice_size, x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    inputs = keras.Input((1,))
    output = tfl.layers.PWLCalibration(
        np.linspace(0, 1, q_keypoints), monotonicity=FLAGS.q_monotonicity
    )(inputs)
    return keras.Model(inputs, output)

  def generate_training_data(self, num_training_examples):
    return [], np.random.normal(0.5, 1, size=num_training_examples)

  def compute_quantile_mses(self, model):
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.norm.ppf(q, 0.5, 1)
      y_pred = model(np.reshape(q, (-1, 1)))
      q_mses.append((y_pred.numpy().flatten() - y_true)**2)
    return q_mses

  def compute_sample_mses(self, y):
    sample_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.norm.ppf(q, 0.5, 1)
      y_sample = scipy.stats.mstats.mquantiles(y, q)[0]
      sample_mses.append((y_sample - y_true)**2)
    return sample_mses

  def compute_harrell_davis_mses(self, y):
    hd_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.norm.ppf(q, 0.5, 1)
      y_hd = scipy.stats.mstats.hdquantiles(y, q)[0]
      hd_mses.append((y_hd - y_true)**2)
    return hd_mses


class LaplaceTestCase(Simulation):
  """Laplace distribution (no x-features)."""

  def num_dims(self):
    return 0

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=2):
    del x_lattice_sizes, q_lattice_size, x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    inputs = keras.Input((1,))
    output = tfl.layers.PWLCalibration(
        np.linspace(0, 1, q_keypoints), monotonicity=FLAGS.q_monotonicity
    )(inputs)
    return keras.Model(inputs, output)

  def generate_training_data(self, num_training_examples):
    return [], np.random.normal(-0.5, 4, size=num_training_examples)

  def compute_quantile_mses(self, model):
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.laplace.ppf(q, -0.5, 4)
      y_pred = model(np.reshape(q, (-1, 1)))
      q_mses.append((y_pred.numpy().flatten() - y_true)**2)
    return q_mses

  def compute_sample_mses(self, y):
    sample_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.laplace.ppf(q, -0.5, 4)
      y_sample = scipy.stats.mstats.mquantiles(y, q)[0]
      sample_mses.append((y_sample - y_true)**2)
    return sample_mses

  def compute_harrell_davis_mses(self, y):
    hd_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.laplace.ppf(q, -0.5, 4)
      y_hd = scipy.stats.mstats.hdquantiles(y, q)[0]
      hd_mses.append((y_hd - y_true)**2)
    return hd_mses


class BetaTestCase(Simulation):
  """Beta distribution (no x-features)."""

  def num_dims(self):
    return 0

  def create_base_model(self, x_lattice_sizes=5, q_lattice_size=5,
                        x_keypoints=50, q_keypoints=2):
    del x_lattice_sizes, q_lattice_size, x_keypoints
    if FLAGS.q_keypoints is not None:
      q_keypoints = FLAGS.q_keypoints
    inputs = keras.Input((1,))
    output = tfl.layers.PWLCalibration(
        np.linspace(0, 1, q_keypoints), monotonicity=FLAGS.q_monotonicity
    )(inputs)
    return keras.Model(inputs, output)

  def generate_training_data(self, num_training_examples):
    return [], np.random.beta(5, 2, size=num_training_examples)

  def compute_quantile_mses(self, model):
    q_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.beta.ppf(q, 5, 2)
      y_pred = model(np.reshape(q, (-1, 1)))
      q_mses.append((y_pred.numpy().flatten() - y_true)**2)
    return q_mses

  def compute_sample_mses(self, y):
    sample_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.beta.ppf(q, 5, 2)
      y_sample = scipy.stats.mstats.mquantiles(y, q)[0]
      sample_mses.append((y_sample - y_true)**2)
    return sample_mses

  def compute_harrell_davis_mses(self, y):
    hd_mses = []
    for q in np.linspace(0.01, 0.99, num=99):
      y_true = scipy.stats.beta.ppf(q, 5, 2)
      y_hd = scipy.stats.mstats.hdquantiles(y, q)[0]
      hd_mses.append((y_hd - y_true)**2)
    return hd_mses
