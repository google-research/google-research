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

"""Tests for google_research.google_research.cold_posterior_bnn.core.ensemble."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import tempfile

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp

from cold_posterior_bnn.core import ensemble
from cold_posterior_bnn.core import statistics as stats

tfd = tfp.distributions


class EnsembleTest(parameterized.TestCase, tf.test.TestCase):

  def test_empirical_ensemble_load_save(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

    input_shape = (None, 10)
    model.build(input_shape=input_shape)

    orig_weights = model.get_weights()
    # Keras does not provide a good reinit function, just draw random weights:
    weights1 = [np.random.random(w.shape) for w in orig_weights]
    weights2 = [np.random.random(w.shape) for w in orig_weights]

    ens = ensemble.EmpiricalEnsemble(model, input_shape, [weights1, weights2])
    ensemble_dir = tempfile.mkdtemp("ensemble_test")
    ens.save_ensemble(ensemble_dir)

    ens2 = ensemble.EmpiricalEnsemble(model, input_shape, [])
    ens2.load_ensemble(ensemble_dir)
    self.assertEqual(
        len(model.layers), len(ens2.model.layers),
        "Saving and loading model must return same model.")
    self.assertEqual(
        len(ens2.weights_list), len(ens.weights_list),
        "Number of members must be equal.")

    for i, weights in enumerate(ens.weights_list):
      self.assertEqual(
          len(ens2.weights_list[i]), len(weights),
          "Number of weights must be equal.")
      for j, weight in enumerate(weights):
        self.assertEqual(ens2.weights_list[i][j].shape, weight.shape,
                         "Weights should have same shape.")
        self.assertAllEqual(
            ens2.weights_list[i][j], weight,
            "Weights after saving and loading should be equal.")

  def test_empirical_ensemble(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

    input_shape = (None, 10)
    model.build(input_shape=input_shape)

    orig_weights = model.get_weights()
    # Keras does not provide a good reinit function, just draw random weights:
    weights1 = [np.random.random(w.shape) for w in orig_weights]
    weights2 = [np.random.random(w.shape) for w in orig_weights]

    ens = ensemble.EmpiricalEnsemble(model, input_shape, [weights1, weights2])
    self.assertLen(ens, 2, msg="Empirical ensemble len wrong.")

    y_true = np.random.choice(10, 20)
    x = np.random.normal(0, 1, (20, 10))
    dataset = tf.data.Dataset.from_tensor_slices((x, y_true)).batch(4)
    stat_results = ens.evaluate_ensemble(dataset,
                                         [stats.ClassificationLogProb()])
    self.assertLen(
        stat_results,
        1,
        msg="Number of evaluation outputs differ from statistics count.")
    self.assertEqual(
        stat_results[0].shape, (len(x), 10),
        "Statistic result should have valid shape."
    )
    output = ens.predict_ensemble(dataset)
    self.assertEqual(
        output.shape, (len(ens), len(x), 10),
        "Output should have valid shape."
    )

  def test_empirical_ensemble_multi_out_dict(self):
    inp = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(10, activation="relu")(inp)
    out1 = tf.keras.layers.Dense(10, name="a")(x)
    out2 = tf.keras.layers.Dense(10, name="b")(x)
    out3 = tf.keras.layers.Dense(10, name="nolabel")(x)
    # Use the tf.keras functional api with named outputs.
    model = tf.keras.models.Model(inp, [out1, out2, out3])

    orig_weights = model.get_weights()
    # Keras does not provide a good reinit function, just draw random weights:
    weights1 = [np.random.random(w.shape) for w in orig_weights]
    weights2 = [np.random.random(w.shape) for w in orig_weights]

    input_shape = (None, 10)
    ens = ensemble.EmpiricalEnsemble(model, input_shape, [weights1, weights2])
    self.assertLen(ens, 2, msg="Empirical ensemble len wrong.")

    y_true = np.random.choice(10, 20)
    x = np.random.normal(0, 1, (20, 10))
    dataset = tf.data.Dataset.from_tensor_slices((x, {
        "a": y_true,
        "b": y_true
    })).batch(4)

    stat_results = ens.evaluate_ensemble(dataset, ({
        "a": [stats.Accuracy()],
        "b": [stats.ClassificationLogProb(),
              stats.Accuracy()],
        "nolabel": [stats.ClassificationLogProb()],
    }))
    self.assertLen(
        stat_results, 3, msg="Number of returned statistic_list should be 2")
    self.assertLen(
        stat_results["a"], 1, msg="Number of returned statistics should be 1")
    self.assertEqual(
        stat_results["b"][0].shape, (len(x), 10),
        "Statistic result should have valid shape."
    )
    self.assertEqual(
        stat_results["nolabel"][0].shape, (len(x), 10),
        "Statistic result should have valid shape."
    )

    outputs = ens.predict_ensemble(dataset)
    self.assertLen(outputs, 3)
    for output in outputs:
      self.assertEqual(
          output.shape, (len(ens), len(x), 10),
          "Predicted output should have valid shape."
      )

  def test_fresh_reservoir(self):
    res = ensemble.FreshReservoir(10000, 80.0)
    for i in range(1, 1000000 + 1):
      res.append_maybe(lambda: i)  # pylint: disable=cell-var-from-loop

    items = np.array([item for item in res]).astype(np.float64)
    desired_mean = 0.5 * (200000.0 + 1000000.0)
    computed_mean = np.mean(items)
    logging.info("Fresh reservoir, true mean %.2f, computed mean %.2f",
                 desired_mean, computed_mean)
    self.assertAlmostEqual(
        desired_mean,
        computed_mean,
        delta=1000.0,
        msg="Fresh reservoir item freshness is off.")

  @parameterized.parameters(
      itertools.product([10.0, 50.0, 75.0],
                        [10000, 100000],
                        [100, 250, 1000, 10000]))
  def test_fresh_reservoir_distribution(self, freshness, n, capacity):
    res = ensemble.FreshReservoir(capacity, freshness=freshness)
    for item in range(1, n+1):
      res.append_maybe(lambda: item)  # pylint: disable=cell-var-from-loop

    items = np.asarray(list(res))

    # Variance of the uniform discrete distr. on {(freshness/100)*n,...,n},
    # from https://en.wikipedia.org/wiki/Discrete_uniform_distribution
    var_true = ((n - ((100.0-freshness)/100.0)*n + 1)**2.0 - 1.0)/12.0

    # The number of fresh items, len(items), varies stochastically in fresh
    # reservoir sampling.
    sample_mean_stddev = math.sqrt(var_true) / math.sqrt(len(items))

    # Check mean is in +/- 4 sampling stddev
    mean_true = 0.5*(((100.0-freshness)/100.0)*n + n)
    mean_estimate = np.mean(items)
    logging.info("Reservoir(n=%d, cap=%d, freshness=%.1f) has "
                 "sample mean %.1f, true mean %.1f, sample stddev %.2f",
                 n, capacity, freshness, mean_estimate, mean_true,
                 sample_mean_stddev)
    self.assertAlmostEqual(mean_true, mean_estimate,
                           delta=4.0*sample_mean_stddev,
                           msg="Mean %.1f deviates from true mean %.1f by "
                           "more than allowed 4 sigma tolerance (%.2f)" % (
                               mean_estimate, mean_true,
                               4.0*sample_mean_stddev))

    # Check sample variance agrees with true variance
    var_var_est = 2.0*(var_true**2.0)/(len(items)-1)  # var of the sample var
    sample_var_stddev = math.sqrt(var_var_est)

    var_estimate = np.var(items)
    logging.info("Reservoir(n=%d, cap=%d, freshness=%.1f) has "
                 "sample var %.1f, true var %.1f, sample var stddev %.2f",
                 n, capacity, freshness, var_estimate, var_true,
                 sample_var_stddev)
    self.assertAlmostEqual(var_true, var_estimate,
                           delta=4.0*sample_var_stddev,
                           msg="Sample variance %.1f deviates from true "
                           "variance %.1f by more than allowed 4 sigma "
                           "tolerance (%.2f)" % (
                               var_estimate, var_true,
                               4.0*sample_var_stddev))

  def test_fresh_reservoir_ensemble(self):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(10)
    ])

    input_shape = (None, 10)
    model.build(input_shape=input_shape)
    orig_weights = model.get_weights()
    # Keras does not provide a good reinit function, just draw random weights:
    weights1 = [np.random.random(w.shape) for w in orig_weights]
    weights2 = [np.random.random(w.shape) for w in orig_weights]

    ens = ensemble.EmpiricalEnsemble(model, input_shape, [weights1, weights2])
    self.assertLen(ens, 2, msg="Empirical ensemble len wrong.")

    y_true = np.random.choice(10, 20)
    x = np.random.normal(0, 1, (20, 10))

    ens = ensemble.FreshReservoirEnsemble(model,
                                          input_shape,
                                          capacity=2,
                                          freshness=50)
    ens.append(weights1)
    ens.append(weights2)
    self.assertLen(ens, 1, msg="Fresh reservoir ensemble len wrong.")

    statistics = [stats.ClassificationLogProb()]
    ens_pred = ens.evaluate_ensemble(x, statistics)
    self.assertLen(
        statistics,
        len(ens_pred),
        msg="Number of prediction outputs differ from statistics count.")

    self.assertLen(
        x,
        int(ens_pred[0].shape[0]),
        msg="Ensemble prediction statistics output has wrong shape.")

    statistics = [stats.Accuracy(), stats.ClassificationCrossEntropy()]
    ens_eval = ens.evaluate_ensemble((x, y_true), statistics)
    self.assertLen(
        statistics,
        len(ens_eval),
        msg="Number of evaluation outputs differ from statistics count.")


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
