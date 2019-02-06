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

"""Trains a Bayesian logistic regression model using No-U-Turn Sampler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cProfile
import functools
import os
import pstats
import time
from absl import flags
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
import numpy as np
import tensorflow as tf

from tensorflow_probability import edward2 as ed
from simple_probabilistic_programming import no_u_turn_sampler

tfe = tf.contrib.eager

flags.DEFINE_integer("max_steps",
                     default=5,
                     help="Number of training steps to run.")
flags.DEFINE_string("model_dir",
                    default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"),
                                         "logistic_regression/"),
                    help="Path to write plots to.")
flags.DEFINE_bool("fake_data",
                  default=None,
                  help="If True, uses fake data. Defaults to real data.")

FLAGS = flags.FLAGS


def profile(func):
  """Decorator for profiling the execution of a function."""
  @functools.wraps(func)
  def func_wrapped(*args, **kwargs):
    """Function which wraps original function with start/stop profiling."""
    pr = cProfile.Profile()
    pr.enable()
    start = time.time()
    output = func(*args, **kwargs)
    print("Elapsed", time.time() - start)
    pr.disable()
    ps = pstats.Stats(pr).sort_stats("cumulative")
    ps.print_stats()
    return output
  return func_wrapped


def logistic_regression(features):
  """Bayesian logistic regression, which returns labels given features."""
  coeffs = ed.MultivariateNormalDiag(
      loc=tf.zeros(features.shape[1]), name="coeffs")
  labels = ed.Bernoulli(
      logits=tf.tensordot(features, coeffs, [[1], [0]]), name="labels")
  return labels


def covertype():
  """Builds the Covertype data set."""
  import sklearn.datasets  # pylint: disable=g-import-not-at-top
  data = sklearn.datasets.covtype.fetch_covtype()
  features = data.data
  labels = data.target

  # Normalize features and append a column of ones for the intercept.
  features -= features.mean(0)
  features /= features.std(0)
  features = np.hstack([features, np.ones([features.shape[0], 1])])
  features = tf.cast(features, dtype=tf.float32)

  # Binarize outcomes on whether it is a specific category.
  _, counts = np.unique(labels, return_counts=True)
  specific_category = np.argmax(counts)
  labels = (labels == specific_category)
  labels = tf.cast(labels, dtype=tf.int32)
  return features, labels


def main(argv):
  del argv  # unused
  if tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warning(
        "Warning: deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  tf.enable_eager_execution()
  print("Number of available GPUs", tfe.num_gpus())

  if FLAGS.fake_data:
    features = tf.random_normal([20, 55])
    labels = tf.random_uniform([20], minval=0, maxval=2, dtype=tf.int32)
  else:
    features, labels = covertype()
  print("Data set size", features.shape[0])
  print("Number of features", features.shape[1])

  log_joint = ed.make_log_joint_fn(logistic_regression)
  def target_log_prob_fn(coeffs):
    return log_joint(features=features, coeffs=coeffs, labels=labels)

  # Initialize using a sample from 20 steps of NUTS. It is roughly a MAP
  # estimate and is written explicitly to avoid differences in warm-starts
  # between different implementations (e.g., Stan, PyMC3).
  coeffs = tf.constant(
      [+2.03420663e+00, -3.53567265e-02, -1.49223924e-01, -3.07049364e-01,
       -1.00028366e-01, -1.46827862e-01, -1.64167881e-01, -4.20344204e-01,
       +9.47479829e-02, -1.12681836e-02, +2.64442056e-01, -1.22087866e-01,
       -6.00568838e-02, -3.79419506e-01, -1.06668741e-01, -2.97053963e-01,
       -2.05253899e-01, -4.69537191e-02, -2.78072730e-02, -1.43250525e-01,
       -6.77954629e-02, -4.34899796e-03, +5.90927452e-02, +7.23133609e-02,
       +1.38526391e-02, -1.24497898e-01, -1.50733739e-02, -2.68872194e-02,
       -1.80925727e-02, +3.47936489e-02, +4.03552800e-02, -9.98773426e-03,
       +6.20188080e-02, +1.15002751e-01, +1.32145107e-01, +2.69109547e-01,
       +2.45785132e-01, +1.19035013e-01, -2.59744357e-02, +9.94279515e-04,
       +3.39266285e-02, -1.44057125e-02, -6.95222765e-02, -7.52013028e-02,
       +1.21171586e-01, +2.29205526e-02, +1.47308692e-01, -8.34354162e-02,
       -9.34122875e-02, -2.97472421e-02, -3.03937674e-01, -1.70958012e-01,
       -1.59496680e-01, -1.88516974e-01, -1.20889175e+00])

  # Initialize step size via result of 50 warmup steps from Stan.
  step_size = 0.00167132

  kernel = profile(no_u_turn_sampler.kernel)
  coeffs_samples = []
  target_log_prob = None
  grads_target_log_prob = None
  for step in range(FLAGS.max_steps):
    print("Step", step)
    [
        [coeffs],
        target_log_prob,
        grads_target_log_prob,
    ] = kernel(target_log_prob_fn=target_log_prob_fn,
               current_state=[coeffs],
               step_size=[step_size],
               seed=step,
               current_target_log_prob=target_log_prob,
               current_grads_target_log_prob=grads_target_log_prob)
    coeffs_samples.append(coeffs)

  for coeffs_sample in coeffs_samples:
    plt.plot(coeffs_sample.numpy())

  filename = os.path.join(FLAGS.model_dir, "coeffs_samples.png")
  plt.savefig(filename)
  print("Figure saved as", filename)
  plt.close()

if __name__ == "__main__":
  tf.app.run()
