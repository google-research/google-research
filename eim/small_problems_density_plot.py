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

"""Plot density from model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import numpy as np
from six.moves import range
import tensorflow as tf
import tensorflow_probability as tfp
from eim.models import his
from eim.models import lars
from eim.models import nis
from eim.models import rejection_sampling
import eim.small_problems_dists as dists
tfd = tfp.distributions

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_enum(
    "algo", "lars", ["lars", "nis", "his", "density", "rejection_sampling"],
    "The algorithm to run. Density draws the targeted density")
tf.app.flags.DEFINE_enum("target", dists.NINE_GAUSSIANS_DIST,
                         dists.TARGET_DISTS, "Distribution to draw data from.")
tf.app.flags.DEFINE_string(
    "energy_fn_sizes", "20,20",
    "List of hidden layer sizes for energy function as as comma "
    "separated list.")
tf.app.flags.DEFINE_float("proposal_variance", 1.0, "Variance for proposal distribution")
tf.app.flags.DEFINE_integer(
    "his_t", 5, "Number of steps for hamiltonian importance sampling.")
tf.app.flags.DEFINE_float("his_stepsize", 1e-2,
                          "Stepsize for hamiltonian importance sampling.")
tf.app.flags.DEFINE_float("his_alpha", 0.995,
                          "Alpha for hamiltonian importance sampling.")
tf.app.flags.DEFINE_boolean("his_learn_stepsize", False,
                            "Allow HIS to learn the stepsize")
tf.app.flags.DEFINE_boolean("his_learn_alpha", False,
                            "Allow HIS to learn alpha.")
tf.app.flags.DEFINE_integer("K", 1024,
                            "The number of samples for NIS and LARS.")
tf.app.flags.DEFINE_integer("num_bins", 500,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_integer("num_samples", 10000000,
                            "Number of samples to use when plotting density.")
tf.app.flags.DEFINE_integer("batch_size", 100000, "The batch size.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                           "Directory for summaries and checkpoints.")

FLAGS = tf.app.flags.FLAGS


def make_sample_density_summary(session,
                                data,
                                max_samples_per_batch=100000,
                                num_samples=1000000,
                                num_bins=100):
  """Plot approximate density based on samples."""
  bounds = (-2, 2)
  num_batches = int(math.ceil(num_samples / float(max_samples_per_batch)))
  hist = None
  for i in range(num_batches):
    tf.logging.info("Processing batch %d / %d of samples for density image." %
                    (i + 1, num_batches))
    s = session.run(data)
    if hist is None:
      hist = np.histogram2d(
          s[:, 0], s[:, 1], bins=num_bins, range=[bounds, bounds])[0]
    else:
      hist += np.histogram2d(
          s[:, 0], s[:, 1], bins=num_bins, range=[bounds, bounds])[0]
  with tf.io.gfile.GFile(os.path.join(FLAGS.logdir, "density"), "w") as out:
    np.save(out, hist)
  tf.logging.info("Density image saved to %s" %
                  os.path.join(FLAGS.logdir, "density.npy"))


def reduce_logavgexp(input_tensor, axis=None, keepdims=None, name=None):
  dims = tf.shape(input_tensor)
  if axis is not None:
    dims = tf.gather(dims, axis)
  denominator = tf.reduce_prod(dims)
  return (tf.reduce_logsumexp(
      input_tensor, axis=axis, keepdims=keepdims, name=name) -
          tf.log(tf.to_float(denominator)))


def make_density_summary(log_density_fn, num_bins=100):
  """Plot density."""
  bounds = (-2, 2)

  x = tf.range(
      bounds[0], bounds[1], delta=(bounds[1] - bounds[0]) / float(num_bins))
  grid_x, grid_y = tf.meshgrid(x, x, indexing="ij")
  grid_xy = tf.stack([grid_x, grid_y], axis=-1)

  log_z = log_density_fn(grid_xy)
  log_bigz = reduce_logavgexp(log_z)
  z = tf.exp(log_z - log_bigz)

  plot = tf.reshape(z, [num_bins, num_bins])
  return plot


def main(unused_argv):
  g = tf.Graph()
  with g.as_default():
    energy_fn_layers = [
        int(x.strip()) for x in FLAGS.energy_fn_sizes.split(",")
    ]
    if FLAGS.algo == "density":
      target = dists.get_target_distribution(FLAGS.target)
      plot = make_density_summary(target.log_prob, num_bins=FLAGS.num_bins)
      with tf.train.SingularMonitoredSession(
          checkpoint_dir=FLAGS.logdir) as sess:
        plot = sess.run(plot)
        with tf.io.gfile.GFile(os.path.join(FLAGS.logdir, "density"),
                               "w") as out:
          np.save(out, plot)
    elif FLAGS.algo == "lars":
      tf.logging.info("Running LARS")
      model = lars.SimpleLARS(
          K=FLAGS.K, data_dim=[2], accept_fn_layers=energy_fn_layers,
          proposal_variance=FLAGS.proposal_variance)
      plot = make_density_summary(
          lambda x: tf.squeeze(model.accept_fn(x)) + model.proposal.log_prob(x),
          num_bins=FLAGS.num_bins)
      with tf.train.SingularMonitoredSession(
          checkpoint_dir=FLAGS.logdir) as sess:
        plot = sess.run(plot)
        with tf.io.gfile.GFile(os.path.join(FLAGS.logdir, "density"),
                               "w") as out:
          np.save(out, plot)
    else:
      if FLAGS.algo == "nis":
        tf.logging.info("Running NIS")
        model = nis.NIS(
            K=FLAGS.K, data_dim=[2], energy_hidden_sizes=energy_fn_layers,
            proposal_variance=FLAGS.proposal_variance)
      elif FLAGS.algo == "his":
        tf.logging.info("Running HIS")
        model = his.FullyConnectedHIS(
            T=FLAGS.his_t,
            data_dim=[2],
            energy_hidden_sizes=energy_fn_layers,
            proposal_variance=FLAGS.proposal_variance,
            q_hidden_sizes=energy_fn_layers,
            init_step_size=FLAGS.his_stepsize,
            learn_stepsize=FLAGS.his_learn_stepsize,
            init_alpha=FLAGS.his_alpha,
            learn_temps=FLAGS.his_learn_alpha)
      elif FLAGS.algo == "rejection_sampling":
        logit_accept_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_size, activation="tanh")
            for layer_size in energy_fn_layers
        ] + [tf.keras.layers.Dense(1, activation=None)])
        model = rejection_sampling.RejectionSampling(
            T=FLAGS.K, data_dim=[2], logit_accept_fn=logit_accept_fn,
            proposal_variance=FLAGS.proposal_variance)
      samples = model.sample(FLAGS.batch_size)
      with tf.train.SingularMonitoredSession(
          checkpoint_dir=FLAGS.logdir) as sess:
        make_sample_density_summary(
            sess,
            samples,
            max_samples_per_batch=FLAGS.batch_size,
            num_samples=FLAGS.num_samples,
            num_bins=FLAGS.num_bins)


if __name__ == "__main__":
  tf.app.run(main)
