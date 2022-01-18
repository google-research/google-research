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

"""Runner for neural clustering process experiments.
"""
import itertools
import os
import sys
import timeit

from . import em
from . import gmm
from . import ncp as ncp_models
from . import plotting

from absl import app
from absl import flags
import jax
from jax import jit
from jax import vmap
import jax.experimental.optimizers
import jax.numpy as np
import matplotlib.pyplot as plt

MODE_SHAPE_GAUSSIAN = "gaussian"
MODE_SHAPE_BANANA = "banana"
MODE_SHAPES = [MODE_SHAPE_GAUSSIAN, MODE_SHAPE_BANANA]

flags.DEFINE_integer("data_dim", 2,
                     "The dimension of the points to cluster.")
flags.DEFINE_integer("num_data_points", 25,
                     "The number of points to sample per MM instance.")
flags.DEFINE_integer("num_modes", 2,
                     "The true number of modes in the data.")
flags.DEFINE_enum("mode_shape", MODE_SHAPE_GAUSSIAN, MODE_SHAPES,
                  "The shape of modes.")
flags.DEFINE_float("mu_prior_mean", 0,
                   "The mean of the prior distribution over mixture means.")
flags.DEFINE_float("mu_prior_scale", 1.,
                   "The mean of the prior distribution over mixture means.")
flags.DEFINE_float("mode_scale", 0.2,
                   "The true scale of each mixture mode.")
flags.DEFINE_integer("h_dim", 16,
                     "The dimension of NCP's h representation.")
flags.DEFINE_integer("u_dim", 16,
                     "The dimension of NCP's u representation.")
flags.DEFINE_integer("g_dim", 16,
                     "The dimension of NCP's g representation.")
flags.DEFINE_integer("hidden_layer_dim", 256,
                     "The number of features for the hidden layers in NCP.")
flags.DEFINE_integer("num_hidden_layers", 3,
                     "The number of hidden layers in NCP.")
flags.DEFINE_integer("batch_size", 64,
                     "The batch size.")
flags.DEFINE_integer("eval_batch_size", 128,
                     "The batch size to use for computing average accuracies.")
flags.DEFINE_list("eval_num_data_points", [25, 50, 100],
                  "A list of numbers of data points to use for computing "
                  "accuracies.")
flags.DEFINE_integer("num_steps", int(1e6), "The number of steps to train for.")
flags.DEFINE_float("lr", 1e-3, "The learning rate for ADAM.")
flags.DEFINE_integer("summarize_every", 100,
                     "Number of steps between summaries.")
flags.DEFINE_string("logdir", "/tmp/ncp",
                    "The directory to put summaries and checkpoints.")
FLAGS = flags.FLAGS


def train_ncp(data_dim=2,
              num_data_points=25,
              num_modes=2,
              mode_shape=MODE_SHAPE_GAUSSIAN,
              mu_prior_mean=0.,
              mu_prior_scale=1.,
              mode_scale=0.2,
              h_dim=16,
              u_dim=16,
              g_dim=16,
              hidden_layer_dim=256,
              num_hidden_layers=3,
              batch_size=16,
              eval_batch_size=128,
              eval_num_data_points=[25],
              lr=1e-4,
              num_steps=100000,
              summarize_every=100):

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  ncp = ncp_models.NCP(h_dim, u_dim, g_dim, data_dim, hidden_layer_dim,
                       num_hidden_layers, subkey)

  def sample_batch(key, num_data_points, batch_size):
    keys = jax.random.split(key, num=(1 + batch_size))
    mus = jax.random.normal(
        keys[0], shape=[batch_size, num_modes, data_dim]
        ) * mu_prior_scale + mu_prior_mean
    if mode_shape == MODE_SHAPE_GAUSSIAN:
      xs, cs = vmap(
          gmm.sample,
          in_axes=(0, None, None, None, 0))(
              mus, mode_scale, np.ones([num_modes]) / num_modes,
              num_data_points, keys[1:])
    else:
      xs, cs = vmap(
          gmm.sample_bananas,
          in_axes=(0, None, None, None, None, 0))(
              mus, mode_scale, 1., np.ones([num_modes]) / num_modes,
              num_data_points, keys[1:])
    return xs, cs

  sample_batch = jit(sample_batch, static_argnums=(1, 2))

  def kl(params, key):
    xs, cs = sample_batch(key, num_data_points, batch_size)
    lls = vmap(
        lambda x, c, p: ncp._ll(x, c, p), in_axes=(0, 0, None))(xs, cs, params)
    return -np.mean(lls)

  kl_and_grad = jit(jax.value_and_grad(kl, argnums=0))

  def ncp_accuracy(xs, cs, params, key):
    ncp_predicted_cs = ncp._sample(xs, params, key)
    permutations = np.array(list(itertools.permutations(range(7))))
    ncp_permuted_cs = jax.lax.map(lambda p: p[ncp_predicted_cs], permutations)
    ncp_acc = np.max(jax.lax.map(
        lambda pcs: np.mean(cs == pcs), ncp_permuted_cs))
    return ncp_acc

  def avg_ncp_accuracy(num_data_points, params, key):
    keys = jax.random.split(key, num=eval_batch_size+1)
    xs, cs = sample_batch(keys[0], num_data_points, eval_batch_size)
    ncp_acc = vmap(ncp_accuracy, in_axes=(0, 0, None, 0))(
        xs, cs, params, keys[1:])
    return np.mean(ncp_acc)

  avg_ncp_accuracy = jit(avg_ncp_accuracy, static_argnums=0)

  def em_accuracy(xs, cs, key):
    _, _, em_log_membership_weights = em.em(xs, num_modes, 25, key)
    em_predicted_cs = np.argmax(em_log_membership_weights, axis=1)
    permutations = np.array(list(itertools.permutations(range(num_modes))))
    permuted_cs = jax.lax.map(lambda p: p[cs], permutations)
    em_acc = np.max(jax.lax.map(
        lambda pcs: np.mean(pcs == em_predicted_cs), permuted_cs))
    return em_acc

  def avg_em_accuracy(batch_size, key):
    keys = jax.random.split(key, num=batch_size+1)
    xs, cs = sample_batch(keys[0], num_data_points, batch_size)
    em_acc = vmap(em_accuracy)(xs, cs, keys[1:])
    return np.mean(em_acc)

  avg_em_accuracy = jit(avg_em_accuracy, static_argnums=0)

  def plot(num_data_points, writer, step, params, key):
    keys = jax.random.split(key, num=4)
    mus = jax.random.normal(
        keys[0], shape=[num_modes, data_dim]) * mu_prior_scale + mu_prior_mean
    if mode_shape == MODE_SHAPE_GAUSSIAN:
      xs, _ = gmm.sample(mus, mode_scale,
                         np.ones([num_modes]) / num_modes, num_data_points,
                         keys[1])
    else:
      xs, _ = gmm.sample_bananas(mus, mode_scale, 1.,
                                 np.ones([num_modes]) / num_modes,
                                 num_data_points, keys[1])
    ncp_predicted_cs = ncp._sample(xs, params, keys[2])
    num_predicted_modes = np.max(ncp_predicted_cs) + 1
    writer.scalar(
        "num_predicted_modes_%d_points" % num_data_points,
        num_predicted_modes,
        step=step)
    _, _, em_log_membership_weights = em.em(xs, num_modes, 25, keys[3])
    em_predicted_cs = np.argmax(em_log_membership_weights, axis=1)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
    for i in range(num_predicted_modes):
      ncp_mode_i_xs = xs[ncp_predicted_cs == i]
      ax[0].plot(ncp_mode_i_xs[:, 0], ncp_mode_i_xs[:, 1], "o")
    for i in range(num_modes):
      em_mode_i_xs = xs[em_predicted_cs == i]
      ax[1].plot(em_mode_i_xs[:, 0], em_mode_i_xs[:, 1], "o")
      ax[1].plot(mus[i, 0], mus[i, 1], "r*")
      ax[0].plot(mus[i, 0], mus[i, 1], "r*")
    ax[0].set_title("NCP")
    ax[1].set_title("EM")
    plot_img = plotting.plot_to_numpy_image(plt)
    writer.image("plot_%d_points" % num_data_points, plot_img, step=step)
    plt.close(fig)

  def summarize(writer, step, params, key):
    if step == 0:
      key, subkey = jax.random.split(key)
      em_acc = avg_em_accuracy(10*eval_batch_size, subkey)
      writer.scalar("em_accuracy", em_acc, step=step)
      print("EM Accuracy: %0.2f" % (em_acc * 100))

    for num_pts in eval_num_data_points:
      key, subkey = jax.random.split(key)
      ncp_acc = avg_ncp_accuracy(num_pts, params, subkey)
      writer.scalar("ncp_accuracy_at_%d_points" % num_pts, ncp_acc, step=step)
      print("NCP Accuracy @ %d: %0.2f" % (num_pts, ncp_acc * 100))

    if data_dim == 2:
      plot(eval_num_data_points[-1], writer, step, params, key)

  def train_step(t, opt_state, key):
    params = opt_get_params(opt_state)
    kl_val, kl_grad = kl_and_grad(params, key)
    opt_state = opt_update(t, kl_grad, opt_state)
    return kl_val, opt_state

  def train_many_steps(t, num_steps, opt_state, key):

    def step(i, state):
      key, opt_state, _ = state
      key, subkey = jax.random.split(key)
      kl, new_opt_state = train_step(t + i, opt_state, subkey)
      return (key, new_opt_state, kl)

    _, new_opt_state, kl = jax.lax.fori_loop(
        0, num_steps, step, (key, opt_state, 0.))
    return new_opt_state, kl

  train_many_steps = jit(train_many_steps, static_argnums=1)

  sw = None
  opt_init, opt_update, opt_get_params = jax.experimental.optimizers.adam(lr)
  opt_state = opt_init(ncp.params)

  start = timeit.default_timer()
  t = 0
  while t < num_steps:
    key, subkey1, subkey2 = jax.random.split(key, num=3)
    opt_state, kl_val = train_many_steps(t, summarize_every, opt_state, subkey1)
    t += summarize_every
    print("Step %d KL: %0.4f" % (t, kl_val))
    sw.scalar("kl", kl_val, step=t)
    summarize(sw, t, opt_get_params(opt_state), subkey2)
    end = timeit.default_timer()
    steps_per_sec = summarize_every / (end - start)
    print("Steps/sec: %0.2f" % steps_per_sec)
    sw.scalar("steps_per_sec", steps_per_sec, step=t)
    start = end
    sw.flush()
    sys.stdout.flush()
  sw.close()


def make_logdir(config):
  basedir = config.logdir
  exp_dir = "num_pts_%d_data_dim_%d_num_modes_%d_mode_scale_%0.2f" % (
      FLAGS.num_data_points, FLAGS.data_dim, FLAGS.num_modes, FLAGS.mode_scale)
  return os.path.join(basedir, exp_dir)


def main(unused_argv):
  train_ncp(
      data_dim=FLAGS.data_dim,
      num_data_points=FLAGS.num_data_points,
      num_modes=FLAGS.num_modes,
      mode_shape=FLAGS.mode_shape,
      mu_prior_mean=FLAGS.mu_prior_mean,
      mu_prior_scale=FLAGS.mu_prior_scale,
      mode_scale=FLAGS.mode_scale,
      h_dim=FLAGS.h_dim,
      u_dim=FLAGS.u_dim,
      g_dim=FLAGS.g_dim,
      hidden_layer_dim=FLAGS.hidden_layer_dim,
      num_hidden_layers=FLAGS.num_hidden_layers,
      batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      eval_num_data_points=[int(x) for x in FLAGS.eval_num_data_points],
      lr=FLAGS.lr,
      num_steps=FLAGS.num_steps,
      summarize_every=FLAGS.summarize_every)


if __name__ == "__main__":
  app.run(main)
