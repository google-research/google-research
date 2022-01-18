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

"""Runner for transformer experiments.
"""
import os

from . import metrics
from . import plotting
from . import ring_dist
from . import ring_models
from . import train

from absl import app
from absl import flags
import jax
from jax.config import config
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as onp


flags.DEFINE_integer("num_encoders", 6,
                     "Number of encoder modules in the transformer.")
flags.DEFINE_integer("num_decoders", 6,
                     "Number of decoder modules in the transformer.")
flags.DEFINE_integer("num_heads", 8,
                     "Number of attention heads in the transformer.")
flags.DEFINE_integer("key_dim", 32,
                     "The dimension of the keys in the transformer.")
flags.DEFINE_integer("value_dim_per_head", 32,
                     "The dimension of the values in the transformer for each head.")
flags.DEFINE_integer("k", 2,
                     "The number of modes in the data.")
flags.DEFINE_integer("data_points_per_mode", 25,
                     "Number of data points to include per mode in the data.")
flags.DEFINE_boolean("parallel", True,
                     "If possible, train in parallel across devices.")
flags.DEFINE_integer("batch_size", 64,
                     "The batch size.")
flags.DEFINE_integer("eval_batch_size", 256,
                     "The batch size for evaluation.")
flags.DEFINE_integer("num_steps", int(1e6),
                     "The number of steps to train for.")
flags.DEFINE_float("lr", 1e-3,
                   "The learning rate for ADAM.")
flags.DEFINE_integer("summarize_every", 100,
                     "Number of steps between summaries.")
flags.DEFINE_integer("checkpoint_every", 5000,
                     "Number of steps between checkpoints.")
flags.DEFINE_boolean("clobber_checkpoint", False,
                     "If true, remove any existing summaries and checkpoints in logdir.")
flags.DEFINE_string("logdir", "/tmp/transformer",
                    "The directory to put summaries and checkpoints.")
flags.DEFINE_boolean("debug_nans", False,
                     "If true, run in debug mode and fail on nans.")

FLAGS = flags.FLAGS


def make_model(key,
               num_encoders=4,
               num_decoders=4,
               num_heads=8,
               value_dim=128,
               data_points_per_mode=25,
               k=10):

  model = ring_models.RingInferenceMachine(
      max_k=k,
      max_num_data_points=k*data_points_per_mode, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim)
  params = model.init_params(key)

  return model, params


def sample_batch(key, batch_size, k, data_points_per_mode):
  keys = jax.random.split(key, num=batch_size)
  xs, cs, params = jax.vmap(
      ring_dist.sample_params_and_points,
      in_axes=(0, None, None, None, None, None, None, None, None,
               None))(keys, k * data_points_per_mode, k, 1., 0.5, 2, .02,
                      jnp.zeros([2]), jnp.eye(2), 0.1)
  return xs, cs, params


def make_loss(model,
              k=2,
              data_points_per_mode=25,
              batch_size=128):

  def sample_train_batch(key):
    xs, _, params = sample_batch(key, batch_size, k, data_points_per_mode)
    return xs, params

  def loss(params, key):
    key, subkey = jax.random.split(key)
    xs, ring_params = sample_train_batch(key)
    ks = jnp.full([batch_size], k)
    losses = model.loss(
        params, xs, ks*data_points_per_mode, ring_params, ks, subkey)
    return jnp.mean(losses)

  return jax.jit(loss)


def make_summarize(
    model,
    k=2,
    data_points_per_mode=25,
    eval_batch_size=256):

  def sample_eval_batch(key):
    return sample_batch(key, eval_batch_size, k, data_points_per_mode)

  sample_eval_batch = jax.jit(sample_eval_batch)

  def sample_single(key):
    xs, cs, params = sample_batch(key, 1, k, data_points_per_mode)
    return xs[0], cs[0], (params[0][0], params[1][0], params[2][0],
                          params[3][0])

  def model_classify(params, inputs, batch_size):
    return model.classify(params, inputs,
                          jnp.full([batch_size], k*data_points_per_mode),
                          jnp.full([batch_size], k))

  def sample_and_classify_eval_batch(key, params):
    xs, cs, true_ring_params = sample_eval_batch(key)
    tfmr_cs, tfmr_ring_params = model_classify(params, xs, eval_batch_size)
    return xs, cs, true_ring_params, tfmr_cs, tfmr_ring_params

  def sample_and_classify_single_mm(key, params):
    xs, cs, ring_params = sample_single(key)
    tfmr_cs, tfmr_ring_params = model_classify(params, xs[jnp.newaxis], 1)
    return xs, cs, ring_params, tfmr_cs, tfmr_ring_params

  sample_and_classify_eval_batch = jax.jit(sample_and_classify_eval_batch)

  sample_and_classify_single_mm= jax.jit(sample_and_classify_single_mm)

  def summarize_baselines(writer, step, key):
    key, subkey = jax.random.split(key)
    xs, cs, _ = sample_eval_batch(subkey)
    ks = onp.full([eval_batch_size], k)
    baseline_metrics = metrics.compute_masked_baseline_metrics(
        xs, cs, ks, ks*data_points_per_mode)
    for method_name, method_metrics in baseline_metrics.items():
      for metric_name, metric_val in method_metrics.items():
        writer.scalar("%s/%s" % (method_name, metric_name),
                      metric_val, step=step)
        print("%s %s: %0.3f" % (method_name, metric_name, metric_val))

  def plot_params(num_data_points, writer, step, params, key):
    outs = sample_and_classify_single_mm(key, params)
    xs, true_cs, true_params, pred_cs, pred_params = outs
    pred_cs = pred_cs[0]
    pred_params = (pred_params[0][0], pred_params[1][0],
                   pred_params[2][0], pred_params[3][0])
    fig = plotting.plot_rings(
        xs, k, true_cs, true_params, pred_cs, pred_params)
    plot_image = plotting.plot_to_numpy_image(plt)
    writer.image(
        "%d_modes_%d_points" % (k, num_data_points), plot_image, step=step)
    plt.close(fig)

  def comparison_inference(params):
    rings_inputs, true_cs = plotting.make_comparison_rings()
    rings_inputs = rings_inputs[jnp.newaxis, Ellipsis]
    new_model = ring_models.RingInferenceMachine(
        max_k=2, max_num_data_points=1500, num_heads=FLAGS.num_heads,
        num_encoders=FLAGS.num_encoders, num_decoders=FLAGS.num_decoders,
        qkv_dim=FLAGS.value_dim_per_head*FLAGS.num_heads)
    pred_cs, pred_params = new_model.classify(
        params, rings_inputs, jnp.array([1500]), jnp.array([2]))
    pred_cs = pred_cs[0]
    pred_params = (pred_params[0][0], pred_params[1][0],
                   pred_params[2][0], pred_params[3][0])
    return rings_inputs[0], true_cs, pred_cs, pred_params

  comparison_inference = jax.jit(comparison_inference)

  def plot_sklearn_comparison(writer, step, params):
    ring_xs, true_cs, pred_cs, pred_params = comparison_inference(params)
    fig = plotting.plot_comparison_rings(ring_xs, true_cs, pred_cs, pred_params)
    writer.image(
        "sklearn_comparison", plotting.plot_to_numpy_image(plt), step=step)
    plt.close(fig)

  def summarize(writer, step, params, key):
    k1, k2, k3 = jax.random.split(key, num=3)
    _, cs, _, tfmr_cs, _ = sample_and_classify_eval_batch(k1, params)
    ks = onp.full([eval_batch_size], k)
    tfmr_metrics = metrics.compute_masked_metrics(
        cs, tfmr_cs, ks, ks*data_points_per_mode,
        metrics=["pairwise_accuracy", "pairwise_f1",
                 "pairwise_macro_f1", "pairwise_micro_f1"])
    for metric_name, metric_val in tfmr_metrics.items():
      writer.scalar("transformer/%s" % metric_name,
                    metric_val, step=step)
      print("Transformer %s: %0.3f" % (metric_name, metric_val))

    plot_params(k*data_points_per_mode, writer, step, params, k2)
    plot_sklearn_comparison(writer, step, params)
    if step == 0:
      summarize_baselines(writer, step, k3)

  return summarize


def make_logdir(config):
  basedir = config.logdir
  exp_dir = (
      "ring_nheads_%d_nencoders_%d_ndecoders_%d_num_modes_%d"
      % (config.num_heads, config.num_encoders, config.num_decoders, config.k))
  return os.path.join(basedir, exp_dir)


def main(unused_argv):
  if FLAGS.debug_nans:
    config.update("jax_debug_nans", True)

  if FLAGS.parallel and train.can_train_parallel():
    assert FLAGS.batch_size % jax.local_device_count(
    ) == 0, "Device count must evenly divide batch_size"
    FLAGS.batch_size = int(FLAGS.batch_size / jax.local_device_count())

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model, init_params = make_model(
      key,
      num_encoders=FLAGS.num_encoders,
      num_decoders=FLAGS.num_decoders,
      num_heads=FLAGS.num_heads,
      value_dim=FLAGS.value_dim_per_head*FLAGS.num_heads,
      data_points_per_mode=FLAGS.data_points_per_mode,
      k=FLAGS.k)
  loss_fn = make_loss(
      model,
      k=FLAGS.k,
      data_points_per_mode=FLAGS.data_points_per_mode,
      batch_size=FLAGS.batch_size)
  summarize_fn = make_summarize(
      model,
      k=FLAGS.k,
      data_points_per_mode=FLAGS.data_points_per_mode,
      eval_batch_size=FLAGS.eval_batch_size)
  train.train_loop(
      subkey,
      init_params,
      loss_fn,
      parallel=FLAGS.parallel,
      lr=FLAGS.lr,
      num_steps=FLAGS.num_steps,
      summarize_fn=summarize_fn,
      summarize_every=FLAGS.summarize_every,
      checkpoint_every=FLAGS.checkpoint_every,
      clobber_checkpoint=FLAGS.clobber_checkpoint,
      logdir=make_logdir(FLAGS))

if __name__ == "__main__":
  app.run(main)
