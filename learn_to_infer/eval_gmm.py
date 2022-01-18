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

"""Script for running evaluations of GMM transformers.
"""
import os

from . import gmm_dist
from . import gmm_models
from . import metrics
from . import util

from absl import app
from absl import flags
import jax
import jax.numpy as jnp


flags.DEFINE_enum("model_name", "mean",
                  ["mean", "mean_scale", "mean_scale_weight"],
                  "Model to run")
flags.DEFINE_integer("num_encoders", 6,
                     "Number of encoder modules in the transformer.")
flags.DEFINE_integer("num_decoders", 6,
                     "Number of decoder modules in the transformer.")
flags.DEFINE_integer("num_heads", 8,
                     "Number of attention heads in the transformer.")
flags.DEFINE_integer("key_dim", 32,
                     "The dimension of the keys in the transformer.")
flags.DEFINE_integer("value_dim_per_head", 32,
                     "The dimension of the values in the transformer for "
                     "each head.")
flags.DEFINE_integer("data_dim", 2,
                     "The dimension of the points to cluster.")
flags.DEFINE_list("eval_ks", [2],
                  "The modes to eval on.")
flags.DEFINE_integer("max_k", 10,
                     "The maximum number of modes in the data.")
flags.DEFINE_list("eval_num_data_points", [100],
                  "The dataset sizes to eval on.")
flags.DEFINE_integer("max_num_data_points", 100,
                     "Maximum number of data points in the data.")
flags.DEFINE_integer("cov_dof", 10,
                     "Degrees of freedom in sampling the random covariances.")
flags.DEFINE_float("separation_mult", 2.,
                   "The value to multiply the max cov diag element by when "
                   "computing the minimum mean separation.")
flags.DEFINE_float("mode_var", 1.,
                   "The variance of the modes in the GMM used when not "
                   "sampling.")
flags.DEFINE_integer("eval_batch_size", 256,
                     "The batch size for evaluation.")
flags.DEFINE_string("logdir", "/tmp/transformer",
                    "The directory to load checkpoints from.")

FLAGS = flags.FLAGS


def make_model(key,
               model_name="mean",
               num_encoders=4,
               num_decoders=4,
               num_heads=8,
               value_dim=128,
               max_num_data_points=200,
               max_k=10,
               data_dim=2):
  class_dict = {
      "mean": gmm_models.MeanInferenceMachine,
      "mean_scale": gmm_models.MeanScaleInferenceMachine,
      "mean_scale_weight": gmm_models.MeanScaleWeightInferenceMachine}

  model = class_dict[model_name](
      data_dim=data_dim, max_k=max_k,
      max_num_data_points=max_num_data_points, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim)
  params = model.init_params(key)

  return model, params


def make_summarize(
    model,
    model_name="mean",
    eval_ks=[2],
    eval_num_data_points=[100],
    max_k=10,
    max_num_data_points=100,
    cov_dof=10,
    separation_mult=2.,
    data_dim=2,
    mode_var=1.,
    eval_batch_size=256):

  def sample_eval_batch(key, k, num_data_points):
    return gmm_dist.sample_batch_fixed_ks2(
        key, model_name, jnp.full([eval_batch_size], k), max_k,
        max_num_data_points, data_dim, mode_var, cov_dof, separation_mult)

  def model_classify(params, inputs, input_length, k):
    return gmm_models.classify_with_defaults(
        model, params, inputs, eval_batch_size,
        jnp.full([eval_batch_size], input_length, dtype=jnp.int32),
        jnp.full([eval_batch_size], k, dtype=jnp.int32),
        max_k, jnp.eye(data_dim)*mode_var)

  def sample_and_classify_eval_batch(key, params, k, num_data_points):
    xs, cs, true_gmm_params = sample_eval_batch(key, k, num_data_points)
    tfmr_cs, tfmr_gmm_params = model_classify(params, xs, num_data_points, k)
    return xs, cs, true_gmm_params, tfmr_cs, tfmr_gmm_params

  # sample_and_classify_eval_batch = jax.jit(
  #    sample_and_classify_eval_batch, static_argnums=(2,3))

  def summarize(params, key):
    for k, num_data_points in zip(eval_ks, eval_num_data_points):
      ks = jnp.full([eval_batch_size], k, dtype=jnp.int32)
      ndps = jnp.full([eval_batch_size], num_data_points, dtype=jnp.int32)
      _, cs, _, tfmr_cs, _ = sample_and_classify_eval_batch(
          key, params, k, num_data_points)
      tfmr_metrics = metrics.compute_masked_metrics(
          cs, tfmr_cs, ks, ndps,
          metrics=["pairwise_accuracy", "pairwise_f1",
                   "pairwise_macro_f1", "pairwise_micro_f1"])
      for metric_name, metric_val in tfmr_metrics.items():
        print("Transformer %d modes %d data points %s: %0.3f" %
              (k, num_data_points, metric_name, metric_val))

  return summarize


def make_logdir(config):
  basedir = config.logdir
  exp_dir = (
      "%s_nheads_%d_nencoders_%d_ndecoders_%d"
      % (config.model_name, config.num_heads, config.num_encoders,
         config.num_decoders))
  return os.path.join(basedir, exp_dir)


def main(unused_argv):
  logdir = make_logdir(FLAGS)

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model, init_params = make_model(
      key,
      model_name=FLAGS.model_name,
      num_encoders=FLAGS.num_encoders,
      num_decoders=FLAGS.num_decoders,
      num_heads=FLAGS.num_heads,
      value_dim=FLAGS.value_dim_per_head*FLAGS.num_heads,
      max_num_data_points=FLAGS.max_num_data_points,
      max_k=FLAGS.max_k,
      data_dim=FLAGS.data_dim)

  params = util.load_parameters(logdir, init_params)
  assert params is not None

  summarize_fn = make_summarize(
      model,
      model_name=FLAGS.model_name,
      max_k=FLAGS.max_k,
      max_num_data_points=FLAGS.max_num_data_points,
      eval_ks=[int(x) for x in FLAGS.eval_ks],
      eval_num_data_points=[int(x) for x in FLAGS.eval_num_data_points],
      cov_dof=FLAGS.cov_dof,
      separation_mult=FLAGS.separation_mult,
      data_dim=FLAGS.data_dim,
      mode_var=FLAGS.mode_var,
      eval_batch_size=FLAGS.eval_batch_size)

  summarize_fn(params, subkey)

if __name__ == "__main__":
  app.run(main)
