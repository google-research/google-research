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

"""Binary to run a neutra experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import timeit
import traceback
import time

import simplejson
import tensorflow as tf
import numpy as np
import gin

from neutra import utils
from neutra import neutra

from absl import app
from absl import flags

flags.DEFINE_string("neutra_log_dir", "/tmp/neutra",
                    "Output directory for experiment artifacts.")
flags.DEFINE_enum(
    "mode", "standard", ["standard", "eval", "all", "benchmark"],
    "Mode for this run. Standard trains bijector, tunes the "
    "chain parameters and does the evals. Benchmark uses "
    "the tuned parameters and benchmarks the chain.")
flags.DEFINE_boolean(
    "restore_from_config", False,
    "Whether to restore the hyperparameters from the "
    "previous run.")
flags.DEFINE(utils.YAMLDictParser(), "hparams", "",
             "Hyperparameters to override.")
flags.DEFINE_string("tune_outputs_name", "tune_outputs", "Name of the tune_outputs file.")
flags.DEFINE_string("eval_suffix", "", "Suffix for the eval outputs.")

FLAGS = flags.FLAGS


def Train(exp, sess):
  log_dir = FLAGS.neutra_log_dir

  global_step = sess.run(exp.global_step)
  if global_step == 0:
    tf.logging.info("Training")
    q_stats, secs_per_step = exp.TrainBijector(sess)
    utils.SaveJSON(q_stats, os.path.join(log_dir, "q_stats" + FLAGS.eval_suffix))
    utils.SaveJSON(secs_per_step, os.path.join(log_dir, "secs_per_train_step" + FLAGS.eval_suffix))

  tf.logging.info("Tuning")
  tune_outputs = exp.Tune(
      sess,
      feed={exp.test_num_steps: 500})

  utils.SaveJSON(tune_outputs, os.path.join(log_dir, FLAGS.tune_outputs_name))


def Benchmark(exp, sess):
  log_dir = FLAGS.neutra_log_dir

  with tf.gfile.Open(os.path.join(log_dir, FLAGS.tune_outputs_name)) as f:
    tune_outputs = neutra.TuneOutputs(**simplejson.load(f))

  tf.logging.info("Benchmarking")
  feed = {
      exp.test_num_leapfrog_steps: tune_outputs.num_leapfrog_steps,
      exp.test_step_size: tune_outputs.step_size,
      exp.test_num_steps: 100,
  }
  seconds_per_step = exp.Benchmark(sess, feed=feed)

  utils.SaveJSON(seconds_per_step, os.path.join(log_dir, "secs_per_hmc_step" + FLAGS.eval_suffix))


@gin.configurable("eval_mode")
def Eval(exp, sess, batch_size=256, total_batch=4096):
  log_dir = FLAGS.neutra_log_dir

  with tf.gfile.Open(os.path.join(log_dir, FLAGS.tune_outputs_name)) as f:
    tune_outputs = neutra.TuneOutputs(**simplejson.load(f))

  results = []
  for i in range(total_batch // batch_size):
    tf.logging.info("Evaluating batch %d", i)
    feed = {
        exp.test_num_leapfrog_steps: tune_outputs.num_leapfrog_steps,
        exp.test_step_size: tune_outputs.step_size,
        exp.test_chain_batch_size: batch_size,
    }
    res = exp.Eval(sess, feed=feed, p_accept_only=True)
    results.append(res)

  def classify(path):
    if "ess" in path:
      return lambda x: 1. / np.mean(1. / np.array(x), 0)
    else:
      return lambda x: np.mean(x, 0)

  avg_type = [classify("".join(str(p) for p in path)) for path in tf.contrib.framework.nest.yield_flat_paths(results[0])]
  flat_results = [tf.contrib.framework.nest.flatten(r) for r in results]
  trans_results = zip(*flat_results)
  trans_mean_results = [avg(r) for avg, r in zip(avg_type, trans_results)]
  neutra_stats, p_accept = tf.contrib.framework.nest.pack_sequence_as(
      results[0], trans_mean_results)

  utils.SaveJSON(neutra_stats, os.path.join(log_dir, "neutra_stats" + FLAGS.eval_suffix))
  utils.SaveJSON(p_accept, os.path.join(log_dir, "p_accept" + FLAGS.eval_suffix))


def main(argv):
  del argv

  log_dir = FLAGS.neutra_log_dir
  utils.BindHParams(FLAGS.hparams)
  if FLAGS.restore_from_config:
    with tf.gfile.Open(os.path.join(log_dir, "config")) as f:
      gin.parse_config(f.read())

  tf.gfile.MakeDirs(log_dir)
  summary_writer = tf.contrib.summary.create_file_writer(
      log_dir, flush_millis=10000)
  summary_writer.set_as_default()
  with tf.contrib.summary.always_record_summaries():
    exp = neutra.NeuTraExperiment(log_dir=log_dir)
    with tf.gfile.Open(os.path.join(log_dir, "config"), "w") as f:
      f.write(gin.operative_config_str())
      tf.logging.info("Config:\n%s", gin.operative_config_str())

    with tf.Session() as sess:
      exp.Initialize(sess)
      tf.contrib.summary.initialize(graph=tf.get_default_graph())

      checkpoint = tf.train.latest_checkpoint(log_dir)
      if checkpoint:
        tf.logging.info("Restoring from %s", checkpoint)
        exp.saver.restore(sess, checkpoint)

      if FLAGS.mode == "standard":
        Train(exp, sess)
        Benchmark(exp, sess)
      elif FLAGS.mode == "benchmark":
        Benchmark(exp, sess)
      elif FLAGS.mode == "eval":
        Benchmark(exp, sess)
        Eval(exp, sess)
      elif FLAGS.mode == "all":
        Train(exp, sess)
        Benchmark(exp, sess)
        Eval(exp, sess)


if __name__ == "__main__":
  app.run(main)
