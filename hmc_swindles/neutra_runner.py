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

# python3
"""Binary to run a neutra experiment."""
# pylint: disable=invalid-name,missing-docstring
import os

from absl import app
from absl import flags
from absl import logging
import gin
import tensorflow.compat.v2 as tf
from hmc_swindles import neutra
from hmc_swindles import utils

tf.enable_v2_behavior()

flags.DEFINE_string("neutra_log_dir", "/tmp/neutra",
                    "Output directory for experiment artifacts.")
flags.DEFINE_string("checkpoint_log_dir", None,
                    "Output directory for checkpoints, if specified.")
flags.DEFINE_enum(
    "mode", "train", ["eval", "benchmark", "train", "objective"],
    "Mode for this run. Standard trains bijector, tunes the "
    "chain parameters and does the evals. Benchmark uses "
    "the tuned parameters and benchmarks the chain.")
flags.DEFINE_boolean(
    "restore_from_config", False,
    "Whether to restore the hyperparameters from the "
    "previous run.")
flags.DEFINE(utils.YAMLDictParser(), "hparams", "",
             "Hyperparameters to override.")
flags.DEFINE_string("tune_outputs_name", "tune_outputs",
                    "Name of the tune_outputs file.")
flags.DEFINE_string("eval_suffix", "", "Suffix for the eval outputs.")

FLAGS = flags.FLAGS


def Train(exp):
  log_dir = (
      FLAGS.checkpoint_log_dir
      if FLAGS.checkpoint_log_dir else FLAGS.neutra_log_dir)
  logging.info("Training")
  q_stats, secs_per_step = exp.Train()
  tf.io.gfile.makedirs(log_dir)
  utils.save_json(q_stats, os.path.join(log_dir, "q_stats"))
  utils.save_json(secs_per_step, os.path.join(log_dir, "secs_per_train_step"))


def TuneObjective(exp):
  log_dir = FLAGS.neutra_log_dir
  tf.io.gfile.makedirs(log_dir)
  objective = exp.TuneObjective()
  utils.save_json(objective, os.path.join(log_dir, "objective"))


def Benchmark(exp):
  log_dir = FLAGS.neutra_log_dir
  tf.io.gfile.makedirs(log_dir)

  tune_outputs = utils.load_json(os.path.join(log_dir, FLAGS.tune_outputs_name))
  if isinstance(tune_outputs, dict):
    tune_outputs = neutra.TuneOutputs(**tune_outputs)

  logging.info("Benchmarking")
  benchmark = exp.Benchmark(
      test_num_leapfrog_steps=tune_outputs.num_leapfrog_steps,
      test_step_size=tune_outputs.step_size,
      test_num_steps=100,
      test_batch_size=16384 * 8,
  )

  utils.save_json(benchmark,
                  os.path.join(log_dir, "bechmark" + FLAGS.eval_suffix))


@gin.configurable("eval_mode")
def Eval(exp, batch_size=256, total_batch=4096):
  log_dir = FLAGS.neutra_log_dir
  tf.io.gfile.makedirs(log_dir)

  tune_outputs = utils.load_json(os.path.join(log_dir, FLAGS.tune_outputs_name))
  if isinstance(tune_outputs, dict):
    tune_outputs = neutra.TuneOutputs(**tune_outputs)

  results = []
  for i in range(total_batch // batch_size):
    logging.info("Evaluating batch %d", i)
    res = exp.Eval(
        test_num_leapfrog_steps=tune_outputs.num_leapfrog_steps,
        test_step_size=tune_outputs.step_size,
        batch_size=batch_size,
    )

    def to_numpy(t):
      if isinstance(t, tf.Tensor):
        return t.numpy()
      else:
        return t

    res = tf.nest.map_structure(to_numpy, res)
    results.append(res)

  neutra_stats = neutra.AverageStats(results)

  utils.save_json(neutra_stats,
                  os.path.join(log_dir, "neutra_stats" + FLAGS.eval_suffix))


def main(argv):
  del argv
  if not hasattr(FLAGS.hparams, "items"):
    FLAGS.hparams = utils.YAMLDictParser().parse(FLAGS.hparams)

  log_dir = FLAGS.neutra_log_dir
  utils.BindHParams(FLAGS.hparams)
  if FLAGS.restore_from_config:
    with tf.io.gfile.GFile(os.path.join(log_dir, "config")) as f:
      gin.parse_config(f.read())

  tf.io.gfile.makedirs(log_dir)
  summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=10000)
  summary_writer.set_as_default()
  tf.summary.experimental.set_step(0)

  for i in range(10):
    try:
      checkpoint_log_dir = (
          FLAGS.checkpoint_log_dir
          if FLAGS.checkpoint_log_dir else FLAGS.neutra_log_dir)
      exp = neutra.NeuTraExperiment(log_dir=checkpoint_log_dir)
      with tf.io.gfile.GFile(os.path.join(log_dir, "config"), "w") as f:
        f.write(gin.config_str())
      logging.info("Config:\n%s", gin.config_str())

      checkpoint = checkpoint_log_dir + "/model.ckpt"
      if tf.io.gfile.exists(checkpoint + ".index"):
        logging.info("Restoring from %s", checkpoint)
        exp.checkpoint.restore(checkpoint)

      with utils.use_xla(False):
        if FLAGS.mode == "train":
          Train(exp)
        elif FLAGS.mode == "objective":
          TuneObjective(exp)
        elif FLAGS.mode == "benchmark":
          Benchmark(exp)
        elif FLAGS.mode == "eval":
          Eval(exp)
        break
    except tf.errors.InvalidArgumentError as e:
      if "NaN" in e.message:
        logging.error(e.message)
        logging.error("Got a NaN, try: %d", i)
      else:
        raise e


if __name__ == "__main__":
  app.run(main)
