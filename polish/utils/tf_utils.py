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

"""APIs for TF utiliy functions and runtime profiling."""
import contextlib
import math
import pickle
import time
from absl import flags
from absl import logging
import gin
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
FLAGS = flags.FLAGS

# Set logging verbosity
logging.set_verbosity(logging.INFO)


def load_machine_state(file_name):
  """Deserialize data from a file.

  Args:
    file_name: file name to read data.

  Returns:
    Expect to get an integer value representing the model version number.
    However, this API can be used to read any data in the file.
    If the file does not exist, we return `0` as the initial model
    version number.
  """
  try:
    with open(file_name, 'r') as f:
      return pickle.load(f)
  except Exception as e:  # pylint: disable=broad-except
    logging.info('Train state encountered an error %s', e)
    return 0


def save_machine_state(data, file_name):
  """Serialize data into a file.

  Args:
    data: array of data to store.
    file_name: file name to store data.
  """
  with open(file_name, 'w') as f:
    pickle.dump(data, f)


def export_model(working_dir, model_path, model_fn, serving_input):
  """Take the latest checkpoint & export it to path.

  Args:
    working_dir: The directory where tf.estimator keeps its checkpoints.
    model_path: The path to export the model to.
    model_fn: model_fn of model.
    serving_input: function for processing input.
  """
  estimator = tf.estimator.Estimator(model_fn, model_dir=working_dir)
  estimator.export_saved_model(
      model_path, serving_input_receiver_fn=serving_input)


@contextlib.contextmanager
def summary_timer(name, message, step=None, summarywriter=None):
  """Context manager for timing snippets of code and write to a TF summary.

  Echos to logging module.

  Args:
    name: summary name.
    message: message for printing.
    step: current step.
    summarywriter: handle for summarywriter.

  Yields:
    yield the operation.

  """
  tick = time.time()
  yield
  tock = time.time()

  logging.info('%s: %.5f seconds', message, (tock - tick))
  if summarywriter:
    assert (step), 'Provide a value for the step function'
    add_summary(tock - tick, name, step, summarywriter)


def add_summary(value, name, index, summarywriter):
  """Add a scalar summary to the summarywriter.

  Args:
    value: scalar value of the summary.
    name: name to be used for summary.
    index: x-axis in summary (usually global step).
    summarywriter: the handle for the summary writer.
  """
  summary_result = tf.Summary()
  summary_result.value.add(tag=name, simple_value=None)
  summary_result.value[0].simple_value = value
  summarywriter.add_summary(summary_result, index)


def summary_stats(in_list, category, tag, index, summarywriter, reduce_en=True):
  """Add a mix of stats (mean, min, max, and variance) to the summarywriter.

  Args:
    in_list: input array to calculate stats.
    category: category for summary (each category is a tab in TensorBoard).
    tag: used for presenting the summary in TensorBoard.
    index: step for summary.
    summarywriter: handle for the Tf summary writer.
    reduce_en: if set, uses reduce operation.
  """
  summary_format = '{category}/{type} {tag}'
  add_summary(
      np.mean(in_list),
      summary_format.format(category=category, type='Mean', tag=tag), index,
      summarywriter)
  add_summary(
      np.median(in_list),
      summary_format.format(category=category, type='Median', tag=tag), index,
      summarywriter)
  add_summary(
      np.min(in_list),
      summary_format.format(category=category, type='Min', tag=tag), index,
      summarywriter)
  add_summary(
      np.max(in_list),
      summary_format.format(category=category, type='Max', tag=tag), index,
      summarywriter)
  if reduce_en:
    add_summary(
        np.mean(np.var(in_list, axis=0)),
        summary_format.format(category=category, type='Std', tag=tag), index,
        summarywriter)
  else:
    add_summary(
        np.mean(np.std(in_list)) / math.sqrt(len(in_list)),
        summary_format.format(category=category, type='Std Err', tag=tag),
        index, summarywriter)


@gin.configurable
def get_tpu_estimator(
    working_dir,
    model_fn,
    iterations_per_loop=320,
    keep_checkpoint_max=20,
    use_tpu=False,
    train_batch_size=64):

  """Obtain an TPU estimator from a directory.

  Args:
    working_dir: the directory for holding checkpoints.
    model_fn: an estimator model function.
    iterations_per_loop: number of steps to run on TPU before outfeeding
        metrics to the CPU. If the number of iterations in the loop would exceed
        the number of train steps, the loop will exit before reaching
        --iterations_per_loop. The larger this value is, the higher
        the utilization on the TPU. For CPU-only training, this flag is equal to
        `num_epochs * num_minibatches`.
    keep_checkpoint_max: the maximum number of checkpoints to save in checkpoint
      directory.
    use_tpu: if True, training happens on TPU.
    train_batch_size: minibatch size for training which is equal to total number
      of data // number of batches.

  Returns:
    Returns a TPU estimator.
  """
  # If `TPUConfig.per_host_input_for_training` is `True`, `input_fn` is
  # invoked per host rather than per core. In this case, a global batch size
  # is transformed a per-host batch size in params for `input_fn`,
  # but `model_fn` still gets per-core batch size.
  run_config = tf.estimator.tpu.RunConfig(
      master=FLAGS.master,
      evaluation_master=FLAGS.master,
      model_dir=working_dir,
      save_checkpoints_steps=iterations_per_loop,
      save_summary_steps=iterations_per_loop,
      keep_checkpoint_max=keep_checkpoint_max,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          per_host_input_for_training=True,
          tpu_job_name=FLAGS.tpu_job_name))

  return tf.estimator.tpu.TPUEstimator(
      use_tpu=use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=train_batch_size)


def create_predictor(estimator, serving_input_func):
  """Create an estimator for prediction.

  Args:
    estimator: TF estimator.
    serving_input_func: placeholder for input to model.
  """
  return tf.contrib.predictor.from_estimator(estimator, serving_input_func)


@gin.configurable
def create_estimator(
    working_dir,
    model_fn,
    keep_checkpoint_max=20,
    iterations_per_loop=320,
    warmstart=None):
  """Create a TF estimator. Used when not using TPU.

  Args:
    working_dir: working directory for loading the model.
    model_fn: an estimator model function.
    keep_checkpoint_max: the maximum number of checkpoints to save in checkpoint
      directory.
    iterations_per_loop: number of steps to run on TPU before outfeeding
      metrics to the CPU. If the number of iterations in the loop would exceed
      the number of train steps, the loop will exit before reaching
      --iterations_per_loop. The larger this value is, the higher
      the utilization on the TPU. For CPU-only training, this flag is equal to
      `num_epochs * num_minibatches`.
    warmstart: if not None, warm start the estimator from an existing
      checkpoint.
  """
  run_config = tf.estimator.RunConfig(
      save_checkpoints_steps=iterations_per_loop,
      save_summary_steps=iterations_per_loop,
      keep_checkpoint_max=keep_checkpoint_max)

  if warmstart is not None:
    return tf.estimator.Estimator(
        model_fn, model_dir=working_dir, config=run_config,
        warm_start_from=warmstart)
  else:
    return tf.estimator.Estimator(
        model_fn, model_dir=working_dir, config=run_config)


@gin.configurable
def serving_input_fn(env_state_space=2):
  """Serving input function for creating predictor.

  Args:
    env_state_space: size of environment state space
      (the first dimension of the state space) used as input
      to the policy network.
  Returns:
    A serving input receiver function.
  """
  x = tf.placeholder(dtype=tf.float32, shape=[None, env_state_space])
  features = {'mcts_features': x, 'policy_features': x}
  return tf.estimator.export.ServingInputReceiver(features, features)
