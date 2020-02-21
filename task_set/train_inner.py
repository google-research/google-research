# coding=utf-8
# Copyright 2020 The Google Research Authors.
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
r"""Train a task with a given optimizer and monitor results when done.

Example usage:
```
  binary_path/train_inner --task_name="mlp_family_seed12" \
  --optimizer_name="adam8p_wide_grid_seed21" \
  --output_directory="/disk2/tmp/optfolder" \
  --alsologtostderr
```
"""
import json
import os
import time
from typing import Dict, Text, Tuple

from absl import app
from absl import flags
from absl import logging

import dataclasses

import numpy as np

from task_set import datasets
from task_set import registry
from task_set.optimizers import all_optimizers  # pylint: disable=unused-import
from task_set.tasks import all_tasks  # pylint: disable=unused-import
from task_set.tasks import base
import tensorflow.compat.v1 as tf


FLAGS = flags.FLAGS

flags.DEFINE_string("optimizer_name", None, "Name of optimizer to run.")
flags.DEFINE_string("task_name", None, "Name of task to run.")
flags.DEFINE_integer("training_steps", 10000,
                     "Number of training steps to run.")
flags.DEFINE_integer("eval_every_n", 200, "Number of steps between each eval.")
flags.DEFINE_integer("replica", 0, "Replica of run.")

flags.DEFINE_string("output_directory", None,
                    "Training directory to save summaries/checkpoints.")

NamedTensorDict = Dict[Text, tf.Tensor]
FourTensorTuple = Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]
FourDictTensorTuple = Tuple[NamedTensorDict, NamedTensorDict, NamedTensorDict,
                            NamedTensorDict]


def compute_averaged_loss(
    task,
    params,
    num_batches = 2,
    with_metrics = False):
  """Computes inner-task loss and metrics with num_batches mini-batches.

  Returns values for each of the 4 splits (train, valid-inner, valid-outer,
  test.) For each split, perform num_batches evaluations averaging the results
  (both losses, and optionally metrics).

  Args:
    task: Task used to compute the loss.
    params: Parameters for the task.
    num_batches: Number of batches to compute averages over.
    with_metrics: bool additionally compute and return averages over aux
      metrics.

  Returns:
    losses: len 4 tuple containing loss values for each split
    metrics: len 4 tuple containing dictionaries of metrics for each split.
      If with_metrics is false, these dictionaries are empty.
  """

  def compute_loss(split):
    inner_loss_and_maybe_aux = task.call_split(
        params, split, with_metrics=with_metrics)
    if not with_metrics:
      inner_loss_and_maybe_aux = inner_loss_and_maybe_aux, {}

    inner_loss_and_maybe_aux = inner_loss_and_maybe_aux  #  type: Tuple[tf.Tensor, Dict[Text, tf.Tensor]]
    return inner_loss_and_maybe_aux

  # Run a forward pass to get a dictionary with metrics of the right dtype.
  # This is needed ahead of time before tf.map_fn is called.
  # Because we are in graph mode, this does not incur overhead.
  _, tmp_aux = compute_loss(datasets.Split.TRAIN)
  dummy_aux = {
      k: tf.zeros(shape=[num_batches], dtype=v.dtype)
      for k, v in tmp_aux.items()
  }

  splits = [
      datasets.Split.TRAIN, datasets.Split.VALID_INNER,
      datasets.Split.VALID_OUTER, datasets.Split.TEST
  ]

  return_losses = []
  return_metrics = []
  for split in splits:
    # pylint: disable=cell-var-from-loop
    losses, metrics = tf.map_fn(lambda _: compute_loss(split),
                                (tf.to_float(tf.range(num_batches)), dummy_aux))
    # pylint: enable=cell-var-from-loop
    avg_loss = tf.reduce_mean(losses)
    avg_metric = {k: tf.reduce_mean(v) for k, v in metrics.items()}

    return_losses.append(avg_loss)
    return_metrics.append(avg_metric)

  return tuple(return_losses), tuple(return_metrics)


@dataclasses.dataclass(frozen=True)
class GraphEndpoints:
  """Class containing endpoints used for inner-training."""
  train_op: tf.Operation
  global_step: tf.Tensor
  init_op: tf.Operation
  test_loss: tf.Tensor
  valid_inner_loss: tf.Tensor
  valid_outer_loss: tf.Tensor
  train_loss: tf.Tensor


def build_training_graph(task_name,
                         optimizer_name,
                         num_batchs_per_evaluation = 5):
  """Build the tensorflow graph.

  Args:
    task_name: Name of task to build.
    optimizer_name: Name of the optimizer to use.
    num_batchs_per_evaluation: Number of batches to use when running a
      single evaluation op. Note, this op is run multiple times per evauation by
      training code.

  Returns:
    A dict containing TensorFlow tensors and operations used for training.
  """

  global_step = tf.train.get_or_create_global_step()

  task_mod = registry.task_registry.get_instance(task_name)
  params = task_mod.current_params()
  loss = task_mod.call_split(params, datasets.Split.TRAIN)
  opt = registry.optimizers_registry.get_instance(optimizer_name)

  train_op = opt.minimize(
      loss, var_list=list(params.values()), global_step=global_step)

  train_op = tf.group(train_op, name="train_op")

  (train_loss, valid_inner_loss, valid_outer_loss,
   test_loss), _ = compute_averaged_loss(task_mod, params,
                                         num_batchs_per_evaluation)

  init_op = tf.initialize_variables(task_mod.get_variables())

  return GraphEndpoints(
      train_op=train_op,
      global_step=global_step,
      init_op=init_op,
      test_loss=test_loss,
      valid_inner_loss=valid_inner_loss,
      valid_outer_loss=valid_outer_loss,
      train_loss=train_loss)


def train(
    train_log_dir,
    task_name,
    optimizer_name,
    training_steps = 10000,
    eval_every_n = 200,
    minibatch_per_evaluation = 50,
    parallel_evaluations = 5,
):
  """Train a model and monitor results.

  This function trains a task specified by the task_name using the optimizer
  from optimizer_name. It logs out 2 files, result and time_per_step to the
  train_log_dir for later processing.

  Args:
    train_log_dir: str Directory to write summaries out to.
    task_name: str Name of task to train.
    optimizer_name: Name of the optimizer to train with.
    training_steps: Number of training steps to perform.
    eval_every_n: Number of steps to run between each evaluation.
    minibatch_per_evaluation: Number of minibatches to run per evalulation
    parallel_evaluations: Number of minibatches to run in parallel in graph.
      Must cleanly devide into minibatch_per_evaluation.

  Returns:
    The resulting learning curves encoded as a json string.
  """

  if minibatch_per_evaluation % parallel_evaluations != 0:
    raise ValueError("minibatch_per_evaluation must be divisible by"
                     "parallel_evaluations")
  tf.gfile.MakeDirs(train_log_dir)

  g = build_training_graph(task_name, optimizer_name, parallel_evaluations)

  state = {"losses": {}, "time_per_step": []}

  config = tf.ConfigProto(
      intra_op_parallelism_threads=20, inter_op_parallelism_threads=20)

  with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(tf.get_collection(tf.GraphKeys.LOCAL_INIT_OP))

    step = sess.run(g.global_step)

    logging.info("Running init op")
    sess.run(g.init_op)

    while step <= training_steps:
      if step % eval_every_n == 0 or step == training_steps:
        logging.info("Evaluating %d", step)
        losses = []
        for _ in range(minibatch_per_evaluation // parallel_evaluations):
          tr, vai, vao, te = sess.run([
              g.train_loss, g.valid_inner_loss, g.valid_outer_loss, g.test_loss
          ])
          losses.append((tr, vai, vao, te))
        state["losses"][str(step)] = [float(np.mean(x)) for x in zip(*losses)]

      # Only log 10 steps to not flood info logs.
      if step < 10:
        logging.info("Running train_op %d (only logging first 10)", step)
      start_time = time.time()
      _, step = sess.run([g.train_op, g.global_step])
      state["time_per_step"].append(time.time() - start_time)

  # compute aggregate values for timing information.
  mean_time = np.mean(state["time_per_step"])
  num_steps = len(state["time_per_step"])
  mean_time_last_half = np.mean(state["time_per_step"][num_steps // 2:])
  median_time = np.median(state["time_per_step"])
  time_per_step = json.dumps({
      "mean_time": mean_time,
      "median_time": median_time,
      "mean_last_half": mean_time_last_half,
  })

  result = json.dumps(state["losses"])
  with tf.gfile.GFile(os.path.join(train_log_dir, "result"), "w") as f:
    f.write(result.encode("utf-8"))

  with tf.gfile.GFile(os.path.join(train_log_dir, "time_per_step"), "w") as f:
    f.write(time_per_step.encode("utf-8"))

  return result


def main(_):
  if not FLAGS.optimizer_name:
    raise ValueError("Must pass `optimizer_name`")
  if not FLAGS.task_name:
    raise ValueError("Must pass `task_name`")
  if not FLAGS.output_directory:
    raise ValueError("Must pass `output_directory`")
  train_log_dir = os.path.join(FLAGS.output_directory, FLAGS.task_name,
                               FLAGS.optimizer_name, str(FLAGS.training_steps),
                               str(FLAGS.replica))
  train(
      train_log_dir=train_log_dir,
      optimizer_name=FLAGS.optimizer_name,
      task_name=FLAGS.task_name,
      training_steps=FLAGS.training_steps,
      eval_every_n=FLAGS.eval_every_n)


if __name__ == "__main__":
  app.run(main)
