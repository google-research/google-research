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

"""Given a learner, perform an unroll on a given inner-task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
import gin
import numpy as np
import setup_experiment
import tensorflow as tf

import tf_utils

nest = tf.contrib.framework.nest

flags.DEFINE_string("checkpoint_dir", None, "Checkpoint dir")

flags.DEFINE_string("master", "local", "Tensorflow target")

FLAGS = flags.FLAGS


def report_score(train_log_dir, name, value):
  filename = os.path.join(train_log_dir, name)
  with open(filename, "w") as f:
    f.write(str(value) + "\n")


def compute_train_meta_loss(loss_mod, params, num=1):
  """Compute the train (inner) and validation (outer, meta) loss.

  Computes an average over num batches.
  Args:
    loss_mod: sonnet loss module
    params: loss mod params
    num: int
  Returns:
    train_loss, meta_loss
  """

  def compute_train_loss(_):
    return loss_mod.call_inner(params)

  train_loss = tf.reduce_mean(
      tf.map_fn(compute_train_loss, tf.to_float(tf.range(num))))

  def compute_meta_loss(_):
    return loss_mod.call_outer(params)

  meta_loss = tf.reduce_mean(
      tf.map_fn(compute_meta_loss, tf.to_float(tf.range(num))))
  return train_loss, meta_loss


@gin.configurable
def build_evaluation_graph(loss_module_fn=gin.REQUIRED,
                           learner_fn=gin.REQUIRED):
  """Build the evaluation graph for inner-training."""
  global_step = tf.train.get_or_create_global_step()

  loss_module = loss_module_fn()
  learner, theta_mod = learner_fn(loss_module)

  initial_state = learner.initial_state()
  reset_state_op = learner.assign_state(initial_state)
  state = learner.current_state()
  state = tf_utils.force_copy(state)

  with tf.control_dependencies(nest.flatten(state)):
    last_loss, new_state = learner.loss_and_next_state(state)

  with tf.control_dependencies([last_loss] + nest.flatten(new_state)):
    train_op = learner.assign_state(new_state)

  update_global_step = global_step.assign_add(1)
  train_op = tf.group(train_op, update_global_step, name="train_op")

  load_vars = list(theta_mod.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))
  meta_loss = learner.meta_loss(state)
  meta_loss = tf.Print(meta_loss, [meta_loss], "meta_loss")

  inner_loss = learner.inner_loss(state)
  inner_loss = tf.Print(inner_loss, [inner_loss], "inner_loss")
  # TODO(lmetz) this should only be scalars.
  train_op = tf.group(train_op, name="train_op")

  return {
      "train_op": train_op,
      "global_step": global_step,
      "init_op": reset_state_op,
      "checkpoint_vars": load_vars,
      "meta_loss": meta_loss,
      "train_loss": inner_loss,
      "current_state": state,
      "next_state": new_state,
  }


@gin.configurable
def train(
    train_log_dir,
    loss_fn=gin.REQUIRED,
    checkpoint_dir=None,  # pylint: disable=unused-argument
    training_steps=10000,
    save_summary_every=50,  # pylint: disable=unused-argument
    save_checkpoint_every=100,  # pylint: disable=unused-argument
):
  """Train a model!

  Args:
    train_log_dir: str Directory to write summaries out to
    loss_fn: function to create loss module
    checkpoint_dir: str
    training_steps: int Number of training steps to perform
    save_summary_every: int Save summary once every `save_summary_ever` steps.
    save_checkpoint_every: int Save summary once every `save_checkpoint_ever`
      steps.
  """
  g = build_evaluation_graph(loss_module_fn=loss_fn)

  with tf.SingularMonitoredSession(master="local") as sess:
    step = sess.run(g["global_step"])
    sess.run(
        g["init_op"]
    )  # NOTE this is required as weights are random from wrong distribution

    # NOTINCLUDED
    # Load the weights of a learned optimizer.
    # NOTINCLUDED

    while (step <= training_steps) and (not sess.should_stop()):
      if step in [0, 10, 100, 1000, 10000, 100000]:
        meta_losses = []
        train_losses = []
        for _ in range(10):
          ml, te = sess.run([g["meta_loss"], g["train_loss"]])
          meta_losses.append(ml)
          train_losses.append(te)
        report_score(train_log_dir, "%d_meta_loss" % step, np.mean(meta_losses))
        report_score(train_log_dir, "%d_train_loss" % step,
                     np.mean(train_losses))

      _, step = sess.run([g["train_op"], g["global_step"]])


def main(_):
  train_log_dir = setup_experiment.setup_experiment("run_single_eval")
  return train(train_log_dir=train_log_dir, checkpoint_dir=FLAGS.checkpoint_dir)


if __name__ == "__main__":
  tf.app.run(main)
