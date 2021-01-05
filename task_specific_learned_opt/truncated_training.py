# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Manage truncated training."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time

from absl import flags
from absl import logging
import base_trainer as meta_trainers_base
import device_utils
import gin
import session_creators
import tensorflow.compat.v1 as tf

nest = tf.contrib.framework.nest

FLAGS = flags.FLAGS


@gin.configurable
def build_graph(
    loss_module_fn,
    learner_fn,
    trainer_class,
    np_global_step=None,  # pylint: disable=unused-argument
):
  """Build the inner / outer training graph."""
  local_device, remote_device, index_remote_device =\
      device_utils.get_local_remote_device_fn(FLAGS.ps_tasks)
  with tf.device(local_device):
    with tf.device(remote_device):
      global_step = tf.train.get_or_create_global_step()

    loss_module = loss_module_fn()

    learner, theta_mod = learner_fn(
        loss_module=loss_module,
        remote_device=remote_device,
    )

    trainer = trainer_class(
        local_device=local_device,
        remote_device=remote_device,
        index_remote_device=index_remote_device,
        learner=learner,
    )

    truncated_trainer_endpoints = trainer.build_endpoints()

    trainable_theta_vars = theta_mod.get_variables(
        tf.GraphKeys.TRAINABLE_VARIABLES)

    logging.info("GOT %d trainable variables", len(trainable_theta_vars))

    # The following is a sort of accounting of variables. It ensures variables
    # are where you think they are, one is not creating extras.
    # Variable management in distributed setting has caused great
    # issues in the past.
    # While verbose, this seems to mitigate a lot of it by being very explicit
    # and throwing errors.
    local_vars = list(learner.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))
    local_vars += list(
        learner.loss_module.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))

    local_vars += trainer.get_local_variables()

    saved_remote_vars = list(
        theta_mod.get_variables(tf.GraphKeys.GLOBAL_VARIABLES))
    saved_remote_vars += [global_step] + trainer.get_saved_remote_variables()

    not_saved_remote_vars = trainer.get_not_saved_remote_variables()
    # TODO(lmetz) remove this line. For now, the meta_opt places variables in
    # the wrong scopes
    saved_remote_vars = list(set(saved_remote_vars))

    all_remote_vars = saved_remote_vars + not_saved_remote_vars

    logging.info("Remote Saved Variables")
    for v in saved_remote_vars:
      logging.info("    %s\t\t %s \t %s", v.shape.as_list(), v.device,
                   v.op.name)

    logging.info("Remote Not Saved Variables")
    for v in not_saved_remote_vars:
      logging.info("    %s\t\t %s \t %s", v.shape.as_list(), v.device,
                   v.op.name)

    logging.info("Local Variables")
    for v in local_vars:
      logging.info("    %s\t\t %s \t %s", v.shape.as_list(), v.device,
                   v.op.name)

    logging.info("Trainable Theta Variables")
    for v in theta_mod.get_variables(tf.GraphKeys.TRAINABLE_VARIABLES):
      logging.info("    %s\t\t %s \t %s", v.shape.as_list(), v.device,
                   v.op.name)

    device_utils.check_variables_accounting(local_vars, all_remote_vars)
    device_utils.check_variables_are_local(local_vars)
    device_utils.check_variables_are_remote(all_remote_vars)

    chief_summary_op = tf.summary.merge([
        tf.summary.scalar("global_step", global_step),
    ])

    # Ops to run when a parameter server is reset.
    ps_was_reset_ops = [tf.initialize_variables(not_saved_remote_vars)]
    ps_was_reset_ops.append(trainer.ps_was_reset_op())
    ps_was_reset_op = tf.group(*ps_was_reset_ops, name="ps_was_reset_op")

  if FLAGS.ps_tasks == 0:
    chief_device = ""
  else:
    chief_device = "/job:chief"
  with tf.device(chief_device):
    chief_is_ready = tf.get_variable(
        name="chief_is_ready", initializer=tf.constant(False))
    set_chief_is_ready = chief_is_ready.assign(True)

  # this dictionary is result of / merged with trainer.
  return dict(
      global_step=global_step,
      chief_summary_op=chief_summary_op,
      saved_remote_vars=saved_remote_vars,
      remote_vars=all_remote_vars,
      local_vars=local_vars,
      chief_is_ready=chief_is_ready,
      set_chief_is_ready=set_chief_is_ready,
      trainer_trainer_ops=truncated_trainer_endpoints,
      ps_was_reset_op=ps_was_reset_op,
  )


@gin.configurable
def chief_run(train_dir,  # pylint: disable=unused-argument
              graph_dict,
              meta_training_steps=None,
              save_checkpoint_minutes=0.1):  # pylint: disable=unused-argument
  """Train a model."""
  g = graph_dict

  # NOTINCLUDED
  # checkpointing for saving and restoring models.
  # NOTINCLUDED

  hooks = []

  # turn off summaries.
  no_summary_op = tf.summary.merge([tf.summary.scalar("dummy", 0.)])
  scaffold = tf.train.Scaffold(summary_op=no_summary_op)
  session_creator = tf.train.ChiefSessionCreator(
      master=FLAGS.master, scaffold=scaffold)

  with tf.MonitoredSession(
      session_creator=session_creator, hooks=hooks) as sess:
    step = sess.run(g["global_step"])
    logging.info("Chief is ready!!!")
    sess.run(g["set_chief_is_ready"])
    step = sess.run(g["global_step"])

    while (meta_training_steps is None or
           step <= meta_training_steps) and (not sess.should_stop()):
      step, _ = sess.run([g["global_step"], g["chief_summary_op"]])
      tf.logging.info("Global step %d" % (step))
      time.sleep(10)


@gin.configurable
def worker_run(train_dir, graph_dict):  # pylint: disable=unused-argument
  """Run the worker."""
  g = graph_dict

  config = tf.ConfigProto()
  session_creator = session_creators.WorkerSessionCreator(
      local_vars=g["local_vars"],
      remote_vars=g["remote_vars"],
      master=FLAGS.master,
      config=config)

  with tf.MonitoredSession(session_creator=session_creator, hooks=[]) as sess:
    is_ready = sess.run(g["chief_is_ready"])
    while not is_ready:
      logging.info("chief not ready")
      time.sleep(1)
      is_ready = sess.run(g["chief_is_ready"])

    runner = meta_trainers_base.TruncatedTrainerLoopRunner(
        graph_dict["trainer_trainer_ops"], sess)

    for _ in runner.worker_iterations():
      if sess.should_stop():
        break
