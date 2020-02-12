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

"""Trains Q function using a TF Dataset.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
import tensorflow.compat.v1 as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import training as contrib_training


@gin.configurable
def train_ddpg(dataset,
               policy,
               actor_optimizer=None,
               critic_optimizer=None,
               pack_transition_fn=None,
               ddpg_graph_fn=None,
               log_dir=None,
               master='local',
               task=0,
               training_steps=None,
               max_training_steps=100000,
               reuse=False,
               init_checkpoint=None,
               update_target_every_n_steps=50,
               log_every_n_steps=None,
               save_checkpoint_steps=500,
               save_summaries_steps=500):
  """Self-contained learning loop for offline Q-learning.

  Code inspired by OpenAI Baselines' deepq.build_train. This function is
  compatible with discrete Q-learning graphs, continuous Q learning graphs, and
  SARSA.

  Args:
    dataset: tf.data.Dataset providing transitions.
    policy: Instance of TFDQNPolicy class that provides functor for building the
      critic function.
    actor_optimizer: Optional instance of an optimizer for the actor network.
      If not specified, creates an AdamOptimizer using the default constructor.
    critic_optimizer: Optional instance of an optimizer for the critic network.
      If not specified, creates an AdamOptimizer using the default constructor.
    pack_transition_fn: Optional function that performs additional processing
      of the transition. This is a convenience method for ad-hoc manipulation of
      transition data passed to the learning function after parsing.
    ddpg_graph_fn: Function used to construct training objectives w.r.t. critic
      outputs.
    log_dir: Where to save model checkpoints and tensorboard summaries.
    master: Optional address of master worker. Specify this when doing
      distributed training.
    task: Optional worker task for distributed training. Defaults to solo master
      task on a single machine.
    training_steps: Optional number of steps to run training before terminating
      early. Max_training_steps remains unchanged - training will terminate
      after max_training_steps whether or not training_steps is specified.
    max_training_steps: maximum number of training iters.
    reuse: If True, reuse existing variables for all declared variables by this
      function.
    init_checkpoint: Optional checkpoint to restore prior to training. If not
      provided, variables are initialized using global_variables_initializer().
    update_target_every_n_steps: How many global steps (training) between
      copying the Q network weights (scope='q_func') to target network
      (scope='target_q_func').
    log_every_n_steps: How many global steps between logging loss tensors.
    save_checkpoint_steps: How many global steps between saving TF variables
      to a checkpoint file.
    save_summaries_steps: How many global steps between saving TF summaries.

  Returns:
    (int) Current `global_step` reached after training for training_steps, or
    `max_training_steps` if `global_step` has reached `max_training_steps`.

  """
  data_iterator = dataset.make_one_shot_iterator()

  transition = data_iterator.get_next()
  if pack_transition_fn:
    transition = pack_transition_fn(transition)

  if actor_optimizer is None:
    actor_optimizer = tf.train.AdamOptimizer()
  if critic_optimizer is None:
    critic_optimizer = tf.train.AdamOptimizer()

  a_func = policy.get_a_func(is_training=True, reuse=reuse)
  q_func = policy.get_q_func(is_training=True, reuse=reuse)
  actor_loss, critic_loss, all_summaries = ddpg_graph_fn(
      a_func, q_func, transition)

  a_func_vars = contrib_framework.get_trainable_variables(scope='a_func')
  q_func_vars = contrib_framework.get_trainable_variables(scope='q_func')
  target_q_func_vars = contrib_framework.get_trainable_variables(
      scope='target_q_func')

  # with tf.variable_scope('ddpg', use_resource=True):
  global_step = tf.train.get_or_create_global_step()

  # CRITIC OPTIMIZATION
  # Only optimize q_func and update its batchnorm params.
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='q_func')
  critic_train_op = contrib_training.create_train_op(
      critic_loss,
      critic_optimizer,
      global_step=global_step,
      update_ops=update_ops,
      summarize_gradients=True,
      variables_to_train=q_func_vars,
  )

  # ACTOR OPTIMIZATION
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='a_func')
  actor_train_op = contrib_training.create_train_op(
      actor_loss,
      actor_optimizer,
      global_step=None,
      summarize_gradients=True,
      variables_to_train=a_func_vars,
  )
  # Combine losses to train both actor and critic simultaneously.
  train_op = critic_train_op + actor_train_op

  chief_hooks = []
  hooks = []
  # Save summaries periodically.
  if save_summaries_steps is not None:
    chief_hooks.append(tf.train.SummarySaverHook(
        save_steps=save_summaries_steps,
        output_dir=log_dir, summary_op=all_summaries))

  # Stop after training_steps
  if max_training_steps:
    hooks.append(tf.train.StopAtStepHook(last_step=max_training_steps))

  # Report if loss tensor is NaN.
  hooks.append(tf.train.NanTensorHook(actor_loss))
  hooks.append(tf.train.NanTensorHook(critic_loss))

  if log_every_n_steps is not None:
    tensor_dict = {
        'global_step': global_step,
        'actor loss': actor_loss,
        'critic_loss': critic_loss
    }
    chief_hooks.append(
        tf.train.LoggingTensorHook(tensor_dict, every_n_iter=log_every_n_steps))

    # Measure how fast we are training per sec and save to summary.
    chief_hooks.append(tf.train.StepCounterHook(
        every_n_steps=log_every_n_steps, output_dir=log_dir))

  # If target network exists, periodically update target Q network with new
  # weights (frozen target network). We hack this by
  # abusing a LoggingTensorHook for this.
  if target_q_func_vars and update_target_every_n_steps is not None:
    update_target_expr = []
    for var, var_t in zip(sorted(q_func_vars, key=lambda v: v.name),
                          sorted(target_q_func_vars, key=lambda v: v.name)):
      update_target_expr.append(var_t.assign(var))
    update_target_expr = tf.group(*update_target_expr)

    with tf.control_dependencies([update_target_expr]):
      update_target = tf.constant(0)
    chief_hooks.append(
        tf.train.LoggingTensorHook({'update_target': update_target},
                                   every_n_iter=update_target_every_n_steps))

  # Save checkpoints periodically, save all of them.
  saver = tf.train.Saver(max_to_keep=None)
  chief_hooks.append(tf.train.CheckpointSaverHook(
      log_dir, save_steps=save_checkpoint_steps, saver=saver,
      checkpoint_basename='model.ckpt'))

  # Save our experiment params to checkpoint dir.
  chief_hooks.append(gin.tf.GinConfigSaverHook(log_dir, summarize_config=True))

  session_config = tf.ConfigProto(log_device_placement=True)

  init_fn = None
  if init_checkpoint:
    assign_fn = contrib_framework.assign_from_checkpoint_fn(
        init_checkpoint, contrib_framework.get_model_variables())
    init_fn = lambda _, sess: assign_fn(sess)
  scaffold = tf.train.Scaffold(saver=saver, init_fn=init_fn)
  with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=(task == 0),
      config=session_config,
      checkpoint_dir=log_dir,
      scaffold=scaffold,
      hooks=hooks,
      chief_only_hooks=chief_hooks) as sess:
    np_step = 0
    while not sess.should_stop():
      np_step, _ = sess.run([global_step, train_op])
      if training_steps and np_step % training_steps == 0:
        break
    done = np_step >= max_training_steps
  return np_step, done
