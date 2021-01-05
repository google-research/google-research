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

"""Multi GPU model (sync gradient updates.)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow.compat.v1 as tf
from capsule_em import em_model
from capsule_em import layers
from capsule_em import simple_model
from capsule_em import utils

FLAGS = tf.app.flags.FLAGS


def _average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.

  Note that this function provides a synchronization point across all towers.

  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.

  Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    print(len(grad_and_vars))
    for g, v in grad_and_vars:
      if g is None:
        print(v)
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    print(len(grad_and_vars))
    for g, v in grad_and_vars:
      if g is not None:
        print(v)
    for g, v in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      print(v)
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)
    capped_grad = tf.clip_by_value(grad, -200., 200.)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (capped_grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def multi_gpu_model(features):
  """Build the Graph and train the model on multiple gpus."""
  if FLAGS.use_caps:
    if FLAGS.use_em:
      inference = em_model.inference
    else:
      print('not supported')
  else:
    inference = simple_model.conv_inference
  with tf.device('/cpu:0'):
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0),
        trainable=False)

    lr = tf.train.exponential_decay(
        FLAGS.learning_rate,
        global_step,
        FLAGS.decay_steps,
        FLAGS.decay_rate,
        staircase=FLAGS.staircase)
    if FLAGS.clip_lr:
      lr = tf.maximum(lr, 1e-6)

    if FLAGS.adam:
      opt = tf.train.AdamOptimizer(lr)
    else:
      opt = tf.train.GradientDescentOptimizer(lr)

    tower_grads = []
    corrects = []
    almosts = []
    result = {}
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(FLAGS.num_gpus):
        with tf.device('/gpu:%d' % i):
          with tf.name_scope('tower_%d' % (i)) as scope:
            label_ = features[i]['labels']
            y, result['recons_1'], result['recons_2'], result[
                'mid_act'] = inference(features[i])
            result['logits'] = y

            losses, correct, almost = layers.optimizer(
                logits=y,
                labels=label_,
                multi=FLAGS.multi and FLAGS.data_set == 'mnist',
                scope=scope,
                softmax=FLAGS.softmax,
                rate=FLAGS.loss_rate,
                step=global_step,
            )
            tf.get_variable_scope().reuse_variables()
            corrects.append(correct)
            almosts.append(almost)
            #           summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
            grads = opt.compute_gradients(
                losses,
                gate_gradients=tf.train.Optimizer.GATE_NONE,
            )
            tower_grads.append(grads)

    with utils.maybe_jit_scope(), tf.name_scope('average_gradients'):
      grads = _average_gradients(tower_grads)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    if FLAGS.verbose:
      for grad, var in grads:
        if grad is not None:
          summaries.append(
              tf.summary.histogram(var.op.name + '/gradients', grad))
    summaries.append(tf.summary.scalar('learning_rate', lr))
    result['summary'] = tf.summary.merge(summaries)
    result['train'] = opt.apply_gradients(grads, global_step=global_step)
    # result['train'] = y

    cors = tf.stack(corrects)
    alms = tf.stack(almosts)
    result['correct'] = tf.reduce_sum(cors, 0)
    result['almost'] = tf.reduce_sum(alms, 0)

    return result
