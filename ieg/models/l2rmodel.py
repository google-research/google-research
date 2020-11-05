# coding=utf-8
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Baseline Learning-to-Reweight method."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ieg.models import networks
from ieg.models.basemodel import BaseModel

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

FLAGS = flags.FLAGS


class L2R(BaseModel):
  """Learning to Reweight method much simpler reimplementation.

  https://arxiv.org/pdf/1803.09050.pdf
  """

  def __init__(self, sess, strategy, dataset):
    super(L2R, self).__init__(sess, strategy, dataset)
    tf.logging.info('Init L2R model')
    self.strategy = strategy

    if FLAGS.use_ema:
      self.ema = tf.train.ExponentialMovingAverage(0.999, self.global_step)

  def set_input(self):
    train_ds = self.dataset.train_dataflow.shuffle(
        buffer_size=self.batch_size*10).repeat().batch(
            self.batch_size, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
    probe_ds = self.dataset.probe_dataflow.repeat().batch(
        self.batch_size, drop_remainder=True).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)
    val_ds = self.dataset.val_dataflow.batch(
        FLAGS.val_batch_size, drop_remainder=True).prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE)

    joint_ds = tf.data.Dataset.zip((train_ds, probe_ds))
    self.train_input_iterator = self.strategy.make_dataset_iterator(joint_ds)
    self.eval_input_iterator = self.strategy.make_dataset_iterator(val_ds)

  def set_dataset(self, dataset):
    with self.strategy.scope():
      self.dataset = dataset.create_loader()

  def create_graph(self):
    # splitted two graph with no connection
    with self.strategy.scope():
      self.train_op = self.train_step()
      self.eval_op = self.eval_step()

  def meta_optimize(self, net_cost):
    """Meta optimization step."""
    probe_images, probe_labels = self.probe_images, self.probe_labels
    net = self.net
    gate_gradients = 1

    batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)
    # initial data weight is zero
    init_eps_val = 0.0

    meta_net = networks.MetaImage(self.net, name='meta_model')

    target = tf.constant(
        [init_eps_val] * batch_size, dtype=np.float32, name='weight')

    lookahead_loss = tf.reduce_sum(tf.multiply(target, net_cost))
    lookahead_loss = lookahead_loss + net.regularization_loss

    with tf.control_dependencies([lookahead_loss]):
      train_vars = net.trainable_variables
      var_grads = tf.gradients(
          lookahead_loss, train_vars, gate_gradients=gate_gradients)

      static_vars = []
      for i in range(len(train_vars)):
        static_vars.append(
            tf.math.subtract(train_vars[i], FLAGS.meta_stepsize * var_grads[i]))
        meta_net.add_variable_alias(
            static_vars[-1], var_name=train_vars[i].name)

      for uv in net.updates_variables:
        meta_net.add_variable_alias(
            uv, var_name=uv.name, var_type='updates_variables')
      meta_net.verbose()

    with tf.control_dependencies(static_vars):
      g_logits = meta_net(
          probe_images, name='meta_model', reuse=True, training=True)

      desired_y = tf.one_hot(probe_labels, self.dataset.num_classes)
      meta_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
          desired_y, g_logits)
      meta_loss = tf.reduce_mean(meta_loss, name='meta_loss')
      meta_loss = meta_loss + meta_net.get_regularization_loss(net.wd)
      meta_acc, meta_acc_op = tf.metrics.accuracy(probe_labels,
                                                  tf.argmax(g_logits, axis=1))

    with tf.control_dependencies([meta_loss] + [meta_acc_op]):
      meta_train_vars = meta_net.trainable_variables
      # sanity: save memory for partial graph backpropagate
      grad_meta_vars = tf.gradients(
          meta_loss, meta_train_vars, gate_gradients=gate_gradients)
      grad_target = tf.gradients(
          static_vars,
          target,
          grad_ys=grad_meta_vars,
          gate_gradients=gate_gradients)[0]

    unorm_weight = tf.clip_by_value(
        -grad_target, clip_value_min=0, clip_value_max=float('inf'))
    norm_c = tf.reduce_sum(unorm_weight)
    weight = tf.divide(unorm_weight, norm_c + 0.00001)

    return tf.stop_gradient(weight), meta_loss, meta_acc

  def train(self):
    self.set_input()
    self.build_graph()

    with self.strategy.scope():
      self.sess.run([
          tf.local_variables_initializer(),
          tf.global_variables_initializer(),
          self.train_input_iterator.initializer
      ])
      self.sess.run([self.eval_input_iterator.initializer])
      iter_epoch = self.iter_epoch

      self.saver = tf.train.Saver(max_to_keep=4)
      if FLAGS.restore_step != 0:
        self.load_model()
        FLAGS.restore_step = self.global_step.eval()

      pbar = tqdm(total=(FLAGS.max_iteration - FLAGS.restore_step))
      tf.logging.info('Starts to train')
      for iteration in range(FLAGS.restore_step + 1, FLAGS.max_iteration + 1):
        self.update_learning_rate(iteration)
        lr, net_loss, meta_loss, acc, meta_acc,\
                          merged_summary, weights =\
                          self.sess.run([self.learning_rate]+self.train_op)
        pbar.update(1)
        message = ('Epoch {}[{}/{}] lr{:.3f} meta_loss:{:.2f} loss:{:.2f} '
                   'weight{:.2f}({:.2f}) acc:{:.2f} mata_acc{:.2f}').format(
                       iteration // iter_epoch, iteration % iter_epoch,
                       iter_epoch, lr, float(meta_loss), float(net_loss),
                       float(np.mean(weights)), float(np.std(weights)),
                       float(acc), float(meta_acc))
        pbar.set_description(message)
        self.summary_writer.add_summary(merged_summary, iteration)

        # checkpoint
        if self.time_for_evaluation(iteration, lr):
          tf.logging.info(message)
          self.evaluate(iteration, lr)
          self.save_model(iteration)
          self.summary_writer.flush()
      # end of iterations
      pbar.close()

  def train_step(self):

    def step_fn(inputs):
      """Step function."""

      net = self.net
      (images, labels), (self.probe_images, self.probe_labels) = inputs
      self.images, self.labels = images, labels

      logits = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      self.logits = logits

      net_cost = tf.losses.sparse_softmax_cross_entropy(
          labels, logits, reduction=tf.losses.Reduction.NONE)
      weight, meta_loss, meta_acc = self.meta_optimize(net_cost)

      net_loss = tf.reduce_sum(tf.math.multiply(net_cost, weight))
      net_loss += net.regularization_loss
      net_loss /= self.strategy.num_replicas_in_sync
      # rescale by gpus
      net_grads = tf.gradients(net_loss, net.trainable_variables)
      minimizer_op = self.optimizer.apply_gradients(
          zip(net_grads, net.trainable_variables), global_step=self.global_step)
      if FLAGS.use_ema:
        ema_op = self.ema.apply(net.trainable_variables)
        optimizer_op = tf.group([net.updates, minimizer_op, ema_op])
      else:
        optimizer_op = tf.group([net.updates, minimizer_op])
      acc_op, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))

      with tf.control_dependencies([optimizer_op, acc_update_op]):
        return tf.identity(net_loss), tf.identity(meta_loss),\
               tf.identity(meta_acc), tf.identity(acc_op),\
               tf.identity(weight), tf.identity(labels)

    # end of parallel
    (pr_net_loss, pr_metaloss, pr_metaacc, pr_acc,
     pr_weight, pr_labels) = self.strategy.run(
         step_fn, args=(next(self.train_input_iterator),))

    # collect device variables
    weights = self.strategy.unwrap(pr_weight)
    weights = tf.concat(weights, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_metaacc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaacc)
    net_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_net_loss)
    meta_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaloss)

    merges = []
    merges.append(tf.summary.scalar('acc/train', mean_acc))
    merges.append(tf.summary.scalar('loss/net', net_loss))
    merges.append(tf.summary.scalar('loss/meta', meta_loss))
    merges.append(tf.summary.scalar('acc/meta', mean_metaacc))

    zw_inds = tf.squeeze(
        tf.where(tf.less_equal(weights, 0), name='zero_weight_index'))
    merges.append(
        tf.summary.scalar(
            'weights/zeroratio',
            tf.math.divide(
                tf.cast(tf.size(zw_inds), tf.float32),
                tf.cast(tf.size(weights), tf.float32))))

    self.epoch_var = tf.cast(
        self.global_step / self.iter_epoch, tf.float32, name='epoch')
    merges.append(tf.summary.scalar('epoch', self.epoch_var))
    merges.append(tf.summary.scalar('learningrate', self.learning_rate))
    merges.append(
        tf.summary.scalar('acc/eval_on_train', self.eval_acc_on_train[0]))
    merges.append(
        tf.summary.scalar('acc/eval_on_train_top5', self.eval_acc_on_train[1]))
    merges.append(tf.summary.scalar('acc/num_eval', self.eval_acc_on_train[2]))
    summary = tf.summary.merge(merges)

    return [net_loss, meta_loss, mean_acc, mean_metaacc, summary, weights]
