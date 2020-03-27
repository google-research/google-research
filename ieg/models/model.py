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
"""The proposed model training code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

from ieg import utils
from ieg.dataset_utils.utils import autoaug_batch_process_map_fn
from ieg.models import networks
from ieg.models.basemodel import BaseModel
from ieg.models.custom_ops import logit_norm
from ieg.models.custom_ops import MixMode

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm


FLAGS = flags.FLAGS
logging = tf.logging


class IEG(BaseModel):
  """Model training class."""

  def __init__(self, sess, strategy, dataset):
    super(IEG, self).__init__(sess, strategy, dataset)
    logging.info('Init IEG model')

    self.augment = MixMode()
    self.beta = 0.5  # MixUp hyperparam
    self.nu = 2      # K value for label guessing

  def set_input(self):
    with self.strategy.scope():

      train_ds = self.dataset.train_dataflow.shuffle(
          buffer_size=self.batch_size * 10).repeat().batch(
              self.batch_size, drop_remainder=True).map(
                  # strong augment each batch data and expand to 5D [Bx2xHxWx3]
                  # TODO(zizhaoz): can be faster if processing before .batch()
                  autoaug_batch_process_map_fn,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE).prefetch(
                      buffer_size=tf.data.experimental.AUTOTUNE)

      # no shuffle for probe, so a batch is class balanced.
      probe_ds = self.dataset.probe_dataflow.repeat().batch(
          self.batch_size, drop_remainder=True).prefetch(
              buffer_size=tf.data.experimental.AUTOTUNE)

      val_ds = self.dataset.val_dataflow.batch(
          FLAGS.val_batch_size, drop_remainder=False).prefetch(
              buffer_size=tf.data.experimental.AUTOTUNE)

      self.train_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              train_ds).make_initializable_iterator())
      self.probe_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              probe_ds).make_initializable_iterator())

      self.eval_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              val_ds).make_initializable_iterator())

  def meta_momentum_update(self, grad, var_name, optimizer):
    # Finds corresponding momentum of a var name
    accumulation = utils.get_var(optimizer.variables(), var_name.split(':')[0])
    if len(accumulation) != 1:
      raise ValueError('length of accumulation {}'.format(len(accumulation)))
    new_grad = tf.math.add(
        tf.stop_gradient(accumulation[0]) * FLAGS.meta_momentum, grad)
    return new_grad

  def guess_label(self, logit, temp=0.5):
    logit = tf.reshape(logit, [-1, self.dataset.num_classes])
    logit = tf.split(logit, self.nu, axis=0)
    logit = [logit_norm(x) for x in logit]
    logit = tf.concat(logit, 0)
    ## Done with logit norm
    p_model_y = tf.reshape(
        tf.nn.softmax(logit), [self.nu, -1, self.dataset.num_classes])
    p_model_y = tf.reduce_mean(p_model_y, axis=0)

    p_target = tf.pow(p_model_y, 1.0 / temp)
    p_target /= tf.reduce_sum(p_target, axis=1, keepdims=True)

    return p_target

  def crossentropy_minimize(self,
                            u_logits,
                            u_images,
                            l_images,
                            l_labels,
                            u_labels=None):
    """Cross-entropy optimization step implementation for TPU."""
    batch_size = self.batch_size // self.strategy.num_replicas_in_sync
    guessed_label = self.guess_label(u_logits)
    self.guessed_label = guessed_label

    guessed_label = tf.reshape(
        tf.stop_gradient(guessed_label), shape=(-1, self.dataset.num_classes))

    l_labels = tf.reshape(
        tf.one_hot(l_labels, self.dataset.num_classes),
        shape=(-1, self.dataset.num_classes))
    augment_images, augment_labels = self.augment(
        [l_images, u_images], [l_labels] + [guessed_label] * self.nu,
        [self.beta, self.beta])
    logit = self.net(augment_images, name='model', training=True)

    zbs = batch_size * 2
    halfzbs = batch_size

    split_pos = [tf.shape(l_images)[0], halfzbs, halfzbs]

    logit = [logit_norm(lgt) for lgt in tf.split(logit, split_pos, axis=0)]
    u_logit = tf.concat(logit[1:], axis=0)

    split_pos = [tf.shape(l_images)[0], zbs]
    l_augment_labels, u_augment_labels = tf.split(
        augment_labels, split_pos, axis=0)

    u_loss = tf.losses.softmax_cross_entropy(u_augment_labels, u_logit)
    l_loss = tf.losses.softmax_cross_entropy(l_augment_labels, logit[0])

    loss = tf.math.add(
        l_loss, u_loss * FLAGS.ce_factor, name='crossentropy_minimization_loss')

    return loss

  def consistency_loss(self, logit, aug_logit):

    def kl_divergence(q_logits, p_logits):
      q = tf.nn.softmax(q_logits)
      per_example_kl_loss = q * (
          tf.nn.log_softmax(q_logits) - tf.nn.log_softmax(p_logits))
      return tf.reduce_mean(per_example_kl_loss) * self.dataset.num_classes

    return tf.math.multiply(
        kl_divergence(tf.stop_gradient(logit), aug_logit),
        FLAGS.consistency_factor,
        name='consistency_loss')

  def unsupervised_loss(self):
    """Creates unsupervised losses.

    Here we create two cross-entropy losses and a KL-loss defined in the paper.

    Returns:
      A list of losses.
    """

    if FLAGS.ce_factor == 0 and FLAGS.consistency_factor == 0:
      return [tf.constant(0, tf.float32), tf.constant(0, tf.float32)]
    logits = self.logits
    images = self.images
    aug_images = self.aug_images
    probe_images, probe_labels = self.probe_images, self.probe_labels
    im_shape = (-1, int(probe_images.shape[1]), int(probe_images.shape[2]),
                int(probe_images.shape[3]))

    aug_logits = self.net(aug_images, name='model', training=True)

    n_probe_to_mix = tf.shape(aug_images)[0]
    probe = tf.tile(tf.constant([[10.]]), [1, tf.shape(probe_images)[0]])
    idx = tf.squeeze(tf.random.categorical(probe, n_probe_to_mix))

    l_images = tf.reshape(tf.gather(probe_images, idx), im_shape)
    l_labels = tf.reshape(tf.gather(probe_labels, idx), (-1,))

    u_logits = tf.concat([logits, aug_logits], axis=0)
    u_images = tf.concat([images, aug_images], axis=0)

    losses = []
    if FLAGS.ce_factor > 0:
      logging.info('Use crossentropy minimization loss {}'.format(
          FLAGS.ce_factor))
      ce_min_loss = self.crossentropy_minimize(u_logits, u_images, l_images,
                                               l_labels)
      losses.append(ce_min_loss)
    else:
      losses.append(tf.constant(0, tf.float32))

    if FLAGS.consistency_factor > 0:
      logging.info('Use consistency loss {}'.format(
          FLAGS.consistency_factor))
      consis_loss = self.consistency_loss(logits, aug_logits)
      losses.append(consis_loss)

    else:
      losses.append(tf.constant(0, tf.float32))

    return losses

  def meta_optimize(self):
    """Meta optimization step."""

    probe_images, probe_labels = self.probe_images, self.probe_labels
    labels = self.labels
    net = self.net
    logits = self.logits
    gate_gradients = 1

    batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)
    init_eps_val = float(1) / batch_size

    meta_net = networks.MetaImage(self.net, name='meta_model')

    if FLAGS.meta_momentum and not self.optimizer.variables():
      # Initializing momentum state of optimizer for meta momentum update.
      # It is a hacky implementation
      logging.info('Pre-initialize optimizer momentum states.')
      idle_net_cost = tf.losses.sparse_softmax_cross_entropy(
          self.labels, logits)
      tmp_var_grads = self.optimizer.compute_gradients(
          tf.reduce_mean(idle_net_cost), net.trainable_variables)
      self.optimizer.apply_gradients(tmp_var_grads)

    with tf.name_scope('coefficient'):
      # Data weight coefficient
      target = tf.constant(
          [init_eps_val] * batch_size,
          shape=(batch_size,),
          dtype=np.float32,
          name='weight')
      # Data re-labeling coefficient
      eps = tf.constant(
          [FLAGS.grad_eps_init] * batch_size,
          shape=(batch_size,),
          dtype=tf.float32,
          name='eps')

    onehot_labels = tf.one_hot(labels, self.dataset.num_classes)
    onehot_labels = tf.cast(onehot_labels, tf.float32)
    eps_k = tf.reshape(eps, [batch_size, 1])

    mixed_labels = eps_k * onehot_labels + (1 - eps_k) * self.guessed_label
    # raw softmax loss
    log_softmax = tf.nn.log_softmax(logits)
    net_cost = -tf.reduce_sum(mixed_labels * log_softmax, 1)

    lookahead_loss = tf.reduce_sum(tf.multiply(target, net_cost))
    lookahead_loss = lookahead_loss + net.regularization_loss

    with tf.control_dependencies([lookahead_loss]):
      train_vars = net.trainable_variables
      var_grads = tf.gradients(
          lookahead_loss, train_vars, gate_gradients=gate_gradients)

      static_vars = []
      for i in range(len(train_vars)):
        if FLAGS.meta_momentum > 0:
          actual_grad = self.meta_momentum_update(var_grads[i],
                                                  train_vars[i].name,
                                                  self.optimizer)
          static_vars.append(
              tf.math.subtract(train_vars[i],
                               FLAGS.meta_stepsize * actual_grad))
        else:
          static_vars.append(
              tf.math.subtract(train_vars[i],
                               FLAGS.meta_stepsize * var_grads[i]))
        # new style
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
      grad_meta_vars = tf.gradients(
          meta_loss, meta_train_vars, gate_gradients=gate_gradients)
      grad_target, grad_eps = tf.gradients(
          static_vars, [target, eps],
          grad_ys=grad_meta_vars,
          gate_gradients=gate_gradients)
    # updates weight
    raw_weight = target - grad_target
    raw_weight = raw_weight - init_eps_val
    unorm_weight = tf.clip_by_value(
        raw_weight, clip_value_min=0, clip_value_max=float('inf'))
    norm_c = tf.reduce_sum(unorm_weight)
    weight = tf.divide(unorm_weight, norm_c + 0.00001)

    # gets new lambda by the sign of gradient
    new_eps = tf.where(grad_eps < 0, x=tf.ones_like(eps), y=tf.zeros_like(eps))

    return tf.stop_gradient(weight), tf.stop_gradient(
        new_eps), meta_loss, meta_acc

  def train_step(self):

    def step_fn(inputs):
      """Step functon.

      Args:
        inputs: inputs from data iterator

      Returns:
        a set of variables want to observe in Tensorboard
      """

      net = self.net
      (all_images, labels), (self.probe_images, self.probe_labels) = inputs
      assert len(all_images.shape) == 5
      images, self.aug_images = all_images[:, 0], all_images[:, 1]

      self.images, self.labels = images, labels
      batch_size = int(self.batch_size / self.strategy.num_replicas_in_sync)

      logits = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      self.logits = logits

      # other losses
      # initialized first to use self.guessed_label for meta step
      xe_loss, cs_loss = self.unsupervised_loss()

      # meta optimization
      weight, eps, meta_loss, meta_acc = self.meta_optimize()

      ## losses w.r.t new weight and loss
      onehot_labels = tf.one_hot(labels, self.dataset.num_classes)
      onehot_labels = tf.cast(onehot_labels, tf.float32)
      eps_k = tf.reshape(eps, [batch_size, 1])

      mixed_labels = tf.math.add(
          eps_k * onehot_labels, (1 - eps_k) * self.guessed_label,
          name='mixed_labels')
      net_cost = tf.losses.softmax_cross_entropy(
          mixed_labels, logits, reduction=tf.losses.Reduction.NONE)
      # loss with initial weight
      net_loss1 = tf.reduce_mean(net_cost)

      # loss with initial eps
      init_eps = tf.constant(
          [FLAGS.grad_eps_init] * batch_size, dtype=tf.float32)
      init_eps = tf.reshape(init_eps, (-1, 1))
      init_mixed_labels = tf.math.add(
          init_eps * onehot_labels, (1 - init_eps) * self.guessed_label,
          name='init_mixed_labels')

      net_cost2 = tf.losses.softmax_cross_entropy(
          init_mixed_labels, logits, reduction=tf.losses.Reduction.NONE)
      net_loss2 = tf.reduce_sum(tf.math.multiply(net_cost2, weight))

      net_loss = (net_loss1 + net_loss2) / 2

      net_loss = net_loss + tf.add_n([xe_loss, cs_loss])
      net_loss += net.regularization_loss
      net_loss /= self.strategy.num_replicas_in_sync

      # rescale by gpus
      with tf.control_dependencies(net.updates):
        net_grads = tf.gradients(net_loss, net.trainable_variables)
        minimizer_op = self.optimizer.apply_gradients(
            zip(net_grads, net.trainable_variables),
            global_step=self.global_step)

      with tf.control_dependencies([minimizer_op]):
        train_op = self.ema.apply(net.trainable_variables)

      acc_op, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))

      with tf.control_dependencies([train_op, acc_update_op]):
        return (tf.identity(net_loss), tf.identity(xe_loss),
                tf.identity(cs_loss), tf.identity(meta_loss),
                tf.identity(meta_acc), tf.identity(acc_op), tf.identity(weight),
                tf.identity(labels))

    # end of parallel
    (pr_net_loss, pr_xe_loss, pr_cs_loss, pr_metaloss, pr_metaacc, pr_acc,
     pr_weight, pr_labels) = self.strategy.experimental_run_v2(
         step_fn,
         args=((next(self.train_input_iterator),
                next(self.probe_input_iterator)),))
    # collect device variables
    weights = self.strategy.unwrap(pr_weight)
    weights = tf.concat(weights, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_metaacc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaacc)
    net_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_net_loss)
    xe_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_xe_loss)
    cs_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_cs_loss)
    meta_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_metaloss)

    # The following add variables for tensorboard visualization
    merges = []
    merges.append(tf.summary.scalar('acc/train', mean_acc))
    merges.append(tf.summary.scalar('loss/xemin', xe_loss))
    merges.append(tf.summary.scalar('loss/consistency', cs_loss))
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
    summary = tf.summary.merge(merges)

    return [
        net_loss, meta_loss, xe_loss, cs_loss, mean_acc, mean_metaacc, summary,
        weights
    ]

  def train(self):
    self.set_input()
    self.build_graph()

    with self.strategy.scope():
      self.initialize_variables()

      self.sess.run([
          self.train_input_iterator.initialize(),
          self.probe_input_iterator.initialize()
      ])
      self.sess.run([self.eval_input_iterator.initialize()])

      logging.info('Finishes variables initializations')
      iter_epoch = self.iter_epoch

      self.saver = tf.train.Saver(max_to_keep=4)
      self.load_model()
      FLAGS.restore_step = self.global_step.eval()

      pbar = tqdm(total=(FLAGS.max_iteration - FLAGS.restore_step))
      for iteration in range(FLAGS.restore_step + 1, FLAGS.max_iteration + 1):
        self.update_learning_rate(iteration)
        (lr, net_loss, meta_loss, xe_loss, cs_loss, acc, meta_acc,
         merged_summary, weights) = (
             self.sess.run([self.learning_rate] + self.train_op))
        pbar.update(1)
        message = ('Epoch {}[{}/{}] lr{:.3f} meta_loss:{:.2f} loss:{:.2f} '
                   'mc_loss:{:.2f} uc_loss:{:.2f} weight{:.2f}({:.2f}) '
                   'acc:{:.2f} mata_acc{:.2f}').format(iteration // iter_epoch,
                                                       iteration % iter_epoch,
                                                       iter_epoch, lr,
                                                       float(meta_loss),
                                                       float(net_loss),
                                                       float(xe_loss),
                                                       float(cs_loss),
                                                       float(np.mean(weights)),
                                                       float(np.std(weights)),
                                                       float(acc),
                                                       float(meta_acc))
        pbar.set_description(message)
        self.summary_writer.add_summary(merged_summary, iteration)

        # checkpoint
        if self.time_for_evaluation(iteration, lr):
          logging.info(message)
          self.evaluate(iteration, lr)
          self.save_model(iteration)
          self.summary_writer.flush()
      # end of iterations
      pbar.close()
