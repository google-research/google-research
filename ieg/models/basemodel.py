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
"""BaseModel that implements basics to support training pipeline."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from absl import flags

from ieg import utils
from ieg.models import resnet
from ieg.models import wrn

import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm


FLAGS = flags.FLAGS
logging = tf.logging


def create_network(name, num_classes):
  """Creates networks."""
  net = None
  logging.info('create network [{}] ...'.format(name))

  if name == 'resnet29':
    net = resnet.ResNet(depth=29, num_classes=num_classes)
  elif name == 'wrn28-10':
    net = wrn.WRN(
        num_classes=num_classes,
        wrn_size=160)
  else:
    raise ValueError('{} net is not implemented'.format(name))
  return net


class Trainer(object):
  """Trainer with basic utility functions."""

  def set_lr_schedule(self):
    """Setup learning rate schedule."""

    if FLAGS.lr_schedule == 'custom_step':
      logging.info(
          'Using custom step for learning rate decay schedule {}'.format([
              (a // self.iter_epoch, FLAGS.learning_rate**(k + 1))
              for k, a in enumerate(self.decay_steps)
          ]))
      self.learning_rate = tf.get_variable(
          'learning_rate',
          dtype=tf.float32,
          trainable=False,
          initializer=tf.constant(FLAGS.learning_rate))

    elif FLAGS.lr_schedule == 'cosine_one':
      logging.info(
          'using cosine_one step learning rate decay, step: {}'.format(
              self.decay_steps))

      self.learning_rate = tf.compat.v1.train.cosine_decay(
          FLAGS.learning_rate,
          self.global_step,
          FLAGS.max_iteration,
          name='learning_rate')

    elif FLAGS.lr_schedule.startswith('cosine'):
      ## Cosine learning rate
      logging.info('Use cosine learning rate decay, step: {}'.format(
          self.decay_steps))
      cond_cosine_lr = 'warmup' in FLAGS.lr_schedule
      if cond_cosine_lr:
        warmup_steps = self.iter_epoch * FLAGS.warmup_epochs
        global_step = tf.math.maximum(
            tf.constant(0, tf.int64), self.global_step - warmup_steps)
      else:
        global_step = self.global_step
      assert len(self.decay_steps) == 1
      self.cos_eval_step = self.decay_steps[0]
      self.cos_eval_tot_step = 1
      learning_rate = tf.train.cosine_decay_restarts(
          FLAGS.learning_rate,
          global_step,
          self.decay_steps[0],
          t_mul=FLAGS.cos_t_mul,
          m_mul=FLAGS.cos_m_mul,
          name='learning_rate')

      if cond_cosine_lr:
        logging.info(
            'Enable warmup with warmup_steps: {}'.format(warmup_steps))
        warmup_learning_rate = 0.0
        slope = (FLAGS.learning_rate - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(self.global_step,
                                      tf.float32) + warmup_learning_rate
        learning_rate = tf.where(self.global_step < warmup_steps, warmup_rate,
                                 learning_rate)

      self.learning_rate = learning_rate

    elif FLAGS.lr_schedule == 'exponential':
      logging.info(
          'using exponential learning rate decay, step: {}'.format(
              self.decay_steps))
      self.learning_rate = tf.train.exponential_decay(
          FLAGS.learning_rate,
          self.global_step,
          self.decay_steps[0],
          0.9,
          staircase=True,
          name='learning_rate')
    else:
      raise NotImplementedError

  def calibrate_flags(self):
    """Adjusts all parameter for multiple GPUs."""

    strategy = self.strategy
    logging.info(
        'Adjust hyperparameters based on num_replicas_in_sync {}'.format(
            strategy.num_replicas_in_sync))
    FLAGS.batch_size *= strategy.num_replicas_in_sync
    self.iter_epoch = self.dataset.train_dataset_size // FLAGS.batch_size
    self.decay_steps = [int(a) for a in FLAGS.decay_steps.split(',')]
    FLAGS.val_batch_size *= strategy.num_replicas_in_sync
    FLAGS.eval_freq = int(FLAGS.eval_freq / strategy.num_replicas_in_sync)

    logging.info('\t FLAGS.eval_freq {}'.format(FLAGS.eval_freq))
    logging.info('\t FLAGS.learning_rate {}'.format(FLAGS.learning_rate))
    logging.info('\t FLAGS.max_iteration {}'.format(FLAGS.max_iteration))
    logging.info('\t self.decay_steps {}'.format(self.decay_steps))
    logging.info('\t self.batch_size {}'.format(FLAGS.batch_size))
    logging.info('\t self.val_batch_size {}'.format(FLAGS.val_batch_size))
    logging.info('\t self.iter_epoch {}'.format(self.iter_epoch))

  def check_checkpoint(self, path=None):
    """Check if a checkpoint exists."""

    if FLAGS.restore_step == 0:
      path = utils.get_latest_checkpoint(FLAGS.checkpoint_path)
      if path is None:
        return None
      logging.warning('load latest checkpoint ' + path)
    else:
      path = '{}/checkpoint.ckpt-{}'.format(FLAGS.checkpoint_path,
                                            FLAGS.restore_step)
    if not tf.gfile.Exists(path + '.meta') and FLAGS.restore_step != 0:
      raise NotImplementedError('{} not exists'.format(path))

    return path


class BaseModel(Trainer):
  """BaseModel class that includes full training pipeline."""

  def __init__(self, sess, strategy, dataset):
    self.sess = sess
    self.strategy = strategy
    self.set_dataset(dataset)
    self.calibrate_flags()

    with self.strategy.scope():
      logging.info('[BaseModel] Parallel training in {} devices'.format(
          strategy.num_replicas_in_sync))
      self.net = create_network(FLAGS.network_name, dataset.num_classes)
      self.batch_size = FLAGS.batch_size
      self.val_batch_size = FLAGS.val_batch_size
      logging.info('[BaseModel] actual batch size {}'.format(
          self.batch_size))
      self.global_step = tf.train.get_or_create_global_step()

      self.set_lr_schedule()

      self.optimizer = tf.train.MomentumOptimizer(
          learning_rate=self.learning_rate, momentum=FLAGS.momentum)
      self.acc_func = tf.metrics.accuracy

      if FLAGS.use_ema:
        self.ema = tf.train.ExponentialMovingAverage(0.999, self.global_step)

  def set_input(self):
    """Set input function."""
    with self.strategy.scope():
      train_ds = self.dataset.train_dataflow.shuffle(
          buffer_size=self.batch_size * 10).repeat().batch(
              self.batch_size, drop_remainder=True).prefetch(
                  buffer_size=tf.data.experimental.AUTOTUNE)

      assert FLAGS.val_batch_size < self.dataset.val_dataset_size
      val_ds = self.dataset.val_dataflow.batch(
          FLAGS.val_batch_size, drop_remainder=False).prefetch(
              buffer_size=tf.data.experimental.AUTOTUNE)
      self.train_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              train_ds).make_initializable_iterator())
      self.eval_input_iterator = (
          self.strategy.experimental_distribute_dataset(
              val_ds).make_initializable_iterator())

  def set_dataset(self, dataset):
    """Setup datasets."""

    with self.strategy.scope():
      self.dataset = dataset.create_loader()
      # TODO(zizhaoz): not useful.
      if self.strategy.num_replicas_in_sync > 8:
        # If parallel more than 8 cores (when do large scale parallel training)
        # we disable auto_shard_policy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.OFF)
        self.dataset.train_dataflow = (
            self.dataset.train_dataflow.with_options(options))

  def load_model(self, path=None):
    """Load model from disk if there is any or required by FLAGS.restore_step.

    Args:
      path: Optional. The path of checkpoints.
      If not provided, it will infer automatically by FLAGS.restore_step.
    """
    if path is None:
      path = self.check_checkpoint()
    if path is not None:
      self.saver.restore(self.sess, save_path=path)
      logging.info(
          'Load model checkpoint {}, learning_rate {:3f} global_step {}'
          .format(path, self.learning_rate.eval(), self.global_step.eval()))
    else:
      if FLAGS.mode == 'evaluation':
        raise ValueError('Checkpoint not found for evaluation')

  def save_model(self, iteration):
    """Saves model."""

    path = '{}/checkpoint.ckpt'.format(FLAGS.checkpoint_path)
    save_path = self.saver.save(self.sess, path, global_step=iteration)
    logging.info('Save weights {} at iteration {}'.format(
        save_path, iteration))

  def update_learning_rate(self, global_step):
    """Updates learning rate.

    Args:
      global_step: optimizer global step
    """

    if global_step in self.decay_steps and FLAGS.lr_schedule == 'custom_step':
      timer = self.decay_steps.index(global_step) + 1
      learning_rate = FLAGS.learning_rate * (FLAGS.decay_rate**timer)
      self.learning_rate.assign(learning_rate).eval()
      logging.info('Decay learning rate to {:6f}'.format(learning_rate))

  def train(self):
    """Main train loop."""

    self.set_input()
    self.build_graph()
    iter_epoch = self.iter_epoch

    with self.strategy.scope():
      self.initialize_variables()
      self.sess.run([
          self.train_input_iterator.initialize()
      ])
      self.sess.run([self.eval_input_iterator.initialize()])
      self.saver = tf.train.Saver(max_to_keep=5)

      self.load_model()
      FLAGS.restore_step = self.global_step.eval()

      pbar = tqdm(total=(FLAGS.max_iteration - FLAGS.restore_step))
      for iteration in range(FLAGS.restore_step + 1, FLAGS.max_iteration + 1):
        self.update_learning_rate(iteration)
        lr, net_loss, merged_summary, acc = self.sess.run([self.learning_rate] +
                                                          self.train_op)
        pbar.update(1)
        pbar.set_description(
            'Epoch {}[{}/{}] lr {:.3f} loss {:.2f} acc {:.3f}'.format(
                iteration // iter_epoch, iteration % iter_epoch, iter_epoch,
                float(lr), float(net_loss), acc))
        self.summary_writer.add_summary(merged_summary, iteration)

        # test and checkpoint
        if self.time_for_evaluation(iteration, lr):
          self.evaluate(iteration, lr)
          self.save_model(
              iteration)
          self.summary_writer.flush()
      pbar.close()

  def time_for_evaluation(self, iteration, lr):
    """Decides whether need to do evaluation.

    For cosine learning rate we want to evaluate when learning rate gets to 0.

    Args:
      iteration: current iteration
      lr: learning rate

    Returns:
      Boolean that determines whether need to do evaluation or not.
    """
    if iteration == 0: return False
    do = False
    if FLAGS.lr_schedule in ('cosine', 'cosine_warmup'):
      # estimate the cosine annealing bahavior
      if iteration == max(1, self.cos_eval_tot_step - 1):
        self.cos_eval_tot_step += self.cos_eval_step
        self.cos_eval_step = round(self.cos_eval_step * FLAGS.cos_t_mul)
        logging.info('[Cicle end] lr {}, steps {} nextsteps {}'.format(
            lr, iteration, self.cos_eval_tot_step - 1))
        do = True
    return (iteration % FLAGS.eval_freq == 0 or iteration == 1) or do

  def evaluation(self):
    """Perform evaluation."""
    self.set_input()
    self.build_graph()

    with self.strategy.scope():
      self.initialize_variables()
      self.saver = tf.train.Saver()
      self.load_model()
      self.evaluate(self.global_step.eval())

  def evaluate(self, iteration, lr=0):
    """Evalation for each epoch.

    Args:
      iteration: current iteration
      lr: learning rate
    Returns:
      Bool whether it is the best model at this point.
    """
    self.clean_acc_history()
    labels, preds, logits = [], [], []
    with self.strategy.scope():
      self.sess.run(self.eval_input_iterator.initialize())
      vds, vbs = self.dataset.val_dataset_size, FLAGS.val_batch_size
      total = vds // vbs + (vds % vbs != 0)
      pbar = tqdm(total=total)
      for _ in range(total):
        try:
          test_acc, logit, label, merged_summary = self.sess.run(self.eval_op)
        except tf.errors.OutOfRangeError:
          break
        labels.append(label)
        preds.append(np.argmax(logit, 1))
        logits.append(logit)
        pbar.update(1)
        pbar.set_description('batch {} accuracy {:.3f} ({:.3f})'.format(
            label.shape[0],
            float(
                utils.topk_accuracy(
                    logit, label, topk=1,
                    ignore_label_above=self.dataset.num_classes)), test_acc))
      pbar.close()
      if FLAGS.mode != 'evaluation':
        self.eval_summary_writer.add_summary(merged_summary,
                                             self.global_step.eval())
      # Updates this variable and update in next round train
      labels, preds, logits = np.concatenate(labels, 0), np.concatenate(
          preds, 0), np.concatenate(logits, 0)
      offline_accuracy, num_evaluated = utils.topk_accuracy(
          logits,
          labels,
          topk=1,
          ignore_label_above=self.dataset.num_classes,
          return_counts=True)
      top5acc = utils.topk_accuracy(
          logits, labels, topk=5, ignore_label_above=self.dataset.num_classes)

      self.clean_acc_history()
      logging.info('[Evaluation] lr {:.5f} global_step {} total {} acc '
                   '{:.3f} (top-5 {:.3f})'.format(
                       float(lr), iteration, num_evaluated, offline_accuracy,
                       float(top5acc)))

  def initialize_variables(self):
    """Initialize global variables."""
    train_vars = tf.trainable_variables()
    other_vars = [
        var for var in tf.global_variables() + tf.local_variables()
        if var not in train_vars
    ]
    self.sess.run([v.initializer for v in train_vars])
    self.sess.run([v.initializer for v in other_vars])

  def clean_acc_history(self):
    """Cleans accumulated counter in metrics.accuracy."""

    if not hasattr(self, 'clean_accstate_op'):
      self.clean_accstate_op = [
          a.assign(0) for a in utils.get_var(tf.local_variables(), 'accuracy')
      ]
      logging.info(
          'Create {} clean_accstate_op for state ops: {}'.format(
              len(self.clean_accstate_op), self.clean_accstate_op))

    self.sess.run(self.clean_accstate_op)

  def build_graph(self):
    """Builds graph."""
    self.create_graph()
    logging.info('Save checkpoint to {}'.format(
        FLAGS.checkpoint_path))
    self.summary_writer = tf.summary.FileWriter(os.path.join(
        FLAGS.checkpoint_path, 'train'))
    self.eval_summary_writer = tf.summary.FileWriter(os.path.join(
        FLAGS.checkpoint_path, 'eval'))

  def create_graph(self):
    logging.info('Build train graph')
    with self.strategy.scope():
      self.train_op = self.train_step()
      self.eval_op = self.eval_step()

  def train_step(self):
    """A single train step with strategy."""

    def step_fn(inputs):
      """Step function for training.

      Args:
        inputs: inputs data

      Returns:
        a list of observable tensors
      """
      images, labels = inputs
      net = self.net
      logits = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      logits = tf.cast(logits, tf.float32)
      loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
      loss = tf.reduce_mean(loss) + net.regularization_loss
      loss /= self.strategy.num_replicas_in_sync
      extra_ops = []
      if FLAGS.use_ema:
        ema_op = self.ema.apply(net.trainable_variables)
        extra_ops.append(ema_op)
      with tf.control_dependencies(net.updates + extra_ops):
        minimizer_op = self.optimizer.minimize(
            loss, global_step=self.global_step)
      acc, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))

      with tf.control_dependencies([minimizer_op, acc_update_op]):
        return tf.identity(loss), tf.identity(acc)

    pr_losses, pr_acc = self.strategy.experimental_run_v2(
        step_fn, args=(next(self.train_input_iterator),))

    mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_losses)
    acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    self.epoch_var = tf.cast(
        self.global_step / self.iter_epoch, tf.float32, name='epoch')

    merges = []
    merges.append(tf.summary.scalar('acc/train', acc))
    merges.append(tf.summary.scalar('loss/net', mean_loss))
    merges.append(tf.summary.scalar('epoch', self.epoch_var))
    merges.append(tf.summary.scalar('learningrate', self.learning_rate))
    summary = tf.summary.merge(merges)

    return [mean_loss, summary, acc]

  def eval_step(self):
    """Evaluate step."""

    def ema_getter(getter, name, *args, **kwargs):
      # for ExponentialMovingAverage
      var = getter(name, *args, **kwargs)
      ema_var = self.ema.average(var)
      return ema_var if ema_var else var  # for batchnorm use the original one

    def step_fn(inputs):
      """Step function."""

      images, labels = inputs
      net = self.net
      if FLAGS.use_ema:
        logits = net(
            images,
            name='model',
            reuse=True,
            training=False,
            custom_getter=ema_getter)
      else:
        logits = net(images, name='model', reuse=True, training=False)
      loss = tf.reduce_mean(
          tf.losses.sparse_softmax_cross_entropy(labels, logits))
      acc_op, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))
      with tf.control_dependencies([acc_update_op]):
        return tf.identity(loss), tf.identity(acc_op),\
               tf.identity(logits), tf.identity(labels)

    pr_loss, pr_acc, pr_logits, pr_labels = self.strategy.experimental_run(
        step_fn, self.eval_input_iterator)

    logits = self.strategy.unwrap(pr_logits)
    logits = tf.concat(logits, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_loss)
    merges = []
    merges.append(tf.summary.scalar('acc/eval', mean_acc))
    merges.append(tf.summary.scalar('loss/eval', mean_loss))
    summary = tf.summary.merge(merges)

    return [mean_acc, logits, labels, summary]
