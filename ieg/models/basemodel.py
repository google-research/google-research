# coding=utf-8
# coding=utf-8
"""BaseModel that implements basics to support training pipeline."""
from __future__ import absolute_import
from __future__ import division

import os

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tqdm import tqdm

from ieg import options
from ieg import utils
from ieg.nets import cifar_resnetv1
from ieg.nets import cifar_resnetv2
from ieg.nets import resnet50
from ieg.nets import wrn

FLAGS = flags.FLAGS
logging = tf.logging


def create_network(name, num_classes):
  """Creates networks."""
  net = None
  logging.info('Create network [{}] ...'.format(name))

  if name == 'resnet29':
    net = cifar_resnetv2.ResNet(depth=29, num_classes=num_classes)
  elif name == 'resnet32':
    net = cifar_resnetv1.ResNetBuilder(num_layers=32, num_classes=num_classes)
  elif name == 'resnet18':
    net = cifar_resnetv1.ResNetBuilder(
        num_layers=20, version='v2', num_classes=num_classes)
  elif name == 'wrn28-10':
    net = wrn.WRN(num_classes=num_classes, wrn_size=160)
  elif name == 'resnet50':
    net = resnet50.ImagenetModelv2(
        num_classes=num_classes, weight_decay_rate=FLAGS.l2_weight_decay)
  else:
    raise ValueError('{} net is not implemented'.format(name))
  return net


class Trainer(object):
  """Trainer with basic utility functions."""

  def set_lr_schedule(self):
    """Setup learning rate schedule."""

    if FLAGS.lr_schedule == 'custom_step':
      # TODO(zizhaoz): Enable FLAGS.warmup_epochs.
      logging.info(
          'Using custom step for learning rate decay schedule {}'.format([
              (a // self.iter_epoch, FLAGS.learning_rate**(k + 1))
              for k, a in enumerate(self.decay_steps)
          ]))
      self.learning_rate = tf.get_variable(
          'learning_rate',
          dtype=tf.float32,
          trainable=False,
          initializer=tf.constant(FLAGS.learning_rate),
          aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA)

    elif FLAGS.lr_schedule == 'cosine_one':
      logging.info('using cosine_one step learning rate decay, step: {}'.format(
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
      if FLAGS.warmup_epochs:
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

      if FLAGS.warmup_epochs:
        logging.info('Enable warmup with warmup_steps: {}'.format(warmup_steps))
        warmup_learning_rate = 0.0
        slope = (FLAGS.learning_rate - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * tf.cast(self.global_step,
                                      tf.float32) + warmup_learning_rate
        learning_rate = tf.where(self.global_step < warmup_steps, warmup_rate,
                                 learning_rate)

      self.learning_rate = learning_rate

    elif FLAGS.lr_schedule == 'exponential':
      logging.info('using exponential learning rate decay, step: {}'.format(
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
    """Adjusts all parameters for multiple GPUs."""
    strategy = self.strategy
    logging.info(
        'Adjust hyperparameters based on num_replicas_in_sync {}'.format(
            strategy.num_replicas_in_sync))
    FLAGS.batch_size *= strategy.num_replicas_in_sync
    self.iter_epoch = self.dataset.train_dataset_size // FLAGS.batch_size

    if FLAGS.max_epoch:
      logging.info(
          'Use epoch scale hyperparam and reset max_iteration and decay_epochs.'
      )
      FLAGS.max_iteration = FLAGS.max_epoch * self.iter_epoch
      FLAGS.decay_steps = ','.join([
          str(int(int(a) * self.iter_epoch))
          for a in FLAGS.decay_epochs.split(',')
      ])
    if FLAGS.lr_schedule == 'cosine':
      self.decay_steps = [self.iter_epoch]
    else:
      self.decay_steps = [int(a) for a in FLAGS.decay_steps.split(',')]
    if FLAGS.eval_freq == options.EVAL_FREQ:
      # If eval_freq is default, we set eval_freq as one epoch.
      FLAGS.eval_freq = self.iter_epoch
    FLAGS.val_batch_size *= strategy.num_replicas_in_sync

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
      if not path:
        return None
      logging.info('Load the latest checkpoint ' + path)
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
    self.enable_test_eval = False

    with self.strategy.scope():
      logging.info('[BaseModel] Parallel training in {} devices'.format(
          strategy.num_replicas_in_sync))
      self.net = create_network(FLAGS.network_name, dataset.num_classes)
      self.batch_size = FLAGS.batch_size
      self.val_batch_size = FLAGS.val_batch_size
      logging.info('[BaseModel] actual batch size {}'.format(self.batch_size))
      self.global_step = tf.train.get_or_create_global_step()

      self.set_lr_schedule()

      self.optimizer = tf.train.MomentumOptimizer(
          learning_rate=self.learning_rate, momentum=FLAGS.momentum)
      self.acc_func = tf.metrics.accuracy
      self.acc_top5_func = lambda labels, predictions: tf.metrics.mean(   # pylint: disable=g-long-lambda
          tf.nn.in_top_k(predictions=predictions, targets=labels, k=5),
          name='accuracy_top5')

      if FLAGS.use_ema:
        self.ema = tf.train.ExponentialMovingAverage(0.999, self.global_step)

      if FLAGS.summary_eval_to_train:
        # Summarize eval results calculated by numpy ane export to train events.
        self.eval_acc_on_train_pl = tf.placeholder(tf.float32, shape=(3,))
        self.eval_acc_on_train = tf.Variable(
            [0.0, 0.0, 0.0],  # [top-1, top-5, num_evaluated]
            trainable=False,
            dtype=tf.float32,
            name='eval_acc_train',
            aggregation=tf.compat.v1.VariableAggregation.ONLY_FIRST_REPLICA)
        self.eval_acc_on_train_assign_op = self.eval_acc_on_train.assign(
            self.eval_acc_on_train_pl)

  def set_input(self):
    """Set input function."""
    train_ds = self.dataset.train_dataflow.shuffle(
        buffer_size=1024).repeat().batch(
            self.batch_size, drop_remainder=True).prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
    if FLAGS.val_batch_size > self.dataset.val_dataset_size:
      raise ValueError(
          'FLAGS.val_batch_size should smaller than dataset.val_dataset_size')
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

  def load_model(self, path=None):
    """Load model from disk if there is any or required by FLAGS.restore_step.

    Args:
      path: The path of checkpoints. If not provided, it will infer
        automatically by FLAGS.restore_step.
    """
    if path is None:
      path = self.check_checkpoint()
    if path is not None:
      self.saver.restore(self.sess, save_path=path)
      logging.info(
          'Load model checkpoint {}, learning_rate {:3f} global_step {}'.format(
              path, self.learning_rate.eval(), self.global_step.eval()))
    else:
      if FLAGS.mode == 'evaluation' and not FLAGS.pretrained_ckpt:
        raise ValueError('Checkpoint not found for evaluation')

  def save_model(self, iteration):
    """Saves model."""
    path = '{}/checkpoint.ckpt'.format(FLAGS.checkpoint_path)
    save_path = self.saver.save(self.sess, path, global_step=iteration)
    logging.info('Save weights {} at iteration {}'.format(save_path, iteration))

  def update_learning_rate(self, global_step):
    """Updates learning rate."""
    if global_step in self.decay_steps and FLAGS.lr_schedule == 'custom_step':
      timer = self.decay_steps.index(global_step) + 1
      learning_rate = FLAGS.learning_rate * (FLAGS.decay_rate**timer)
      self.learning_rate.assign(learning_rate).eval()
      logging.info('Decay learning rate to {:6f}'.format(learning_rate))

  def train(self):
    """Main train loop."""
    self.set_input()
    self.build_graph()

    with self.strategy.scope():
      self.initialize_variables()
      self.sess.run([self.train_input_iterator.initializer])
      self.sess.run([self.eval_input_iterator.initializer])
      self.saver = tf.train.Saver(max_to_keep=5)

      self.load_model()
      FLAGS.restore_step = self.global_step.eval()

      for iteration in range(FLAGS.restore_step + 1, FLAGS.max_iteration + 1):
        self.update_learning_rate(iteration)
        lr, _, merged_summary, _ = self.sess.run([self.learning_rate] +
                                                 self.train_op)
        if iteration % 10 == 0 or iteration == 1:
          self.summary_writer.add_summary(merged_summary, iteration)

        # test and checkpoint
        if self.time_for_evaluation(iteration, lr):
          self.evaluate(iteration, lr)
          self.save_model(iteration)
          self.summary_writer.add_summary(merged_summary, iteration)
          self.summary_writer.flush()

  def profile_model(self):
    """Profiles model performance."""
    self.set_input()
    self.build_graph()

    with self.strategy.scope():
      self.initialize_variables()
      self.sess.run([self.train_input_iterator.initializer])
      self.saver = tf.train.Saver(max_to_keep=3)

      self.load_model()
      restore_step = self.global_step.eval()
      profile_enabled = 0
      for iteration in tqdm(range(restore_step + 1, FLAGS.max_iteration + 1)):
        self.update_learning_rate(iteration)
        if FLAGS.profile_training and profile_enabled == 0:
          profile_enabled = 1
          tf.compat.v2.profiler.experimental.start(
              os.path.join(FLAGS.checkpoint_path, 'profile'))
        (_, merged_summary, _) = self.sess.run(self.train_op)

        if iteration % 10 == 0 or iteration == 1:
          # Reduce write frequency to reduce size of tf events.
          self.summary_writer.add_summary(merged_summary, iteration)

        if FLAGS.profile_training:
          if profile_enabled == 1 and iteration > (FLAGS.restore_step +
                                                   FLAGS.profile_training):
            # Profile no more than 1 epoch
            tf.compat.v2.profiler.experimental.stop()
            profile_enabled = 2
            logging.info('Profiler stopped.')

  def time_for_evaluation(self, iteration, lr):
    """Decides whether need to do evaluation.

    For cosine learning rate we want to evaluate when learning rate gets to 0.

    Args:
      iteration: current iteration
      lr: learning rate

    Returns:
      Boolean that determines whether need to do evaluation or not.
    """
    if iteration == 0:
      return False
    do = False
    if FLAGS.lr_schedule in ('cosine',):
      warmup_steps = 0
      if FLAGS.warmup_epochs:
        warmup_steps = self.iter_epoch * FLAGS.warmup_epochs
      # estimate the cosine annealing bahavior
      if (iteration - warmup_steps) == max(1, self.cos_eval_tot_step - 1):
        self.cos_eval_tot_step += self.cos_eval_step
        self.cos_eval_step = round(self.cos_eval_step * FLAGS.cos_t_mul)
        logging.info('[Cicle end] lr {:.3f} steps {} next steps {}'.format(
            lr, iteration, self.cos_eval_tot_step - 1))
        do = True
    return (iteration % FLAGS.eval_freq == 0 or iteration == 1) or do

  def cosine_lr_early_stopping(self):
    if FLAGS.lr_schedule in ('cosine',):
      if FLAGS.max_iteration < self.cos_eval_tot_step:
        return True
    return False

  def evaluation(self):
    """Perform evaluation."""
    self.set_input()
    self.build_graph()

    with self.strategy.scope():
      self.initialize_variables()
      self.saver = tf.train.Saver()
      self.load_model()
      self.evaluate(self.global_step.eval())

  def write_to_summary(self, summary_writer, summary, iteration, flush=False):
    summary_writer.add_summary(summary, iteration)
    if flush:
      summary_writer.flush()

  def evaluate(self, iteration, lr=0, op=None, op_scope=''):
    """Evalation for each epoch.

    Args:
      iteration: current iteration
      lr: learning rate
      op: alternative to self.eval_op for a certain dataset
      op_scope: dataset scope name of the op
    """
    self.clean_acc_history()
    labels, logits = [], []
    if op is not None:
      assert op_scope
      eval_op = op
    else:
      eval_op = self.eval_op
      op_scope = 'val'

    with self.strategy.scope():
      self.sess.run(self.eval_input_iterator.initializer)
      vds, vbs = self.dataset.val_dataset_size, FLAGS.val_batch_size
      total = vds // vbs + (vds % vbs != 0)
      for _ in range(total):
        try:
          online_acc, logit, label, merged_summary = self.sess.run(eval_op)
        except tf.errors.OutOfRangeError:
          break
        if FLAGS.summary_eval_to_train:
          labels.append(label)
          logits.append(logit)
        else:
          del logit, label

      if FLAGS.mode != 'evaluation':
        self.write_to_summary(
            self.eval_summary_writer,
            merged_summary,
            self.global_step.eval(),
            flush=True)
      if FLAGS.summary_eval_to_train:
        # Updates this variable and update in next round train
        labels, logits = np.concatenate(labels, 0), np.concatenate(logits, 0)
        offline_accuracy, num_evaluated = utils.topk_accuracy(
            logits,
            labels,
            topk=1,
            # Useful for eval imagenet on webvision mini 50 classes.
            ignore_label_above=self.dataset.num_classes,
            return_counts=True)
        top5acc = utils.topk_accuracy(
            logits, labels, topk=5, ignore_label_above=self.dataset.num_classes)
        if op is None and FLAGS.mode != 'evaluation':
          # We only expoert validation op results to self.eval_acc_on_train.
          self.sess.run(
              self.eval_acc_on_train_assign_op,
              feed_dict={
                  self.eval_acc_on_train_pl:
                      np.array([
                          float(offline_accuracy),
                          float(top5acc), num_evaluated
                      ])
              })
      else:
        num_evaluated = -1
        offline_accuracy = online_acc
        top5acc = -1
      self.clean_acc_history()
      logging.info('Evaluation ({}): lr {:.5f} global_step {} total {} acc '
                   '{:.3f} (top-5 {:.3f})'.format(op_scope, float(lr),
                                                  iteration, num_evaluated,
                                                  offline_accuracy,
                                                  float(top5acc)))

  def initialize_variables(self):
    """Initializes global variables."""
    if FLAGS.pretrained_ckpt:
      # Used for imagenet pretraining
      self.net.init_model_weight(
          FLAGS.pretrained_ckpt,
          include_top=FLAGS.mode == 'evaluation',
          mode=FLAGS.pretrained_ckpt_mode)
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
      logging.info('Create {} clean accuracy state ops'.format(
          len(self.clean_accstate_op)))
    self.sess.run(self.clean_accstate_op)

  def build_graph(self):
    """Builds graph."""
    self.create_graph()
    logging.info('Save checkpoint to {}'.format(FLAGS.checkpoint_path))
    self.summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.checkpoint_path, 'train'))
    self.eval_summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.checkpoint_path, 'eval'))

  def create_graph(self):
    with self.strategy.scope():
      self.train_op = self.train_step()
      self.eval_op = self.eval_step()
      assert isinstance(self.train_op, list)
      assert isinstance(self.eval_op, list)
      if self.enable_test_eval:
        self.test_op = self.eval_step(self.test_input_iterator, 'test')
        assert isinstance(self.test_op, list)

  def train_step(self):
    """A single train step with strategy."""

    def step_fn(inputs):
      """Step function for training."""
      images, labels = inputs
      net = self.net
      logits = net(images, name='model', reuse=tf.AUTO_REUSE, training=True)
      logits = tf.cast(logits, tf.float32)
      oh_labels = tf.one_hot(labels, self.dataset.num_classes)
      loss = tf.losses.softmax_cross_entropy(
          oh_labels, logits, label_smoothing=FLAGS.label_smoothing)
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

    pr_losses, pr_acc = self.strategy.run(
        step_fn, args=(next(self.train_input_iterator),))

    mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_losses)
    acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    self.epoch_var = tf.cast(self.global_step / self.iter_epoch, tf.float32)

    merges = []
    merges.append(tf.summary.scalar('acc/train', acc))
    merges.append(tf.summary.scalar('loss/net', mean_loss))
    merges.append(tf.summary.scalar('epoch', self.epoch_var))
    merges.append(tf.summary.scalar('learningrate', self.learning_rate))
    if hasattr(self, 'eval_acc_on_train'):
      merges.append(
          tf.summary.scalar('acc/eval_on_train', self.eval_acc_on_train[0]))
      merges.append(
          tf.summary.scalar('acc/eval_on_train_top5',
                            self.eval_acc_on_train[1]))
      merges.append(
          tf.summary.scalar('acc/num_eval', self.eval_acc_on_train[2]))
    summary = tf.summary.merge(merges)

    return [mean_loss, summary, acc]

  def eval_step(self, alt_iterator=None, alt_iterator_scope=''):
    """Evaluate step."""

    if alt_iterator:
      iterator = alt_iterator
      assert alt_iterator_scope
      scope = '_' + alt_iterator_scope
    else:
      iterator = self.eval_input_iterator
      scope = ''

    def ema_getter(getter, name, *args, **kwargs):
      # Replace ExponentialMovingAverage variables
      var = getter(name, *args, **kwargs)
      ema_var = self.ema.average(var)
      return ema_var if ema_var else var  # for batchnorm use the original one

    def step_fn(inputs):
      """Step function."""
      images, labels = inputs
      net = self.net
      logits = net(images, name='model', reuse=True, training=False)
      loss = tf.reduce_mean(
          tf.losses.sparse_softmax_cross_entropy(labels, logits))
      acc, acc_update_op = self.acc_func(labels, tf.argmax(logits, axis=1))
      acctop5, acctop5_update_op = self.acc_top5_func(labels,
                                                      tf.nn.softmax(logits, -1))
      ops = [acc_update_op, acctop5_update_op]

      if FLAGS.use_ema:
        ema_logits = net(
            images,
            name='model',
            reuse=True,
            training=False,
            custom_getter=ema_getter)
        ema_acc, ema_acc_update_op = self.acc_func(
            labels, tf.argmax(ema_logits, axis=1))
        ema_acctop5, ema_acctop5_update_op = self.acc_top5_func(
            labels, tf.nn.softmax(ema_logits, -1))
        ops.extend([ema_acc_update_op, ema_acctop5_update_op])

        with tf.control_dependencies(ops):
          return (tf.identity(loss), tf.identity(acc), tf.identity(ema_acc),
                  tf.identity(acctop5), tf.identity(ema_acctop5),
                  tf.identity(ema_logits), tf.identity(labels))
      else:
        with tf.control_dependencies(ops):
          return (tf.identity(loss), tf.identity(acc), tf.identity(acctop5),
                  tf.identity(logits), tf.identity(labels))

    reduced_results = self.strategy.run(step_fn, args=(next(iterator),))
    if FLAGS.use_ema:
      (pr_loss, pr_acc, pr_ema_acc, pr_acctop5, pr_ema_acctop5, pr_logits,
       pr_labels) = reduced_results
    else:
      pr_loss, pr_acc, pr_acctop5, pr_logits, pr_labels = reduced_results

    logits = self.strategy.unwrap(pr_logits)
    logits = tf.concat(logits, axis=0)
    labels = self.strategy.unwrap(pr_labels)
    labels = tf.concat(labels, axis=0)

    mean_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acc)
    mean_acctop5 = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_acctop5)
    mean_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, pr_loss)
    merges = []
    merges.append(tf.summary.scalar('acc%s/eval' % scope, mean_acc))
    merges.append(tf.summary.scalar('acc%s/top5_eval' % scope, mean_acctop5))
    merges.append(
        tf.summary.scalar('acc%s/error' % scope, (1.0 - mean_acc) * 100))
    merges.append(tf.summary.scalar('loss%s/eval' % scope, mean_loss))
    if FLAGS.use_ema:
      mean_ema_acc = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                          pr_ema_acc)
      mean_ema_acctop5 = self.strategy.reduce(tf.distribute.ReduceOp.MEAN,
                                              pr_ema_acctop5)
      merges.append(tf.summary.scalar('acc%s/ema_eval' % scope, mean_ema_acc))
      merges.append(
          tf.summary.scalar('acc%s/top5_ema_eval' % scope, mean_ema_acctop5))
      merges.append(
          tf.summary.scalar('acc%s/ema_error' % scope,
                            (1.0 - mean_ema_acc) * 100))

    summary = tf.summary.merge(merges)
    return [mean_acc, logits, labels, summary]
