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

r"""Train and evaluate models: optimize FNR at a fixed FPR.

It implements the method proposed in https://arxiv.org/abs/2107.10960 for
optimizing FNR at a fixed FPR. It also implments the baseline of cross-entropy
loss which can be run using the 'method' flag.
"""

import functools
import logging
import os
import sys

from absl import app
from absl import flags
import ml_collections
import numpy as np
from sklearn import metrics
import tensorflow as tf
import tensorflow_datasets as tfds

from implicit_constrained_optimization import co_utils
from implicit_constrained_optimization import models

FLAGS = flags.FLAGS
num_eval_samples = 19962

flags.DEFINE_string(
    'model_dir', '/usr/local/google/home/abhishk/logs/constrained_opt',
    'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_string(
    'proxy_fn_obj', 'softplus',
    'proxy function for 0-1 loss (objective): sigmoid/softplus')
flags.DEFINE_string(
    'proxy_fn_cons', 'softplus',
    'proxy function for 0-1 loss (constraint): sigmoid/softplus')
flags.DEFINE_float(
    'sigmoid_temp', 0.001,
    'Temperature used in the sigmoid proxy for loss and constraint')
flags.DEFINE_string('method', 'ico', 'loss type: ce/ico')
flags.DEFINE_string('attr', 'Black_Hair', 'Attribute name for CelebA')
flags.DEFINE_float('target_fpr', 0.01, 'target false positive rate')
flags.DEFINE_integer('n_batches_for_threshold', 10,
                     'Number of batches used for computing theshold')
flags.DEFINE_integer(
    'th_project_freq', 1000,
    'Number of minibatches after which threshold is set to the operating point')
flags.DEFINE_integer('grad_update_threshold', 1,
                     'Update threshold using gradient (0/1)')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('train_iters', 50000, 'training iterations')
flags.DEFINE_float('clip_grad_min', 1e-5,
                   'lower limit on gradient of constraint wrt. threshold')
flags.DEFINE_integer('eval_freq', 317, 'evaluation periodicity in iterations')


def preprocess(features):
  """Preprocess function for images."""
  image = features['image']
  image = tf.image.resize_with_crop_or_pad(image, 160, 160)
  image = tf.image.resize(image, [32, 32])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.cast(features['attributes'][FLAGS.attr], tf.float32)
  return dict(image=image, label=label)


def get_dataset(batch_size, data='celeb_a'):
  """TFDS Dataset."""
  ds, _ = tfds.load(data, split='train', with_info=True)
  ds = ds.repeat().shuffle(
      batch_size * 4, seed=1).batch(
          batch_size,
          drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
  ds = ds.map(preprocess)

  ds_valid, _ = tfds.load(data, split='validation', with_info=True)
  ds_valid = ds_valid.shuffle(
      10000, seed=1).batch(
          batch_size,
          drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  ds_valid = ds_valid.map(preprocess)

  ds_tst, _ = tfds.load(data, split='test', with_info=True)
  ds_tst = ds_tst.shuffle(
      10000, seed=1).batch(
          batch_size,
          drop_remainder=False).prefetch(tf.data.experimental.AUTOTUNE)
  ds_tst = ds_tst.map(preprocess)
  return ds, ds_valid, ds_tst


def binary_cross_entropy(y_true, y_pred, from_logits=False):
  """Binary CrossEntropy Loss."""
  bce = tf.keras.losses.BinaryCrossentropy(from_logits=from_logits)(y_true,
                                                                    y_pred)
  return tf.reduce_sum(bce)


def get_threshold_for_metric_ub(metric, target_val, thresholds):
  """Compute threshold where metric is less than the target value."""
  ind = np.array(metric) <= target_val  # constraint less than target value
  ind_metric = np.where(metric == np.max(metric[ind]))[0][0]
  target_th = thresholds[ind_metric]
  return target_th


def update_threshold(model, train_iterator, num_batches, training):
  """Update threshold such that constraint is satisfied on the specified number of minibatches."""
  predictions = np.zeros(num_batches * FLAGS.batch_size)
  labels = np.zeros(num_batches * FLAGS.batch_size)
  for i in range(num_batches):
    batch = next(train_iterator)
    images = batch['image']
    predictions[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size] = model(
        [images, training])
    labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size] = batch['label']
  fprs, _, thresholds = metrics.roc_curve(labels, predictions)
  return get_threshold_for_metric_ub(fprs, FLAGS.target_fpr, thresholds)


def eval_step(model, threshold, loss_op, constraint_op, proxy_funcs,
              keras_metrics, images, labels):
  """Evaluation step on a minibatch."""
  predictions = model([images, False])
  one_hot_labels = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), depth=2))
  pred_probs = tf.sigmoid(predictions - threshold)
  pred_probs_twoclass = tf.stack([1 - pred_probs, pred_probs], axis=1)
  num_pos = tf.reduce_sum(labels)
  num_neg = tf.reduce_sum(1 - labels)
  if FLAGS.method == 'ce':
    loss = loss_op(one_hot_labels, pred_probs_twoclass) / FLAGS.batch_size
  else:
    loss = loss_op(threshold, predictions, labels)
  fp_exact = co_utils.fp_func(threshold, predictions, labels)
  fn_exact = co_utils.fn_func(threshold, predictions, labels)
  fp_proxy = proxy_funcs['fp'](threshold, predictions, labels)
  fn_proxy = proxy_funcs['fn'](threshold, predictions, labels)
  constraint = constraint_op(threshold, predictions, labels)
  # Update states.
  keras_metrics['eval_loss'].update_state(loss)
  keras_metrics['eval_acc'].update_state(one_hot_labels, pred_probs_twoclass)
  return (predictions, loss, constraint, fp_exact, fn_exact, fp_proxy, fn_proxy,
          num_pos, num_neg)


def eval_metrics_on_data_source(ds,
                                model,
                                threshold,
                                loss_op,
                                constraint_op,
                                proxy_funcs,
                                keras_metrics,
                                target_th=None):
  """Evaluate metric on the given TFDS data source."""
  iterator = iter(ds)
  fp_total = 0
  fn_total = 0
  fp_proxy_total = 0
  fn_proxy_total = 0
  total_pos = 0
  total_neg = 0
  avg_loss = 0
  labels_all = np.zeros(num_eval_samples, dtype=np.float32)
  preds_all = np.zeros(num_eval_samples, dtype=np.float32)
  batch_ind = 0
  results = ml_collections.ConfigDict()
  for batch in iterator:
    images, labels = batch['image'], batch['label']
    labels_all[batch_ind * FLAGS.batch_size:batch_ind * FLAGS.batch_size +
               len(labels)] = labels.numpy()
    preds, loss, _, fp_exact, fn_exact, fp_proxy, fn_proxy, num_pos, num_neg = eval_step(
        model, threshold, loss_op, constraint_op, proxy_funcs, keras_metrics,
        images, labels)
    preds_all[batch_ind * FLAGS.batch_size:batch_ind * FLAGS.batch_size +
              len(preds)] = preds.numpy()
    fp_total += fp_exact
    fn_total += fn_exact
    fp_proxy_total += fp_proxy
    fn_proxy_total += fn_proxy
    total_pos += num_pos
    total_neg += num_neg
    avg_loss += loss
    batch_ind += 1
  results.fpr_exact = fp_total / total_neg
  results.fnr_exact = fn_total / total_pos
  results.fpr_proxy = fp_proxy_total / total_neg
  results.fnr_proxy = fn_proxy_total / total_pos
  results.avg_loss = avg_loss / batch_ind

  results.acc = np.mean(
      np.logical_not(
          np.logical_xor(preds_all - threshold.numpy() > 0, labels_all > 0)))
  if target_th is None:
    fprs, _, thresholds = metrics.roc_curve(labels_all, preds_all)
    target_th = get_threshold_for_metric_ub(fprs, FLAGS.target_fpr, thresholds)
  results.fpr_exact_th = co_utils.fpr_func(target_th, preds_all, labels_all)
  results.fnr_exact_th = co_utils.fnr_func(target_th, preds_all, labels_all)
  results.fpr_proxy_th = proxy_funcs['fpr'](target_th, preds_all, labels_all)
  results.fnr_proxy_th = proxy_funcs['fnr'](target_th, preds_all, labels_all)
  results.target_th = target_th
  return results


def eval_metrics(model, threshold, valid_ds, eval_ds, loss_op, constraint_op,
                 proxy_funcs, keras_metrics, it, eval_summary_writer,
                 best_results, logger):
  """Evaluate the current model on metrics of interest."""

  def log_results(results):
    info_str = ('Iter %s: loss=%s, acc=%s, '
                'far=%s, fpr_proxy=%s, frr=%s, fnr_proxy=%s, '
                'threshold=%s') % (it, results.avg_loss, round(
                    results.acc,
                    4), results.fpr_exact, results.fpr_proxy, results.fnr_exact,
                                   results.fnr_proxy, threshold)
    logger.info(info_str)
    info_str = ('Iter %s: At corrected threshold: far=%s, frr=%s, '
                'threshold=%s') % (it, round(results.fpr_exact_th.numpy(), 4),
                                   round(results.fnr_exact_th.numpy(),
                                         4), round(results.target_th, 4))
    logger.info(info_str)

  logger.info('==================== Eval iter %d =====================', it)
  # validation set
  valid_results = eval_metrics_on_data_source(valid_ds, model, threshold,
                                              loss_op, constraint_op,
                                              proxy_funcs, keras_metrics)
  logger.info('Validation set -------')
  log_results(valid_results)
  # test set
  eval_results = eval_metrics_on_data_source(eval_ds, model, threshold, loss_op,
                                             constraint_op, proxy_funcs,
                                             keras_metrics)
  logger.info('\nTest set (at tuned threshold) -------')
  log_results(eval_results)
  # best eval result so far
  if best_results.fnr_valid > valid_results.fnr_exact_th.numpy():
    best_results.fnr_valid = float(valid_results.fnr_exact_th.numpy())
    best_results.fpr_valid = float(valid_results.fpr_exact_th.numpy())
    best_results.fnr_eval_at_valid = float(eval_results.fnr_exact_th.numpy())
    best_results.fpr_eval_at_valid = float(eval_results.fpr_exact_th.numpy())
  info_str = ('\nFor best validated model: (@ '
              'tuned th) far=%s, frr=%s') % (best_results.fpr_eval_at_valid,
                                             best_results.fnr_eval_at_valid)
  logger.info(info_str)
  logger.info('========================================')

  with eval_summary_writer.as_default():
    tf.summary.scalar('loss/eval', eval_results.avg_loss, step=it)
    tf.summary.scalar('acc/eval', eval_results.acc, step=it)
    tf.summary.scalar('fpr_proxy/eval', eval_results.fpr_proxy, step=it)
    tf.summary.scalar('fnr_proxy/eval', eval_results.fnr_proxy, step=it)
    tf.summary.scalar(
        'fnr_th/eval_at_best_valid', best_results.fnr_eval_at_valid, step=it)
    tf.summary.scalar(
        'fpr_th/eval_at_best_valid', best_results.fpr_eval_at_valid, step=it)

  keras_metrics['eval_loss'].reset_states()
  keras_metrics['eval_acc'].reset_states()
  return best_results


def train_step(model, threshold, loss_op, constraint_op, optimizer,
               keras_metrics, images, labels):
  """Performs single training step."""

  with tf.GradientTape(persistent=True) as tape:
    predictions = model([images, True])
    if FLAGS.method == 'ico':
      tape.watch(threshold)
    one_hot_labels = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), depth=2))
    pred_probs = tf.sigmoid(predictions - threshold)
    pred_probs_twoclass = tf.stack([1 - pred_probs, pred_probs], axis=1)
    if FLAGS.method == 'ce':
      loss = loss_op(one_hot_labels, pred_probs_twoclass) / FLAGS.batch_size
      constraint = constraint_op(tf.constant(0.), predictions, labels)
    else:
      loss = loss_op(threshold, predictions, labels)
      with tf.GradientTape() as tape_cons:
        tape_cons.watch(threshold)
        constraint = constraint_op(threshold, predictions, labels)

  if FLAGS.method == 'ce':
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
  else:
    train_vars_old = [var.numpy() for var in model.trainable_variables]
    grad_loss_w = tape.gradient(loss, model.trainable_variables)
    grad_loss_th = tape.gradient(loss, threshold)
    grad_constraint_w = tape.gradient(constraint, model.trainable_variables)
    grad_constraint_th = tape.gradient(constraint, threshold)
    grad_th_w = [None] * len(grad_constraint_w)
    for gi, grad_cons_w_layer in enumerate(grad_constraint_w):
      grad_constraint_th_clip = tf.sign(grad_constraint_th) * tf.clip_by_value(
          tf.abs(grad_constraint_th), FLAGS.clip_grad_min, np.infty)
      grad_th_w[gi] = -1 * tf.math.divide_no_nan(grad_cons_w_layer,
                                                 grad_constraint_th_clip)
    # grad_th_w = [
    #     -1 * tf.math.divide_no_nan(
    #         grad_cons_w_layer,
    #         tf.sign(grad_constraint_th) * tf.clip_by_value(
    #             tf.abs(grad_constraint_th), FLAGS.clip_grad_min, np.infty))
    #     for grad_cons_w_layer in grad_constraint_w
    # ]

    # final gradient wrt. model parameters
    final_grad_w = [
        grad_loss_w[i] + grad_loss_th * grad_th_w[i]
        for i in range(len(grad_loss_w))
    ]
    optimizer.apply_gradients(zip(final_grad_w, model.trainable_variables))
    # update threshold
    train_vars_update = [
        var - train_vars_old[i]
        for i, var in enumerate(model.trainable_variables)
    ]
    est_step_th = tf.reduce_sum([
        tf.reduce_sum(tf.multiply(delta_w, grad_th_w[i]))
        for i, delta_w in enumerate(train_vars_update)
    ])

    if FLAGS.grad_update_threshold:
      threshold.assign_add(est_step_th)

  del tape

  # Update states.
  keras_metrics['train_loss'].update_state(loss)
  keras_metrics['train_acc'].update_state(one_hot_labels, pred_probs_twoclass)
  return loss, constraint


def train_and_eval_model(logger):
  """Trains and evaluates the model."""

  train_ds, valid_ds, eval_ds = get_dataset(batch_size=FLAGS.batch_size)

  # Building model.
  logger.info('Building model...')
  if FLAGS.method == 'ce':
    model = models.build_model(image_size=32, bias_last=True)
  else:
    model = models.build_model(image_size=32, bias_last=False)
  threshold = tf.Variable(0., trainable=False)
  optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.lr)

  # Metrics
  train_acc = tf.keras.metrics.CategoricalAccuracy(
      'train_acc', dtype=tf.float32)
  eval_acc = tf.keras.metrics.CategoricalAccuracy('eval_acc', dtype=tf.float32)
  train_loss = tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32)
  eval_loss = tf.keras.metrics.Mean(name='eval_loss', dtype=tf.float32)
  keras_metrics = dict()
  keras_metrics['train_loss'] = train_loss
  keras_metrics['eval_loss'] = eval_loss
  keras_metrics['train_acc'] = train_acc
  keras_metrics['eval_acc'] = eval_acc

  # proxy or surrogate function for FNR
  if FLAGS.proxy_fn_obj == 'sigmoid':
    fnr_proxy_func = functools.partial(
        co_utils.fnr_sigmoid_proxy_func, temperature=FLAGS.sigmoid_temp)
    fn_proxy_func = functools.partial(
        co_utils.fn_sigmoid_proxy_func, temperature=FLAGS.sigmoid_temp)
  elif FLAGS.proxy_fn_obj == 'softplus':
    fnr_proxy_func = functools.partial(
        co_utils.fnr_softplus_proxy_func, temperature=FLAGS.sigmoid_temp)
    fn_proxy_func = functools.partial(
        co_utils.fn_softplus_proxy_func, temperature=FLAGS.sigmoid_temp)
  else:
    raise NotImplementedError('Proxy function {} not implemented'.format(
        FLAGS.proxy_fn))
  # proxy or surrogate function for FPR
  if FLAGS.proxy_fn_cons == 'sigmoid':
    fpr_proxy_func = functools.partial(
        co_utils.fpr_sigmoid_proxy_func, temperature=FLAGS.sigmoid_temp)
    fp_proxy_func = functools.partial(
        co_utils.fp_sigmoid_proxy_func, temperature=FLAGS.sigmoid_temp)
  elif FLAGS.proxy_fn_cons == 'softplus':
    fpr_proxy_func = functools.partial(
        co_utils.fpr_softplus_proxy_func, temperature=FLAGS.sigmoid_temp)
    fp_proxy_func = functools.partial(
        co_utils.fp_softplus_proxy_func, temperature=FLAGS.sigmoid_temp)
  else:
    raise NotImplementedError('Proxy fn {} not implemented'.format(
        FLAGS.proxy_fn))
  proxy_funcs = dict()
  proxy_funcs['fnr'] = fnr_proxy_func
  proxy_funcs['fpr'] = fpr_proxy_func
  proxy_funcs['fp'] = fp_proxy_func
  proxy_funcs['fn'] = fn_proxy_func

  # loss and constraint ops
  if FLAGS.method == 'ce':
    loss_op = binary_cross_entropy
  else:
    loss_op = fnr_proxy_func
  constraint_op = fpr_proxy_func

  # Create summary writers
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/train'))
  eval_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/eval'))

  # eval performance at best validated model
  best_results = ml_collections.ConfigDict()
  best_results.fnr_valid = np.inf
  best_results.fpr_valid = np.inf
  best_results.fnr_eval_at_valid = np.inf
  best_results.fpr_eval_at_valid = np.inf

  # Main training loop.
  train_iterator = iter(train_ds)
  initial_epoch = 0
  for it in range(FLAGS.train_iters):
    batch = next(train_iterator)
    images, labels = batch['image'], batch['label']
    loss, constraint = train_step(model, threshold, loss_op, constraint_op,
                                  optimizer, keras_metrics, images, labels)
    if FLAGS.method == 'ico' and (it + 1) % FLAGS.th_project_freq == 0:
      # Correction step
      threshold_new = update_threshold(
          model, train_iterator, FLAGS.n_batches_for_threshold, training=True)
      logger.info('updating threshold: old {}, new {}'.format(
          threshold, threshold_new))
      threshold.assign(threshold_new)
    if it % 100 == 0:
      with train_summary_writer.as_default():
        tf.summary.scalar('threshold', threshold.numpy(), step=it)
        tf.summary.scalar(
            'loss/train', keras_metrics['train_loss'].result().numpy(), step=it)
        tf.summary.scalar(
            'acc/train', keras_metrics['train_acc'].result().numpy(), step=it)
        tf.summary.scalar('fpr_proxy/train', constraint, step=it)
      info_str = ('Train iter %s: loss=%s, acc=%s, constraint=%s') % (
          it, round(keras_metrics['train_loss'].result().numpy(),
                    8), round(keras_metrics['train_acc'].result().numpy(),
                              4), round(constraint.numpy(), 4))
      logger.info(info_str)

    keras_metrics['train_loss'].reset_states()
    keras_metrics['train_acc'].reset_states()

    # Evaluation
    if (it + 1) % FLAGS.eval_freq == 0:
      best_results = eval_metrics(model, threshold, valid_ds, eval_ds, loss_op,
                                  constraint_op, proxy_funcs, keras_metrics, it,
                                  eval_summary_writer, best_results, logger)


def main(_):
  if not tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.makedirs(FLAGS.model_dir)

  # Train and eval.
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)
  stdout_handler = logging.StreamHandler(sys.stdout)
  logger.addHandler(stdout_handler)
  logger.info('Start training and eval')
  train_and_eval_model(logger)


if __name__ == '__main__':
  app.run(main)
