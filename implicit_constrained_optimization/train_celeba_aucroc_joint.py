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

r"""Train and evaluate models: Optimize for partial AUCROC.

It is possible to train a joint model for multiple attributes by specifying them
in the 'attr' flag. It implements the method proposed in
https://arxiv.org/abs/2107.10960
for optimizing partial AUCROC. It also implements the two baselines of
cross-entropy
loss and pairwise proxy loss, which can be selected using the 'method' flag.
"""

import ast
import functools
import logging
import os
import sys
import time

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

flags.DEFINE_string(
    'model_dir', '/usr/local/google/home/abhishk/logs/constrained_opt',
    'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_string(
    'proxy_fn_obj', 'sigmoid',
    'proxy function for 0-1 loss (objective): sigmoid/softplus')
flags.DEFINE_string(
    'proxy_fn_cons', 'sigmoid',
    'proxy function for 0-1 loss (constraint): sigmoid/softplus')
flags.DEFINE_float(
    'sigmoid_temp', 1.,
    'Temperature used in the sigmoid proxy for loss and constraint')
flags.DEFINE_string('method', 'ico', 'loss type: ce/pairwise/ico')
flags.DEFINE_string('attr', "['High_Cheekbones']",
                    'Attribute name(s) for CelebA, separeted by commas')
flags.DEFINE_integer('n_batches_for_threshold', 100,
                     'Number of batches used for computing theshold')
flags.DEFINE_integer(
    'th_project_freq', 1000,
    'Number of minibatches after which threshold is set to the operating point')
flags.DEFINE_integer('grad_update_threshold', 0,
                     'Update threshold using gradient (0/1)')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_integer('batch_size', 512, 'batch size')
flags.DEFINE_integer('train_iters', 50000, 'training iterations')
flags.DEFINE_float('clip_grad_min_ratio', 1e-5,
                   'lower limit on gradient of constraint wrt. threshold')
flags.DEFINE_float('fpr_low', 0., 'fpr lower limit for area under ROC curve')
flags.DEFINE_float('fpr_high', 0.01, 'fpr upper limit for area under ROC curve')
flags.DEFINE_integer('num_bins', 10, 'number of bins')
flags.DEFINE_integer('eval_freq', 317, 'evaluation periodicity in iterations')

num_classes = len(ast.literal_eval(FLAGS.attr))
num_eval_samples = 19962
bin_width = (FLAGS.fpr_high - FLAGS.fpr_low) / float(FLAGS.num_bins)
fpr_targets = np.arange(FLAGS.fpr_low, FLAGS.fpr_high,
                        bin_width) + bin_width / 2.


def preprocess(features):
  """Preprocess function for images."""
  image = features['image']
  image = tf.image.resize_with_crop_or_pad(image, 160, 160)
  image = tf.image.resize(image, [32, 32])
  image = tf.cast(image, tf.float32) / 255.0
  label = tf.cast(
      tf.stack(
          [features['attributes'][key] for key in ast.literal_eval(FLAGS.attr)],
          axis=1), tf.float32)
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


def ce_loss(labels, preds):
  """Binary CrossEntropy Loss."""
  bce = tf.keras.backend.binary_crossentropy(labels, preds)
  bce_mean = tf.reduce_mean(bce, axis=1)  # mean along label dimension
  return tf.reduce_sum(bce_mean)


def pairwise_auc_loss(labels,
                      logits,
                      beta=1.,
                      proxy='softplus',
                      temperature=1.):
  r"""Computes Partial ROC-AUC loss for the FPR range [0, beta].

  The partial AUC can be written as a pairwise loss between the positively
  labeled examples and the top beta-fraction of the negatively labeled examples.

    sum_{i \in Pos} sum_{j \in TopNeg} celoss(logit_i - logit_j).

  where i ranges over the positive datapoints, j ranges over the top beta
  fraction of the negative datapoints (sorted according to logits), logit_k
  denotes the logit (or score) of the k-th datapoint, and celoss is the sigmoid
  cross-entropy loss.

  Args:
    labels: A `Tensor` of shape [batch_size].
    logits: A `Tensor` with the same shape and dtype as `labels`.
    beta: A float value specifying an upper bound on the FPR. Defaults to 1.0
    proxy: Proxy function (softplus/sigmoid)
    temperature: Temperature parameter for the proxy function

  Returns:
    loss: A scalar loss `Tensor`.
  """
  if (beta <= 0.0) or (beta > 1.0):
    raise ValueError("'beta' needs to be in (0, 1].")

  # Convert inputs to tensors and standardize dtypes.
  labels = tf.reshape(tf.cast(labels, tf.float32), [-1, 1])
  logits = tf.reshape(tf.cast(logits, tf.float32), [-1, 1])

  # Separate out logits positively and negatively labeled examples.
  positive_logits = logits[labels > 0]
  negative_logits = logits[labels <= 0]

  # Pick top "beta" fraction of negatives to cover the FPR range [0, \beta].
  # Equivalently, we pick the top ceil(num_negatives x beta) negatives sorted by
  # logits.
  num_negatives = tf.reduce_sum(tf.cast(labels <= 0, dtype=tf.float32))
  num_beta_negatives = tf.cast(tf.math.ceil(num_negatives * beta), tf.int32)
  sorted_negative_logits = tf.sort(negative_logits, direction='DESCENDING')
  top_negative_logits = sorted_negative_logits[0:num_beta_negatives]

  # Create tensors of pairwise differences between positive logits and top
  # negative logits. These have shapes [num_positives, num_beta_negatives].
  logits_difference = tf.expand_dims(positive_logits, 0) - tf.expand_dims(
      top_negative_logits, 1)

  # Calculate proxy loss.
  if proxy == 'softplus':
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(logits_difference),
        logits=logits_difference * temperature)
  elif proxy == 'sigmoid':
    loss = tf.nn.sigmoid(-1 * logits_difference * temperature)
  else:
    raise ValueError('Unknown proxy {}'.format(proxy))
  return tf.reduce_mean(loss)


def get_threshold_for_metric_ub(metric, target_val, thresholds):
  """Compute threshold where metric is less than the target value."""
  ind = np.array(metric) <= target_val  # constraint less than target value
  ind_metric = np.where(metric == np.max(metric[ind]))[0][0]
  target_th = thresholds[ind_metric]
  return target_th


def update_threshold(model, train_iterator, num_batches, training):
  """Update threshold such that constraints are satisfied on the specified number of minibatches."""
  predictions = np.zeros((num_batches * FLAGS.batch_size, num_classes))
  labels = np.zeros((num_batches * FLAGS.batch_size, num_classes))
  th_updated = np.zeros((num_classes, FLAGS.num_bins))
  for i in range(num_batches):
    batch = next(train_iterator)
    images = batch['image']
    predictions[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size] = model(
        [images, training])
    labels[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size] = batch['label']
  for j in range(num_classes):
    fprs, _, thresholds = metrics.roc_curve(labels[:, j], predictions[:, j])
    for k, fpr_target in enumerate(fpr_targets):
      th = get_threshold_for_metric_ub(fprs, fpr_target, thresholds)
      th_updated[j, k] = th
  return th_updated


def train_step(model, threshold_var, loss_op, constraint_op, optimizer,
               keras_metrics, images, labels):
  """Performs single training step."""
  with tf.GradientTape(persistent=True) as tape:
    predictions = model([images, True])  # batchsize x attrs
    if FLAGS.method == 'aucroc':
      tape.watch(threshold_var)
    if FLAGS.method == 'ce':
      pred_probs = tf.sigmoid(predictions)
      loss = loss_op(labels, pred_probs) / FLAGS.batch_size
      constraint = tf.reduce_mean(
          constraint_op(
              tf.constant(
                  np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)),
              predictions, labels))
    elif FLAGS.method == 'pairwise':
      loss = loss_op(labels, predictions)
      constraint = tf.reduce_mean(
          constraint_op(
              tf.constant(
                  np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)),
              predictions, labels))
    else:
      with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(threshold_var)
        loss_per_label = loss_op(threshold_var, predictions,
                                 labels)  # classes x bins
        loss = tf.reduce_mean(loss_per_label)
        constraint_per_label = constraint_op(threshold_var, predictions,
                                             labels)  # classes x bins
        constraint = tf.reduce_sum(constraint_per_label)
      with tape.stop_recording():
        grad_loss_th = tape2.gradient(loss, threshold_var)
        grad_constraint_th = tape2.gradient(constraint, threshold_var)
        ratio_grads_th = tf.math.divide_no_nan(
            -1 * grad_loss_th,
            tf.sign(grad_constraint_th + 1e-10) * tf.clip_by_value(
                tf.abs(grad_constraint_th), FLAGS.clip_grad_min_ratio,
                np.infty))
      constraint_weighted = tf.reduce_sum(
          tf.multiply(constraint_per_label, ratio_grads_th))
      del tape2

  if FLAGS.method == 'ce' or FLAGS.method == 'pairwise':
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
  else:
    grad_loss_w = tape.gradient(loss, model.trainable_variables)
    grad_weighted_constraint_w = tape.gradient(constraint_weighted,
                                               model.trainable_variables)
    # final gradient wrt. model parameters
    final_grad_w = [
        grad_loss_w[i] + grad_weighted_constraint_w[i]
        for i in range(len(grad_loss_w))
    ]
    optimizer.apply_gradients(zip(final_grad_w, model.trainable_variables))
    if FLAGS.grad_update_threshold:
      # update threshold using forward pass
      predictions = model([images, False])  # batchsize x attrs
      th_updated = np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)
      for j in range(num_classes):
        fprs, _, thresholds = metrics.roc_curve(labels.numpy()[:, j],
                                                predictions.numpy()[:, j])
        for k, fpr_target in enumerate(fpr_targets):
          th = get_threshold_for_metric_ub(fprs, fpr_target, thresholds)
          th_updated[j, k] = th
      threshold_var.assign(th_updated)

  del tape

  # Update states.
  keras_metrics['train_loss'].update_state(loss)
  return loss, constraint


def eval_step(model, threshold_var, loss_op, constraint_op, proxy_funcs,
              keras_metrics, images, labels):
  """Evaluation step on a minibatch."""
  predictions = model([images, False])
  num_pos = tf.reduce_sum(labels, axis=0)
  num_neg = tf.reduce_sum(1 - labels, axis=0)
  if FLAGS.method == 'ce':
    pred_probs = tf.sigmoid(predictions)
    loss = loss_op(labels, pred_probs) / FLAGS.batch_size
  elif FLAGS.method == 'pairwise':
    loss = loss_op(labels, predictions)
  else:
    loss_per_label = loss_op(threshold_var, predictions,
                             labels)  # classes x bins
    loss = tf.reduce_mean(loss_per_label)
  tp_exact = co_utils.tp_func_multi(threshold_var, predictions, labels)
  fp_exact = co_utils.fp_func_multi_th(threshold_var, predictions, labels)
  tp_proxy = proxy_funcs['tp'](threshold_var, predictions, labels)
  fp_proxy = proxy_funcs['fp'](threshold_var, predictions, labels)
  constraint = constraint_op(threshold_var, predictions,
                             labels)  # classes x bins
  # Update states.
  keras_metrics['eval_loss'].update_state(loss)
  return (predictions, loss, constraint, tp_exact, tp_proxy, fp_exact, fp_proxy,
          num_pos, num_neg)


def eval_metrics_on_data_source(ds,
                                model,
                                threshold_var,
                                loss_op,
                                constraint_op,
                                proxy_funcs,
                                keras_metrics,
                                target_th=None):
  """Evaluate metric on the given TFDS data source."""
  iterator = iter(ds)
  tp_total = np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)
  fp_total = np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)
  tp_proxy_total = np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)
  fp_proxy_total = np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)
  total_pos = np.zeros(num_classes, dtype=np.int32)
  total_neg = np.zeros(num_classes, dtype=np.int32)
  avg_loss = 0
  labels_all = np.zeros((num_eval_samples, num_classes), dtype=np.float32)
  preds_all = np.zeros((num_eval_samples, num_classes), dtype=np.float32)
  batch_ind = 0
  results = ml_collections.ConfigDict()
  for batch in iterator:
    # batch = next(iterator)
    images, labels = batch['image'], batch['label']
    labels_all[batch_ind * FLAGS.batch_size:batch_ind * FLAGS.batch_size +
               len(labels)] = labels.numpy()
    preds, loss, _, tp_exact, tp_proxy, fp_exact, fp_proxy, num_pos, num_neg = eval_step(
        model, threshold_var, loss_op, constraint_op, proxy_funcs,
        keras_metrics, images, labels)
    preds_all[batch_ind * FLAGS.batch_size:batch_ind * FLAGS.batch_size +
              len(preds)] = preds.numpy()
    tp_total += tp_exact.numpy()
    fp_total += fp_exact.numpy()
    tp_proxy_total += tp_proxy
    fp_proxy_total += fp_proxy
    total_pos += num_pos
    total_neg += num_neg
    avg_loss += loss
    batch_ind += 1
  results.fpr_exact = fp_total / np.expand_dims(total_neg, axis=1)
  results.tpr_exact = tp_total / np.expand_dims(total_pos, axis=1)
  results.tpr_proxy = tp_proxy_total / np.expand_dims(total_pos, axis=1)
  results.fpr_proxy = fp_proxy_total / np.expand_dims(total_neg, axis=1)
  results.avg_loss = avg_loss / batch_ind

  fprs_all_classes = []
  tprs_all_classes = []
  if target_th is None:
    target_th = np.zeros((num_classes, FLAGS.num_bins), dtype=np.float32)
    for i in range(num_classes):
      fprs, tprs, thresholds = metrics.roc_curve(labels_all[:, i], preds_all[:,
                                                                             i])
      fprs_all_classes.append(fprs)
      tprs_all_classes.append(tprs)
      for j, fpr_target in enumerate(fpr_targets):
        target_th[i, j] = get_threshold_for_metric_ub(fprs, fpr_target,
                                                      thresholds)
  results.tpr_exact_th = co_utils.tpr_func_multi(target_th, preds_all,
                                                 labels_all)
  results.fpr_exact_th = co_utils.fpr_func_multi_th(target_th, preds_all,
                                                    labels_all)
  results.tpr_proxy_th = proxy_funcs['tpr'](target_th, preds_all, labels_all)
  results.fpr_proxy_th = proxy_funcs['fpr'](target_th, preds_all, labels_all)
  results.fprs_all = fprs_all_classes
  results.tprs_all = tprs_all_classes
  aucroc = np.zeros(num_classes, dtype=np.float32)
  for i in range(num_classes):
    aucroc[i] = metrics.roc_auc_score(
        labels_all[:, i], preds_all[:, i], max_fpr=FLAGS.fpr_high)
  results.aucroc = aucroc
  results.target_th = target_th
  return results


def eval_metrics(model, threshold_var, valid_ds, eval_ds, loss_op,
                 constraint_op, proxy_funcs, keras_metrics, it,
                 eval_summary_writer, best_results, logger, fp_log):
  """Evaluate the current model on metrics of interest."""

  def log_results(results):
    info_str = ('It %s: loss=%s, aucroc=%s, fprs=%s'
                'threshold=%s') % (it, results.avg_loss, results.aucroc, [
                    round(t, 4) for t in results.fpr_exact[0]
                ], threshold_var.numpy()[0])
    logger.info(info_str)
    info_str = ('It %s: At corrected threshold: aucroc=%s, fprs=%s, '
                'threshold=%s') % (it, results.aucroc, [
                    round(t, 4)
                    for t in np.mean(results.fpr_exact_th.numpy(), axis=0)
                ], [round(t, 4) for t in results.target_th[0]])
    logger.info(info_str)

  logger.info('==================== Eval iter %d =====================', it)
  # validation set
  valid_results = eval_metrics_on_data_source(valid_ds, model, threshold_var,
                                              loss_op, constraint_op,
                                              proxy_funcs, keras_metrics)
  logger.info('Validation set -------')
  log_results(valid_results)
  # test set
  eval_results = eval_metrics_on_data_source(eval_ds, model, threshold_var,
                                             loss_op, constraint_op,
                                             proxy_funcs, keras_metrics)
  logger.info('\nTest set (at tuned threshold) -------')
  log_results(eval_results)
  # best eval result so far
  if np.mean(best_results.aucroc_valid) < np.mean(valid_results.aucroc):
    best_results.aucroc_valid = valid_results.aucroc
    best_results.tprs_valid = valid_results.tpr_exact_th.numpy()
    best_results.aucroc_eval_at_valid = eval_results.aucroc
    best_results.tprs_eval_at_valid = eval_results.tpr_exact_th.numpy()
    best_results.eval_fprs_valid = eval_results.fprs_all
    best_results.eval_tprs_valid = eval_results.tprs_all
  info_str = (
      '\nFor best validated model:\n (@ '
      'tuned th) aucroc=%s, tprs=%s') % (best_results.aucroc_eval_at_valid,
                                         best_results.tprs_eval_at_valid)
  fp_log.write(info_str + '\n')
  logger.info(info_str)
  logger.info('========================================')

  with eval_summary_writer.as_default():
    tf.summary.scalar('loss/eval', eval_results.avg_loss, step=it)
    for i in range(num_classes):
      tf.summary.scalar(
          'aucroc/eval_at_best_valid_' + ast.literal_eval(FLAGS.attr)[i],
          best_results.aucroc_eval_at_valid[i],
          step=it)
    tf.summary.scalar(
        'aucroc/eval_at_best_valid',
        np.mean(best_results.aucroc_eval_at_valid),
        step=it)
    for i in range(FLAGS.num_bins):
      tf.summary.scalar(
          'tprs/eval_at_best_valid_bin' + str(i),
          np.mean(best_results.tprs_eval_at_valid[:, i]),
          step=it)

  keras_metrics['eval_loss'].reset_states()
  keras_metrics['eval_acc'].reset_states()
  return best_results


def train_and_eval_model(logger):
  """Trains and evaluates the model."""

  # discretize the FPR axis and set FPR targets
  num_bins = FLAGS.num_bins

  threshold_new = np.zeros((num_classes, num_bins), dtype=np.float32)

  train_ds, valid_ds, eval_ds = get_dataset(batch_size=FLAGS.batch_size)

  # Building model.
  logger.info('Building model...')
  if FLAGS.method == 'ce' or FLAGS.method == 'pairwise':
    model = models.build_model(
        image_size=32, bias_last=True, num_classes=num_classes, squeeze=False)
  else:
    model = models.build_model(
        image_size=32, bias_last=False, num_classes=num_classes, squeeze=False)

  threshold_var = tf.Variable(
      np.zeros((num_classes, num_bins), dtype=np.float32), trainable=False)

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

  if FLAGS.proxy_fn_obj == 'softplus':
    tpr_proxy_func = functools.partial(
        co_utils.tpr_softplus_proxy_func_multi, temperature=FLAGS.sigmoid_temp)
    tp_proxy_func = functools.partial(
        co_utils.tp_softplus_proxy_func_multi, temperature=FLAGS.sigmoid_temp)
  elif FLAGS.proxy_fn_obj == 'sigmoid':
    tpr_proxy_func = functools.partial(
        co_utils.tpr_sigmoid_proxy_func_multi, temperature=FLAGS.sigmoid_temp)
    tp_proxy_func = functools.partial(
        co_utils.tp_sigmoid_proxy_func_multi, temperature=FLAGS.sigmoid_temp)
  else:
    raise NotImplementedError('Proxy function {} not implemented'.format(
        FLAGS.proxy_fn_obj))

  if FLAGS.proxy_fn_cons == 'softplus':
    fpr_proxy_func = functools.partial(
        co_utils.fpr_softplus_proxy_func_multi_th,
        temperature=FLAGS.sigmoid_temp)
    fp_proxy_func = functools.partial(
        co_utils.fp_softplus_proxy_func_multi_th,
        temperature=FLAGS.sigmoid_temp)
  elif FLAGS.proxy_fn_cons == 'sigmoid':
    fpr_proxy_func = functools.partial(
        co_utils.fpr_sigmoid_proxy_func_multi_th,
        temperature=FLAGS.sigmoid_temp)
    fp_proxy_func = functools.partial(
        co_utils.fp_sigmoid_proxy_func_multi_th, temperature=FLAGS.sigmoid_temp)
  else:
    raise NotImplementedError('Proxy function {} not implemented'.format(
        FLAGS.proxy_fn_cons))
  proxy_funcs = dict()
  proxy_funcs['tpr'] = tpr_proxy_func
  proxy_funcs['fpr'] = fpr_proxy_func
  proxy_funcs['tp'] = tp_proxy_func
  proxy_funcs['fp'] = fp_proxy_func

  # loss and constraint ops
  if FLAGS.method == 'ce':
    loss_op = ce_loss
  elif FLAGS.method == 'pairwise':
    loss_op = functools.partial(
        pairwise_auc_loss,
        beta=FLAGS.fpr_high,
        proxy=FLAGS.proxy_fn_obj,
        temperature=FLAGS.sigmoid_temp)
  else:
    if FLAGS.proxy_fn_obj == 'sigmoid':
      loss_op = co_utils.fnr_sigmoid_proxy_func_multi_th
    elif FLAGS.proxy_fn_obj == 'softplus':
      loss_op = co_utils.fnr_softplus_proxy_func_multi_th
  constraint_op = fpr_proxy_func

  # Create summary writers
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/train'))
  eval_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/eval'))

  # log file
  fp_log = tf.io.gfile.GFile(
      os.path.join(FLAGS.model_dir, 'training_log.txt'), 'w')

  # eval performance at best validated model
  best_results = ml_collections.ConfigDict()
  best_results.aucroc_valid = -1 * np.array([np.inf])
  best_results.tprs_valid = -1 * np.array([np.inf])
  best_results.aucroc_eval_at_valid = -1 * np.array([np.inf])
  best_results.tprs_eval_at_valid = -1 * np.array([np.inf])
  best_results.eval_fprs_valid = []
  best_results.eval_tprs_valid = []

  # Main training loop.
  train_iterator = iter(train_ds)
  initial_ts = time.time()
  for it in range(FLAGS.train_iters):
    batch = next(train_iterator)
    images, labels = batch['image'], batch['label']
    loss, constraint = train_step(model, threshold_var, loss_op, constraint_op,
                                  optimizer, keras_metrics, images, labels)
    if FLAGS.method == 'ico' and (it + 1) % FLAGS.th_project_freq == 0:
      threshold_new = update_threshold(
          model, train_iterator, FLAGS.n_batches_for_threshold, training=True)
      logger.info('updating threshold: old {}, new {}'.format(
          threshold_var, threshold_new))
      threshold_var.assign(threshold_new)
    if it % 100 == 0:
      with train_summary_writer.as_default():
        tf.summary.scalar('threshold', threshold_var.numpy()[0, 0], step=it)
        tf.summary.scalar(
            'loss/train', keras_metrics['train_loss'].result().numpy(), step=it)
      info_str = ('Train Iter %s: loss=%s, constraint=%s') % (
          it, round(keras_metrics['train_loss'].result().numpy(),
                    8), round(constraint.numpy(), 4))
      logger.info(info_str)
    keras_metrics['train_loss'].reset_states()

    # Evaluation
    if (it + 1) % FLAGS.eval_freq == 0:
      best_results = eval_metrics(model, threshold_var, valid_ds, eval_ds,
                                  loss_op, constraint_op, proxy_funcs,
                                  keras_metrics, it, eval_summary_writer,
                                  best_results, logger, fp_log)

  fp_log.close()


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
