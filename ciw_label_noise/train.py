# coding=utf-8
# Copyright 2022 The Google Research Authors.
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

r"""Train and evaluate model: learning with noisy labels.

This code trains a model using Constrained Instance reWeighting (CIW) proposed
in https://arxiv.org/abs/2111.05428 for learning with noisy labels.
"""

from absl import app
from absl import flags
import os
import functools

import numpy as np
import tensorflow as tf
from ciw_label_noise import preact_resnet_models as resnet_models
from ciw_label_noise import utils
from ciw_label_noise import losses
from ciw_label_noise import cifar
from scipy import interpolate
import time

FLAGS = flags.FLAGS

flags.DEFINE_string('model_dir', './ciw',
                    'directory where model and logs are stored')
flags.DEFINE_string('dataset', 'cifar10', 'dataset')
flags.DEFINE_integer('num_classes', 10, 'number of classes')
flags.DEFINE_float('lr', 0.1, 'initial learning rate')
flags.DEFINE_integer('num_iters', 50000, 'number of training iterations')
flags.DEFINE_integer('decay_steps', 5000, 'decay lr every many steps.')
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_string('loss', 'ce', 'loss type: ce/dcl')
flags.DEFINE_string('div_type', 'alpha', 'divergence type: alpha divergence')
flags.DEFINE_string('w_type', 'normalized',
                    'weight normalization: normalized/unnormalized')
flags.DEFINE_float('lambda_hyp', 0.1,
                   'hyperparameter controlling the radius on the divergence')
flags.DEFINE_float('alpha', 0.1, 'alpha-parameter for alpha divergence')
flags.DEFINE_integer('eval_freq', 1000, 'eval frequency')
flags.DEFINE_string(
    'noise_type', 'none',
    'label noise type: none/random/random_flip/random_flip_next/random_flip_asym'
)
flags.DEFINE_float('noisy_frac', 0.1, 'fraction of noisy samples')
flags.DEFINE_string(
    'mixup_type', 'none',
    'type of mixup: none/simple/sample_w/weight_w/sample_weight_w/label_smoothing_w'
)
flags.DEFINE_float('mixup_alpha', 2.0, 'beta distribution parameter for mixup')
flags.DEFINE_integer('dcl_w_mixup', 0, 'recompute dcl loss after mixup')
flags.DEFINE_integer(
    'burnin', 4000,
    'burn-in iterations when all examples are uniformly weighted')
flags.DEFINE_string('div_type_cls', 'none',
                    'divergence type for class reweighting: none/l1')
flags.DEFINE_float('gamma', 0.1, 'gamma parameter for class reweighting')
flags.DEFINE_float('m_pow', 1, 'exponent for weights used for mixup')
flags.DEFINE_integer('train_on_full', 0,
                     'whether to train on full training set')
flags.DEFINE_integer('run_id', 0, 'run id')

EPOCH_SIZE = 50000  # number of data points in an epoch
IMAGE_SIZE = 32  # spatial dimension of images


def main(_):

  if tf.io.gfile.exists(FLAGS.model_dir):
    tf.io.gfile.makedirs(FLAGS.model_dir)
  save_dir = os.path.join(FLAGS.model_dir, 'model')
  tf.io.gfile.makedirs(save_dir)

  # log file
  fp_log_res = tf.io.gfile.GFile(
      os.path.join(FLAGS.model_dir, 'results_log.txt'), 'w')

  # get dataset
  train_ds, valid_ds, eval_ds = cifar.get_dataset(
      FLAGS.batch_size,
      data=FLAGS.dataset,
      num_classes=FLAGS.num_classes,
      image_size=IMAGE_SIZE,
      noise_type=FLAGS.noise_type,
      noisy_frac=FLAGS.noisy_frac,
      train_on_full=FLAGS.train_on_full)

  # build model
  model = resnet_models.create_resnet18(
      input_shape=(32, 32, 3), num_classes=FLAGS.num_classes, norm='batch')

  # loss
  if FLAGS.loss == 'ce':
    loss_op = losses.ce_loss
  else:
    loss_op = functools.partial(
        losses.div_constrained_loss,
        div_type=FLAGS.div_type,
        alpha=FLAGS.alpha,
        w_type=FLAGS.w_type,
        div_type_cls=FLAGS.div_type_cls,
        gamma=FLAGS.gamma,
        burnin=FLAGS.burnin,
        mixup_type=FLAGS.mixup_type,
        mixup_alpha=FLAGS.mixup_alpha,
        m_pow=FLAGS.m_pow,
        dcl_w_mixup=FLAGS.dcl_w_mixup)

  # set up optimizer
  boundaries = [(30 * EPOCH_SIZE) // FLAGS.batch_size,
                (80 * EPOCH_SIZE) // FLAGS.batch_size,
                (110 * EPOCH_SIZE) // FLAGS.batch_size]
  values = [0.1, 0.01, 0.001, 0.0001]
  lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries, values)
  optimizer = tf.keras.optimizers.SGD(
      learning_rate=lr_schedule, momentum=0.9, nesterov=True)

  # summary writers
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/train'))
  eval_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.model_dir, 'summaries/eval'))

  def train_step(images, labels, lambda_hyp, iteration):
    """Perform single training step."""
    with tf.GradientTape() as tape:
      if FLAGS.mixup_type != 'label_smoothing_w' and (
          FLAGS.mixup_type == 'simple' or
          (FLAGS.mixup_type != 'none' and iteration <= FLAGS.burnin)):
        images, labels = utils.mixup(images, labels, FLAGS.num_classes,
                                     FLAGS.mixup_alpha)
      logits = model([images, True])
      if FLAGS.loss == 'ce':
        loss = loss_op(labels, logits)
      else:
        loss = loss_op(
            labels,
            logits,
            lambda_hyp=lambda_hyp,
            iteration=iteration,
            model=model,
            images=images)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

  def eval_metrics(ds, lambda_hyp, iteration):
    """Evaluate accuracy on test set."""
    iterator = iter(ds)
    avg_loss = 0.
    num_samples = 0
    num_correct = 0
    for batch in iterator:
      images, labels = batch['image'], batch['label']
      logits = model([images, False])
      if FLAGS.loss == 'ce':
        avg_loss += loss_op(labels, logits).numpy()
      else:
        loss = loss_op(
            labels,
            logits,
            lambda_hyp=lambda_hyp,
            iteration=iteration,
            model=model,
            images=images,
            is_train=False)
        avg_loss += loss.numpy()
      num_samples += len(labels)
      num_correct += tf.reduce_sum(
          tf.cast(
              tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
              tf.int32)).numpy()
    avg_loss /= num_samples
    acc = num_correct / float(num_samples) * 100.
    return avg_loss, acc

  # training
  train_iterator = iter(train_ds)
  best_acc_valid = 0
  best_acc_eval_at_valid = 0
  burnin_interp_fn = interpolate.interp1d(
      [FLAGS.burnin, FLAGS.burnin + 3000, FLAGS.num_iters],
      [FLAGS.lambda_hyp * 10, FLAGS.lambda_hyp, FLAGS.lambda_hyp])

  t0 = time.time()
  for it in range(1, FLAGS.num_iters + 1):
    batch = next(train_iterator)
    images, labels = batch['image'], batch['label']
    if FLAGS.burnin > 0 and it > FLAGS.burnin:
      lambda_hyp = burnin_interp_fn(it)
    else:
      lambda_hyp = FLAGS.lambda_hyp
    loss = train_step(images, labels, lambda_hyp, it)

    if it % 100 == 0:
      with train_summary_writer.as_default():
        tf.summary.scalar('loss/train', loss.numpy(), step=it)
      info_str = 'It: {}, loss: {:.5f}, time elapsed: {:.3f}'.format(
          it, loss.numpy(),
          time.time() - t0)
      print(info_str)
      fp_log_res.write(info_str + '\n')

    if it % FLAGS.eval_freq == 0:
      loss_valid, acc_valid = eval_metrics(valid_ds, lambda_hyp, it)
      loss_eval, acc_eval = eval_metrics(eval_ds, lambda_hyp, it)
      if acc_valid > best_acc_valid:
        best_acc_valid = acc_valid
        best_acc_eval_at_valid = acc_eval
      with eval_summary_writer.as_default():
        tf.summary.scalar('loss/valid', loss_valid, step=it)
        tf.summary.scalar('loss/eval', loss_eval, step=it)
        tf.summary.scalar('acc/valid', acc_valid, step=it)
        tf.summary.scalar('acc/eval', acc_eval, step=it)
        tf.summary.scalar(
            'acc/best_eval_at_valid', best_acc_eval_at_valid, step=it)

      info_str = (
          'It: {}, lambda_hyp: {:.2f}, Valid loss: {:.3f}, Valid acc: {:.3f}, '
          'Eval loss: {:.3f}, Eval acc: {:.3f}, Best Valid acc: {:.3f}, Best '
          'Eval acc: {:.3f}, time elapsed: {:.3f}').format(
              it, lambda_hyp, loss_valid, acc_valid, loss_eval, acc_eval,
              best_acc_valid, best_acc_eval_at_valid,
              time.time() - t0)
      print(info_str)
      fp_log_res.write(info_str + '\n')

  fp_log_res.close()


if __name__ == '__main__':
  app.run(main)
