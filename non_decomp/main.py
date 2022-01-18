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

"""Code for the paper: "Training Over-parameterized Models ...", NeurIPS 2021.

Paper: https://arxiv.org/pdf/2107.04641

Example usage:
    $ python -m non_decomp.main --dataset=cifar10-lt
"""

import os
from absl import app
from absl import flags
from non_decomp import models
from non_decomp import utils
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10-lt', 'Dataset to use.')
flags.DEFINE_string('data_home', 'non_decomp/data',
                    'Directory where data files are stored.')
flags.DEFINE_integer('train_batch_size', 128, 'Train batch size.')
flags.DEFINE_integer('vali_batch_size', 4096, 'Vali batch size.')
flags.DEFINE_integer('test_batch_size', 100, 'Test batch size.')
flags.DEFINE_enum('mode', 'proposed',
                  ['erm', 'balanced', 'reweighted', 'proposed'],
                  'Logit-adjustment mode. See paper for details.')
flags.DEFINE_float('tau', 1.0, 'Tau parameter for logit adjustment.')
flags.DEFINE_float('eg_lr', 0.1,
                   'Learning rate for exponentiated gradient update on class'
                   'weights.')
flags.DEFINE_integer('update_freq', 32, 'Update class weights once in '
                     '"update_freq" number of gradient steps.')
flags.DEFINE_string('tb_log_dir', 'non_decomp/log',
                    'Path to write Tensorboard summaries.')


def main(_):

  # Prepare the datasets.
  dataset = utils.dataset_mappings()[FLAGS.dataset]
  num_classes = dataset.num_classes
  batches_per_epoch = int(dataset.num_train / FLAGS.train_batch_size)
  train_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                          FLAGS.train_batch_size, split='train')
  vali_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                         FLAGS.vali_batch_size, split='vali')
  test_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                         FLAGS.test_batch_size, split='test')
  vali_iter = iter(vali_dataset)

  # Model to be trained.
  model = models.cifar_resnet32(dataset.num_classes)

  # Read the base probabilities to use for logit adjustment.
  base_probs_path = os.path.join(FLAGS.data_home,
                                 f'{FLAGS.dataset}_base_probs.txt')
  try:
    with tf.io.gfile.GFile(base_probs_path, mode='r') as fin:
      base_probs = np.loadtxt(fin)
  except tf.errors.NotFoundError:
    if FLAGS.mode != 'erm':
      raise app.UsageError(
          f'{base_probs_path} must exist when `mode` is {FLAGS.mode}.')
    else:
      base_probs = None

  # Build the loss function.
  if FLAGS.mode == 'erm':
    loss_type = 'standard'
  elif FLAGS.mode == 'reweighted':
    loss_type = 'reweighted'
  else:
    loss_type = 'logit_adjusted'
  loss_fn = utils.build_loss_fn(loss_type, base_probs, FLAGS.tau)

  # Prepare the metrics, the optimizer, etc.
  train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

  vali_recall_list = [tf.keras.metrics.Recall() for _ in range(num_classes)]
  vali_min_recall = utils.MinRecall(vali_recall_list)

  test_recall_list = [tf.keras.metrics.Recall() for _ in range(num_classes)]
  test_min_recall = utils.MinRecall(test_recall_list)

  learning_rate = utils.LearningRateSchedule(
      schedule=dataset.lr_schedule,
      steps_per_epoch=batches_per_epoch,
      base_learning_rate=0.1,
  )

  optimizer = tf.keras.optimizers.SGD(
      learning_rate,
      momentum=0.9,
      nesterov=True,
  )

  # Prepare Tensorboard summary writers.
  train_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.tb_log_dir, 'train'))
  test_summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.tb_log_dir, 'test'))

  # Multiplier variables.
  if FLAGS.mode == 'proposed':
    class_weights = tf.Variable(
        tf.ones(num_classes) / tf.cast(num_classes, tf.float32),
        dtype=tf.float32)
  else:
    class_weights = None

  if FLAGS.mode == 'proposed':
    # Setup for tracking per-class false negative rates (i.e. 1 - recall).
    fnrs = utils.FalseNegativeRates(num_classes)

  # Train for num_epochs iterations over the train set.
  for epoch in range(dataset.num_epochs):

    # Iterate over the train dataset.
    for step, (x, y) in enumerate(train_dataset):
      labels_vec = tf.one_hot(y, depth=num_classes)

      with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(labels_vec, logits, class_weights)
        loss_value = loss_value + tf.reduce_sum(model.losses)

      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      if step % FLAGS.update_freq == 0:
        x_val, y_val = next(vali_iter)
        labels_vec_val = tf.one_hot(y_val, depth=num_classes)
        logits_val = model(x_val, training=False)

        if FLAGS.mode == 'proposed':
          # Exponentiated gradient update on class weights using vali batch.
          fnrs.update_state(labels_vec_val, logits_val)
          exp_class_weights = class_weights * tf.math.exp(
              FLAGS.eg_lr * fnrs.result())
          class_weights.assign(
              exp_class_weights / tf.reduce_sum(exp_class_weights))

        # Track per-class recall on vali data.
        preds_vec_val = tf.one_hot(
            tf.argmax(logits_val, axis=1), depth=num_classes)
        for ii in range(num_classes):
          vali_recall_list[ii].update_state(
              labels_vec_val[:, ii], preds_vec_val[:, ii])

      train_acc_metric.update_state(y, logits)

      # Log every 1000 batches.
      if step % 10 == 0:
        print(f'Training loss (for one batch) at step {epoch} / {step}: '
              f'{loss_value:.4f}')
        vali_perf = vali_min_recall.result()
        print(f'Validation min recall (moving average) at step {epoch} / '
              f'{step}: {vali_perf:.4f}')
        if class_weights is not None:
          print(fnrs.result().numpy())
          print(class_weights.numpy())
        with train_summary_writer.as_default():
          tf.summary.scalar(
              'batch loss', loss_value, step=epoch * batches_per_epoch + step)
          tf.summary.scalar(
              'min recall', vali_perf, step=epoch * batches_per_epoch + step)

    # Display train metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    print(f'Training accuracy over epoch: {train_acc:.4f}')
    with train_summary_writer.as_default():
      tf.summary.scalar(
          'train accuracy', train_acc, step=(epoch + 1) * batches_per_epoch)

    # Display validation metrics at the end of each epoch.
    vali_perf = vali_min_recall.result()
    for ii in range(num_classes):
      vali_recall_list[ii].reset_states()
    print(f'Validation min recall over epoch: {vali_perf:.4f}')
    with train_summary_writer.as_default():
      tf.summary.scalar(
          'vali min recall', vali_perf, step=(epoch + 1) * batches_per_epoch)

    # Run a test loop at the end of each epoch.
    for x_test, y_test in test_dataset:
      logits_test = model(x_test, training=False)
      test_acc_metric.update_state(y_test, logits_test)

      labels_vec_test = tf.one_hot(y_test, depth=num_classes)
      preds_vec_test = tf.one_hot(
          tf.argmax(logits_test, axis=1), depth=num_classes)
      for ii in range(num_classes):
        test_recall_list[ii].update_state(
            labels_vec_test[:, ii], preds_vec_test[:, ii])

    # Display test metrics.
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print(f'Test accuracy: {test_acc:.4f}')
    with test_summary_writer.as_default():
      tf.summary.scalar(
          'accuracy', test_acc, step=(epoch + 1) * batches_per_epoch)

if __name__ == '__main__':
  app.run(main)
