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

"""Code for the paper: Long-tail learning via logit adjustment, ICLR 2021.

Paper: https://arxiv.org/abs/2007.07314

Example usage:
    $ python -m logit_adjustment.main --dataset=cifar10-lt
"""

import os

from absl import app
from absl import flags
from logit_adjustment import models
from logit_adjustment import utils
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'cifar10-lt', 'Dataset to use.')
flags.DEFINE_string('data_home', 'logit_adjustment/data',
                    'Directory where data files are stored.')
flags.DEFINE_integer('train_batch_size', 128, 'Train batch size.')
flags.DEFINE_integer('test_batch_size', 100, 'Test batch size.')
flags.DEFINE_enum('mode', 'posthoc', ['baseline', 'posthoc', 'loss'],
                  'Logit-adjustment mode. See paper for details.')
flags.DEFINE_float('tau', 1.0, 'Tau parameter for logit adjustment.')
flags.DEFINE_string('tb_log_dir', 'logit_adjustment/log',
                    'Path to write Tensorboard summaries.')


def main(_):

  # Prepare the datasets.
  dataset = utils.dataset_mappings()[FLAGS.dataset]
  batches_per_epoch = int(dataset.num_train / FLAGS.train_batch_size)
  train_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                          FLAGS.train_batch_size, True)
  test_dataset = utils.create_tf_dataset(dataset, FLAGS.data_home,
                                         FLAGS.test_batch_size, False)

  # Model to be trained.
  model = models.cifar_resnet32(dataset.num_classes)

  # Read the base probabilities to use for logit adjustment.
  base_probs_path = os.path.join(FLAGS.data_home,
                                 f'{FLAGS.dataset}_base_probs.txt')
  try:
    with tf.io.gfile.GFile(base_probs_path, mode='r') as fin:
      base_probs = np.loadtxt(fin)
  except tf.errors.NotFoundError:
    if FLAGS.mode in ['posthoc', 'loss']:
      raise app.UsageError(
          f'{base_probs_path} must exist when `mode` is {FLAGS.mode}.')
    else:
      base_probs = None

  # Build the loss function.
  loss_fn = utils.build_loss_fn(FLAGS.mode == 'loss', base_probs, FLAGS.tau)

  # Prepare the metrics, the optimizer, etc.
  train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  test_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
  posthoc_adjusting = FLAGS.mode == 'posthoc'
  if posthoc_adjusting:
    test_adj_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

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

  # Train for num_epochs iterations over the train set.
  for epoch in range(dataset.num_epochs):

    # Iterate over the train dataset.
    for step, (x, y) in enumerate(train_dataset):
      with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss_value = loss_fn(y, logits)
        loss_value = loss_value + tf.reduce_sum(model.losses)

      grads = tape.gradient(loss_value, model.trainable_weights)
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      train_acc_metric.update_state(y, logits)

      # Log every 1000 batches.
      if step % 1000 == 0:
        print(f'Training loss (for one batch) at step {step}: {loss_value:.4f}')
        with train_summary_writer.as_default():
          tf.summary.scalar(
              'batch loss', loss_value, step=epoch * batches_per_epoch + step)

    # Display train metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    train_acc_metric.reset_states()
    print(f'Training accuracy over epoch: {train_acc:.4f}')
    with train_summary_writer.as_default():
      tf.summary.scalar(
          'accuracy', train_acc, step=(epoch + 1) * batches_per_epoch)

    # Run a test loop at the end of each epoch.
    for x, y in test_dataset:
      logits = model(x, training=False)
      test_acc_metric.update_state(y, logits)

      if posthoc_adjusting:
        # Posthoc logit-adjustment.
        adjusted_logits = logits - tf.math.log(
            tf.cast(base_probs**FLAGS.tau + 1e-12, dtype=tf.float32))
        test_adj_acc_metric.update_state(y, adjusted_logits)

    # Display test metrics.
    test_acc = test_acc_metric.result()
    test_acc_metric.reset_states()
    print(f'Test accuracy: {test_acc:.4f}')
    with test_summary_writer.as_default():
      tf.summary.scalar(
          'accuracy', test_acc, step=(epoch + 1) * batches_per_epoch)

    if posthoc_adjusting:
      test_adj_acc = test_adj_acc_metric.result()
      test_adj_acc_metric.reset_states()
      print(f'Logit-adjusted test accuracy: {test_adj_acc:.4f}')

      with test_summary_writer.as_default():
        tf.summary.scalar(
            'logit-adjusted accuracy',
            test_adj_acc,
            step=(epoch + 1) * batches_per_epoch)


if __name__ == '__main__':
  app.run(main)
