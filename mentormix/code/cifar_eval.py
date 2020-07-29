# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Evaluates a trained MentorMix model.

See the README.md file for compilation and running instructions.
"""

import math
import os
import cifar_data_provider
import resnet_model
import tensorflow as tf
import tensorflow.contrib.slim as slim

flags = tf.app.flags

flags.DEFINE_integer('batch_size', 25, 'The number of images in each batch.')

flags.DEFINE_string('data_dir', '', 'Data dir')

flags.DEFINE_string('dataset_name', 'cifar100', 'cifar10 or cifar100')

flags.DEFINE_string('studentnet', 'resnet32', 'Resnet 32 model.')

flags.DEFINE_string('master', None, 'BNS name of the TensorFlow master to use.')

flags.DEFINE_string('checkpoint_dir', '',
                    'Directory where the model was written to.')

flags.DEFINE_string('eval_dir', '', 'Directory where the results are saved to.')

flags.DEFINE_integer(
    'eval_interval_secs', 600,
    'The frequency, in seconds, with which evaluation is run.')

flags.DEFINE_string('split_name', 'test', """Either 'train' or 'test'.""")

flags.DEFINE_string('output_csv_file', '',
                    'The csv file where the results are saved.')

flags.DEFINE_string('device_id', '0', 'GPU device ID to run the job.')

FLAGS = flags.FLAGS

# Turn this on if there are no log outputs.
tf.logging.set_verbosity(tf.logging.INFO)


def eval_resnet():
  """Evaluates the resnet model."""
  if not os.path.exists(FLAGS.eval_dir):
    os.makedirs(FLAGS.eval_dir)
  g = tf.Graph()
  with g.as_default():
    tf_global_step = slim.get_or_create_global_step()
    (images, one_hot_labels, num_samples,
     num_of_classes) = cifar_data_provider.provide_resnet_data(
         FLAGS.dataset_name,
         FLAGS.split_name,
         FLAGS.batch_size,
         dataset_dir=FLAGS.data_dir,
         num_epochs=None)

    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=num_of_classes,
        min_lrn_rate=0.0001,
        lrn_rate=0,
        num_residual_units=5,
        use_bottleneck=False,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer='mom')

    # Define the model:
    images.set_shape([FLAGS.batch_size, 32, 32, 3])
    resnet = resnet_model.ResNet(hps, images, one_hot_labels, mode='eval')

    with tf.variable_scope('ResNet32'):
      logits = resnet.build_model()

    variable_averages = tf.train.ExponentialMovingAverage(
        0.9999, tf_global_step)
    for var in tf.get_collection('moving_vars'):
      tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
    for var in slim.get_model_variables():
      tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

    variables_to_restore = variable_averages.variables_to_restore()

    total_loss = tf.nn.softmax_cross_entropy_with_logits(
        labels=one_hot_labels, logits=logits)
    total_loss = tf.reduce_mean(total_loss, name='xent')

    slim.summaries.add_scalar_summary(
        total_loss, 'total_loss', print_summary=True)

    # Define the metrics:
    predictions = tf.argmax(logits, 1)
    labels = tf.argmax(one_hot_labels, 1)

    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': tf.metrics.accuracy(predictions, labels),
    })

    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(
          value, name, prefix='eval', print_summary=True)

    # This ensures that we make a single pass over all of the data.
    num_batches = math.ceil(num_samples / float(FLAGS.batch_size))

    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        variables_to_restore=variables_to_restore,
        eval_interval_secs=FLAGS.eval_interval_secs)


def main(_):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.device_id
  if FLAGS.studentnet == 'resnet32':
    eval_resnet()
  else:
    tf.logging.error('unknown backbone student network %s', FLAGS.studentnet)


if __name__ == '__main__':
  tf.app.run()
