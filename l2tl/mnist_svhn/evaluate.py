# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Finetuning the pre-trained model on the target set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds

import model
import model_utils

flags.DEFINE_string('ckpt_path', '', 'Path to evaluation checkpoint')
flags.DEFINE_string('cls_dense_name', '', 'Final dense layer name')

flags.DEFINE_integer('train_batch_size', 128,
                     'The batch size for the target dataset.')

FLAGS = flags.FLAGS

NUM_EVAL_IMAGES = {
    'mnist': 60000,
    'svhn_cropped': 26032,
    'svhn_cropped_small': 600,
}


def get_model_fn(run_config):
  """Returns the model definition."""
  src_num_classes = 10

  def resnet_model_fn(features, labels, mode, params):
    """Returns the model function."""
    feature = features['feature']
    labels = labels['label']
    one_hot_labels = model_utils.get_label(
        labels, params, src_num_classes, batch_size=FLAGS.train_batch_size)

    def get_logits():
      """Return the logits."""
      avg_pool = model.conv_model(feature, mode)
      name = FLAGS.cls_dense_name
      with tf.variable_scope('target_CLS'):
        logits = tf.layers.dense(
            inputs=avg_pool,
            units=src_num_classes,
            name=name)
      return logits

    logits = get_logits()
    logits = tf.cast(logits, tf.float32)

    dst_loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,)
    loss = dst_loss

    eval_metrics = model_utils.metric_fn(labels, logits)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=None,
        eval_metric_ops=eval_metrics,
    )

  return resnet_model_fn


def main(unused_argv):
  config = tf.estimator.RunConfig()

  classifier = tf.estimator.Estimator(
      get_model_fn(config), config=config)

  def _merge_datasets(train_batch):
    feature, label = train_batch['image'], train_batch['label'],
    features = {'feature': feature,}
    labels = {'label': label,}
    return (features, labels)

  def get_dataset(dataset_split):
    def make_input_dataset(params):
      """Returns input dataset."""
      train_data = tfds.load(name=FLAGS.target_dataset, split=dataset_split)
      train_data = train_data.batch(FLAGS.train_batch_size)
      dataset = tf.data.Dataset.zip((train_data,))
      dataset = dataset.map(_merge_datasets)
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return dataset
    return make_input_dataset

  num_eval_images = NUM_EVAL_IMAGES[FLAGS.target_dataset]
  eval_steps = num_eval_images // FLAGS.train_batch_size

  classifier.evaluate(
      input_fn=get_dataset('test'),
      steps=eval_steps,
      checkpoint_path=FLAGS.ckpt_path,
  )


if __name__ == '__main__':
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  app.run(main)
