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

"""Evaluates the model based on a performance metric."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import model
import model_utils
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow import estimator as tf_estimator
import tensorflow_datasets as tfds

flags.DEFINE_string('ckpt_path', '', 'Path to evaluation checkpoint')
flags.DEFINE_string('cls_dense_name', '', 'Final dense layer name')

flags.DEFINE_integer('train_batch_size', 600,
                     'The batch size for the target dataset.')

FLAGS = flags.FLAGS

NUM_EVAL_IMAGES = {
    'mnist': 10000,
    'svhn_cropped_small': 6000,
}


def get_model_fn():
  """Returns the model definition."""

  def model_fn(features, labels, mode, params):
    """Returns the model function."""
    feature = features['feature']
    labels = labels['label']
    one_hot_labels = model_utils.get_label(
        labels,
        params,
        FLAGS.src_num_classes,
        batch_size=FLAGS.train_batch_size)

    def get_logits():
      """Return the logits."""
      network_output = model.conv_model(
          feature,
          mode,
          target_dataset=FLAGS.target_dataset,
          src_hw=FLAGS.src_hw,
          target_hw=FLAGS.target_hw)
      name = FLAGS.cls_dense_name
      with tf.variable_scope('target_CLS'):
        logits = tf.layers.dense(
            inputs=network_output, units=FLAGS.src_num_classes, name=name)
      return logits

    logits = get_logits()
    logits = tf.cast(logits, tf.float32)

    dst_loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
    )
    loss = dst_loss

    eval_metrics = model_utils.metric_fn(labels, logits)

    return tf_estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=None,
        eval_metric_ops=eval_metrics,
    )

  return model_fn


def main(unused_argv):
  config = tf_estimator.RunConfig()

  classifier = tf_estimator.Estimator(get_model_fn(), config=config)

  def _merge_datasets(test_batch):
    feature, label = test_batch['image'], test_batch['label'],
    features = {
        'feature': feature,
    }
    labels = {
        'label': label,
    }
    return (features, labels)

  def get_dataset(dataset_split):
    """Returns dataset creation function."""

    def make_input_dataset():
      """Returns input dataset."""
      test_data = tfds.load(name=FLAGS.target_dataset, split=dataset_split)
      test_data = test_data.batch(FLAGS.train_batch_size)
      dataset = tf.data.Dataset.zip((test_data,))
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
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
