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

"""Finetunes the pre-trained model on the target set."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

from absl import app
from absl import flags
import model
import model_utils
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tensorflow.python.estimator import estimator
import tensorflow_datasets as tfds

flags.DEFINE_string(
    'model_dir',
    None,
    help=('The directory where the model and training/evaluation summaries are'
          ' stored.'))
flags.DEFINE_string(
    'warm_start_ckpt_path', None, 'The path to the checkpoint '
    'that will be used before training.')
flags.DEFINE_integer(
    'log_step_count_steps', 200, 'The number of steps at '
    'which the global step information is logged.')
flags.DEFINE_integer('train_steps', 100, 'Number of steps for training.')
flags.DEFINE_float('target_base_learning_rate', 0.001,
                   'Target base learning rate.')
flags.DEFINE_integer('train_batch_size', 256,
                     'The batch size for the target dataset.')
flags.DEFINE_float('weight_decay', 0.0005, 'The value for weight decay.')

FLAGS = flags.FLAGS


def lr_schedule():
  """Learning rate scheduling."""
  target_lr = FLAGS.target_base_learning_rate
  current_step = tf.train.get_global_step()

  if FLAGS.target_dataset == 'mnist':
    return tf.train.piecewise_constant(current_step, [
        500,
        1500,
    ], [target_lr, target_lr * 0.1, target_lr * 0.01])
  else:
    return tf.train.piecewise_constant(current_step, [
        800,
    ], [target_lr, target_lr * 0.1])


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
      avg_pool = model.conv_model(
          feature,
          mode,
          target_dataset=FLAGS.target_dataset,
          src_hw=FLAGS.src_hw,
          target_hw=FLAGS.target_hw)
      name = 'final_dense_dst'
      with tf.variable_scope('target_CLS'):
        logits = tf.layers.dense(
            inputs=avg_pool,
            units=FLAGS.src_num_classes,
            name=name,
            kernel_initializer=tf.random_normal_initializer(stddev=.05),
        )
      return logits

    logits = get_logits()
    logits = tf.cast(logits, tf.float32)

    dst_loss = tf.losses.softmax_cross_entropy(
        logits=logits,
        onehot_labels=one_hot_labels,
    )
    dst_l2_loss = FLAGS.weight_decay * tf.add_n([
        tf.nn.l2_loss(v)
        for v in tf.trainable_variables()
        if 'batch_normalization' not in v.name and 'kernel' in v.name
    ])

    loss = dst_loss + dst_l2_loss

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      cur_finetune_step = tf.train.get_global_step()
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        finetune_learning_rate = lr_schedule()
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=finetune_learning_rate,
            momentum=0.9,
            use_nesterov=True)
        train_op = tf.contrib.slim.learning.create_train_op(loss, optimizer)
        with tf.variable_scope('finetune'):
          train_op = optimizer.minimize(loss, cur_finetune_step)
    else:
      train_op = None

    eval_metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
      eval_metrics = model_utils.metric_fn(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
      with tf.control_dependencies([train_op]):
        tf.summary.scalar('classifier/finetune_lr', finetune_learning_rate)
    else:
      train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metrics,
    )

  return model_fn


def main(unused_argv):
  tf.set_random_seed(FLAGS.random_seed)

  save_checkpoints_steps = 100
  run_config_args = {
      'model_dir': FLAGS.model_dir,
      'save_checkpoints_steps': save_checkpoints_steps,
      'log_step_count_steps': FLAGS.log_step_count_steps,
      'keep_checkpoint_max': 200,
  }

  config = tf.estimator.RunConfig(**run_config_args)

  if FLAGS.warm_start_ckpt_path:
    var_names = []
    checkpoint_path = FLAGS.warm_start_ckpt_path
    reader = tf.train.NewCheckpointReader(checkpoint_path)
    for key in reader.get_variable_to_shape_map():
      keep_str = 'Momentum|global_step|finetune_global_step|Adam|final_dense_dst'
      if not re.findall('({})'.format(keep_str,), key):
        var_names.append(key)

    tf.logging.info('Warm-starting tensors: %s', sorted(var_names))

    vars_to_warm_start = var_names
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=checkpoint_path,
        vars_to_warm_start=vars_to_warm_start)
  else:
    warm_start_settings = None

  classifier = tf.estimator.Estimator(
      get_model_fn(), config=config, warm_start_from=warm_start_settings)

  def _merge_datasets(train_batch):
    feature, label = train_batch['image'], train_batch['label'],
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
      train_data = tfds.load(name=FLAGS.target_dataset, split=dataset_split)
      train_data = train_data.shuffle(1024).repeat().batch(
          FLAGS.train_batch_size)
      dataset = tf.data.Dataset.zip((train_data,))
      dataset = dataset.map(_merge_datasets)
      dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
      return dataset

    return make_input_dataset

  # pylint: disable=protected-access
  current_step = estimator._load_global_step_from_checkpoint_dir(
      FLAGS.model_dir)

  train_steps = FLAGS.train_steps
  while current_step < train_steps:
    print('Run {}'.format(current_step))
    next_checkpoint = current_step + 500
    classifier.train(input_fn=get_dataset('train'), max_steps=next_checkpoint)
    current_step = next_checkpoint


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
